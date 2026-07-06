# SPDX-License-Identifier: Apache-2.0
"""Belief-gated page selection policies (host-side control plane).

This module is the heart of the experiment. It turns a sequence's *full* logical
page list into a *compact* chronological subset that the existing
``paged_attention_primitive`` can attend over exactly (see
``IMPLEMENTATION_AUDIT.md`` §4 and ``MATH.md`` for the correctness proof).

The decode attention kernel (``pagedattention.metal``) is position-agnostic with
respect to the block table for RoPE-only decode: it walks the block table by
iteration index and masks causally **by token count** (``token_idx >=
effective_context_len``). RoPE is baked into cached K at write time. Therefore a
chronological subset of physical pages + a matching ``context_len`` produces
exact attention over exactly the selected tokens — provided the
**full-page-except-tail invariant** holds:

    Every selected page must be a full BLOCK_SIZE page, except at most the final
    selected page (which may be the sequence's original partial tail). The
    reported context_len must equal the count of valid tokens across the
    selected pages.

Because in a live sequence only the *last* logical page is ever partial, any
chronological subset that keeps the partial tail page last (or selects only full
pages) satisfies this automatically. ``build_selection`` enforces it and raises
on violation, so a buggy policy fails loudly instead of returning wrong-but-
plausible attention.

All policies are **equal-budget**: given a page budget ``J`` they select exactly
``J`` pages (clamped to ``[required, P]``), so memory/latency are compared at
identical retention. The ``full_pages`` policy ignores the budget and selects
everything (the numeric-equivalence reference).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequencePages:
    """Full logical page layout of one sequence at one decode step.

    Attributes:
        block_ids: physical page ids in chronological order, length P.
        context_len: true number of valid KV tokens (<= P*block_size).
        block_size: tokens per page (B).
    """

    block_ids: tuple[int, ...]
    context_len: int
    block_size: int

    @property
    def num_pages(self) -> int:
        return len(self.block_ids)

    def valid_tokens(self, page_index: int) -> int:
        """Valid tokens in logical page ``page_index`` (full except the tail)."""
        last = self.num_pages - 1
        if page_index < last:
            return self.block_size
        # Final page: remainder (block_size if context_len is an exact multiple).
        rem = self.context_len - last * self.block_size
        return rem if rem > 0 else self.block_size

    def __post_init__(self) -> None:
        expected_pages = (self.context_len + self.block_size - 1) // self.block_size
        if self.num_pages != expected_pages:
            raise ValueError(
                f"block_ids has {self.num_pages} pages but context_len="
                f"{self.context_len} with block_size={self.block_size} implies "
                f"{expected_pages} pages"
            )


@dataclass(frozen=True)
class PageSelection:
    """Result of a page-selection policy for one sequence.

    ``selected_block_ids`` is a chronological subset of the full block ids;
    ``selected_context_len`` is the count of valid tokens across them (what the
    kernel must receive so causal tail-masking is exact).
    """

    selected_block_ids: tuple[int, ...]
    selected_context_len: int
    selected_page_indices: tuple[int, ...]  # indices into the original page list

    @property
    def num_selected_pages(self) -> int:
        return len(self.selected_block_ids)


def build_selection(pages: SequencePages, page_indices: Sequence[int]) -> PageSelection:
    """Build a validated :class:`PageSelection` from chosen logical page indices.

    Enforces the full-page-except-tail invariant and chronological order. Raises
    ``ValueError`` on any violation so a bad policy cannot silently corrupt
    attention.
    """
    idx = sorted(set(int(i) for i in page_indices))
    if not idx:
        raise ValueError("page selection must be non-empty")
    if idx[0] < 0 or idx[-1] >= pages.num_pages:
        raise ValueError(f"page indices {idx} out of range [0,{pages.num_pages})")

    # Invariant: every selected page except the LAST selected must be full.
    last_sel = idx[-1]
    for i in idx[:-1]:
        if pages.valid_tokens(i) != pages.block_size:
            raise ValueError(
                f"invariant violation: interior selected page {i} is partial "
                f"({pages.valid_tokens(i)} < {pages.block_size}); only the final "
                f"selected page may be partial"
            )

    selected_block_ids = tuple(pages.block_ids[i] for i in idx)
    selected_context_len = sum(pages.valid_tokens(i) for i in idx)
    # Sanity: the only way context_len is not a multiple of block_size is if the
    # last selected page is the original partial tail.
    if selected_context_len % pages.block_size != 0 and last_sel != pages.num_pages - 1:
        raise ValueError(
            "non-multiple context_len but last selected page is not the tail"
        )
    return PageSelection(selected_block_ids, selected_context_len, tuple(idx))


# ---------------------------------------------------------------------------
# Budget + always-keep machinery
# ---------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """Shared configuration controlling the always-keep set and the budget.

    The always-keep set (sinks + recent window + required evidence pages + the
    partial tail) is honored by *every* non-full policy, per the spec.
    """

    budget_pages: int = 0  # J; 0 means "use budget_fraction"
    budget_fraction: float = 1.0  # fraction of P, used when budget_pages == 0
    recent_window: int = 1  # always keep this many most-recent pages
    sink_pages: int = 0  # always keep this many earliest pages
    seed: int = 0

    def resolve_budget(self, num_pages: int) -> int:
        if self.budget_pages > 0:
            j = self.budget_pages
        else:
            j = max(1, round(self.budget_fraction * num_pages))
        return max(1, min(j, num_pages))


def _floor(pages: SequencePages, cfg: PolicyConfig) -> set[int]:
    """The mandatory floor kept by EVERY non-full policy, identical across
    policies so it does not bias the comparison: the current partial tail page,
    a recent-page window, and sink pages. Evidence pages are NOT added here —
    they compete for the discretionary budget so all policies use exactly J
    pages (the spec's 'identical page budgets')."""
    P = pages.num_pages
    keep: set[int] = {P - 1}  # current partial tail (query must see itself)
    keep.update(range(min(cfg.sink_pages, P)))  # sinks
    keep.update(range(max(0, P - cfg.recent_window), P))  # recent window
    return keep


def _select_exact(pages: SequencePages, cfg: PolicyConfig,
                  preference: Sequence[int]) -> PageSelection:
    """Return EXACTLY J pages: the mandatory floor, then the highest-preference
    non-floor pages until the budget J is reached (J clamped to [|floor|, P]).
    ``preference`` is the policy's ranked list of candidate page indices,
    best-first; the floor is always included regardless of ranking."""
    P = pages.num_pages
    budget = cfg.resolve_budget(P)
    floor = _floor(pages, cfg)
    budget = max(budget, len(floor))  # cannot go below the mandatory floor
    sel = set(floor)
    for i in preference:
        if len(sel) >= budget:
            break
        if 0 <= i < P:
            sel.add(i)
    # If preference exhausted before budget, top up with most-recent pages.
    if len(sel) < budget:
        for i in range(P - 1, -1, -1):
            if len(sel) >= budget:
                break
            sel.add(i)
    return build_selection(pages, sorted(sel))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
#
# Each policy is a function (pages, cfg, ctx) -> PageSelection. ``ctx`` carries
# policy-specific signals (attention proxy scores, belief evidence pages, etc.).


@dataclass
class SelectionContext:
    """Per-step signals available to policies (all from <= current time)."""

    # attention_proxy: accumulated attention mass per logical page (incremental).
    page_attention_mass: Optional[list[float]] = None
    # keyword/entity and belief policies: page indices flagged as relevant.
    keyword_pages: Sequence[int] = field(default_factory=tuple)
    oracle_pages: Sequence[int] = field(default_factory=tuple)
    inferred_pages: Sequence[int] = field(default_factory=tuple)


def full_pages(pages: SequencePages, cfg: PolicyConfig,
               ctx: SelectionContext) -> PageSelection:
    """Select every page — the numeric-equivalence reference (100% budget)."""
    return build_selection(pages, range(pages.num_pages))


def recent_pages(pages: SequencePages, cfg: PolicyConfig,
                 ctx: SelectionContext) -> PageSelection:
    """Keep the J most-recent pages (a strong, cheap locality baseline)."""
    P = pages.num_pages
    return _select_exact(pages, cfg, list(range(P - 1, -1, -1)))


def seeded_random_pages(pages: SequencePages, cfg: PolicyConfig,
                        ctx: SelectionContext) -> PageSelection:
    """Deterministic random subset. Null hypothesis for 'does *which* pages
    matter, or just how many?'"""
    P = pages.num_pages
    rng = random.Random(cfg.seed * 1_000_003 + P)  # deterministic, length-varied
    candidates = list(range(P))
    rng.shuffle(candidates)
    return _select_exact(pages, cfg, candidates)


def attention_proxy_pages(pages: SequencePages, cfg: PolicyConfig,
                          ctx: SelectionContext) -> PageSelection:
    """Heavy-hitter style: keep pages with the most accumulated attention mass.

    ``ctx.page_attention_mass`` is maintained incrementally by the harness, so
    this is O(P) per step, not a re-scan of history."""
    P = pages.num_pages
    mass = ctx.page_attention_mass or [0.0] * P
    order = sorted(range(P), key=lambda i: mass[i] if i < len(mass) else 0.0,
                   reverse=True)
    return _select_exact(pages, cfg, order)


def keyword_entity_pages(pages: SequencePages, cfg: PolicyConfig,
                         ctx: SelectionContext) -> PageSelection:
    """Non-belief heuristic: prefer pages flagged by surface keyword/entity
    match. This is the falsifier baseline (#1): inferred belief must beat THIS."""
    P = pages.num_pages
    flagged = [i for i in ctx.keyword_pages if 0 <= i < P]
    rest = [i for i in range(P - 1, -1, -1) if i not in set(flagged)]
    return _select_exact(pages, cfg, flagged + rest)


def oracle_belief_pages(pages: SequencePages, cfg: PolicyConfig,
                        ctx: SelectionContext) -> PageSelection:
    """Upper bound on the *retention objective*: prefix-derived task state picks
    the evidence pages (no future info). Separates page-selection failure from
    belief-inference failure (falsifier #4)."""
    P = pages.num_pages
    flagged = [i for i in ctx.oracle_pages if 0 <= i < P]
    rest = [i for i in range(P - 1, -1, -1) if i not in set(flagged)]
    return _select_exact(pages, cfg, flagged + rest)


def inferred_belief_pages(pages: SequencePages, cfg: PolicyConfig,
                          ctx: SelectionContext) -> PageSelection:
    """The hypothesis: online-inferred belief evidence picks the pages."""
    P = pages.num_pages
    flagged = [i for i in ctx.inferred_pages if 0 <= i < P]
    rest = [i for i in range(P - 1, -1, -1) if i not in set(flagged)]
    return _select_exact(pages, cfg, flagged + rest)


POLICIES: dict[str, Callable[[SequencePages, PolicyConfig, SelectionContext],
                             PageSelection]] = {
    "full_pages": full_pages,
    "recent_pages": recent_pages,
    "seeded_random_pages": seeded_random_pages,
    "attention_proxy_pages": attention_proxy_pages,
    "keyword_entity_pages": keyword_entity_pages,
    "oracle_belief_pages": oracle_belief_pages,
    "inferred_belief_pages": inferred_belief_pages,
}


def select(policy: str, pages: SequencePages, cfg: PolicyConfig,
           ctx: Optional[SelectionContext] = None) -> PageSelection:
    """Dispatch to a named policy."""
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; known: {sorted(POLICIES)}")
    return POLICIES[policy](pages, cfg, ctx or SelectionContext())
