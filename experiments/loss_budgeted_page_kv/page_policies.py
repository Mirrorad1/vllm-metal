# SPDX-License-Identifier: Apache-2.0
"""Page-selection policies for the loss-budgeted page KV cache (KV-only).

Every non-full policy is EQUAL-BUDGET (exactly J pages) with an identical
mandatory floor (current tail + recent window + sinks), so policies are compared
at the same memory footprint and differ only in how they spend the discretionary
budget. The full-page-except-tail invariant (MATH.md §2 / IMPLEMENTATION_AUDIT.md
§3) is enforced by build_selection — a bad policy fails loudly, never returns
wrong-but-plausible attention.

Policies are pure ranking functions over per-page SIGNALS supplied by the harness
(attention mass, KL-damage, page norm). No semantics, no future leakage — the
harness only ever passes signals derived from ≤ t information (oracle damage is
computed from full-cache *logits* on calibration prompts, never from answer
labels; flagged ORACLE).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class SequencePages:
    block_ids: tuple[int, ...]
    context_len: int
    block_size: int

    @property
    def num_pages(self) -> int:
        return len(self.block_ids)

    def valid_tokens(self, i: int) -> int:
        last = self.num_pages - 1
        if i < last:
            return self.block_size
        rem = self.context_len - last * self.block_size
        return rem if rem > 0 else self.block_size

    def __post_init__(self):
        exp = (self.context_len + self.block_size - 1) // self.block_size
        if self.num_pages != exp:
            raise ValueError(f"{self.num_pages} pages != ceil({self.context_len}/"
                             f"{self.block_size})={exp}")


@dataclass(frozen=True)
class PageSelection:
    selected_block_ids: tuple[int, ...]
    selected_context_len: int
    selected_page_indices: tuple[int, ...]

    @property
    def num_selected_pages(self) -> int:
        return len(self.selected_block_ids)


def build_selection(pages: SequencePages, idx: Sequence[int]) -> PageSelection:
    idx = sorted(set(int(i) for i in idx))
    if not idx:
        raise ValueError("empty selection")
    if idx[0] < 0 or idx[-1] >= pages.num_pages:
        raise ValueError(f"indices {idx} out of range [0,{pages.num_pages})")
    for i in idx[:-1]:  # interior pages must be full
        if pages.valid_tokens(i) != pages.block_size:
            raise ValueError(f"interior page {i} partial; only last may be partial")
    bids = tuple(pages.block_ids[i] for i in idx)
    clen = sum(pages.valid_tokens(i) for i in idx)
    return PageSelection(bids, clen, tuple(idx))


@dataclass
class PolicyConfig:
    budget_fraction: float = 1.0
    budget_pages: int = 0
    recent_window: int = 1
    sink_pages: int = 1
    seed: int = 0

    def resolve_budget(self, P: int) -> int:
        j = self.budget_pages if self.budget_pages > 0 else max(1, round(self.budget_fraction * P))
        return max(1, min(j, P))


@dataclass
class Signals:
    """Per-page signals from the harness (length == num_pages, or None)."""
    attention_mass: Optional[list[float]] = None
    damage: Optional[list[float]] = None        # ORACLE KL-damage per page
    page_norm: Optional[list[float]] = None      # ||K|| proxy per page
    reuse_count: Optional[list[float]] = None


def _floor(pages: SequencePages, cfg: PolicyConfig) -> set[int]:
    P = pages.num_pages
    keep = {P - 1}
    keep.update(range(min(cfg.sink_pages, P)))
    keep.update(range(max(0, P - cfg.recent_window), P))
    return keep


def _select_exact(pages: SequencePages, cfg: PolicyConfig,
                  preference: Sequence[int]) -> PageSelection:
    P = pages.num_pages
    budget = max(cfg.resolve_budget(P), len(_floor(pages, cfg)))
    sel = set(_floor(pages, cfg))
    for i in preference:
        if len(sel) >= budget:
            break
        if 0 <= i < P:
            sel.add(i)
    if len(sel) < budget:
        for i in range(P - 1, -1, -1):
            if len(sel) >= budget:
                break
            sel.add(i)
    return build_selection(pages, sorted(sel))


def _by_score(P: int, score: Optional[list[float]]) -> list[int]:
    s = score or [0.0] * P
    return sorted(range(P), key=lambda i: s[i] if i < len(s) else 0.0, reverse=True)


# --- policies -------------------------------------------------------------

def full_pages(p, cfg, sig):
    return build_selection(p, range(p.num_pages))


def recent_pages(p, cfg, sig):
    return _select_exact(p, cfg, list(range(p.num_pages - 1, -1, -1)))


def seeded_random_pages(p, cfg, sig):
    rng = random.Random(cfg.seed * 1_000_003 + p.num_pages)
    cand = list(range(p.num_pages))
    rng.shuffle(cand)
    return _select_exact(p, cfg, cand)


def sink_recent_pages(p, cfg, sig):
    # floor already includes sinks+recent+tail; fill remaining alternating
    # earliest-then-latest so the discretionary budget is sink/recent-biased.
    P = p.num_pages
    order = []
    lo, hi = 0, P - 1
    while lo <= hi:
        order.append(lo); order.append(hi); lo += 1; hi -= 1
    return _select_exact(p, cfg, order)


def attention_proxy_pages(p, cfg, sig):
    return _select_exact(p, cfg, _by_score(p.num_pages, sig.attention_mass))


def page_norm_pages(p, cfg, sig):
    return _select_exact(p, cfg, _by_score(p.num_pages, sig.page_norm))


def loss_budgeted_oracle(p, cfg, sig):
    """Keep the highest-KL-damage pages (oracle: damage from full-cache logits on
    calibration, no answer labels). Diagnostic, not deployable."""
    if sig.damage is None:
        raise ValueError("loss_budgeted_oracle requires per-page damage signal")
    return _select_exact(p, cfg, _by_score(p.num_pages, sig.damage))


def loss_budgeted_online(p, cfg, sig):
    """Causal online approximation of page damage from runtime signals only:
    z-normalized blend of attention mass, page norm, and recency. No labels, no
    future tokens."""
    P = p.num_pages
    import numpy as np
    def zn(x):
        if x is None:
            return np.zeros(P)
        a = np.asarray(x[:P], dtype=np.float64)
        if a.size < P:
            a = np.pad(a, (0, P - a.size))
        sd = a.std()
        return (a - a.mean()) / sd if sd > 1e-9 else a * 0.0
    recency = np.linspace(0.0, 1.0, P)  # later pages slightly favored
    score = zn(sig.attention_mass) + 0.5 * zn(sig.page_norm) + 0.25 * recency
    return _select_exact(p, cfg, list(np.argsort(-score)))


POLICIES: dict[str, Callable] = {
    "full_pages": full_pages,
    "recent_pages": recent_pages,
    "seeded_random_pages": seeded_random_pages,
    "sink_recent_pages": sink_recent_pages,
    "attention_proxy_pages": attention_proxy_pages,
    "page_norm_pages": page_norm_pages,
    "loss_budgeted_oracle": loss_budgeted_oracle,
    "loss_budgeted_online": loss_budgeted_online,
}


def select(policy: str, pages: SequencePages, cfg: PolicyConfig,
           sig: Optional[Signals] = None) -> PageSelection:
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}")
    return POLICIES[policy](pages, cfg, sig or Signals())
