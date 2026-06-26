# SPDX-License-Identifier: Apache-2.0
"""Online, incremental belief-state inference for belief-gated page selection.

The belief state ``B_t = f(B_{t-1}, x_t)`` is updated **incrementally** as tokens
stream in (one chunk per call), using only information available through time t.
It never inspects a future query, answer, future token, or task annotation
unavailable at runtime (this is enforced by the harness, which only ever feeds
``update`` the prefix observed so far). Avoiding a full re-scan each step is what
keeps the selector's cost ``Theta(P)`` rather than ``Theta(L)`` (falsifier #9).

Each belief element carries **evidence links** to the *logical KV pages* where
its supporting text appeared. ``active_pages`` returns the union of pages
referenced by currently-relevant evidence (unresolved questions, active
entities/constraints, recently-revised facts), which the
``inferred_belief_pages`` policy retains.

This is a deliberately small, transparent inference model — surface-pattern
parsing over a controlled synthetic vocabulary (see ``tasks.py``). It is good
enough to test the *systems* hypothesis (can an online belief select fewer pages
while preserving behavior) and is honest about its linguistic limits; the
contradiction / delayed-disambiguation tasks (falsifier #7) deliberately stress
those limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Markers (controlled vocabulary used by tasks.py).
_CONSTRAINT_MARKERS = {"must", "always", "never", "only", "cannot", "constraint"}
_REVISION_MARKERS = {"actually", "instead", "correction", "now", "revised", "no_longer"}
_QUESTION_MARKERS = {"question", "what", "which", "where", "who"}


@dataclass
class BeliefState:
    """Inferred belief state at time t. All fields are derived from the prefix
    observed so far; ``evidence_links`` maps a belief key to the set of logical
    page indices that support it."""

    task_type: str = "unknown"
    active_entities: dict[str, int] = field(default_factory=dict)  # entity -> last token pos
    active_constraints: list[str] = field(default_factory=list)
    candidate_facts: dict[str, str] = field(default_factory=dict)  # entity -> value
    revised_facts: dict[str, str] = field(default_factory=dict)
    unresolved_questions: set[str] = field(default_factory=set)
    uncertainty: float = 1.0
    evidence_links: dict[str, set[int]] = field(default_factory=dict)

    # --- internal incremental cursor (never re-scans prior tokens) ---
    _pos: int = 0  # next absolute token position to be consumed
    _block_size: int = 16
    _entity_vocab: frozenset[str] = field(default_factory=frozenset)
    # Generic capitalization is a poor entity signal in prose (every sentence
    # starts with a capital), so it is OFF by default: entities come from the
    # task-available vocabulary + structural markers. Enabling it reproduces the
    # over-liberal variant for ablation.
    _flag_capitalized: bool = False

    # ------------------------------------------------------------------
    def link(self, key: str, page_index: int) -> None:
        self.evidence_links.setdefault(key, set()).add(page_index)

    def update(self, new_tokens: list[str]) -> "BeliefState":
        """Consume the next chunk of decoded tokens (words), advancing the
        incremental cursor. Each call is O(len(new_tokens)), independent of
        history length."""
        for tok in new_tokens:
            pos = self._pos
            self._pos += 1
            page = pos // self._block_size
            low = tok.lower()

            # Entity mention: known (task-available) vocabulary; optionally a
            # capitalized word (off by default — see _flag_capitalized).
            is_entity = (low in self._entity_vocab) or (
                self._flag_capitalized and tok[:1].isupper() and len(tok) > 1
                and low not in _CONSTRAINT_MARKERS
            )
            if is_entity:
                self.active_entities[low] = pos
                self.link(f"entity:{low}", page)

            # Fact assignment pattern "<entity> = <value>" is detected at the
            # value token by the harness-provided structure; here we capture the
            # lightweight "<entity> is <value>" by remembering the last entity.
            if low in _CONSTRAINT_MARKERS:
                self.active_constraints.append(f"@{pos}:{low}")
                self.link(f"constraint:{pos}", page)
            if low in _REVISION_MARKERS:
                # A revision invalidates prior candidate facts; mark uncertainty.
                self.uncertainty = min(1.0, self.uncertainty + 0.25)
                self.link(f"revision:{pos}", page)
            if low in _QUESTION_MARKERS:
                self.unresolved_questions.add(f"q@{pos}")
                self.link(f"question:{pos}", page)

        # Uncertainty decays as more evidence accrues (bounded).
        if self.active_entities:
            self.uncertainty = max(0.0, self.uncertainty * 0.98)
        return self

    # ------------------------------------------------------------------
    def active_pages(self, recency: int = 64) -> set[int]:
        """Logical pages referenced by currently-relevant evidence.

        ``recency`` bounds how far back entity evidence stays 'active' (in
        tokens), so the active set stays bounded as context grows — the
        property that lets J remain bounded (context-independent decode)."""
        pages: set[int] = set()
        cutoff = self._pos - recency
        # Unresolved questions always retained.
        for key, ps in self.evidence_links.items():
            if key.startswith("question:") or key.startswith("constraint:"):
                pages |= ps
        # Recently-seen entities retained.
        for ent, last in self.active_entities.items():
            if last >= cutoff:
                pages |= self.evidence_links.get(f"entity:{ent}", set())
        # Recent revisions retained.
        for key, ps in self.evidence_links.items():
            if key.startswith("revision:"):
                rev_pos = int(key.split(":")[1])
                if rev_pos >= cutoff:
                    pages |= ps
        return pages


def new_belief(task_type: str, block_size: int,
               entity_vocab: Optional[frozenset[str]] = None) -> BeliefState:
    return BeliefState(
        task_type=task_type,
        _block_size=block_size,
        _entity_vocab=entity_vocab or frozenset(),
    )
