# SPDX-License-Identifier: Apache-2.0
"""Belief-gated paged-attention control-plane seam (experimental, default off).

This module is the *only* core-repo change the belief-gating experiment needs
beyond a feature flag. It exposes a registry for a host-side page selector and a
pure transform that rewrites the per-sequence block table + context length for
the **decode attention read** when ``VLLM_METAL_BELIEF_PAGED_ATTENTION=1`` and a
selector is registered.

Safety contract (see experiments/belief_paged_attention/IMPLEMENTATION_AUDIT.md):
  * Decode-only: only segments that are decode (one query token) are gated;
    prefill segments are passed through unchanged (the tiled prefill kernel is
    position-indexed and unsafe for compaction).
  * The selector must return, per sequence, a CHRONOLOGICAL subset of the page
    ids and a matching valid-token count, honoring the full-page-except-tail
    invariant. ``apply_gate`` re-validates the count parity and falls back to the
    full table on any violation (fail safe, never fail wrong).
  * Read-only: this never touches slot_mapping (the KV write path) or the
    scheduler's block allocations — no physical page is reclaimed.
  * Off by default: with the flag unset or no selector registered, ``apply_gate``
    returns its inputs unchanged (exact numeric equivalence with the baseline).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

# selector(block_tables, context_lens, num_decode_requests, block_size)
#   -> (new_block_tables, new_context_lens)
Selector = Callable[
    [list[list[int]], list[int], int, int],
    tuple[list[list[int]], list[int]],
]

_SELECTOR: Optional[Selector] = None


def register_selector(selector: Optional[Selector]) -> None:
    """Install (or clear, with ``None``) the active page selector."""
    global _SELECTOR
    _SELECTOR = selector


def get_selector() -> Optional[Selector]:
    return _SELECTOR


def is_enabled() -> bool:
    from vllm_metal import envs

    # Generic page-gating seam: enabled by either experiment's flag. Both only
    # rewrite the decode attention read; the seam itself is policy-agnostic.
    flag = bool(envs.VLLM_METAL_BELIEF_PAGED_ATTENTION) or bool(
        envs.VLLM_METAL_LOSS_BUDGETED_PAGE_KV
    )
    return flag and _SELECTOR is not None


def apply_gate(
    block_tables: Sequence[Sequence[int]],
    context_lens: Sequence[int],
    num_decode_requests: int,
    block_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Return possibly-compacted (block_tables, context_lens).

    No-op (returns copies of the inputs) unless gating is enabled. On any
    selector error or invariant violation, falls back to the full table for the
    offending sequence so attention is never silently corrupted."""
    full_bt = [list(bt) for bt in block_tables]
    full_cl = [int(c) for c in context_lens]
    if not is_enabled():
        return full_bt, full_cl
    try:
        new_bt, new_cl = _SELECTOR(full_bt, full_cl, num_decode_requests, block_size)  # type: ignore[misc]
    except Exception:
        return full_bt, full_cl

    out_bt: list[list[int]] = []
    out_cl: list[int] = []
    for i, (obt, ocl) in enumerate(zip(full_bt, full_cl)):
        # Only gate decode segments; validate the result, else fall back.
        if i < num_decode_requests and i < len(new_bt):
            cand_bt = [int(b) for b in new_bt[i]]
            cand_cl = int(new_cl[i])
            n_pages = len(cand_bt)
            # Validity: pages cover exactly cand_cl tokens with full interior
            # pages (cand_cl in ((n-1)*B, n*B]) and cand_cl <= original length.
            lo = (n_pages - 1) * block_size
            if (
                n_pages >= 1
                and lo < cand_cl <= n_pages * block_size
                and cand_cl <= ocl
                and all(0 <= b for b in cand_bt)
            ):
                out_bt.append(cand_bt)
                out_cl.append(cand_cl)
                continue
        out_bt.append(obt)
        out_cl.append(ocl)
    _log_capacity(full_bt, out_bt)
    return out_bt, out_cl


def dropped_blocks(full_block_tables: Sequence[Sequence[int]],
                   gated_block_tables: Sequence[Sequence[int]]) -> dict[int, set[int]]:
    """Per-segment set of physical blocks present in the full table but dropped by
    the gate. Read-only; feed to reclaim_safety.reclaim_accounting (with ref_cnts)
    to quantify the would-be CAPACITY reclaim. Does NOT free anything."""
    out: dict[int, set[int]] = {}
    for i, (f, g) in enumerate(zip(full_block_tables, gated_block_tables)):
        out[i] = set(int(b) for b in f) - set(int(b) for b in g)
    return out


def _log_capacity(full_bt, out_bt) -> None:
    """Read-only debug log of how many blocks the gate dropped (capacity-for-reuse
    signal). Never frees a block. Coarse count only (true safety needs ref_cnts via
    reclaim_safety)."""
    try:
        import logging
        log = logging.getLogger(__name__)
        if not log.isEnabledFor(logging.DEBUG):
            return
        dropped = sum(len(set(f) - set(g)) for f, g in zip(full_bt, out_bt))
        if dropped:
            log.debug("belief_gate: gate dropped %d block-refs from the decode read "
                      "(capacity-for-reuse only; no physical reclamation)", dropped)
    except Exception:
        pass
