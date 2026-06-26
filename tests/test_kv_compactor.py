# SPDX-License-Identifier: Apache-2.0
"""Numeric-equivalence + transient-peak tests for the staged KV compactor."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vllm_metal.attention.kv_compactor import stage_compact, transient_peak_bound


def _arrays(n_layers, P, B, Hkv, d, seed=0):
    rng = np.random.default_rng(seed)
    return [mx.array(rng.standard_normal((P, B, Hkv, d)).astype(np.float32), dtype=mx.float16)
            for _ in range(n_layers)]


def test_stage_compact_equivalence():
    arrays = _arrays(6, 64, 16, 2, 64)
    keep = [0, 3, 7, 10, 63]
    # keep a reference copy before stage_compact frees inputs
    ref = [np.array(a[mx.array(keep, dtype=mx.int32)].astype(mx.float32)) for a in arrays]
    out = stage_compact(arrays, keep, free_inputs=True)
    assert len(out) == 6
    for o, r in zip(out, ref):
        assert tuple(o.shape) == (len(keep), 16, 2, 64)
        assert np.abs(np.array(o.astype(mx.float32)) - r).max() < 1e-3


def test_stage_compact_preserves_order():
    arrays = _arrays(2, 32, 16, 2, 64)
    keep = [5, 1, 9]  # function gathers exactly in given order
    out = stage_compact([a for a in arrays], keep, free_inputs=False)
    base = np.array(arrays[0].astype(mx.float32))
    got = np.array(out[0].astype(mx.float32))
    for j, k in enumerate(keep):
        assert np.abs(got[j] - base[k]).max() < 1e-3


def test_staged_peak_below_all_at_once():
    # staged (free_inputs=True) should peak well below holding full+full.
    mx.clear_cache(); mx.reset_peak_memory()
    arrays = _arrays(24, 128, 16, 2, 64)
    mx.eval(arrays)
    full = mx.get_active_memory()
    mx.reset_peak_memory()
    _ = stage_compact(arrays, list(range(8)), free_inputs=True)  # keep 8/128
    staged_peak = mx.get_peak_memory()
    # staged peak must be far below 2x full (all-at-once would approach full+full)
    assert staged_peak < 1.5 * full


def test_transient_peak_bound_formula():
    full_per_layer = 1_000_000
    nl = 24
    staged = transient_peak_bound(nl, full_per_layer, keep_frac=0.5, staged=True)
    allatonce = transient_peak_bound(nl, full_per_layer, keep_frac=0.5, staged=False)
    assert staged < allatonce
    assert staged == nl * full_per_layer + int(full_per_layer * 0.5)
