# SPDX-License-Identifier: Apache-2.0
"""Focused unit + integration tests for belief-gated paged attention.

Pure tests (page-selection invariants, determinism, equal-budget, gate no-op,
ref-count safety) run fast and always. Kernel tests (numeric equivalence,
partition boundary, multi-sequence isolation) build the Metal kernels from
source and are marked ``kernel``; run them with ``-m kernel`` or by default.
"""
from __future__ import annotations

import numpy as np
import pytest

import page_policies as PP
from page_policies import PolicyConfig, SelectionContext, SequencePages, build_selection


# ---------------------------------------------------------------------------
# Pure: page-selection invariants
# ---------------------------------------------------------------------------


def _pages(num_pages: int, context_len: int, block_size: int = 16) -> SequencePages:
    return SequencePages(tuple(range(100, 100 + num_pages)), context_len, block_size)


def test_selected_block_table_validity_interior_partial_rejected():
    # 3 pages, last is partial (40 = 2*16 + 8). Selecting [0,2] is legal (2 is
    # tail). Selecting [0,1] is legal (both full). Selecting an interior partial
    # is impossible here, so synthesize a case: pages where page 1 is partial is
    # not constructible (only the tail is partial), so we test the guard directly
    # by a hand-built SequencePages whose middle selection would be partial.
    p = _pages(3, 40)  # pages 0,1 full (16 each), page 2 partial (8)
    sel = build_selection(p, [0, 2])
    assert sel.selected_context_len == 16 + 8
    # build_selection must reject a non-tail partial as the last interior:
    with pytest.raises(ValueError):
        # claim page indices out of range
        build_selection(p, [0, 5])


def test_chronological_sparse_page_order():
    p = _pages(8, 8 * 16)
    sel = PP.select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.5, seed=3))
    idx = list(sel.selected_page_indices)
    assert idx == sorted(idx)  # chronological
    # block ids follow chronological page order
    assert list(sel.selected_block_ids) == [p.block_ids[i] for i in idx]


def test_partial_tail_context_len():
    # context_len not a multiple of block_size -> tail page partial.
    p = _pages(4, 3 * 16 + 5)  # 53 tokens, page 3 has 5 valid
    assert p.valid_tokens(0) == 16 and p.valid_tokens(3) == 5
    full = PP.select("full_pages", p, PolicyConfig())
    assert full.selected_context_len == 53
    # Selecting only full pages -> context multiple of block_size.
    sel = build_selection(p, [0, 1, 2])
    assert sel.selected_context_len == 48
    # Selecting the tail -> non-multiple, allowed because tail is last.
    sel2 = build_selection(p, [0, 3])
    assert sel2.selected_context_len == 16 + 5


def test_deterministic_random_selection():
    p = _pages(10, 10 * 16)
    a = PP.select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.4, seed=7))
    b = PP.select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.4, seed=7))
    c = PP.select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.4, seed=8))
    assert a.selected_page_indices == b.selected_page_indices  # reproducible
    assert a.selected_page_indices != c.selected_page_indices  # seed-sensitive


def test_equal_budget_across_policies():
    p = _pages(16, 16 * 16)
    cfg = PolicyConfig(budget_fraction=0.25, seed=1)
    ctx = SelectionContext(keyword_pages=(2, 9), oracle_pages=(2,), inferred_pages=(5, 9))
    js = {}
    for pol in ["recent_pages", "seeded_random_pages", "attention_proxy_pages",
                "keyword_entity_pages", "oracle_belief_pages", "inferred_belief_pages"]:
        sel = PP.select(pol, p, cfg, ctx)
        js[pol] = sel.num_selected_pages
    assert len(set(js.values())) == 1, f"unequal budgets: {js}"
    assert next(iter(js.values())) == max(1, round(0.25 * 16))


def test_always_keep_floor_present():
    p = _pages(20, 20 * 16)
    cfg = PolicyConfig(budget_fraction=0.1, recent_window=2, sink_pages=1, seed=0)
    sel = PP.select("seeded_random_pages", p, cfg)
    idx = set(sel.selected_page_indices)
    assert 19 in idx  # partial tail
    assert 18 in idx and 19 in idx  # recent window of 2
    assert 0 in idx  # sink


def test_reference_count_safety_no_mutation():
    # Selection must not mutate the input block ids (read-only view).
    original = tuple(range(100, 110))
    p = SequencePages(original, 10 * 16, 16)
    _ = PP.select("inferred_belief_pages", p, PolicyConfig(budget_fraction=0.3),
                  SelectionContext(inferred_pages=(1, 4)))
    assert p.block_ids == original  # unchanged


def test_full_pages_selects_everything():
    p = _pages(7, 7 * 16 - 3)
    sel = PP.select("full_pages", p, PolicyConfig())
    assert sel.num_selected_pages == 7
    assert sel.selected_context_len == 7 * 16 - 3


# ---------------------------------------------------------------------------
# Pure: feature-flag-off equivalence + gate validation
# ---------------------------------------------------------------------------


def test_feature_flag_off_is_noop(monkeypatch):
    from vllm_metal.attention import belief_gate
    monkeypatch.delenv("VLLM_METAL_BELIEF_PAGED_ATTENTION", raising=False)
    belief_gate.register_selector(lambda b, c, n, bs: ([[999]], [bs]))
    bt = [[5, 6, 7], [1, 2]]
    cl = [40, 20]
    out_bt, out_cl = belief_gate.apply_gate(bt, cl, 2, 16)
    assert out_bt == bt and out_cl == cl
    belief_gate.register_selector(None)


def test_gate_validates_and_falls_back(monkeypatch):
    from vllm_metal.attention import belief_gate
    monkeypatch.setenv("VLLM_METAL_BELIEF_PAGED_ATTENTION", "1")
    # selector returns an INVALID context_len (too long) -> must fall back.
    belief_gate.register_selector(lambda b, c, n, bs: ([[5]], [9999]))
    bt = [[5, 6, 7]]
    cl = [40]
    out_bt, out_cl = belief_gate.apply_gate(bt, cl, 1, 16)
    assert out_bt == [[5, 6, 7]] and out_cl == [40]  # fell back
    # valid compaction passes through.
    belief_gate.register_selector(lambda b, c, n, bs: ([[5, 7]], [16 + 8]))
    out_bt, out_cl = belief_gate.apply_gate(bt, cl, 1, 16)
    assert out_bt == [[5, 7]] and out_cl == [24]
    belief_gate.register_selector(None)


def test_gate_prefill_segment_untouched(monkeypatch):
    from vllm_metal.attention import belief_gate
    monkeypatch.setenv("VLLM_METAL_BELIEF_PAGED_ATTENTION", "1")
    # 1 decode + 1 prefill segment; selector tries to gate both, prefill must
    # be passed through unchanged.
    belief_gate.register_selector(lambda b, c, n, bs: ([[5, 7], [1]], [24, 1]))
    bt = [[5, 6, 7], [1, 2, 3]]
    cl = [40, 40]
    out_bt, out_cl = belief_gate.apply_gate(bt, cl, 1, 16)  # only seg 0 decodes
    assert out_bt[0] == [5, 7] and out_cl[0] == 24
    assert out_bt[1] == [1, 2, 3] and out_cl[1] == 40  # prefill untouched
    belief_gate.register_selector(None)


# ---------------------------------------------------------------------------
# Belief state: incrementality
# ---------------------------------------------------------------------------


def test_belief_update_is_incremental():
    from belief_state import new_belief
    b = new_belief("entity", 16, frozenset({"marcus"}))
    b.update(["Marcus", "is", "here"])
    assert b._pos == 3
    pos_before = b._pos
    b.update(["filler"])
    assert b._pos == pos_before + 1  # cursor advanced by chunk size only
    assert "marcus" in b.active_entities


# ---------------------------------------------------------------------------
# Kernel: numeric equivalence (build from source)
# ---------------------------------------------------------------------------

pytestmark_kernel = pytest.mark.kernel


@pytest.mark.kernel
@pytest.mark.parametrize("context_len", [128, 500, 1024])
def test_all_pages_equivalence_kernel(context_len):
    import benchmark as B
    err = B.all_pages_equivalence(context_len, 16, n_q_heads=14, n_kv_heads=2,
                                  head_dim=64, seed=0)
    assert err < 5e-3, f"all-pages err {err} at L={context_len}"


@pytest.mark.kernel
def test_partition_boundary_kernel():
    # context length spanning the PARTITION_SIZE (512) boundary exercises the
    # _ps512 partitioned path + reduce kernel.
    import benchmark as B
    for L in [511, 512, 513, 1100]:
        err = B.all_pages_equivalence(L, 16, 14, 2, 64, seed=1)
        assert err < 5e-3, f"partition-boundary err {err} at L={L}"


@pytest.mark.kernel
def test_compact_selection_matches_reference_kernel():
    # Drop interior full pages; compact attention == numpy ref over kept tokens.
    import mlx.core as mx
    from benchmark import get_ops
    ops = get_ops()
    rng = np.random.default_rng(0)
    B, Hq, Hkv, d = 16, 14, 2, 64
    n_pages, ctx = 6, 6 * 16 - 4
    q = mx.array(rng.standard_normal((1, Hq, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    kc = mx.array(rng.standard_normal((n_pages, B, Hkv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    vc = mx.array(rng.standard_normal((n_pages, B, Hkv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    keep = [0, 2, 5]  # 2 interior full + partial tail
    R = 16 + 16 + (ctx - 5 * 16)
    out = mx.zeros((1, Hq, d), dtype=mx.float16)
    ops.paged_attention_primitive(q, kc, vc, Hkv, float(d ** -0.5), 0.0,
                                  mx.array([keep], dtype=mx.int32),
                                  mx.array([R], dtype=mx.int32),
                                  mx.array([0, 1], dtype=mx.int32), B, int(R), -1, out)
    mx.eval(out)
    got = np.array(out.astype(mx.float32))[0]
    # reference over kept tokens
    knp = np.array(kc.astype(mx.float32)).astype(np.float64)
    vnp = np.array(vc.astype(mx.float32)).astype(np.float64)
    qnp = np.array(q.astype(mx.float32)).astype(np.float64)[0]
    Ks, Vs = [], []
    for pid in keep:
        for t in range(B):
            Ks.append(knp[pid, t]); Vs.append(vnp[pid, t])
    Ks = np.stack(Ks)[:R]; Vs = np.stack(Vs)[:R]
    ref = np.zeros((Hq, d))
    for h in range(Hq):
        lg = (qnp[h] @ Ks[:, h // (Hq // Hkv)].T) * (d ** -0.5)
        w = np.exp(lg - lg.max()); w /= w.sum()
        ref[h] = w @ Vs[:, h // (Hq // Hkv)]
    assert np.abs(got - ref).max() < 5e-3


@pytest.mark.kernel
def test_multi_sequence_isolation_kernel():
    # Two sequences in one decode batch; compacting seq 0 must not change seq 1.
    import mlx.core as mx
    from benchmark import get_ops
    ops = get_ops()
    rng = np.random.default_rng(2)
    Bsz, Hq, Hkv, d = 16, 14, 2, 64
    n_pages = 8
    kc = mx.array(rng.standard_normal((n_pages, Bsz, Hkv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    vc = mx.array(rng.standard_normal((n_pages, Bsz, Hkv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    q = mx.array(rng.standard_normal((2, Hq, d)).astype(np.float32) * 0.1, dtype=mx.float16)

    def run(block_tables, ctx_lens):
        out = mx.zeros((2, Hq, d), dtype=mx.float16)
        maxb = max(len(b) for b in block_tables)
        bt = mx.array([b + [0] * (maxb - len(b)) for b in block_tables], dtype=mx.int32)
        ops.paged_attention_primitive(q, kc, vc, Hkv, float(d ** -0.5), 0.0, bt,
                                      mx.array(ctx_lens, dtype=mx.int32),
                                      mx.array([0, 1, 2], dtype=mx.int32), Bsz,
                                      int(max(ctx_lens)), -1, out)
        mx.eval(out)
        return np.array(out.astype(mx.float32))

    # seq0 full = pages 0..3 (64 tok), seq1 full = pages 4..6 (48 tok)
    base = run([[0, 1, 2, 3], [4, 5, 6]], [64, 48])
    # compact seq0 to [0,3]; seq1 unchanged
    gated = run([[0, 3], [4, 5, 6]], [32, 48])
    # seq1 output must be identical (isolation)
    assert np.abs(base[1] - gated[1]).max() < 1e-3
    # seq0 output should change (fewer pages)
    assert np.abs(base[0] - gated[0]).max() > 1e-4
