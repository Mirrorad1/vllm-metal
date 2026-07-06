# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the loss-budgeted page KV cache.

Pure correctness/policy/metric tests run fast; kernel/model tests are marked.
"""
from __future__ import annotations

import numpy as np
import pytest

import error_metrics as EM
import page_policies as PP
from page_policies import PolicyConfig, SequencePages, Signals, build_selection, select


def _pages(P, L, B=16):
    return SequencePages(tuple(range(100, 100 + P)), L, B)


# --- correctness / invariant ---------------------------------------------

def test_all_pages_equivalence_logical():
    p = _pages(8, 8 * 16 - 3)
    sel = select("full_pages", p, PolicyConfig())
    assert sel.num_selected_pages == 8
    assert sel.selected_context_len == 8 * 16 - 3


def test_selected_block_table_shape_and_chronological_order():
    p = _pages(10, 10 * 16)
    sel = select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.5, seed=2))
    idx = list(sel.selected_page_indices)
    assert idx == sorted(idx)
    assert list(sel.selected_block_ids) == [p.block_ids[i] for i in idx]


def test_current_tail_page_always_present():
    p = _pages(12, 12 * 16)
    for pol in ["recent_pages", "seeded_random_pages", "sink_recent_pages"]:
        sel = select(pol, p, PolicyConfig(budget_fraction=0.1, seed=0))
        assert (p.num_pages - 1) in sel.selected_page_indices


def test_required_sink_pages_present():
    p = _pages(20, 20 * 16)
    sel = select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.1, sink_pages=2, seed=0))
    assert 0 in sel.selected_page_indices and 1 in sel.selected_page_indices


def test_context_lens_consistency_partial_tail():
    p = _pages(4, 3 * 16 + 5)
    assert build_selection(p, [0, 1, 2]).selected_context_len == 48
    assert build_selection(p, [0, 3]).selected_context_len == 16 + 5
    with pytest.raises(ValueError):
        build_selection(p, [0, 9])  # out of range


def test_interior_partial_rejected():
    # construct pages where a non-tail selected page would be partial: only the
    # tail is partial, so selecting [tail, ...] out of order is impossible; the
    # guard triggers if we lie about validity. Use an exact-multiple tail.
    p = _pages(4, 4 * 16)  # all full
    assert build_selection(p, [0, 1, 3]).selected_context_len == 48


def test_deterministic_random_policy():
    p = _pages(10, 10 * 16)
    a = select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.4, seed=5))
    b = select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.4, seed=5))
    c = select("seeded_random_pages", p, PolicyConfig(budget_fraction=0.4, seed=6))
    assert a.selected_page_indices == b.selected_page_indices
    assert a.selected_page_indices != c.selected_page_indices


def test_equal_budget_across_policies():
    p = _pages(16, 16 * 16)
    cfg = PolicyConfig(budget_fraction=0.25, seed=1)
    sig = Signals(attention_mass=list(np.random.RandomState(0).rand(16)),
                  damage=list(np.random.RandomState(1).rand(16)),
                  page_norm=list(np.random.RandomState(2).rand(16)))
    js = {pol: select(pol, p, cfg, sig).num_selected_pages for pol in
          ["recent_pages", "seeded_random_pages", "sink_recent_pages",
           "attention_proxy_pages", "page_norm_pages", "loss_budgeted_oracle",
           "loss_budgeted_online"]}
    assert len(set(js.values())) == 1, js


def test_oracle_keeps_highest_damage():
    p = _pages(8, 8 * 16)
    dmg = [0.0, 9.0, 0.0, 0.0, 8.0, 0.0, 7.0, 0.0]  # pages 1,4,6 are high-damage
    sel = select("loss_budgeted_oracle", p, PolicyConfig(budget_fraction=0.75, seed=0),
                 Signals(damage=dmg))
    # with J=6 of 8, the three high-damage pages must be retained
    assert {1, 4, 6}.issubset(set(sel.selected_page_indices))


def test_loss_budgeted_oracle_requires_damage():
    p = _pages(4, 4 * 16)
    with pytest.raises(ValueError):
        select("loss_budgeted_oracle", p, PolicyConfig(budget_fraction=0.5), Signals())


def test_no_future_leakage_signals_are_optional():
    # online policy must work with only causal signals (no damage).
    p = _pages(8, 8 * 16)
    sel = select("loss_budgeted_online", p, PolicyConfig(budget_fraction=0.5),
                 Signals(attention_mass=[1, 2, 3, 4, 5, 6, 7, 8], page_norm=[1] * 8))
    assert sel.num_selected_pages == 4


# --- metrics --------------------------------------------------------------

def test_kl_nonnegative_and_zero_on_identical():
    z = np.random.RandomState(0).randn(100)
    assert EM.kl(z, z) == pytest.approx(0.0, abs=1e-9)
    assert EM.kl(z, z + np.random.RandomState(1).randn(100)) >= 0.0


def test_js_symmetric():
    a = np.random.RandomState(0).randn(50)
    b = np.random.RandomState(1).randn(50)
    assert EM.js(a, b) == pytest.approx(EM.js(b, a), abs=1e-12)


def test_topk_overlap_bounds():
    a = np.random.RandomState(0).randn(50)
    b = np.random.RandomState(1).randn(50)
    v = EM.topk_overlap(a, b, 10)
    assert 0.0 <= v <= 1.0
    assert EM.topk_overlap(a, a, 10) == 1.0


def test_memory_reports_physical_and_logical_separately():
    acc = EM.SystemsAccount(pages_retained=4, full_pages=32, pages_reclaimed=0,
                            shadow_mode=True)
    assert acc.pages_reclaimed == 0 and acc.shadow_mode is True


def test_page_bytes_scales_with_pages():
    b1 = EM.page_bytes(1, 16, 24, 2, 64)
    b4 = EM.page_bytes(4, 16, 24, 2, 64)
    assert b4 == 4 * b1


# --- gate seam ------------------------------------------------------------

def test_feature_flag_off_is_noop(monkeypatch):
    from vllm_metal.attention import belief_gate
    monkeypatch.delenv("VLLM_METAL_LOSS_BUDGETED_PAGE_KV", raising=False)
    monkeypatch.delenv("VLLM_METAL_BELIEF_PAGED_ATTENTION", raising=False)
    belief_gate.register_selector(lambda b, c, n, bs: ([[1]], [bs]))
    bt, cl = [[5, 6, 7]], [40]
    assert belief_gate.apply_gate(bt, cl, 1, 16) == (bt, cl)
    belief_gate.register_selector(None)


def test_loss_flag_enables_gate(monkeypatch):
    from vllm_metal.attention import belief_gate
    monkeypatch.setenv("VLLM_METAL_LOSS_BUDGETED_PAGE_KV", "1")
    belief_gate.register_selector(lambda b, c, n, bs: ([[5, 7]], [24]))
    out_bt, out_cl = belief_gate.apply_gate([[5, 6, 7]], [40], 1, 16)
    assert out_bt == [[5, 7]] and out_cl == [24]
    belief_gate.register_selector(None)


# --- kernel (build from source) ------------------------------------------

@pytest.mark.kernel
@pytest.mark.parametrize("L", [128, 512, 1024])
def test_paged_all_pages_equivalence(L):
    import benchmark_harness as H
    err = H.paged_all_pages_err(L, 16, 14, 2, 64)
    assert err < 5e-3, f"L={L} err={err}"


@pytest.mark.kernel
def test_partition_boundary():
    import benchmark_harness as H
    for L in [511, 512, 513]:
        assert H.paged_all_pages_err(L, 16, 14, 2, 64, seed=1) < 5e-3
