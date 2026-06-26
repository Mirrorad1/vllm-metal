# SPDX-License-Identifier: Apache-2.0
"""Unit + property tests for the advisory reclamation-safety gate (read-only)."""
from __future__ import annotations

import random

import pytest

from vllm_metal.attention import belief_gate
from vllm_metal.attention.reclaim_safety import reclaim_accounting, safe_to_reclaim


def test_safe_subset_of_dropped_and_basic():
    active = {"r0": [0, 5, 6, 7], "r1": [0, 9, 10]}  # block 0 = shared sink
    dropped = {"r0": {5, 6}, "r1": {9}}
    ref = {5: 1, 6: 1, 7: 1, 9: 1, 10: 1, 0: 2}
    safe = safe_to_reclaim(active, dropped, ref, null_block_id=0)
    assert safe["r0"] == {5, 6} and safe["r1"] == {9}
    # always a subset of dropped
    for r in dropped:
        assert safe[r] <= dropped[r]


def test_excludes_shared_refcnt():
    active = {"r0": [5], "r1": [5]}  # block 5 referenced by two reqs
    safe = safe_to_reclaim({"r0": [5]}, {"r0": {5}}, {5: 2})
    assert safe["r0"] == set()  # ref_cnt>1 ⇒ unsafe


def test_excludes_null_block():
    safe = safe_to_reclaim({"r0": [0, 3]}, {"r0": {0, 3}}, {0: 1, 3: 1}, null_block_id=0)
    assert 0 not in safe["r0"] and 3 in safe["r0"]


def test_excludes_prefix_cached():
    safe = safe_to_reclaim({"r0": [3, 4]}, {"r0": {3, 4}}, {3: 1, 4: 1},
                           prefix_cached_blocks={3})
    assert safe["r0"] == {4}


def test_excludes_cross_referenced_even_if_refcnt_stale():
    # block 8 appears in another active table even though ref_cnts says 1 (defense in depth)
    active = {"r0": [8], "r1": [8, 9]}
    safe = safe_to_reclaim(active, {"r0": {8}}, {8: 1, 9: 1})
    assert safe["r0"] == set()


def test_accounting_capacity():
    active = {"r0": [0, 5, 6], "r1": [0, 9]}
    dropped = {"r0": {5, 6}, "r1": {9}}
    ref = {0: 2, 5: 1, 6: 1, 9: 1}
    acc = reclaim_accounting(active, dropped, ref, block_size=16, null_block_id=0)
    assert acc.reclaimable_blocks == 3
    assert acc.headroom_tokens == 3 * 16
    assert acc.per_request == {"r0": 2, "r1": 1}
    assert "CAPACITY" in acc.note and "NOT process-memory" in acc.note


def test_property_safe_excludes_unsafe():
    rng = random.Random(0)
    for _ in range(200):
        nblocks = rng.randint(2, 40)
        nreq = rng.randint(1, 4)
        active = {f"r{i}": rng.sample(range(nblocks), rng.randint(1, nblocks))
                  for i in range(nreq)}
        ref = {b: 0 for b in range(nblocks)}
        for bt in active.values():
            for b in bt:
                ref[b] += 1
        prefix = set(rng.sample(range(nblocks), rng.randint(0, nblocks // 3)))
        null = 0
        dropped = {r: set(rng.sample(bt, rng.randint(0, len(bt)))) for r, bt in active.items()}
        safe = safe_to_reclaim(active, dropped, ref, null_block_id=null,
                               prefix_cached_blocks=prefix)
        owners = {}
        for r, bt in active.items():
            for b in bt:
                owners.setdefault(b, set()).add(r)
        for r, s in safe.items():
            assert s <= dropped[r]                       # subset of dropped
            for b in s:
                assert b != null
                assert ref[b] == 1
                assert b not in prefix
                assert owners[b] == {r}                  # uniquely owned


def test_dropped_blocks_helper():
    full = [[0, 1, 2, 3], [0, 9]]
    gated = [[0, 3], [0, 9]]
    d = belief_gate.dropped_blocks(full, gated)
    assert d[0] == {1, 2} and d[1] == set()
