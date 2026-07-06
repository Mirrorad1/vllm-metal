# SPDX-License-Identifier: Apache-2.0
"""Measurement harness for the loss-budgeted page KV cache (KV-only).

Two faithful, real measurements:
  * BEHAVIORAL (real model, mlx-lm Qwen2.5-0.5B): teacher-forced full-cache vs
    page-gated next-token distributions. A GatedCache keeps the ABSOLUTE RoPE
    offset while storing a compacted key set, so relative positions stay exact
    (full selection ⇒ KL=0, validated). Provides per-page signals:
      - attention_mass: REAL per-page attention probability mass for the current
        query, captured by wrapping the model's SDPA (causal; current query only).
      - damage[p] = KL(full || ablate page p): the loss-budgeted ORACLE signal,
        from full-cache logits only (no answer labels).
      - page_norm: ||K|| per page (cheap runtime proxy).
  * SYSTEMS (real vllm-metal paged_attention_primitive): all-pages numeric
    equivalence + decode kernel latency (warmup excluded). Memory is accounted
    honestly: shadow mode (no page reclaimed) unless an allocator path is added.

Dropping a contiguous K/V block is exactly subset attention under both mlx-lm
SDPA and the paged kernel, so the split is principled.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np

import error_metrics as EM
import page_policies as PP

BLOCK_SIZE = 16
_MODEL = {}


def load_model(model_id="Qwen/Qwen2.5-0.5B-Instruct"):
    if model_id not in _MODEL:
        from mlx_lm import load
        _MODEL[model_id] = load(model_id)
    return _MODEL[model_id]


def model_config(model):
    a0 = model.model.layers[0].self_attn
    return dict(n_layers=len(model.model.layers), n_q=a0.n_heads, n_kv=a0.n_kv_heads,
                head_dim=model.args.hidden_size // a0.n_heads)


class GatedCache:
    """mlx-lm cache holding compacted K/V but reporting the ABSOLUTE position."""
    def __init__(self, keys, values, abs_offset):
        self.keys = keys
        self.values = values
        self._abs = int(abs_offset)

    @property
    def offset(self):
        return self._abs

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        self.keys = mx.concatenate([self.keys, keys], axis=2)
        self.values = mx.concatenate([self.values, values], axis=2)
        self._abs += keys.shape[2]
        return self.keys, self.values


# --- attention-mass capture (real, causal) --------------------------------

_CAPTURE = {"on": False, "mass": []}


@contextlib.contextmanager
def capture_attention_mass(model):
    """Wrap the model's SDPA so each layer's per-key attention mass for the
    current query is recorded. Decode-only (L_q small). Restores on exit."""
    import mlx.core as mx
    import mlx_lm.models.qwen2 as q2
    orig = q2.scaled_dot_product_attention
    _CAPTURE["mass"] = []

    def wrapped(queries, keys, values, cache, scale, mask, sinks=None):
        # GQA: expand kv heads to query heads (q head h attends kv head h//group).
        hq, hkv = queries.shape[1], keys.shape[1]
        k = keys
        if hq != hkv:
            k = mx.repeat(keys, hq // hkv, axis=1)
        # weights for the LAST query row over all keys: [1,Hq,1,S]
        w = mx.softmax((queries[:, :, -1:, :].astype(mx.float32) * scale)
                       @ k.astype(mx.float32).swapaxes(-1, -2), axis=-1)
        per_key = np.array(w[0, :, 0, :].sum(axis=0).astype(mx.float32))  # [S], summed over heads
        _CAPTURE["mass"].append(per_key)
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    q2.scaled_dot_product_attention = wrapped
    try:
        yield
    finally:
        q2.scaled_dot_product_attention = orig


def _slice(keys, seq, mx):
    return keys[:, :, :seq, :]


def _gather(keys, idx, mx):
    return keys[:, :, mx.array(np.asarray(idx, dtype=np.int32)), :]


def full_step(model, full_caches, query, seq, want_mass=False):
    """Next-token logits for `query` over the full cache (no mutation). Returns
    (logits_np, per_page_mass or None)."""
    import mlx.core as mx
    qx = mx.array([[query]])
    caches = [GatedCache(_slice(c.keys, seq, mx), _slice(c.values, seq, mx), seq)
              for c in full_caches]
    if want_mass:
        with capture_attention_mass(model):
            logits = model(qx, cache=caches)
            mx.eval(logits)
        masses = _CAPTURE["mass"]  # list over layers, each [seq+1]
        n_pages = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
        page_mass = np.zeros(n_pages)
        for m in masses:
            for tpos in range(min(seq, len(m))):  # ignore the new query key (last)
                page_mass[tpos // BLOCK_SIZE] += float(m[tpos])
        return np.array(logits[0, -1].astype(mx.float32)), page_mass
    logits = model(qx, cache=caches)
    mx.eval(logits)
    return np.array(logits[0, -1].astype(mx.float32)), None


def gated_step(model, full_caches, query, seq, keep_pages):
    """Next-token logits attending only to the selected pages."""
    import mlx.core as mx
    keep_tok = []
    for pg in keep_pages:
        s = pg * BLOCK_SIZE
        keep_tok.extend(range(s, min(s + BLOCK_SIZE, seq)))
    keep_tok = sorted(set(keep_tok))
    qx = mx.array([[query]])
    caches = [GatedCache(_gather(c.keys, keep_tok, mx), _gather(c.values, keep_tok, mx), seq)
              for c in full_caches]
    logits = model(qx, cache=caches)
    mx.eval(logits)
    return np.array(logits[0, -1].astype(mx.float32)), len(keep_tok)


def page_damage(model, full_caches, query, seq, z_full):
    """ORACLE signal: damage[p] = KL(full || ablate page p). O(P) forwards.
    Uses full-cache logits only — no answer labels, no future tokens."""
    n_pages = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    dmg = np.zeros(n_pages)
    all_pages = list(range(n_pages))
    for p in all_pages:
        keep = [q for q in all_pages if q != p]
        if not keep:
            keep = [p]
        z_abl, _ = gated_step(model, full_caches, query, seq, keep)
        dmg[p] = EM.kl(z_full, z_abl)
    return dmg


def page_norms(full_caches, seq):
    """||K|| per page, averaged over layers (cheap runtime proxy)."""
    import mlx.core as mx
    n_pages = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    norms = np.zeros(n_pages)
    for c in full_caches:
        k = np.array(c.keys[0, :, :seq, :].astype(mx.float32))  # [H_kv, seq, d]
        per_tok = np.linalg.norm(k, axis=(0, 2))  # [seq]
        for t in range(seq):
            norms[t // BLOCK_SIZE] += float(per_tok[t])
    return norms


# --- prompt preparation ---------------------------------------------------

def prefill(model, ids):
    """Prefill ids[:-1]; return (full_caches, first_query, seq). seq = len(ids)-1."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    caches = [KVCache() for _ in model.model.layers]
    _ = model(mx.array([ids[:-1]]), cache=caches)
    mx.eval([c.keys for c in caches])
    return caches, ids[-1], len(ids) - 1


def advance(model, full_caches, query):
    """Append `query`'s key to the full cache (teacher forcing)."""
    import mlx.core as mx
    _ = model(mx.array([[query]]), cache=full_caches)
    mx.eval([full_caches[0].keys])


# --- systems: real paged kernel -------------------------------------------

_OPS = None


def get_ops():
    global _OPS
    if _OPS is None:
        import os
        os.environ.setdefault("VLLM_METAL_BUILD_FROM_SOURCE", "1")
        from vllm_metal.metal import get_ops as g
        _OPS = g()
    return _OPS


def paged_all_pages_err(L, B, n_q, n_kv, d, seed=0):
    import mlx.core as mx
    ops = get_ops()
    rng = np.random.default_rng(seed)
    npg = (L + B - 1) // B
    qn = rng.standard_normal((1, n_q, d)).astype(np.float32) * 0.1
    kn = rng.standard_normal((npg, B, n_kv, d)).astype(np.float32) * 0.1
    vn = rng.standard_normal((npg, B, n_kv, d)).astype(np.float32) * 0.1
    q, kc, vc = (mx.array(x, dtype=mx.float16) for x in (qn, kn, vn))
    out = mx.zeros((1, n_q, d), dtype=mx.float16)
    ops.paged_attention_primitive(q, kc, vc, n_kv, float(d ** -0.5), 0.0,
                                  mx.array([list(range(npg))], dtype=mx.int32),
                                  mx.array([L], dtype=mx.int32),
                                  mx.array([0, 1], dtype=mx.int32), B, int(L), -1, out)
    mx.eval(out)
    got = np.array(out.astype(mx.float32))[0]
    K = kn.reshape(npg * B, n_kv, d)[:L].astype(np.float64)
    V = vn.reshape(npg * B, n_kv, d)[:L].astype(np.float64)
    g = n_q // n_kv
    ref = np.zeros((n_q, d))
    for h in range(n_q):
        lg = (qn[0, h].astype(np.float64) @ K[:, h // g].T) * (d ** -0.5)
        w = np.exp(lg - lg.max()); w /= w.sum()
        ref[h] = w @ V[:, h // g]
    return float(np.abs(got - ref).max())


def paged_latency(L, keep_pages, B, n_q, n_kv, d, iters=30, warmup=5, seed=0):
    import mlx.core as mx
    ops = get_ops()
    rng = np.random.default_rng(seed)
    npg = (L + B - 1) // B
    q = mx.array(rng.standard_normal((1, n_q, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    kc = mx.array(rng.standard_normal((npg, B, n_kv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    vc = mx.array(rng.standard_normal((npg, B, n_kv, d)).astype(np.float32) * 0.1, dtype=mx.float16)
    last = npg - 1
    R = sum(B if p < last else (L - last * B or B) for p in keep_pages)
    bt = mx.array([keep_pages], dtype=mx.int32)
    cu = mx.array([0, 1], dtype=mx.int32)

    def once():
        out = mx.zeros((1, n_q, d), dtype=mx.float16)
        ops.paged_attention_primitive(q, kc, vc, n_kv, float(d ** -0.5), 0.0, bt,
                                      mx.array([R], dtype=mx.int32), cu, B, int(R), -1, out)
        mx.eval(out)
    for _ in range(warmup):
        once()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); once(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[min(len(ts) - 1, int(0.95 * len(ts)))], R
