# SPDX-License-Identifier: Apache-2.0
"""Behavioral-error and systems-accounting metrics (KV-cache only).

No semantics, no task-state inference — purely logit/output/memory/latency math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(softmax(p) || softmax(q)) in nats, >= 0."""
    p, q = softmax(p_logits), softmax(q_logits)
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def js(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p, q = softmax(p_logits), softmax(q_logits)
    m = 0.5 * (p + q)
    _k = lambda a, b: float(np.sum(a * (np.log(a + 1e-12) - np.log(b + 1e-12))))
    return 0.5 * _k(p, m) + 0.5 * _k(q, m)


def top1_agree(p_logits: np.ndarray, q_logits: np.ndarray) -> bool:
    return bool(np.argmax(p_logits) == np.argmax(q_logits))


def topk_overlap(p_logits: np.ndarray, q_logits: np.ndarray, k: int = 10) -> float:
    pk = set(np.argsort(-p_logits)[:k].tolist())
    qk = set(np.argsort(-q_logits)[:k].tolist())
    return len(pk & qk) / float(k)


def rel_output_error(full_out: np.ndarray, gated_out: np.ndarray) -> float:
    num = np.linalg.norm((gated_out - full_out).ravel())
    den = np.linalg.norm(full_out.ravel()) + 1e-12
    return float(num / den)


def page_bytes(num_pages: int, block_size: int, n_layers: int, n_kv_heads: int,
               head_dim: int, bytes_per_scalar: int = 2) -> int:
    """Physical KV bytes for num_pages: 2(K&V)*N*H_kv*d*s*B*pages."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_scalar * block_size * num_pages


@dataclass
class SystemsAccount:
    """Keeps logical, physical-would-be, physical-reclaimed, and latency
    separate so a verdict can never count logical compression as a physical or
    latency win (operating rule)."""
    logical_tokens_attended: int = 0
    full_logical_tokens: int = 0
    pages_retained: int = 0
    full_pages: int = 0
    pages_reclaimed: int = 0          # 0 unless real allocator reclamation (exp009)
    physical_bytes_allocated: int = 0  # ACTUAL (shadow mode: == full)
    would_be_bytes: int = 0            # for pages_retained (not freed in shadow mode)
    full_bytes: int = 0
    shadow_mode: bool = True
    controller_s: float = 0.0          # T_ctl
    table_build_s: float = 0.0         # T_tbl
    kernel_s: float = 0.0              # T_attn_gate (per call)

    @property
    def overhead_s(self) -> float:
        return self.controller_s + self.table_build_s


def mean_ci(values, z: float = 1.96):
    if not values:
        return (float("nan"), float("nan"))
    a = np.asarray(values, dtype=np.float64)
    m = float(a.mean())
    if a.size < 2:
        return (m, 0.0)
    return (m, z * float(a.std(ddof=1)) / math.sqrt(a.size))


@dataclass
class Aggregate:
    records: list = field(default_factory=list)

    def add(self, rec: dict):
        self.records.append(rec)
