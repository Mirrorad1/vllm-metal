# SPDX-License-Identifier: Apache-2.0
"""Correctness and systems metrics for the belief-gated experiment.

Correctness metrics compare a *gated* next-token distribution ``q`` against the
*full-cache* reference ``p`` (teacher-forced): KL, JS, top-1 agreement, top-k
overlap, plus attention-output relative error. Systems metrics are accumulated
by the harness.

All distribution metrics take raw logits (numpy float arrays) and apply a
numerically stable softmax internally so callers never double-normalize.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(p || q) over the next-token distribution, in nats."""
    p = softmax(p_logits)
    q = softmax(q_logits)
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def js_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded by ln 2), in nats."""
    p = softmax(p_logits)
    q = softmax(q_logits)
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * (np.log(a + 1e-12) - np.log(b + 1e-12))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def top1_agree(p_logits: np.ndarray, q_logits: np.ndarray) -> bool:
    return bool(np.argmax(p_logits) == np.argmax(q_logits))


def topk_overlap(p_logits: np.ndarray, q_logits: np.ndarray, k: int = 10) -> float:
    """Fraction of the full model's top-k tokens also in the gated top-k."""
    pk = set(np.argsort(-p_logits)[:k].tolist())
    qk = set(np.argsort(-q_logits)[:k].tolist())
    return len(pk & qk) / float(k)


def attention_rel_error(full_out: np.ndarray, gated_out: np.ndarray) -> float:
    """Relative L2 error of attention output (per-token), ||g-f|| / ||f||."""
    num = np.linalg.norm((gated_out - full_out).ravel())
    den = np.linalg.norm(full_out.ravel()) + 1e-12
    return float(num / den)


# ---------------------------------------------------------------------------
# Systems accounting
# ---------------------------------------------------------------------------


@dataclass
class SystemsAccount:
    """Per-(task, policy, budget, trial) systems metrics.

    Distinguishes the four quantities the spec demands be kept separate:
    logical tokens attended, physical pages retained, physical bytes that *would*
    be allocated for the retained pages, and (in shadow mode) the fact that the
    full physical cache is still allocated.
    """

    logical_tokens_attended: int = 0   # R
    full_logical_tokens: int = 0       # L
    physical_pages_retained: int = 0   # J
    full_physical_pages: int = 0       # P
    pages_reclaimed: int = 0           # 0 in shadow mode (honest)
    block_table_bytes: int = 0         # 4 * J
    full_block_table_bytes: int = 0    # 4 * P
    selected_cache_bytes: int = 0      # bytes for J pages (would-be)
    full_cache_bytes: int = 0          # bytes for P pages (still allocated in shadow)
    shadow_mode: bool = True           # full cache still allocated → no real bytes freed

    # timing (seconds); warmup excluded by the harness
    belief_update_s: float = 0.0       # T_u
    page_score_s: float = 0.0          # T_s
    table_build_s: float = 0.0         # T_tbl
    kernel_attention_s: float = 0.0    # T_attention(R)

    @property
    def control_plane_overhead_s(self) -> float:
        return self.belief_update_s + self.page_score_s + self.table_build_s

    @property
    def total_gated_s(self) -> float:
        return self.control_plane_overhead_s + self.kernel_attention_s


def cache_bytes(num_pages: int, block_size: int, n_layers: int, n_kv_heads: int,
                head_dim: int, bytes_per_scalar: int = 2) -> int:
    """Physical KV bytes for ``num_pages`` pages: 2 (K&V) * N * H_kv * d * s * B * pages."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_scalar * block_size * num_pages


@dataclass
class Aggregate:
    """Accumulates per-trial records for confidence intervals."""

    records: list[dict] = field(default_factory=list)

    def add(self, rec: dict) -> None:
        self.records.append(rec)

    @staticmethod
    def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
        """Return (mean, half-width of 95% CI). Half-width 0 for n<2."""
        if not values:
            return (float("nan"), float("nan"))
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        if arr.size < 2:
            return (mean, 0.0)
        sd = float(arr.std(ddof=1))
        return (mean, z * sd / math.sqrt(arr.size))
