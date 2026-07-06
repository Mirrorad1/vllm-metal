# SPDX-License-Identifier: Apache-2.0
"""KV-cache (prefix-cache) speed benchmark against a running vLLM server.

Measures how much faster a query is when its prompt prefix is already in the
KV cache (a prefix-cache HIT) versus cold. The signal is time-to-first-token
(TTFT): on a warm prefix, vLLM skips re-prefilling the cached blocks, so TTFT
collapses.

Each trial sends the SAME long prompt twice:
  1. COLD  - the prompt begins with a unique nonce, so every prefix-cache block
             hash is new => full prefill => high TTFT.
  2. WARM  - identical prompt; the prefix is now cached => prefill skipped for
             the cached blocks => low TTFT.

The cold/warm TTFT ratio is the KV-cache speedup. Uses the /v1/completions
endpoint (raw prompt, no chat template) with streaming to time the first token.

Run (server must already be up, e.g. `vllm serve <model> --max-model-len 4096`):

    source <repo>/.venv-vllm-metal/bin/activate
    python tools/benchmark/kv_cache_prefix_benchmark.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Requires only `requests` (already in the vllm-metal venv).
"""

from __future__ import annotations

import argparse
import statistics
import time

import requests

# A neutral filler sentence repeated to build a long, deterministic prefix.
# Repetition is fine: the prefix only needs to be long enough to span many
# 16-token cache blocks so a hit saves real prefill work.
_FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer profiles "
    "the paged attention kernel and inspects the key value cache occupancy. "
)


def build_prompt(nonce: str, target_words: int, question: str) -> str:
    """A unique nonce at position 0 (busts the cache) + long shared body + question.

    The nonce sits at the very start so that, on the cold call, the first cache
    block's hash differs and every downstream block hash (which chains from it)
    differs too -- guaranteeing a full miss. The warm call reuses the identical
    string, so the whole prefix hits.
    """
    body_words: list[str] = []
    while len(body_words) < target_words:
        body_words.extend(_FILLER.split())
    body = " ".join(body_words[:target_words])
    return f"[session {nonce}] {body}\n\nQuestion: {question}\nAnswer:"


def ttft_once(base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    """Send one streamed completion; return (TTFT seconds, prompt_tokens).

    TTFT is measured from just before the POST to the arrival of the first
    streamed chunk that carries generated text.
    """
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    prompt_tokens = 0
    start = time.perf_counter()
    ttft: float | None = None
    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[len("data: "):]
            if data == "[DONE]":
                break
            import json

            chunk = json.loads(data)
            if ttft is None:
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    ttft = time.perf_counter() - start
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
    if ttft is None:
        raise RuntimeError("No token was streamed back; cannot measure TTFT.")
    return ttft, prompt_tokens


def get_prefix_hit_rate(base_url: str) -> float | None:
    """Best-effort read of the cumulative prefix-cache hit rate from /metrics."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/metrics", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    queries = hits = None
    for line in resp.text.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total"):
            queries = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:prefix_cache_hits_total"):
            hits = float(line.rsplit(" ", 1)[-1])
    if queries and hits is not None and queries > 0:
        return hits / queries
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True, help="Served model name (see /v1/models)")
    p.add_argument("--trials", type=int, default=5, help="Cold/warm pairs to run")
    p.add_argument("--prefix-words", type=int, default=1500,
                   help="Approx shared-prefix length in words (~1.3 tokens/word)")
    p.add_argument("--max-tokens", type=int, default=8,
                   help="Generated tokens per call (small: we only time TTFT)")
    p.add_argument("--warmup", action="store_true",
                   help="Do one throwaway pair first (loads kernels / JIT paths)")
    args = p.parse_args()

    print(f"Server : {args.base_url}")
    print(f"Model  : {args.model}")
    print(f"Trials : {args.trials}  |  prefix~{args.prefix_words} words  |  max_tokens={args.max_tokens}\n")

    if args.warmup:
        wp = build_prompt("warmup", args.prefix_words, "Warm up the engine.")
        ttft_once(args.base_url, args.model, wp, args.max_tokens)
        ttft_once(args.base_url, args.model, wp, args.max_tokens)

    cold_ttfts: list[float] = []
    warm_ttfts: list[float] = []
    prompt_tokens = 0

    for i in range(args.trials):
        # Unique nonce per trial => trial i's cold call is genuinely uncached.
        nonce = f"t{i}-{time.perf_counter_ns()}"
        prompt = build_prompt(nonce, args.prefix_words, "Summarize the text above in one word.")

        cold, prompt_tokens = ttft_once(args.base_url, args.model, prompt, args.max_tokens)
        warm, _ = ttft_once(args.base_url, args.model, prompt, args.max_tokens)
        cold_ttfts.append(cold)
        warm_ttfts.append(warm)
        speedup = cold / warm if warm else float("inf")
        print(f"  trial {i+1}/{args.trials}: cold TTFT {cold*1000:7.1f} ms | "
              f"warm TTFT {warm*1000:7.1f} ms | speedup {speedup:4.1f}x")

    def stat(xs: list[float]) -> str:
        return f"median {statistics.median(xs)*1000:7.1f} ms  mean {statistics.mean(xs)*1000:7.1f} ms"

    med_cold = statistics.median(cold_ttfts)
    med_warm = statistics.median(warm_ttfts)
    print(f"\nPrompt tokens (measured): {prompt_tokens}")
    print(f"COLD  TTFT: {stat(cold_ttfts)}")
    print(f"WARM  TTFT: {stat(warm_ttfts)}")
    print(f"KV-cache speedup (median cold / median warm): {med_cold/med_warm:.1f}x")

    hit_rate = get_prefix_hit_rate(args.base_url)
    if hit_rate is not None:
        print(f"Server cumulative prefix-cache hit rate: {hit_rate*100:.1f}%")
    else:
        print("Prefix-cache hit rate: (/metrics unavailable; check server log "
              "lines 'Prefix cache hit rate')")


if __name__ == "__main__":
    main()
