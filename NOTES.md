# NOTES — running, testing, and where to look

Personal reference for working on **vllm-metal** (vLLM on Apple Silicon via MLX).
Not part of the upstream docs; safe to edit/delete.

---

## 1. Serving a model

This is a **git checkout**, so the prebuilt Metal kernels are NOT present (they ship
only in release wheels). Building the `.metallib` shaders with `python -m vllm_metal.metal.build`
needs **full Xcode** (for `xcrun metal`) — Command Line Tools alone fail with
`xcrun: error: unable to find utility "metal"`.

**Workaround: serve with `VLLM_METAL_BUILD_FROM_SOURCE=1`**, which rebuilds the `.so` with
`clang++` and JIT-compiles the shaders in-process via MLX (no `xcrun metal`, CLT is enough):

```bash
source .venv-vllm-metal/bin/activate
export VLLM_METAL_BUILD_FROM_SOURCE=1
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 4096
```

Server comes up at `http://localhost:8000` (OpenAI-compatible) when the log shows
`Application startup complete.` Keep the env var set every run while on a checkout
(recompiles each startup, a few seconds). To avoid that: install a release wheel, or
install full Xcode and prebuild once.

Quick check it's alive:
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Gotchas:
- Requires native **arm64 Python 3.12** (Rosetta/x86_64 unsupported).
- Harmless `Found ulimit of 2048 ... fd limit errors` warning; under load can cause
  `OSError: [Errno 24] Too many open files` → pre-empt with `ulimit -n 8192`.
- R1-Distill / reasoning models "think out loud" and may ignore terse instructions —
  that's model behavior, not a serving bug.

---

## 2. KV-cache (prefix-cache) speed test

Measures TTFT for a query whose prefix is **cold** vs **already in the KV cache**. Sends the
same long prompt twice per trial; the cold/warm TTFT ratio is the KV-cache speedup. `~50%`
cumulative hit rate is the built-in self-check (1 miss + 1 hit per trial).

Server must be running, then:
```bash
source .venv-vllm-metal/bin/activate
python tools/benchmark/kv_cache_prefix_benchmark.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --warmup
```
Knobs: `--prefix-words 3000` (widen the gap), `--trials N`, `--max-tokens N`, `--base-url URL`.

Reference result (DeepSeek-R1-Distill-Qwen-1.5B, 1723-token prompt): cold ~1188 ms,
warm ~115 ms → **~10x**, hit rate ~50%.

> Non-obvious: the per-trial cache-buster nonce must be at **token position 0** (prefix-cache
> block hashes chain from the start); use `/v1/completions` (no chat template); stream to time
> the first token. See the script's docstring.

---

## 3. Running the test suite

```bash
source .venv-vllm-metal/bin/activate
pytest                      # all tests (testpaths = ["tests"])
pytest -m "not slow"        # skip slow tests
pytest tests/test_attention_sdpa.py -q   # one file
```

- `scripts/test.sh` — golden-output **smoke test** helper: serves a model with
  `VLLM_METAL_USE_PAGED_ATTENTION=1` and checks generated text against a golden string.
- `scripts/lint.sh` — ruff + mypy. `scripts/release.sh` — release build.
- Golden-token / parity tools live in `tools/` (e.g. `gen_golden_token_ids_for_deterministics.py`,
  `pp_parity_check.py`, `awq_parity.py`).

---

## 4. Where to look (inference: O(n²) attention + KV cache)

Path is layered: **Python orchestration → MLX glue → C++ dispatch → Metal kernel.**
Prefill and decode both run through the unified paged varlen kernel (README v0.2.0).

### The O(n²) attention math (QK·V over the sequence)
- `vllm_metal/metal/kernels_v2/pagedattention.metal` — **core kernel; the actual QK·V** (online softmax).
- `vllm_metal/metal/kernels_v2/pagedattention_tiled.metal` — tiled variant for larger workloads.
- `vllm_metal/metal/kernels_v2/mla.metal` — Multi-head Latent Attention (DeepSeek-style).
- `vllm_metal/metal/paged_ops.cpp` — C++/nanobind dispatch of the shaders through MLX (the `.so`).
- `vllm_metal/attention/impls/sdpa.py` — **Python entry point: read first.** Builds Q/K/V + RoPE,
  invokes the paged-attention primitive (Qwen/Llama/Mistral/Gemma incl. our DeepSeek-distill).
- `vllm_metal/attention/runtime/mha.py` — wires model attention layers to the kernel + owns the
  cache (`runtime/mla.py` for MLA, `runtime/hybrid.py` for GDN/hybrid).

### The KV cache
- `vllm_metal/attention/caches/kv_cache.py` — **the paged KV cache** (`MetalPagedKVCache`), MLX
  arrays `[num_blocks, block_size, num_kv_heads, head_dim]`. Start here.
- `vllm_metal/metal/kernels_v2/reshape_and_cache.metal` — **writes** new K/V into the cache.
- `vllm_metal/metal/kernels_v2/gather_kv_cache.metal`, `copy_blocks.metal` — read/gather + block
  copy (prefix-cache block reuse).
- `vllm_metal/attention/caches/mla_cache.py`, `turboquant.py`, `gdn_cache.py` — latent / quantized
  / GDN-state cache variants.

### Driving loop
`vllm_metal/v1/model_runner.py` → `vllm_metal/v1/worker.py` per step: write new K/V via
`reshape_and_cache.metal`, then run `pagedattention.metal` against cached blocks.

**Reading order:** `impls/sdpa.py` → `caches/kv_cache.py` → `kernels_v2/pagedattention.metal`.
