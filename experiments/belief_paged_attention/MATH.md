# MATH — baselines, the correctness invariant, and the break-even model

Symbols (per the task spec):

| sym | meaning |
|---|---|
| `L` | full effective context length (tokens) |
| `B` | tokens per KV page (block_size; 16 in the behavioral harness, ∈{8,16,32} kernel) |
| `P = ceil(L/B)` | full logical page count |
| `J` | selected page count |
| `R ≤ J·B` | selected valid KV tokens |
| `N` | transformer layers (24 for Qwen2.5-0.5B) |
| `H_q, H_kv` | query / KV heads (14 / 2 for Qwen2.5-0.5B) |
| `d` | head dim (64) |
| `s` | bytes per cached scalar (2, fp16) |
| `G` | generated tokens |
| `T_u, T_s, T_tbl` | belief-update / page-scoring / table-build time |

---

## 1. Standard PagedAttention baseline

PagedAttention does **not** reduce attended tokens. Per layer, decode arithmetic:

    F_full(L) ≈ 4 · H_q · L · d        (QK + weighted-V)

Per-token attention time:

    T_full(L) = Θ(H_q · L · d)  +  Θ(H_q · ceil(L/B))
                └ KV walk ┘        └ page-table traversal ┘

Full KV storage per sequence (includes page-rounding waste):

    M_full(L) = 2 · N · H_kv · d · s · B · ceil(L/B)  +  M_block_table
    M_block_table ≈ 4 · ceil(L/B) bytes

For G generated tokens after context L_0:

    F_full,total = Θ( H_q · d · (G·L_0 + G·(G-1)/2) )

We **measure** latency rather than trust the asymptotics, because the Metal
decode kernel is bandwidth/overhead-bound at the context lengths tested (see §5).

---

## 2. The correctness invariant (the enabling theorem)

**Claim.** For RoPE-only decode on the non-tiled kernel `pagedattention.metal`,
let a sequence have logical pages `0..P-1` (page `i` covering tokens
`[iB, (i+1)B)`, only page `P-1` possibly partial with `r = L-(P-1)B` valid
tokens). Choose a chronological index subset `S = {i_0 < i_1 < … < i_{J-1}}` and
present the kernel a compact block table `[phys(i_0),…,phys(i_{J-1})]` with

    context_len(S) = Σ_{i∈S} valid(i),   valid(i)=B for i<P-1, valid(P-1)=r.

If **every page in S except possibly the last is full** (i.e. `i_k < P-1` ⇒
`valid(i_k)=B`), then the kernel output equals exact attention of the query over
exactly the tokens in the selected pages.

**Proof.** The kernel (audit §4) computes, for compact column `block_idx`,
`token_idx = block_idx·B + offset` and the causal mask `token_idx ≥
effective_context_len` with `effective_context_len = context_len(S)` for decode
(`q_len=1, q_pos=0`). It iterates `num_context_blocks = ceil(context_len(S)/B)`
columns and, per block, processes `block_valid = min(B, context_len(S) −
block_idx·B)` tokens. Since interior selected pages are full, the first `J-1`
compact columns each contribute exactly `B` valid tokens and the last
contributes `context_len(S) − (J-1)B = valid(i_{J-1})`. Thus the kernel visits
exactly `Σ valid(i)` tokens — the selected pages' tokens — with **no** token
masked that should be kept and none kept that should be masked. Each visited key
`k` is read from its physical page with its **write-time RoPE rotation at its
original absolute position**; the query is rotated at its true absolute position
(supplied independently, not derived from the compact layout). Hence every
`q·k` carries the correct relative rotation, and the online-softmax over the
visited set is exactly attention over the selected tokens. ∎

**Why the invariant is automatically satisfiable.** In a live sequence only page
`P-1` is partial. Any chronological subset that keeps `P-1` last (guaranteed by
sorting) — or omits it, leaving only full pages — satisfies the hypothesis.
`page_policies.build_selection` enforces it and raises otherwise.

**Empirical confirmation.** `all_pages_equivalence` (kernel vs numpy):
`max|·| ≈ 4.3e-6` at L=1024; a 3-of-6-page compact selection: `1.5e-5`; the
behavioral harness `full_pages` budget gives `D_h = KL = 0.0000` exactly. The
fp16 noise floor, i.e. exact.

---

## 3. Belief-gated cost and storage

Runtime state `M_t = (B_t, E_t)`; belief update `B_t = f(B_{t-1}, x_t)` uses only
≤ t information (enforced: the harness feeds `update` the prefix only, and the
cursor is incremental — `Θ(|chunk|)`, never `Θ(L)`; falsifier #9).

    T_gate = T_u + T_s + T_tbl + Θ(H_q·R·d) + Θ(H_q·J)
    M_gate = 2·N·H_kv·d·s·B·J + b + M_selected_block_table

**Shadow-mode honesty.** In experiment 1 the full physical cache stays allocated
(no reclamation — audit §5). So `M_gate` is the *would-be* footprint of `J`
pages, reported as **logical/page reduction**, NOT bytes freed. The four
quantities are tracked separately in `metrics.SystemsAccount`: logical tokens
attended `R`, physical pages retained `J`, would-be bytes for `J` pages, and the
still-allocated full bytes.

If every page is rescored every token, `T_s = Θ(P·f_score)`; our selectors are
`Θ(P)` (a sort over pages) and the belief update is `Θ(|chunk|)`, so per-step
control cost is `Θ(P)` — context-independent decode requires `J` (and the active
belief set) bounded, which `BeliefState.active_pages(recency=…)` enforces.

---

## 4. Break-even

Net gating wins on latency iff

    T_u + T_s + T_tbl  <  T_full(L) − T_attention(R)
                       ≈  c_f·H_q·d·(L−R) + c_p·H_q·(P−J)

`c_f`, `c_p` are fit from measured full-vs-gated kernel latency (systems.csv).
Asymptotic savings do **not** guarantee wall-clock savings; at small L the
kernel is overhead-bound and the control plane can dominate (see §5, falsifiers
#2/#3).

---

## 5. Fitted constants (from this run)

Per-call decode kernel latency (single layer, M-series GPU; `paged_kernel_latency`,
warmup excluded, p50), Qwen2.5-0.5B head config (H_q=14, H_kv=2, d=64, B=16):

(Representative values; authoritative numbers in `results/systems.csv`.)

| L | full p50 | gated@25% (J,R) | gated@6.25% (J,R) |
|---|---|---|---|
| 1024 | ~0.210 ms | ~0.166 ms (15, 240) | ~0.143 ms (3, 48) |

A single decode step runs the kernel `N=24` times, so per-step kernel savings
≈ `N·(full−gated)`. At L=1024, 6.25%: `≈ 24·(0.210−0.143) ≈ 1.6 ms/step` saved,
against a measured control-plane overhead `T_u+T_s+T_tbl ≈ 1 ms/step` (dominated
by `tokenizer.decode` in the belief update, an implementation cost, not
fundamental). ⇒ **near break-even at L≈1024**; the margin grows with L (kernel
walk ∝ L) and shrinks the control overhead's relative weight. This is the honest
wall: at the *tested* context lengths the net latency win is small and only
clearly positive at the largest contexts / smallest budgets. The fit
`c_f, c_p` and the crossing point are reported in `results/summary.md`.

---

## 6. Behavioral locus

Full model `p_t(y)=p(y|C_t)`, gated `q_t(y)=q(y|B_t,E_t)`. Teacher-forced
horizon-`h` behavioral error

    D_h = (1/h) Σ_{j=1..h} KL( p_{t+j}^full || q_{t+j}^gate )

measured on the real model (mlx-lm, identical continuation tokens fed to both).
Acceptable locus `L_(ε,δ) = { (B_t,E_t) : D_h ≤ ε ∧ task_error ≤ δ }`. The
empirical Pareto frontier is over (physical/would-be cache bytes, decode latency,
D_h, answer accuracy); gating dominates a baseline only if no worse on all four
and strictly better on one. See `results/summary.md` for the frontier.
