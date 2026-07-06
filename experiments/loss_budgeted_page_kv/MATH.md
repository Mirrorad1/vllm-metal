# MATH — Loss-Budgeted Page KV Cache

Symbols: `L` context tokens, `B` tokens/page, `P=ceil(L/B)`, `J` kept pages,
`R≤J·B` kept tokens, `N` layers, `H_q/H_kv` heads, `d` head dim, `s` bytes/scalar.

## 1. Memory & work

    M_full(L) = 2·N·H_kv·d·s·B·ceil(L/B) + M_block_table
    M_gate(J) = 2·N·H_kv·d·s·B·J + M_selected_block_table + M_meta
    Work_full(L)=Θ(N·H_q·L·d),  Work_gate(R)=Θ(N·H_q·R·d)

Break-even (must hold in WALL-CLOCK, not just asymptotics):

    T_ctl + T_tbl + T_attn_gate < T_attn_full
    ⇔ T_ctl + T_tbl < c_f·N·H_q·d·(L−R) + c_p·N·H_q·(P−J)

`c_f`,`c_p` fit from measured full-vs-gated kernel latency (`systems.csv`). The
operating rule: an asymptotic win is not a win until measured end-to-end.

## 2. Correctness invariant (enabling)

For RoPE-only decode the kernel masks causally **by token count**
(`token_idx ≥ effective_context_len`) and reads K with its write-time
(absolute-position) RoPE; the query's position is supplied independently. So a
chronological page subset `S` with `context_len(S)=Σ_{i∈S} valid(i)` gives EXACT
attention over the selected tokens **iff every selected page except the last is
full** (only the tail is ever partial). Proof: the kernel visits
`ceil(context_len(S)/B)` compact columns, each interior column contributing `B`
valid tokens and the last `context_len(S)−(J−1)B`; with full interiors this is
exactly `Σ valid(i)`. Verified: kernel-vs-numpy err ≈ 4e-6; `full_pages`
behavioral KL = 0. `build_selection` enforces the invariant.

## 3. Page-damage scoring (the loss-budgeted signal)

ORACLE per-page damage (calibration; full-cache logits only, NO labels):

    damage(p) = KL( softmax(z_full) || softmax(z_{ablate p}) )

Horizon form: `damage_h(p)=(1/h)Σ_j KL(p_full,t+j || p_{∖p},t+j)`. Keep top-`J`
pages by `damage`. Per-action value (later action space):

    value(p,a) = damage_increase(p,a) / bytes_saved(p,a)

Initial action space = {keep, evict}. Quantize/offload/recompute deferred.

## 4. Behavioral error & the COMPLETE gate

    D_h = (1/h) Σ_j KL(p_full,t+j || p_gate,t+j)   (teacher-forced)

A method **dominates** a baseline only if, at equal budget, it is no worse on ALL
of {physical memory, end-to-end latency, D_h, task accuracy} and strictly better
on ≥1. Hard rules (never relaxed):
- logical page reduction ≠ physical memory reduction (F4): only count bytes
  actually not-allocated/freed safely (F15).
- kernel-only speedup ≠ end-to-end speedup (F5/S3): controller + table-build +
  sync + copies are included in gated latency.
- `D_h` must hold across context lengths {512,1024,2048,4096,8192} (F8) and
  seeds (F14), and on natural prompts (F12), or the win is scale/noise/synthetic.

## 5. Controller complexity (F9)

If `T_ctl=Θ(P)` per token and the total still doesn't beat full, that's
COMPLEXITY-NEGATIVE. The oracle damage signal is `Θ(P)` *forward passes* per
step — explicitly a diagnostic, never deployable. A deployable controller must be
≈`O(log P)`/incremental (exp008). The attention-mass proxy is `Θ(P)` cheap
arithmetic (one extra softmax-bin during the forward already being run), and an
incremental running per-page mass makes it amortized `O(B)` per token.
