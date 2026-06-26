Experiment: exp001_audit_dispatch
Hypothesis: the active PagedAttention dispatch can be traced.
Result: see IMPLEMENTATION_AUDIT.md — decode dispatches kernels_v2/pagedattention.metal (count-masked, position-agnostic); tiled prefill kernel UNSAFE; allocator owned by upstream vLLM.
Verdict: POSITIVE (audit complete).
Next experiment: exp002_all_pages_equivalence.
