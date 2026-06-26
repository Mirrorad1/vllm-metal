Experiment: exp009_physical_memory
Hypothesis: logical page reduction reduces physical KV memory.
Result: pages_reclaimed=0; physical_bytes_actual == full_bytes (read-only compact block table; allocator owned by upstream vLLM; no reclamation implemented). Would-be bytes shrink ~J/P but are NOT freed.
Verdict: LOGICAL-ONLY POSITIVE (F4) — no real physical reduction.
Next experiment: exp010_latency_frontier (and a real allocator path).
