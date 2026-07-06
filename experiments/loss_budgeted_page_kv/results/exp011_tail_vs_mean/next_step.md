Next experiment (decided by exp011 = KL-ONLY / behaviorally inert):

The page-SCORING question is now closed for single-needle copy retrieval at 0.5B:
attention ≡ oracle on the answer even in the tail; prefill information diffusion
leaves no behavioral headroom. Two mutually exclusive next steps:

A. PIVOT TO SYSTEMS (recommended, default). The honest wall is physical, not
   signal. Build the smallest real physical-reclamation step: free/avoid
   allocating unselected PRIVATE (refcount==1) pages, ref-count- and prefix-safe,
   and measure peak physical bytes + end-to-end latency vs J/P. This is the lacuna
   the field analysis named (importance with byte authority). Only this converts
   the strong LOGICAL result into a real win.

B. STRESS THE ONE UNTESTED REGIME that could revive scoring (do this ONLY before
   pivoting if the multi-hop hypothesis is worth one shot): replace single-needle
   COPY retrieval with MULTI-HOP composition where the answer requires combining
   two distant facts on different pages (so diffusion cannot redundantly encode the
   final answer), at 7B+/GQA scale. Re-run the SAME answer-level McNemar gate. If
   oracle rescues answers attention loses there (McNemar p<0.01, lower bound>3%),
   the behavioral tail is real for hard tasks and scoring is revived; if not,
   scoring is dead across the board and only systems remains.

Recommendation: A. The mean-of-opposites was the strongest remaining doubt about
"scoring is saturated"; it is now falsified behaviorally. Stop optimizing signals.
