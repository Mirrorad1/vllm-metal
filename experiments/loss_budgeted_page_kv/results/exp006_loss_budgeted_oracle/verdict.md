Experiment: exp006_loss_budgeted_oracle
Hypothesis: some pages are much cheaper to drop under logit error; an oracle keeping highest-KL-damage pages dominates baselines.
| policy | bf=0.25 | bf=0.125 | bf=0.0625 |
|---|---|---|---|
| loss_budgeted_oracle | 0.028/1.00 | 0.069/1.00 | 0.549/0.91 |
| attention_proxy_pages | 0.021/1.00 | 0.074/1.00 | 0.548/0.88 |
| recent_pages | 1.843/0.63 | 1.851/0.63 | 1.884/0.63 |
oracle vs recent @12.5% (sig beat): True; oracle vs attention_proxy: False
Verdict: INCONCLUSIVE — KL-damage structure exists (oracle ≫ recent) but does NOT beat attention_proxy (F7): attention mass already captures the structure.
Next experiment: exp007_loss_budgeted_online.
