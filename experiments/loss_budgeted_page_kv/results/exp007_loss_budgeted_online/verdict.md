Experiment: exp007_loss_budgeted_online
| policy | bf=0.25 | bf=0.125 | bf=0.0625 |
|---|---|---|---|
| loss_budgeted_online | 0.047/0.98 | 0.081/0.98 | 0.575/0.86 |
| attention_proxy_pages | 0.021/1.00 | 0.074/1.00 | 0.548/0.88 |
| loss_budgeted_oracle | 0.028/1.00 | 0.069/1.00 | 0.549/0.91 |
| recent_pages | 1.843/0.63 | 1.851/0.63 | 1.884/0.63 |
online vs recent @12.5% (sig): True; online vs attention_proxy: False
Verdict: WEAK POSITIVE / INCONCLUSIVE — online beats recent/random but not attention_proxy (F7); attention mass alone is the recoverable signal.
Next experiment: exp009_physical_memory.
