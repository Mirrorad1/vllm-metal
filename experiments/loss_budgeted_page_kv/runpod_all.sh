#!/usr/bin/env bash
# exp022 one-shot pod driver: s23 then s4, strictly sequential, double-launch-proof.
# Usage on the pod (the ONLY line you need):
#   nohup bash runpod_all.sh > run.log 2>&1 &
# then watch with: tail -f run.log
set -e
cd "$(dirname "$0")"

if [ "$(pgrep -fc exp022_substrates || true)" -gt 0 ]; then
    echo "ABORT: an exp022_substrates process is already running (pgrep below). One at a time."
    pgrep -af exp022_substrates
    exit 1
fi

S23=results/exp022_7b/s23.jsonl
if [ "$(wc -l < "$S23" 2>/dev/null || echo 0)" -eq 225 ]; then
    echo "=== s23 already complete (225 rows) — skipping ==="
else
    echo "=== stage s23 (~25 min; 4 gates must PASS) ==="
    python exp022_substrates.py --stage s23
    [ "$(wc -l < "$S23")" -eq 225 ] || { echo "ABORT: s23 wrote $(wc -l < "$S23") rows, want 225"; exit 1; }
fi

echo "=== stage s4 (~2h; recipe line below must say epochs=10 ... overlap=64) ==="
python exp022_substrates.py --stage s4 --resume

echo "=== ALL DONE — now run:  runpodctl send results/exp022_7b ==="
