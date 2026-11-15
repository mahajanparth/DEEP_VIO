#!/usr/bin/env bash
# Runs run_euroc on EuRoC sequences and computes ATE against ground truth.
# Usage: scripts/eval_euroc.sh [config.yaml] [SEQ ...]   (default: all MH sequences)
# Outputs in results/runs/: <SEQ>.txt (TUM trajectory), <SEQ>.json, <SEQ>.png, summary.md
# Sequences run in parallel (JOBS, default 3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$ROOT/config/euroc_mono.yaml}"
shift || true
SEQS=("$@")
[ ${#SEQS[@]} -eq 0 ] && SEQS=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult)
DATA="${EUROC_ROOT:-$ROOT/datasets/euroc/machine_hall}"
OUT="${OUT:-$ROOT/results/runs}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"

run_one() {
  local s=$1 mav0="$DATA/$1/$1/mav0"
  "$ROOT/build/run_euroc" "$mav0" "$CONFIG" "$OUT/$s.txt" > "$OUT/$s.log" 2>&1
  "$PY" -I "$ROOT/scripts/eval_ate.py" "$mav0/state_groundtruth_estimate0/data.csv" "$OUT/$s.txt" \
    --json "$OUT/$s.json" --plot "$OUT/$s.png" > /dev/null
  echo "done $s"
}
export -f run_one
export ROOT DATA OUT CONFIG PY
printf "%s\n" "${SEQS[@]}" | xargs -P "${JOBS:-3}" -I{} bash -c 'run_one {}'

"$PY" -I - "$OUT" "${SEQS[@]}" <<'EOF'
import json, sys
out, seqs = sys.argv[1], sys.argv[2:]
rows = ["| sequence | length [m] | ATE SE3 RMSE [m] | ATE Sim3 RMSE [m] | scale | drift [%] |",
        "|---|---|---|---|---|---|"]
for s in seqs:
    r = json.load(open(f"{out}/{s}.json"))
    rows.append(f"| {s} | {r['gt_length_m']:.1f} | {r['ate_se3_rmse_m']:.3f} | "
                f"{r['ate_sim3_rmse_m']:.3f} | {r['scale']:.3f} | {r['drift_pct']:.2f} |")
open(f"{out}/summary.md", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
