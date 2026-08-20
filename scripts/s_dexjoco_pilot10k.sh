#!/usr/bin/env bash
# Single-seed 10k pilot from the best bounded 1k checkpoint.
# Keeps strict sample_health thresholds; does NOT start 50k.
set -Eeuo pipefail
ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SRC_GATE="${SRC_GATE:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-gate500}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-pilot10k}"
export DEXORA_STATS="${DEXORA_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
export MAX_TRAIN_STEPS=10000
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=2500,5000,7500,10000
export SAVE_INITIAL_CHECKPOINT=0
export RESUME_FROM_CHECKPOINT=checkpoint-1000
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export SEED=42
HEALTH_DIR="${HEALTH_DIR:-/mnt/hdd/dexora/logs}"

mkdir -p "$OUTPUT_DIR" "$HEALTH_DIR"
if [[ ! -d "$OUTPUT_DIR/checkpoint-1000" ]]; then
  echo "==> seeding pilot dir from $SRC_GATE/checkpoint-1000"
  if [[ ! -d "$SRC_GATE/checkpoint-1000" ]]; then
    echo "Missing source checkpoint-1000 under $SRC_GATE" >&2
    exit 1
  fi
  cp -a "$SRC_GATE/checkpoint-1000" "$OUTPUT_DIR/checkpoint-1000"
  # Keep a copy of the gate 1k health baseline for comparison.
  if [[ -f "$SRC_GATE/config.json" ]]; then
    cp -a "$SRC_GATE/config.json" "$OUTPUT_DIR/config.json" || true
  fi
fi

echo "==> 10k pilot resume from checkpoint-1000 -> $OUTPUT_DIR (GPU ${CUDA_VISIBLE_DEVICES})"
bash scripts/s_dexjoco_finetune.sh

echo "==> multi-seed raw health at 1000,2500,5000,7500,10000 (parallel)"
pids=()
for seed in 0 1 2; do
  (
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -m eval_sim.sample_health \
      --ckpt-root "$OUTPUT_DIR" \
      --steps 1000,2500,5000,7500,10000 \
      --noise-seed "$seed" \
      --stats-file "$DEXORA_STATS" \
      --out "$HEALTH_DIR/dexjoco_relative_rot_vpred_pilot10k_health_seed${seed}.json" \
      > "$HEALTH_DIR/dexjoco_relative_rot_vpred_pilot10k_health_seed${seed}.log" 2>&1
  ) &
  pids+=($!)
done
ec=0
for pid in "${pids[@]}"; do
  wait "$pid" || ec=1
done
if [[ "$ec" -ne 0 ]]; then
  echo "WARN: at least one seed health exited non-zero" >&2
fi

python - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path("/mnt/hdd/dexora/logs")
files = [root / f"dexjoco_relative_rot_vpred_pilot10k_health_seed{s}.json" for s in (0, 1, 2)]
steps = ["1000", "2500", "5000", "7500", "10000"]
print("seed | step | max | bound | mse | rot_p99 | viol_max | pass_last")
for f in files:
    if not f.is_file():
        print(f.name, "MISSING")
        continue
    d = json.loads(f.read_text())
    for st in steps:
        s = d["per_step"][st]["solver"]
        print(
            f"{d['noise_seed']:4d} | {st:>5} | {s['max_abs']:6.3f} | {s['frac_in_m1_2']:.4f} | "
            f"{s['mse_all']:.4f} | {s.get('rotvec_norm_p99', float('nan')):6.3f} | "
            f"{s.get('max_violation_margin', 0):.4f}"
        )
    print(" verdict", d.get("verdict"))
print("\naggregate:")
for st in steps:
    vals = [json.loads(f.read_text())["per_step"][st]["solver"] for f in files if f.is_file()]
    if not vals:
        continue
    for k in ["max_abs", "frac_in_m1_2", "mse_all", "rotvec_norm_p99", "max_violation_margin"]:
        a = np.array([v.get(k, np.nan) for v in vals], dtype=np.float64)
        print(f"  {st} {k}: mean={np.nanmean(a):.4f} [{np.nanmin(a):.4f},{np.nanmax(a):.4f}]")
PY
echo "==> pilot done. Pick best of 1k/2.5k/5k/7.5k/10k before any 50k."
