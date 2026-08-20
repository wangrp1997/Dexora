#!/usr/bin/env bash
# Formal 50k FT from pilot best checkpoint-1000 (relative rotvec + v_prediction).
# Does NOT auto-launch sim eval; run health after train, sim/success after 50k.
set -Eeuo pipefail
ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SRC_PILOT="${SRC_PILOT:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-pilot10k}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k}"
export DEXORA_STATS="${DEXORA_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
export MAX_TRAIN_STEPS=50000
export CHECKPOINTING_PERIOD=0
# Default: save every 5k (~10 ckpts). Override: CHECKPOINT_STEPS=2500,5000,... for 2.5k cadence.
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-5000,10000,15000,20000,25000,30000,35000,40000,45000,50000}"
export SAVE_INITIAL_CHECKPOINT=0
export RESUME_FROM_CHECKPOINT=checkpoint-1000
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export SEED=42
HEALTH_DIR="${HEALTH_DIR:-/mnt/hdd/dexora/logs}"
# Comma-separated steps for post-train multi-seed health (subset; not every ckpt).
HEALTH_STEPS="${HEALTH_STEPS:-5000,10000,20000,30000,40000,50000}"

mkdir -p "$OUTPUT_DIR" "$HEALTH_DIR"
if [[ ! -d "$OUTPUT_DIR/checkpoint-1000" ]]; then
  echo "==> seeding 50k dir from $SRC_PILOT/checkpoint-1000"
  if [[ ! -d "$SRC_PILOT/checkpoint-1000" ]]; then
    echo "Missing source checkpoint-1000 under $SRC_PILOT" >&2
    exit 1
  fi
  cp -a "$SRC_PILOT/checkpoint-1000" "$OUTPUT_DIR/checkpoint-1000"
  if [[ -f "$SRC_PILOT/config.json" ]]; then
    cp -a "$SRC_PILOT/config.json" "$OUTPUT_DIR/config.json" || true
  fi
fi

echo "==> 50k formal FT resume from checkpoint-1000 -> $OUTPUT_DIR (GPU ${CUDA_VISIBLE_DEVICES})"
echo "    checkpoint saves: $CHECKPOINT_STEPS"
bash scripts/s_dexjoco_finetune.sh

echo "==> multi-seed health at ${HEALTH_STEPS} (parallel)"
pids=()
IFS=',' read -r -a _health_step_arr <<< "$HEALTH_STEPS"
for seed in 0 1 2; do
  (
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python -m eval_sim.sample_health \
      --ckpt-root "$OUTPUT_DIR" \
      --steps "$HEALTH_STEPS" \
      --noise-seed "$seed" \
      --stats-file "$DEXORA_STATS" \
      --out "$HEALTH_DIR/dexjoco_relative_rot_vpred_50k_health_seed${seed}.json" \
      > "$HEALTH_DIR/dexjoco_relative_rot_vpred_50k_health_seed${seed}.log" 2>&1
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

python - <<PY
import json
from pathlib import Path
import numpy as np

root = Path("${HEALTH_DIR}")
steps = "${HEALTH_STEPS}".split(",")
files = [root / f"dexjoco_relative_rot_vpred_50k_health_seed{s}.json" for s in (0, 1, 2)]
print("seed | step | max | bound | mse | rot_p99 | viol_max")
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
print("\naggregate (last step):")
last = steps[-1]
vals = [json.loads(f.read_text())["per_step"][last]["solver"] for f in files if f.is_file()]
if vals:
    for k in ["max_abs", "frac_in_m1_2", "mse_all", "rotvec_norm_p99", "max_violation_margin"]:
        a = np.array([v.get(k, np.nan) for v in vals], dtype=np.float64)
        print(f"  {last} {k}: mean={np.nanmean(a):.4f} [{np.nanmin(a):.4f},{np.nanmax(a):.4f}]")
PY
echo "==> 50k train + health done. Next: sim eval / success on best ckpt (not auto-run here)."
