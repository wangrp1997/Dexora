#!/usr/bin/env bash
# Continue C-combined (oversample + horizon weight) from 2k -> 5k.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${OUTPUT_DIR:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-start-align-C-combined-2k}"
: "${LOG_DIR:=/mnt/hdd/dexora/logs/start_align_ab}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${HEALTH_DIR:=/mnt/hdd/dexora/health/start_align_ab}"

mkdir -p "$LOG_DIR" "$HEALTH_DIR"

export OUTPUT_DIR
export MAX_TRAIN_STEPS=5000
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=5000
export RESUME_FROM_CHECKPOINT=checkpoint-2000
export LEARNING_RATE=1e-5
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=1.0
export EXTRA_TRAIN_ARGS="--early_window_prob=0.25 --horizon_loss_weighting"

echo "==> C-combined resume 2k -> 5k on GPU ${CUDA_VISIBLE_DEVICES}"
bash scripts/s_dexjoco_start_align_finetune.sh 2>&1 | tee "$LOG_DIR/C-combined_2k_to_5k_train.log"
ec=${PIPESTATUS[0]}
echo "EXIT=$ec" >> "$LOG_DIR/C-combined_2k_to_5k_train.log"

if [[ "$ec" -ne 0 ]]; then
  exit "$ec"
fi

echo "==> health C-combined @ 5000"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python -m eval_sim.sample_health \
    --ckpt-root "$OUTPUT_DIR" \
    --steps 5000 \
    --noise-seed 0 \
    --stats-file "$DEXORA_STATS" \
    --out "$HEALTH_DIR/C-combined_seed0_step5000.json" \
    2>&1 | tee "$HEALTH_DIR/C-combined_seed0_step5000.log"

python - <<PY
import json
from pathlib import Path

paths = {
    "50k_baseline": Path("/mnt/hdd/dexora/health/dexjoco_relative_rot_vpred_50k_v2/health_seed0_step50k.json"),
    "C_2k": Path("${HEALTH_DIR}/C-combined_seed0_step2000.json"),
    "C_5k": Path("${HEALTH_DIR}/C-combined_seed0_step5000.json"),
}
for name, p in paths.items():
    if not p.is_file():
        print(f"{name}: MISSING")
        continue
    d = json.loads(p.read_text())
    v = d["verdict"]
    step = "5000" if "5000" in d.get("per_step", {}) else "2000" if "2000" in d.get("per_step", {}) else "50000"
    es = d["per_step"][step].get("episode_start", {})
    s = d["per_step"][step]["solver"]
    print(
        f"{name}: start_align={v.get('start_align')} xyz={es.get('first_action_xyz_jump_m', float('nan')):.3f}m "
        f"hand={es.get('first_action_hand_mean', float('nan')):.3f} bound={s.get('frac_in_m1_2'):.4f}"
    )
PY
