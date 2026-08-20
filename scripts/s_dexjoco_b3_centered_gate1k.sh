#!/usr/bin/env bash
# Track B / B3: centered [-1, 1] state-action normalization, stop at 1k.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

SOURCE_STATS="${SOURCE_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
CENTERED_STATS="${CENTERED_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot_centered_m11/dataset_statistics.json}"

mkdir -p "$(dirname "$CENTERED_STATS")"
python - "$SOURCE_STATS" "$CENTERED_STATS" <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
payload = json.loads(source.read_text())
for key in ("state", "action"):
    low = payload[key]["percentile_1"]
    high = payload[key]["percentile_99"]
    payload[key]["percentile_1"] = [(lo + hi) / 2.0 for lo, hi in zip(low, high)]
payload["normalization_contract"] = {
    "mode": "centered_min_max",
    "mapping": "original q01 -> -1, original q99 -> 1",
    "source": str(source),
}
target.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote centered stats: {target}")
PY

export CONFIG_PATH="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml"
export DEXORA_LEROBOT_ROOT="${DEXORA_LEROBOT_ROOT:-/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264}"
export DEXORA_STATS="$CENTERED_STATS"
export PRETRAINED="${PRETRAINED:-/mnt/hdd/dexora/checkpoints/dexora-400m-pretrain-assemble}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-b3-centered-m11-gate1k}"
export MAX_TRAIN_STEPS=1000
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=1000
export SAVE_INITIAL_CHECKPOINT=0
export RESUME_FROM_CHECKPOINT=
export SEED=42
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export NUM_GPUS=1

LOG_DIR="${LOG_DIR:-/mnt/hdd/dexora/logs}"
AUDIT_DIR="${AUDIT_DIR:-/mnt/hdd/dexora/audit/b3_centered_m11_gate1k}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$AUDIT_DIR"

echo "==> B3 centered [-1,1] gate1k GPU=$CUDA_VISIBLE_DEVICES"
bash scripts/s_dexjoco_finetune.sh 2>&1 | tee "$LOG_DIR/dexjoco_b3_centered_m11_gate1k_train.log"

python scripts/probe_dexora_first_plan.py \
  --checkpoint "$OUTPUT_DIR/checkpoint-1000" \
  --repo-dir "$DEXORA_LEROBOT_ROOT" \
  --stats-file "$CENTERED_STATS" \
  --n-episodes 20 \
  --noise-seeds 0,1,2,3,4 \
  --tag b3_centered_m11_1k \
  --out "$AUDIT_DIR/first_plan_20ep_seed0-4.json" \
  2>&1 | tee "$LOG_DIR/dexjoco_b3_centered_m11_gate1k_probe.log"

echo "==> B3 gate1k done $(date -Is)"
