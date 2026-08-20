#!/usr/bin/env bash
# DexJoCo 36->44 transfer: freeze inherited backbone and warm up rebuilt I/O only.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

export CONFIG_PATH="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml"
export DEXORA_LEROBOT_ROOT="${DEXORA_LEROBOT_ROOT:-/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264}"
export DEXORA_STATS="${DEXORA_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
export PRETRAINED="${PRETRAINED:-/mnt/hdd/dexora/checkpoints/dexora-400m-pretrain-assemble}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-io-warmup-gate1k}"
export MAX_TRAIN_STEPS=1000
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=1000
export SAVE_INITIAL_CHECKPOINT=0
export RESUME_FROM_CHECKPOINT=
export SEED=42
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export BACKBONE_LR_MULT=0.0
export IO_LR_MULT="${IO_LR_MULT:-2.0}"
export TRAIN_FRESH_IO_ONLY=1
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export NUM_GPUS=1

LOG_DIR="${LOG_DIR:-/mnt/hdd/dexora/logs}"
AUDIT_DIR="${AUDIT_DIR:-/mnt/hdd/dexora/audit/io_warmup_gate1k}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$AUDIT_DIR"

echo "==> DexJoCo fresh-I/O-only warmup gate1k GPU=$CUDA_VISIBLE_DEVICES"
bash scripts/s_dexjoco_finetune.sh 2>&1 | tee "$LOG_DIR/dexjoco_io_warmup_gate1k_train.log"

python scripts/probe_dexora_first_plan.py \
  --checkpoint "$OUTPUT_DIR/checkpoint-1000" \
  --repo-dir "$DEXORA_LEROBOT_ROOT" \
  --stats-file "$DEXORA_STATS" \
  --n-episodes 20 \
  --noise-seeds 0,1,2,3,4 \
  --tag io_warmup_1k \
  --out "$AUDIT_DIR/first_plan_20ep_seed0-4.json" \
  2>&1 | tee "$LOG_DIR/dexjoco_io_warmup_gate1k_probe.log"

echo "==> fresh-I/O-only gate1k done $(date -Is)"
