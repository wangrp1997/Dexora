#!/usr/bin/env bash
# Diagnostic only: explicit semantic partial-copy adapter, 1k -> 5k.
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
export PARTIAL_COPY_MAP="${PARTIAL_COPY_MAP:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-partial-copy-gate5k}"
export DEXJOCO_ACTION_TARGET=absolute
export MAX_TRAIN_STEPS=5000
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=1000,5000
export SAVE_INITIAL_CHECKPOINT=0
export RESUME_FROM_CHECKPOINT=
export SEED=42
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export TRAIN_FRESH_IO_ONLY=0
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export NUM_GPUS=1

if [[ -z "$PARTIAL_COPY_MAP" || ! -f "$PARTIAL_COPY_MAP" ]]; then
    echo "Set PARTIAL_COPY_MAP to a reviewed semantic 36D->44D JSON map." >&2
    echo "AIRBOT joint and DexJoCo TCP/Allegro fields are not automatically aligned." >&2
    exit 2
fi

LOG_DIR="${LOG_DIR:-/mnt/hdd/dexora/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "==> DexJoCo partial-copy diagnostic gate5k GPU=$CUDA_VISIBLE_DEVICES"
bash scripts/s_dexjoco_finetune.sh 2>&1 | tee "$LOG_DIR/dexjoco_partial_copy_gate5k_train.log"
