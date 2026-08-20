#!/usr/bin/env bash
# 500-step raw I/O-expand gate (not EMA). Saves checkpoint-0/100/250/500 then
# runs eval_sim.sample_health. Does not start 50k.
set -Eeuo pipefail
ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-ioexpand-gate500}"
export MAX_TRAIN_STEPS=500
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS=100,250,500
export SAVE_INITIAL_CHECKPOINT=1
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export SEED=42
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
HEALTH_JSON="${HEALTH_JSON:-/mnt/hdd/dexora/logs/dexjoco_ioexpand_gate500_health.json}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$HEALTH_JSON")"
echo "==> 500-step raw gate -> $OUTPUT_DIR (GPU ${CUDA_VISIBLE_DEVICES})"
bash scripts/s_dexjoco_finetune.sh

echo "==> sample health on raw checkpoints 0/100/250/500"
python -m eval_sim.sample_health \
    --ckpt-root "$OUTPUT_DIR" \
    --steps 0,100,250,500 \
    --out "$HEALTH_JSON"
echo "==> health json: $HEALTH_JSON"
