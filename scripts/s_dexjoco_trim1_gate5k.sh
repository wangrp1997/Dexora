#!/usr/bin/env bash
# Track A trim1 gate: same Dexora recipe, new data/stats, save 1k+5k then stop.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export DEXORA_LEROBOT_ROOT="${DEXORA_LEROBOT_ROOT:-/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_trim1_h264}"
export DEXORA_STATS="${DEXORA_STATS:-/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_trim1_relative_rot/dataset_statistics.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/dexora/checkpoints/dexora-dexjoco-trim1-baseline-5k}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
export PRETRAINED="${PRETRAINED:-/mnt/hdd/dexora/checkpoints/dexora-400m-pretrain-assemble}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
export CHECKPOINTING_PERIOD=0
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000,5000}"
export SAVE_INITIAL_CHECKPOINT=0
export RESUME_FROM_CHECKPOINT=
export SEED=42
export BACKBONE_LR_MULT=0.1
export IO_LR_MULT=2.0
export TRAIN_BATCH_SIZE=2
export GRAD_ACCUM=2
export NUM_GPUS=1

LOG_DIR=/mnt/hdd/dexora/logs
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "==> trim1 gate FT -> $OUTPUT_DIR (GPU $CUDA_VISIBLE_DEVICES)"
bash scripts/s_dexjoco_finetune.sh 2>&1 | tee "$LOG_DIR/trim1_baseline_5k_train.log"

echo "==> gate health on 1000 and 5000"
python -m eval_sim.sample_health \
  --ckpt-root "$OUTPUT_DIR" \
  --steps 1000,5000 \
  --noise-seed 0 \
  --repo-dir "$DEXORA_LEROBOT_ROOT" \
  --stats-file "$DEXORA_STATS" \
  --out /mnt/hdd/dexora/audit/cursor/trim1_gate_health_seed0.json \
  2>&1 | tee "$LOG_DIR/trim1_baseline_5k_health.log"

echo "==> trim1 5k gate DONE $(date -Is)" | tee /mnt/hdd/dexora/audit/cursor/trim1_gate_DONE.log
