#!/usr/bin/env bash
# Start-align repair ablation from Dexora 50k (2k steps each).
#   A = early-window oversample only
#   B = horizon loss weighting only
#   C = both
#
# Default: launch A/B/C in parallel on GPU 0/1/2. Set PARALLEL=0 to run serially on GPU 0.
set -Eeuo pipefail

ROOT=/home/wangrenpeng/Dexora
cd "$ROOT"
export PATH="/home/wangrenpeng/miniconda3/envs/dexora/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${SRC_CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${MAX_TRAIN_STEPS:=2000}"
: "${CHECKPOINT_STEPS:=2000}"
: "${LEARNING_RATE:=1e-5}"
: "${BACKBONE_LR_MULT:=0.1}"
: "${IO_LR_MULT:=1.0}"
: "${EARLY_WINDOW_PROB:=0.25}"
: "${PARALLEL:=1}"
: "${GPUS:=0,1,2}"
: "${LOG_DIR:=/mnt/hdd/dexora/logs/start_align_ab}"
: "${CKPT_ROOT:=/mnt/hdd/dexora/checkpoints}"

mkdir -p "$LOG_DIR" "$CKPT_ROOT"

run_one() {
  local tag="$1"
  local gpu="$2"
  shift 2
  local out="$CKPT_ROOT/dexora-dexjoco-start-align-${tag}-2k"
  local log="$LOG_DIR/${tag}_train.log"
  echo "==> [$tag] GPU $gpu -> $out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OUTPUT_DIR="$out"
    export PRETRAINED="$SRC_CKPT"
    export MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
    export CHECKPOINTING_PERIOD=0
    export CHECKPOINT_STEPS="$CHECKPOINT_STEPS"
    export SAVE_INITIAL_CHECKPOINT=0
    export RESUME_FROM_CHECKPOINT=
    export LEARNING_RATE="$LEARNING_RATE"
    export BACKBONE_LR_MULT="$BACKBONE_LR_MULT"
    export IO_LR_MULT="$IO_LR_MULT"
    export SEED=42
    export EXTRA_TRAIN_ARGS="$*"
    bash scripts/s_dexjoco_start_align_finetune.sh
  ) 2>&1 | tee "$log"
  echo "EXIT=$?" >> "$log"
}

if [[ "$PARALLEL" == "1" ]]; then
  IFS=',' read -r -a gpu_arr <<< "$GPUS"
  if [[ "${#gpu_arr[@]}" -lt 3 ]]; then
    echo "Need 3 GPUs in GPUS for parallel mode, got: $GPUS" >&2
    exit 1
  fi
  run_one "A-oversample" "${gpu_arr[0]}" \
    --early_window_prob="$EARLY_WINDOW_PROB" &
  pid_a=$!
  run_one "B-horizon" "${gpu_arr[1]}" \
    --horizon_loss_weighting &
  pid_b=$!
  run_one "C-combined" "${gpu_arr[2]}" \
    --early_window_prob="$EARLY_WINDOW_PROB" --horizon_loss_weighting &
  pid_c=$!
  ec=0
  wait "$pid_a" || ec=1
  wait "$pid_b" || ec=1
  wait "$pid_c" || ec=1
  exit "$ec"
fi

run_one "A-oversample" "0" --early_window_prob="$EARLY_WINDOW_PROB"
run_one "B-horizon" "0" --horizon_loss_weighting
run_one "C-combined" "0" --early_window_prob="$EARLY_WINDOW_PROB" --horizon_loss_weighting
