#!/usr/bin/env bash
# Closed-loop DexJoCo sim eval for pilot10k checkpoints (1k primary, 10k control).
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${DEXJOCO_ROOT:=/home/wangrenpeng/dexjoco}"
: "${CKPT_ROOT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-pilot10k}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${DEXJOCO_CONFIG:=${DEXJOCO_ROOT}/configs/multi_task/bimanual_assembly.yaml}"
: "${DEXORA_MODEL_CONFIG:=${DEXORA_ROOT}/configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${CUDA_VISIBLE_DEVICES:=2}"
: "${EPISODES:=10}"
: "${MAX_STEPS:=1500}"
: "${REPLAN_STEPS:=24}"
: "${INFERENCE_DTYPE:=bf16}"
: "${OUTPUT_ROOT:=/mnt/hdd/dexjoco/outputs/dexora/pilot10k_sim}"
: "${LOG_DIR:=/mnt/hdd/dexora/logs}"
: "${CKPT_STEPS:=1000 10000}"
: "${SEEDS:=0 1 2}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

export DEXJOCO_ROOT
export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$DEXORA_ROOT"

run_one() {
  local step="$1"
  local seed="$2"
  local ckpt="$CKPT_ROOT/checkpoint-${step}"
  local out="$OUTPUT_ROOT/ckpt${step}_seed${seed}"
  local log="$LOG_DIR/dexjoco_pilot10k_sim_ckpt${step}_seed${seed}.log"

  if [[ ! -f "$ckpt/pytorch_model.bin" && ! -f "$ckpt/model.safetensors" ]]; then
    echo "Missing weights: $ckpt" | tee "$log"
    return 1
  fi

  echo "==> ckpt-${step} seed=${seed} episodes=${EPISODES}" | tee "$log"
  PYTHONPATH="$DEXORA_ROOT:${DEXJOCO_ROOT}/dexjoco:${PYTHONPATH:-}" \
    /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python -m eval_sim.evaluate \
      --config "$DEXJOCO_CONFIG" \
      --checkpoint "$ckpt" \
      --stats-file "$DEXORA_STATS" \
      --model-config "$DEXORA_MODEL_CONFIG" \
      --text-encoder "$DEXORA_T5" \
      --vision-encoder "$DEXORA_SIGLIP" \
      --episodes "$EPISODES" \
      --max-steps "$MAX_STEPS" \
      --replan-steps "$REPLAN_STEPS" \
      --inference-dtype "$INFERENCE_DTYPE" \
      --seed "$seed" \
      --output "$out" \
      --overwrite \
      2>&1 | tee -a "$log"
}

echo "pilot sim eval: steps=${CKPT_STEPS} seeds=${SEEDS} gpu=${CUDA_VISIBLE_DEVICES}"
for step in $CKPT_STEPS; do
  for seed in $SEEDS; do
    run_one "$step" "$seed"
  done
done

echo "==> all pilot sim evals done under $OUTPUT_ROOT"
