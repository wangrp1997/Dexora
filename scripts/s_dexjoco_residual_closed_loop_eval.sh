#!/usr/bin/env bash
# Long-horizon closed-loop evaluation for the DexJoCo residual-action policy.
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${DEXJOCO_ROOT:=/home/wangrenpeng/dexjoco}"
: "${CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-residual-gate5k/ema}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${DEXJOCO_CONFIG:=${DEXJOCO_ROOT}/configs/multi_task/bimanual_assembly.yaml}"
: "${DEXORA_MODEL_CONFIG:=${DEXORA_ROOT}/configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${CUDA_VISIBLE_DEVICES:=1}"
: "${EPISODES:=30}"
: "${MAX_STEPS:=1500}"
: "${REPLAN_STEPS:=24}"
: "${OUTPUT_ROOT:=/mnt/hdd/dexjoco/outputs/dexora/residual_gate5k_closed_loop}"

export DEXJOCO_ROOT CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$DEXORA_ROOT"
mkdir -p "$OUTPUT_ROOT"

run_eval() {
  local name="$1"
  shift
  local output="$OUTPUT_ROOT/$name"
  echo "==> $name: episodes=$EPISODES max_steps=$MAX_STEPS"
  PYTHONPATH="$DEXORA_ROOT:${DEXJOCO_ROOT}/dexjoco:${PYTHONPATH:-}" \
    /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python -m eval_sim.evaluate \
      --config "$DEXJOCO_CONFIG" \
      --checkpoint "$CKPT" \
      --stats-file "$DEXORA_STATS" \
      --model-config "$DEXORA_MODEL_CONFIG" \
      --text-encoder "$DEXORA_T5" \
      --vision-encoder "$DEXORA_SIGLIP" \
      --episodes "$EPISODES" \
      --max-steps "$MAX_STEPS" \
      --replan-steps "$REPLAN_STEPS" \
      --residual-action \
      --output "$output" \
      --overwrite \
      "$@"
}

run_eval standard_seed0 --seed 0
run_eval standard_seed1 --seed 1
run_eval standard_seed2 --seed 2
run_eval randomized_seed0 --seed 0 --randomize-dynamics
run_eval randomized_seed1 --seed 1 --randomize-dynamics
run_eval randomized_seed2 --seed 2 --randomize-dynamics

echo "==> closed-loop residual evaluation complete: $OUTPUT_ROOT"
