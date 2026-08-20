#!/usr/bin/env bash
# A/B: smooth evaluator (default) vs legacy on Dexora 50k checkpoint-50000.
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000}"
: "${CUDA_VISIBLE_DEVICES:=2}"
: "${SEED:=0}"
: "${EPISODES:=1}"
: "${OUTPUT_ROOT:=/mnt/hdd/dexjoco/outputs/dexora/50k_smooth_ab}"

cd "$DEXORA_ROOT"
export PYTHONPATH="$DEXORA_ROOT:${DEXJOCO_ROOT:-/home/wangrenpeng/dexjoco}/dexjoco:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_eval() {
  local tag="$1"
  shift
  local out="$OUTPUT_ROOT/${tag}_seed${SEED}_ep${EPISODES}"
  local log="/mnt/hdd/dexora/logs/dexora_50k_${tag}_eval.log"
  echo "==> $tag -> $out"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python -m eval_sim.evaluate \
      --config /home/wangrenpeng/dexjoco/configs/multi_task/bimanual_assembly.yaml \
      --checkpoint "$CKPT" \
      --stats-file /mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json \
      --model-config configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml \
      --episodes "$EPISODES" --seed "$SEED" --overwrite --output "$out" \
      "$@" 2>&1 | tee "$log"
}

mkdir -p "$OUTPUT_ROOT" /mnt/hdd/dexora/logs
run_eval smooth_default
run_eval legacy --legacy-exec --replan-steps 24
