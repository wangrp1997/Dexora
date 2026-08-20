#!/usr/bin/env bash
# Faithful Dexora closed-loop eval: replan=chunk_size, no settle/smooth/blend.
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SEED:=0}"
: "${EPISODES:=1}"
: "${TAG:=50k_faithful}"
: "${OUTPUT_ROOT:=/mnt/hdd/dexjoco/outputs/dexora}"

cd "$DEXORA_ROOT"
export PYTHONPATH="$DEXORA_ROOT:${DEXJOCO_ROOT:-/home/wangrenpeng/dexjoco}/dexjoco:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

out="$OUTPUT_ROOT/${TAG}_chunk32_seed${SEED}_ep${EPISODES}"
log="/mnt/hdd/dexora/logs/dexora_${TAG}_eval.log"
mkdir -p "$OUTPUT_ROOT" /mnt/hdd/dexora/logs

echo "==> faithful eval -> $out"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python -m eval_sim.evaluate \
    --config /home/wangrenpeng/dexjoco/configs/multi_task/bimanual_assembly.yaml \
    --checkpoint "$CKPT" \
    --stats-file /mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json \
    --model-config configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml \
    --episodes "$EPISODES" --seed "$SEED" --overwrite --output "$out" \
    --faithful-exec 2>&1 | tee "$log"
