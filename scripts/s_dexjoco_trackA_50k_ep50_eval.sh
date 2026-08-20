#!/usr/bin/env bash
# Track A closed-loop report eval: existing relative-rot + v_pred 50k, 50 episodes.
# Not residual / not B3. Matches DexJoCo-style 50-video budget on seed 0.
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${DEXJOCO_ROOT:=/home/wangrenpeng/dexjoco}"
: "${CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${DEXJOCO_CONFIG:=${DEXJOCO_ROOT}/configs/multi_task/bimanual_assembly.yaml}"
: "${DEXORA_MODEL_CONFIG:=${DEXORA_ROOT}/configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
: "${CUDA_VISIBLE_DEVICES:=1}"
: "${SEED:=0}"
: "${EPISODES:=50}"
: "${MAX_STEPS:=1500}"
: "${REPLAN_STEPS:=24}"
: "${OUTPUT:=/mnt/hdd/dexjoco/outputs/dexora/trackA_50k_relative_vpred_seed0}"
: "${LOG:=/mnt/hdd/dexora/logs/dexora_trackA_50k_seed0_ep50.log}"

export DEXJOCO_ROOT CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$DEXORA_ROOT:${DEXJOCO_ROOT}/dexjoco:${PYTHONPATH:-}"

cd "$DEXORA_ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

echo "==> Track A 50k closed-loop: ckpt=$CKPT episodes=$EPISODES seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
echo "==> output=$OUTPUT"
/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python -m eval_sim.evaluate \
  --config "$DEXJOCO_CONFIG" \
  --checkpoint "$CKPT" \
  --stats-file "$DEXORA_STATS" \
  --model-config "$DEXORA_MODEL_CONFIG" \
  --text-encoder google/t5-v1_1-xxl \
  --vision-encoder google/siglip-so400m-patch14-384 \
  --episodes "$EPISODES" \
  --max-steps "$MAX_STEPS" \
  --replan-steps "$REPLAN_STEPS" \
  --inference-dtype bf16 \
  --seed "$SEED" \
  --output "$OUTPUT" \
  --overwrite \
  2>&1 | tee "$LOG"

echo "==> DONE $(date -Is) -> $OUTPUT" | tee -a "$LOG"
test -f "$OUTPUT/summary.json" && cat "$OUTPUT/summary.json" | tee -a "$LOG"
