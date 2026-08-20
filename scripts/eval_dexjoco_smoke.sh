#!/usr/bin/env bash
# Smoke: 1 short episode of Dexora FT ckpt on DexJoCo insert (conda dexjoco).
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${DEXJOCO_ROOT:=/home/wangrenpeng/dexjoco}"
# Force 44-D insert stats (ignore leftover Stage-1 assemble DEXORA_STATS).
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${DEXORA_CKPT_ROOT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-visual-h264}"
: "${USE_EMA:=1}"
: "${DEXJOCO_CONFIG:=${DEXJOCO_ROOT}/configs/multi_task/bimanual_assembly.yaml}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${DEXORA_MODEL_CONFIG:=${DEXORA_ROOT}/configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
: "${CUDA_VISIBLE_DEVICES:=1}"
: "${EPISODES:=1}"
: "${SEED:=0}"
: "${MAX_STEPS:=60}"
: "${REPLAN_STEPS:=24}"
: "${RAND_FULL:=0}"
: "${RANDOMIZE_DYNAMICS:=0}"
: "${INFERENCE_DTYPE:=bf16}"
: "${OVERWRITE:=1}"
: "${OUTPUT:=/mnt/hdd/dexjoco/outputs/dexora/smoke_bimanual_assembly}"

if [[ -z "${DEXORA_CKPT:-}" ]]; then
  if [[ "$USE_EMA" == "1" ]]; then
    DEXORA_CKPT="$DEXORA_CKPT_ROOT/ema"
  else
    DEXORA_CKPT="$DEXORA_CKPT_ROOT"
  fi
fi

if [[ ! -f "$DEXORA_CKPT/pytorch_model.bin" && ! -f "$DEXORA_CKPT/model.safetensors" ]]; then
  echo "Missing Dexora weights under $DEXORA_CKPT" >&2
  exit 1
fi

export DEXJOCO_ROOT
export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-/mnt/hdd/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$DEXORA_ROOT"
mkdir -p "$(dirname "$OUTPUT")"

eval_args=(
  --config "$DEXJOCO_CONFIG"
  --checkpoint "$DEXORA_CKPT"
  --stats-file "$DEXORA_STATS"
  --model-config "$DEXORA_MODEL_CONFIG"
  --text-encoder "$DEXORA_T5"
  --vision-encoder "$DEXORA_SIGLIP"
  --episodes "$EPISODES"
  --max-steps "$MAX_STEPS"
  --replan-steps "$REPLAN_STEPS"
  --inference-dtype "$INFERENCE_DTYPE"
  --seed "$SEED"
  --output "$OUTPUT"
)

if [[ "$RAND_FULL" == "1" ]]; then
  eval_args+=(--rand-full)
fi
if [[ "$RANDOMIZE_DYNAMICS" == "1" ]]; then
  eval_args+=(--randomize-dynamics)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  eval_args+=(--overwrite)
fi

PYTHONPATH="$DEXORA_ROOT:${DEXJOCO_ROOT}/dexjoco:${PYTHONPATH:-}" \
  /home/wangrenpeng/miniconda3/envs/dexjoco/bin/python -m eval_sim.evaluate "${eval_args[@]}"
