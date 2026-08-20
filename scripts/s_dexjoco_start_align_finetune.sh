#!/usr/bin/env bash
# Low-LR start-align repair finetune from Dexora 50k checkpoint (loads weights, fresh step counter).
set -Eeuo pipefail

: "${DEXORA_ROOT:=/home/wangrenpeng/Dexora}"
: "${DEXORA_LEROBOT_ROOT:=/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${CONFIG_PATH:=configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
: "${SRC_CKPT:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000}"
: "${PRETRAINED:=${PRETRAINED:-$SRC_CKPT}}"
: "${OUTPUT_DIR:?OUTPUT_DIR required}"
: "${CUDA_VISIBLE_DEVICES:=0}"

: "${NUM_GPUS:=1}"
: "${TRAIN_BATCH_SIZE:=2}"
: "${GRAD_ACCUM:=2}"
: "${MAX_TRAIN_STEPS:=2000}"
: "${CHECKPOINTING_PERIOD:=0}"
: "${CHECKPOINT_STEPS:=2000}"
: "${SAVE_INITIAL_CHECKPOINT:=0}"
: "${RESUME_FROM_CHECKPOINT:=}"
: "${LEARNING_RATE:=1e-5}"
: "${BACKBONE_LR_MULT:=0.1}"
: "${IO_LR_MULT:=1.0}"
: "${MIXED_PRECISION:=bf16}"
: "${LR_SCHEDULER:=constant}"
: "${DATALOADER_NUM_WORKERS:=4}"
: "${REPORT_TO:=tensorboard}"
: "${SEED:=42}"
: "${EXTRA_TRAIN_ARGS:=}"

cd "$DEXORA_ROOT"
export PYTHONPATH="$DEXORA_ROOT:${PYTHONPATH:-}"
export WANDB_PROJECT=${WANDB_PROJECT:-dexora-dexjoco}
export WANDB_MODE=${WANDB_MODE:-offline}

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "${CHECKPOINT_STEPS}" ]]; then
  EXTRA_ARGS+=(--checkpoint_steps="$CHECKPOINT_STEPS")
fi
if [[ "${SAVE_INITIAL_CHECKPOINT}" == "1" ]]; then
  EXTRA_ARGS+=(--save_initial_checkpoint)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi
# shellcheck disable=SC2206
EXTRA_ARGS+=(${EXTRA_TRAIN_ARGS})

echo "==> start-align repair FT"
echo "    PRETRAINED : $PRETRAINED"
echo "    OUTPUT_DIR : $OUTPUT_DIR"
echo "    STEPS      : $MAX_TRAIN_STEPS"
echo "    LR         : $LEARNING_RATE (backbone x$BACKBONE_LR_MULT, io x$IO_LR_MULT)"
echo "    EXTRA      : ${EXTRA_TRAIN_ARGS:-none}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" accelerate launch --num_processes="$NUM_GPUS" \
  --mixed_precision="$MIXED_PRECISION" \
  -m train.main \
  --config_path="$CONFIG_PATH" \
  --pretrained_text_encoder_name_or_path="$DEXORA_T5" \
  --pretrained_vision_encoder_name_or_path="$DEXORA_SIGLIP" \
  --pretrained_model_name_or_path="$PRETRAINED" \
  --output_dir="$OUTPUT_DIR" \
  --load_from=lerobot \
  --lerobot_root="$DEXORA_LEROBOT_ROOT" \
  --stats_file="$DEXORA_STATS" \
  --state_dim_keep=44 \
  --dataset_type=finetune \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --sample_batch_size=2 \
  --gradient_accumulation_steps="$GRAD_ACCUM" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --checkpointing_period="$CHECKPOINTING_PERIOD" \
  --sample_period=-1 \
  --lr_scheduler="$LR_SCHEDULER" \
  --learning_rate="$LEARNING_RATE" \
  --mixed_precision="$MIXED_PRECISION" \
  --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  --state_noise_snr=40 \
  --image_aug \
  --seed="$SEED" \
  --backbone_lr_mult="$BACKBONE_LR_MULT" \
  --io_lr_mult="$IO_LR_MULT" \
  --report_to="$REPORT_TO" \
  "${EXTRA_ARGS[@]}"
