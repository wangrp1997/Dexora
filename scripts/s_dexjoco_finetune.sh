#!/usr/bin/env bash
# DexJoCo bimanual_assembly finetune: Stage-1 36-D -> policy 44-D (full Allegro).
# Inherit same-shape layers; scale-init state_adaptor.0 + fc2 only.
set -Eeuo pipefail

: "${DEXORA_LEROBOT_ROOT:=/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${DEXORA_STATS:=/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json}"
: "${CONFIG_PATH:=configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml}"
: "${PRETRAINED:=/mnt/hdd/dexora/checkpoints/dexora-400m-pretrain-assemble}"
: "${OUTPUT_DIR:=/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-visual-h264}"

: "${NUM_GPUS:=1}"
: "${TRAIN_BATCH_SIZE:=2}"
: "${GRAD_ACCUM:=2}"
: "${MAX_TRAIN_STEPS:=50000}"
: "${CHECKPOINTING_PERIOD:=5000}"
: "${CHECKPOINT_STEPS:=}"
: "${SAVE_INITIAL_CHECKPOINT:=0}"
: "${RESUME_FROM_CHECKPOINT:=}"
: "${LEARNING_RATE:=5e-5}"
: "${BACKBONE_LR_MULT:=0.1}"
: "${IO_LR_MULT:=2.0}"
: "${TRAIN_FRESH_IO_ONLY:=0}"
: "${PARTIAL_COPY_MAP:=}"
: "${DEXJOCO_ACTION_TARGET:=absolute}"
: "${MIXED_PRECISION:=bf16}"
: "${LR_SCHEDULER:=constant}"
: "${DATALOADER_NUM_WORKERS:=4}"
: "${REPORT_TO:=tensorboard}"
: "${SEED:=42}"

export WANDB_PROJECT=${WANDB_PROJECT:-dexora-dexjoco}
export WANDB_MODE=${WANDB_MODE:-offline}

mkdir -p "$OUTPUT_DIR" "$(dirname "$DEXORA_STATS")"
echo "==> DexJoCo insert finetune (policy 44-D, full Allegro 16)"
echo "    DEXORA_LEROBOT_ROOT : $DEXORA_LEROBOT_ROOT"
echo "    PRETRAINED          : $PRETRAINED"
echo "    OUTPUT_DIR          : $OUTPUT_DIR"
echo "    CONFIG              : $CONFIG_PATH"

if [[ ! -d "$DEXORA_LEROBOT_ROOT/videos" ]]; then
    echo "Missing H.264 DexJoCo dataset at $DEXORA_LEROBOT_ROOT" >&2
    echo "Run: bash scripts/transcode_dexjoco_videos.sh" >&2
    exit 1
fi

echo "==> Verifying that all three training cameras decode to non-black frames ..."
python - <<PY
import numpy as np

from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset

dataset = DexJoCoLeRobotVLADataset(
    repo_dir="${DEXORA_LEROBOT_ROOT}",
    stats_file="${DEXORA_STATS}",
    load_imgs=True,
    state_dim_keep=44,
)
indices = (0, len(dataset) // 2, len(dataset) - 1)
for index in indices:
    sample = dataset.get_item(index)
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = np.asarray(sample[key])[-1]
        if image.std() < 1.0 or image.max() == 0:
            raise RuntimeError(
                f"Invalid decoded image: index={index}, camera={key}, "
                f"min={image.min()}, max={image.max()}, std={image.std()}"
            )
print("DexJoCo video preflight passed")
PY

if [[ ! -f "$DEXORA_STATS" ]]; then
    echo "==> Stats file $DEXORA_STATS missing; generating 44-D stats ..."
    python -m data.lerobot_vla_dataset --stat \
        --num_samples 5000 \
        --output_dir "$(dirname "$DEXORA_STATS")" \
        --repo_dir "$DEXORA_LEROBOT_ROOT"
    # CLI writes dataset_statistics.json into --output_dir
    if [[ ! -f "$DEXORA_STATS" && -f "$(dirname "$DEXORA_STATS")/dataset_statistics.json" ]]; then
        : # already at expected path
    fi
fi

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
if [[ "${TRAIN_FRESH_IO_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--train_fresh_io_only)
fi
if [[ -n "${PARTIAL_COPY_MAP}" ]]; then
    EXTRA_ARGS+=(--partial_copy_map="$PARTIAL_COPY_MAP")
fi

accelerate launch --num_processes="$NUM_GPUS" \
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
    --dexjoco_action_target="$DEXJOCO_ACTION_TARGET" \
    --report_to="$REPORT_TO" \
    "${EXTRA_ARGS[@]}"
