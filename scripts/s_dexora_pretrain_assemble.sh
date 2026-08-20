#!/usr/bin/env bash
# Stage 1 on Dexora official data (paper §III-D). Use assemble — closest to insert.
set -Eeuo pipefail
: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_assemble}"
: "${OUTPUT_DIR:=checkpoints/dexora-400m-pretrain-assemble}"
export DEXORA_LEROBOT_ROOT OUTPUT_DIR
bash "$(dirname "$0")/../s1_pretrain.sh"
