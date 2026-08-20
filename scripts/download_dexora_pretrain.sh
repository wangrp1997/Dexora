#!/usr/bin/env bash
# Download Dexora official LeRobot data for Stage-1 pretrain.
# Default: assemble family only (closest to insert). Set FAMILY=all for 240GB.
set -Eeuo pipefail
: "${FAMILY:=airbot_assemble}"
: "${LOCAL_DIR:=data/Dexora_Real-World_Dataset}"
mkdir -p "$LOCAL_DIR"
if [[ "$FAMILY" == "all" ]]; then
    huggingface-cli download Dexora/Dexora_Real-World_Dataset \
        --repo-type dataset --local-dir "$LOCAL_DIR"
else
    huggingface-cli download Dexora/Dexora_Real-World_Dataset \
        --repo-type dataset --local-dir "$LOCAL_DIR" \
        --include "${FAMILY}/**"
fi
echo "done -> $LOCAL_DIR/$FAMILY"
