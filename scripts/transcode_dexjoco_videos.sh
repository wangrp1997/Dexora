#!/usr/bin/env bash
# Convert DexJoCo's AV1 videos to H.264 for reliable OpenCV random-access decoding.
set -Eeuo pipefail

: "${SOURCE_ROOT:=/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly}"
: "${OUTPUT_ROOT:=/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264}"
: "${JOBS:=3}"
: "${CRF:=18}"
: "${PYTHON:=/home/wangrenpeng/miniconda3/envs/dexora/bin/python}"

mkdir -p "$OUTPUT_ROOT"
ln -sfn "$SOURCE_ROOT/data" "$OUTPUT_ROOT/data"
ln -sfn "$SOURCE_ROOT/meta" "$OUTPUT_ROOT/meta"

transcode_one() {
    local source_file="$1"
    local relative_path="${source_file#${SOURCE_ROOT}/}"
    local output_file="$OUTPUT_ROOT/$relative_path"
    local temp_file="${output_file}.tmp.mp4"

    mkdir -p "$(dirname "$output_file")"
    if [[ -f "$output_file" ]] && \
       [[ "$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$output_file")" == "h264" ]]; then
        echo "skip $relative_path"
        return
    fi

    echo "transcode $relative_path"
    ffmpeg -nostdin -v error -y -i "$source_file" \
        -map 0:v:0 -an -c:v libx264 -preset veryfast -crf "$CRF" \
        -pix_fmt yuv420p -g 30 -keyint_min 30 -sc_threshold 0 \
        -movflags +faststart "$temp_file"
    mv "$temp_file" "$output_file"
}

export SOURCE_ROOT OUTPUT_ROOT CRF
export -f transcode_one
find "$SOURCE_ROOT/videos" -type f -name '*.mp4' -print0 \
    | xargs -0 -n1 -P "$JOBS" bash -c 'transcode_one "$1"' _

export OUTPUT_ROOT
"$PYTHON" - <<'PY'
import os
import numpy as np

from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset

dataset = DexJoCoLeRobotVLADataset(
    repo_dir=os.environ["OUTPUT_ROOT"],
    normalize_mode=None,
    load_imgs=True,
    state_dim_keep=44,
)
for index in (0, len(dataset) // 2, len(dataset) - 1):
    sample = dataset.get_item(index)
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = np.asarray(sample[key])[-1]
        if image.std() < 1.0 or image.max() == 0:
            raise RuntimeError(f"Bad transcoded frame: index={index}, camera={key}")
print(f"Validated H.264 dataset: {len(dataset)} frames")
PY
