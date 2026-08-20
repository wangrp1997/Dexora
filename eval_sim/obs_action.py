"""Map DexJoCo env obs into DexoraPolicy inputs (44-D + 4-cam layout)."""

from __future__ import annotations

from typing import Any

import numpy as np

from data.dexjoco_remap import state46_to_policy44

# Must match ``train/dataset.py`` image_metas order for LeRobot / DexJoCo.
TRAIN_CAMERA_ORDER: tuple[str, ...] = (
    "cam_high",
    "cam_right_wrist",
    "cam_left_wrist",
    "cam_third_view",
)

# DexJoCo raw cam -> Dexora training slot
ENV_TO_DEXORA_CAM: dict[str, str] = {
    "ego": "cam_high",
    "wrist_right": "cam_right_wrist",
    "wrist_left": "cam_left_wrist",
}


def env_raw_to_dexora_images(raw_images: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build the 4-slot image dict DexoraPolicy expects (missing cam omitted)."""
    out: dict[str, np.ndarray] = {}
    for env_name, dex_name in ENV_TO_DEXORA_CAM.items():
        if env_name not in raw_images:
            continue
        img = np.asarray(raw_images[env_name])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        out[dex_name] = img
    # cam_third_view intentionally absent -> SigLIP mean-colour fill in policy
    return out


def env_state_to_policy44(state46: np.ndarray) -> np.ndarray:
    return state46_to_policy44(np.asarray(state46, dtype=np.float32)[..., :46])
