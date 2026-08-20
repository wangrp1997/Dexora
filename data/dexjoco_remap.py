"""Map DexJoCo bimanual_assembly tensors onto the Dexora finetune 44-D canvas.

Policy layout (action / proprio after remap, matches DexJoCo action44):

    [ right_xyz(3) | right_relative_rotvec(3) | right_hand(16)
    | left_xyz(3)  | left_relative_rotvec(3)  | left_hand(16) ]

Native DexJoCo LeRobot:

    action44 — already in the layout above
    state46  — [ right_tcp(7=xyz+quat_wxyz) | left_tcp(7)
                 | right_hand(16) | left_hand(16) ]

Stage-1 stays on official 36-D assemble. Finetune expands
``state_dim`` / ``action_dim`` to 44 and rebuilds only
``state_adaptor.0`` and ``final_layer.fc2`` (scale-matched random).
AIRBOT joint-36 and DexJoCo TCP+Allegro-44 are not the same skill space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POLICY_DIM = 44
DEXJOCO_ACTION_DIM = 44
DEXJOCO_STATE_DIM = 46

# Fixed task-frame orientation references fitted once from the 100-episode
# DexJoCo insert training set. Expressing wrist orientation relative to these
# references removes the rotvec +/-pi branch cut while remaining stateless at
# deployment. Quaternions use scalar-first (w, x, y, z).
RIGHT_ROT_REF_WXYZ = np.array(
    [0.20127589, -0.97252644, -0.10572893, -0.05001720], dtype=np.float64
)
LEFT_ROT_REF_WXYZ = np.array(
    [-0.10578238, -0.94733709, 0.27371961, -0.12821893], dtype=np.float64
)

DEFAULT_PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)


def is_dexjoco_lerobot(repo_dir: str | Path) -> bool:
    info_path = Path(repo_dir) / "meta" / "info.json"
    if not info_path.is_file():
        return False
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    action_shape = feats.get("action", {}).get("shape")
    return "observation.images.ego" in feats and action_shape == [DEXJOCO_ACTION_DIM]


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion (w, x, y, z) to axis-angle rotvec."""
    q = np.asarray(quat, dtype=np.float64)
    lead = q.shape[:-1]
    q = q.reshape(-1, 4)
    q = _normalize_quat_wxyz(q)
    q = np.where(q[:, :1] < 0.0, -q, q)
    w = np.clip(q[:, 0], -1.0, 1.0)
    xyz = q[:, 1:]
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    rot = np.zeros_like(xyz)
    small = s < 1e-8
    rot[small] = 0.0
    rot[~small] = xyz[~small] / s[~small, None] * angle[~small, None]
    return rot.reshape(*lead, 3).astype(np.float32)


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    lead = rotvec.shape[:-1]
    flat = rotvec.reshape(-1, 3)
    angle = np.linalg.norm(flat, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-8, np.sin(half) / np.clip(angle, 1e-8, None), 0.5)
    quat = np.concatenate([np.cos(half), flat * scale], axis=-1)
    return _normalize_quat_wxyz(quat).reshape(*lead, 4).astype(np.float32)


def quat_wxyz_to_relative_rotvec(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat)
    reference = _normalize_quat_wxyz(reference)
    reference_inv = reference * np.array([1.0, -1.0, -1.0, -1.0])
    relative = _quat_multiply_wxyz(reference_inv, quat)
    return quat_wxyz_to_rotvec(relative)


def relative_rotvec_to_absolute_rotvec(rotvec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = rotvec_to_quat_wxyz(rotvec)
    absolute = _quat_multiply_wxyz(_normalize_quat_wxyz(reference), relative)
    return quat_wxyz_to_rotvec(absolute)


def action44_as_policy(action44: np.ndarray) -> np.ndarray:
    """Convert absolute wrist rotvecs to task-reference-relative rotvecs."""
    a = np.asarray(action44, dtype=np.float32)
    if a.shape[-1] != DEXJOCO_ACTION_DIM:
        raise ValueError(f"expected action dim {DEXJOCO_ACTION_DIM}, got {a.shape[-1]}")
    out = a.copy()
    out[..., 3:6] = quat_wxyz_to_relative_rotvec(
        rotvec_to_quat_wxyz(a[..., 3:6]), RIGHT_ROT_REF_WXYZ
    )
    out[..., 25:28] = quat_wxyz_to_relative_rotvec(
        rotvec_to_quat_wxyz(a[..., 25:28]), LEFT_ROT_REF_WXYZ
    )
    return out


def state46_to_policy44(state46: np.ndarray) -> np.ndarray:
    """state46 (quat TCP) -> policy44 (rotvec TCP + full hands)."""
    s = np.asarray(state46, dtype=np.float32)
    if s.shape[-1] != DEXJOCO_STATE_DIM:
        raise ValueError(f"expected state dim {DEXJOCO_STATE_DIM}, got {s.shape[-1]}")
    r_xyz = s[..., 0:3]
    r_rot = quat_wxyz_to_relative_rotvec(s[..., 3:7], RIGHT_ROT_REF_WXYZ)
    l_xyz = s[..., 7:10]
    l_rot = quat_wxyz_to_relative_rotvec(s[..., 10:14], LEFT_ROT_REF_WXYZ)
    r_hand = s[..., 14:30]
    l_hand = s[..., 30:46]
    return np.concatenate([r_xyz, r_rot, r_hand, l_xyz, l_rot, l_hand], axis=-1)


def policy44_to_action44(vec44: np.ndarray) -> np.ndarray:
    """Convert task-reference-relative wrist rotvecs back to absolute rotvecs."""
    v = np.asarray(vec44, dtype=np.float32)
    if v.shape[-1] != POLICY_DIM:
        raise ValueError(f"expected policy dim {POLICY_DIM}, got {v.shape[-1]}")
    out = v.copy()
    out[..., 3:6] = relative_rotvec_to_absolute_rotvec(v[..., 3:6], RIGHT_ROT_REF_WXYZ)
    out[..., 25:28] = relative_rotvec_to_absolute_rotvec(v[..., 25:28], LEFT_ROT_REF_WXYZ)
    return out
