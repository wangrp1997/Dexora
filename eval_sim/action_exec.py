"""Dual-arm action chunk merge + per-step rate limits for DexJoCo rollout."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class TimedAction:
    action: np.ndarray
    timestamp: int


RIGHT_ROT = slice(3, 6)
LEFT_ROT = slice(25, 28)
RIGHT_HAND = slice(6, 22)
LEFT_HAND = slice(28, 44)


def state46_to_action44(state46: np.ndarray) -> np.ndarray:
    """Hold command: map 46-D quat proprio to 44-D absolute rotvec action."""
    s = np.asarray(state46, dtype=np.float64).reshape(-1)
    if s.shape[0] < 46:
        raise ValueError(f"expected state dim >= 46, got {s.shape[0]}")
    r_arm, l_arm = s[:7], s[7:14]
    r_hand, l_hand = s[14:30], s[30:46]
    r_rot = R.from_quat(r_arm[3:7], scalar_first=True).as_rotvec()
    l_rot = R.from_quat(l_arm[3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [r_arm[:3], r_rot, r_hand, l_arm[:3], l_rot, l_hand],
        dtype=np.float64,
    ).astype(np.float32)


def _interp_rotvec(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    if t <= 0.0:
        return r0.copy()
    if t >= 1.0:
        return r1.copy()
    rot0 = R.from_rotvec(r0)
    rot1 = R.from_rotvec(r1)
    delta = (rot0.inv() * rot1).as_rotvec()
    return (rot0 * R.from_rotvec(delta * t)).as_rotvec()


def interp_dual_arm_action44(
    old: np.ndarray, new: np.ndarray, t: float
) -> np.ndarray:
    out = (1.0 - t) * old + t * new
    out[RIGHT_ROT] = _interp_rotvec(old[RIGHT_ROT], new[RIGHT_ROT], t)
    out[LEFT_ROT] = _interp_rotvec(old[LEFT_ROT], new[LEFT_ROT], t)
    return out.astype(np.float32, copy=False)


def rate_limit_dual_arm_action44(
    prev: np.ndarray,
    target: np.ndarray,
    *,
    max_xyz_step_m: float,
    max_rot_step_rad: float,
) -> np.ndarray:
    """Clamp one-step wrist motion; hands pass through unchanged."""
    prev = np.asarray(prev, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    out = target.copy()

    for xyz_sl, rot_sl in ((slice(0, 3), RIGHT_ROT), (slice(22, 25), LEFT_ROT)):
        delta = target[xyz_sl] - prev[xyz_sl]
        dist = float(np.linalg.norm(delta))
        if dist > max_xyz_step_m and dist > 0.0:
            out[xyz_sl] = prev[xyz_sl] + delta * (max_xyz_step_m / dist)

        r0 = prev[rot_sl]
        r1 = target[rot_sl]
        rot_delta = (R.from_rotvec(r0).inv() * R.from_rotvec(r1)).as_rotvec()
        angle = float(np.linalg.norm(rot_delta))
        if angle > max_rot_step_rad and angle > 0.0:
            limited = R.from_rotvec(r0) * R.from_rotvec(rot_delta * (max_rot_step_rad / angle))
            out[rot_sl] = limited.as_rotvec()

    return out.astype(np.float32, copy=False)


def merge_chunk_into_buffer(
    actions_buffer: deque[TimedAction],
    chunk: np.ndarray,
    *,
    now_timestamp: int,
    chunk_origin_timestamp: int,
) -> None:
    """Blend a new env-space chunk into the existing buffer (pi0.5-style overlap)."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 44:
        raise ValueError(f"expected chunk [H, 44], got {chunk.shape}")

    chunk_range = (
        now_timestamp,
        chunk_origin_timestamp + chunk.shape[0],
    )
    if chunk_range[1] <= now_timestamp:
        return

    action = chunk[
        (chunk_range[0] - chunk_origin_timestamp) : (chunk_range[1] - chunk_origin_timestamp)
    ]

    if actions_buffer:
        buffer_range = (
            actions_buffer[0].timestamp,
            actions_buffer[-1].timestamp + 1,
        )
    else:
        buffer_range = (now_timestamp, now_timestamp)

    overlap = (
        max(chunk_range[0], buffer_range[0]),
        min(chunk_range[1], buffer_range[1]),
    )
    overlap_len = overlap[1] - overlap[0]
    for ts in range(overlap[0], overlap[1]):
        buf_i = ts - buffer_range[0]
        act_i = ts - chunk_range[0]
        t = (ts - overlap[0] + 1) / (overlap_len + 1)
        blended = interp_dual_arm_action44(
            actions_buffer[buf_i].action, action[act_i], t
        )
        actions_buffer[buf_i] = TimedAction(action=blended, timestamp=ts)

    for ts in range(buffer_range[1], chunk_range[1]):
        act_i = ts - chunk_range[0]
        actions_buffer.append(
            TimedAction(action=action[act_i].astype(np.float32, copy=False), timestamp=ts)
        )


@dataclass
class ExecStats:
    first_plan_xyz_jump_m: float = 0.0
    first_plan_rot_jump_rad: float = 0.0
    first_plan_hand_mean: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "first_plan_xyz_jump_m": self.first_plan_xyz_jump_m,
            "first_plan_rot_jump_rad": self.first_plan_rot_jump_rad,
            "first_plan_hand_mean": self.first_plan_hand_mean,
        }


def rot_geodesic_rad(r0: np.ndarray, r1: np.ndarray) -> float:
    """SO(3) geodesic distance between two rotvec orientations."""
    r0 = np.asarray(r0, dtype=np.float64).reshape(3)
    r1 = np.asarray(r1, dtype=np.float64).reshape(3)
    return float(np.linalg.norm((R.from_rotvec(r0).inv() * R.from_rotvec(r1)).as_rotvec()))


def measure_first_plan_jump(
    hold_action44: np.ndarray, first_action44: np.ndarray
) -> ExecStats:
    hold = np.asarray(hold_action44, dtype=np.float64)
    first = np.asarray(first_action44, dtype=np.float64)
    xyz = float(
        max(
            np.linalg.norm(first[:3] - hold[:3]),
            np.linalg.norm(first[22:25] - hold[22:25]),
        )
    )
    rot = float(
        max(
            rot_geodesic_rad(hold[RIGHT_ROT], first[RIGHT_ROT]),
            rot_geodesic_rad(hold[LEFT_ROT], first[LEFT_ROT]),
        )
    )
    hand = float(
        max(np.abs(first[RIGHT_HAND]).mean(), np.abs(first[LEFT_HAND]).mean())
    )
    return ExecStats(
        first_plan_xyz_jump_m=xyz,
        first_plan_rot_jump_rad=rot,
        first_plan_hand_mean=hand,
    )
