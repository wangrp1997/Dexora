#!/usr/bin/env python3
"""π0.5 first-plan probe at episode starts (read-only; does not touch training).

Uses the same untrimmed DexJoCo LeRobot root π0.5 trains on.
Compares predicted chunk[0] to hold(state) — same metric family as Dexora.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Prefer the training openpi tree (matches π0.5 handfresh train PYTHONPATH).
OPENPI_ROOT = Path("/home/wangrenpeng/openpi")
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import image_tools

RIGHT_HAND = slice(6, 22)
LEFT_HAND = slice(28, 44)
RIGHT_ROT = slice(3, 6)
LEFT_ROT = slice(25, 28)


def state46_to_action44(state46: np.ndarray) -> np.ndarray:
    s = np.asarray(state46, dtype=np.float64).reshape(-1)
    r_arm, l_arm = s[:7], s[7:14]
    r_hand, l_hand = s[14:30], s[30:46]
    r_rot = R.from_quat(r_arm[3:7], scalar_first=True).as_rotvec()
    l_rot = R.from_quat(l_arm[3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [r_arm[:3], r_rot, r_hand, l_arm[:3], l_rot, l_hand], dtype=np.float64
    ).astype(np.float32)


def rot_geodesic_rad(r0: np.ndarray, r1: np.ndarray) -> float:
    return float(np.linalg.norm((R.from_rotvec(r0).inv() * R.from_rotvec(r1)).as_rotvec()))


def measure_first_plan(hold: np.ndarray, first: np.ndarray) -> dict[str, float]:
    hold = np.asarray(hold, dtype=np.float64).reshape(-1)
    first = np.asarray(first, dtype=np.float64).reshape(-1)
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
    hand = float(max(np.abs(first[RIGHT_HAND]).mean(), np.abs(first[LEFT_HAND]).mean()))
    hold_hand = float(max(np.abs(hold[RIGHT_HAND]).mean(), np.abs(hold[LEFT_HAND]).mean()))
    return {
        "first_plan_xyz_jump_m": xyz,
        "first_plan_rot_jump_rad": rot,
        "first_plan_hand_mean": hand,
        "hold_hand_mean": hold_hand,
    }


def resize_u8(img: np.ndarray) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="/mnt/ssd/checkpoints/openpi_dexjoco/pi05_dexjoco_lora/pi05_dexjoco_lora_state44_handfresh/40000",
    )
    p.add_argument("--config", default="pi05_dexjoco_lora")
    p.add_argument(
        "--repo-dir",
        default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264",
        help="Untrimmed H.264 LeRobot root (same demos π0.5 trains on).",
    )
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument(
        "--out",
        default="/mnt/hdd/dexora/audit/cursor/pi05_first_plan_probe_40000.json",
    )
    p.add_argument(
        "--prompt",
        default="Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.",
    )
    args = p.parse_args()

    # Load DexJoCo frames via Dexora loader (CPU decode only).
    sys.path.insert(0, "/home/wangrenpeng/Dexora")
    from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset

    ds = DexJoCoLeRobotVLADataset(
        repo_dir=args.repo_dir,
        normalize_mode=None,
        load_imgs=True,
        state_dim_keep=44,
    )
    starts = [i for i in range(len(ds)) if int(ds.frame_index[i]) == 0][: args.n_episodes]
    print(f"Loading policy from {args.checkpoint} ...", flush=True)
    train_cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)

    rows = []
    for gi in starts:
        sample = ds.get_item(gi)
        state46 = ds.states46[gi]
        hold = state46_to_action44(state46)
        gt0 = np.asarray(sample["actions"][0], dtype=np.float32)  # absolute rotvec action44 in dataset
        # Dexora dataset remaps to relative rot in get_item; reload raw for GT hand.
        gt_raw = ds.actions44[gi]
        obs = {
            "base": resize_u8(np.asarray(sample["cam_high"])[-1]),
            "wrist_left": resize_u8(np.asarray(sample["cam_left_wrist"])[-1]),
            "wrist_right": resize_u8(np.asarray(sample["cam_right_wrist"])[-1]),
            "state": state46.astype(np.float32),
            "prompt": args.prompt,
        }
        out = policy.infer(obs)
        chunk = np.asarray(out["actions"], dtype=np.float32)
        first = chunk[0]
        m = measure_first_plan(hold, first)
        m.update(
            {
                "global_index": int(gi),
                "episode_index": int(ds.episode_index[gi]),
                "gt_action_hand_mean": float(
                    max(np.abs(gt_raw[6:22]).mean(), np.abs(gt_raw[28:44]).mean())
                ),
                "pred_minus_gt_hand_mae": float(
                    max(
                        np.abs(first[6:22] - gt_raw[6:22]).mean(),
                        np.abs(first[28:44] - gt_raw[28:44]).mean(),
                    )
                ),
                "pred_xyz_minus_gt_m": float(
                    max(
                        np.linalg.norm(first[:3] - gt_raw[:3]),
                        np.linalg.norm(first[22:25] - gt_raw[22:25]),
                    )
                ),
            }
        )
        rows.append(m)
        print(
            f"ep{m['episode_index']:03d}: xyz={m['first_plan_xyz_jump_m']*100:.1f}cm "
            f"rot={m['first_plan_rot_jump_rad']:.3f} hand={m['first_plan_hand_mean']:.3f} "
            f"gt_hand={m['gt_action_hand_mean']:.3f}",
            flush=True,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "repo_dir": args.repo_dir,
        "n": len(rows),
        "mean_first_plan_xyz_jump_m": float(np.mean([r["first_plan_xyz_jump_m"] for r in rows])),
        "mean_first_plan_rot_jump_rad": float(np.mean([r["first_plan_rot_jump_rad"] for r in rows])),
        "mean_first_plan_hand_mean": float(np.mean([r["first_plan_hand_mean"] for r in rows])),
        "mean_gt_action_hand_mean": float(np.mean([r["gt_action_hand_mean"] for r in rows])),
        "mean_hold_hand_mean": float(np.mean([r["hold_hand_mean"] for r in rows])),
        "mean_pred_minus_gt_hand_mae": float(np.mean([r["pred_minus_gt_hand_mae"] for r in rows])),
        "dexora_50k_ref": {
            "faithful_sim_first_plan_xyz_m": 0.199,
            "faithful_sim_first_plan_hand": 0.592,
            "phase_ep0_h0_mse": 0.297,
        },
        "per_episode": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
