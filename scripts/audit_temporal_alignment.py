#!/usr/bin/env python3
"""Read-only GT temporal-alignment audit for DexJoCo LeRobot parquet.

Outputs JSON under /mnt/hdd/dexora/audit/cursor/ (Cursor track; no GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from data.dexjoco_remap import action44_as_policy, state46_to_policy44
from eval_sim.action_exec import LEFT_HAND, RIGHT_HAND, state46_to_action44


def _xyz_jump(a44: np.ndarray, b44: np.ndarray) -> float:
    a = np.asarray(a44, dtype=np.float64).reshape(-1)
    b = np.asarray(b44, dtype=np.float64).reshape(-1)
    return float(
        max(
            np.linalg.norm(a[:3] - b[:3]),
            np.linalg.norm(a[22:25] - b[22:25]),
        )
    )


def _hand_mean(a44: np.ndarray) -> float:
    a = np.asarray(a44, dtype=np.float64).reshape(-1)
    return float(max(np.abs(a[RIGHT_HAND]).mean(), np.abs(a[LEFT_HAND]).mean()))


def _phase_bucket(local_frame: int) -> str:
    if local_frame == 0:
        return "ep0"
    if local_frame <= 7:
        return "ep1_7"
    if local_frame <= 31:
        return "ep8_31"
    if local_frame <= 128:
        return "ep32_128"
    return "ep129p"


def load_dataset(repo_dir: Path) -> dict[str, np.ndarray]:
    tables = []
    for path in sorted((repo_dir / "data" / "chunk-000").glob("*.parquet")):
        tables.append(
            pq.read_table(
                path,
                columns=[
                    "action",
                    "observation.state",
                    "episode_index",
                    "frame_index",
                    "index",
                ],
            )
        )
    import pyarrow as pa

    table = pa.concat_tables(tables)
    order = np.argsort(table.column("index").to_numpy())
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)[order]
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[order]
    episode_index = table.column("episode_index").to_numpy()[order]
    frame_index = table.column("frame_index").to_numpy()[order]
    return {
        "actions": actions,
        "states": states,
        "episode_index": episode_index,
        "frame_index": frame_index,
    }


def summarize_bucket(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264")
    p.add_argument(
        "--out",
        default="/mnt/hdd/dexora/audit/cursor/gt_temporal_alignment.json",
    )
    args = p.parse_args()

    repo = Path(args.repo_dir)
    data = load_dataset(repo)
    actions = data["actions"]
    states = data["states"]
    ep_idx = data["episode_index"]
    frame_idx = data["frame_index"]
    n = len(actions)

    # Episode boundaries
    ep_starts: list[int] = []
    for i in range(n):
        if int(frame_idx[i]) == 0:
            ep_starts.append(i)

    episode_start_samples: list[dict[str, Any]] = []
    bucket_rows: dict[str, list[dict[str, float]]] = {
        k: [] for k in ("ep0", "ep1_7", "ep8_31", "ep32_128", "ep129p")
    }
    off_by_one_scores: list[float] = []

    for gi in range(n):
        act_abs = actions[gi]
        st46 = states[gi]
        act_pol = action44_as_policy(act_abs[None])[0]
        st_pol = state46_to_policy44(st46[None])[0]
        hold_abs = state46_to_action44(st46)

        gt_vs_state_abs_xyz = _xyz_jump(act_abs, hold_abs)
        gt_vs_state_pol_xyz = _xyz_jump(act_pol, st_pol)
        gt_hand = _hand_mean(act_abs)

        aligned = gt_vs_state_abs_xyz
        shifted = aligned
        if gi + 1 < n and int(ep_idx[gi + 1]) == int(ep_idx[gi]):
            act_next = actions[gi + 1]
            hold_next = state46_to_action44(states[gi + 1])
            shifted = _xyz_jump(act_abs, hold_next)
            off_by_one_scores.append(abs(aligned - _xyz_jump(act_next, hold_abs)))

        local = int(frame_idx[gi])
        bucket = _phase_bucket(local)
        bucket_rows[bucket].append(
            {
                "gt_vs_state_abs_xyz_m": gt_vs_state_abs_xyz,
                "gt_vs_state_pol_xyz_m": gt_vs_state_pol_xyz,
                "gt_hand_mean_abs": gt_hand,
                "aligned_vs_shifted_abs_delta_m": abs(aligned - shifted),
            }
        )

        if local == 0:
            rec: dict[str, Any] = {
                "global_index": int(gi),
                "episode_index": int(ep_idx[gi]),
                "raw_action_hand_mean": gt_hand,
                "gt_vs_hold_abs_xyz_m": gt_vs_state_abs_xyz,
                "gt_vs_hold_pol_xyz_m": gt_vs_state_pol_xyz,
            }
            if gi + 1 < n and int(ep_idx[gi + 1]) == int(ep_idx[gi]):
                act1 = actions[gi + 1]
                rec["frame1_hand_mean"] = _hand_mean(act1)
                rec["frame0_to_frame1_xyz_m"] = _xyz_jump(act_abs, act1)
                rec["state0_to_state1_xyz_m"] = _xyz_jump(hold_abs, state46_to_action44(states[gi + 1]))
                rec["action0_vs_state1_xyz_m"] = _xyz_jump(act_abs, state46_to_action44(states[gi + 1]))
                rec["action1_vs_state0_xyz_m"] = _xyz_jump(act1, hold_abs)
            episode_start_samples.append(rec)

    out = {
        "agent": "cursor",
        "repo_dir": str(repo),
        "n_frames": n,
        "n_episodes": len(ep_starts),
        "loader_semantics": {
            "dexora_train": "state[t] + action[t:t+H] (same index)",
            "lerobot_v2_delta": "action offsets [0, 1/fps, ...] from current frame",
            "parquet_export": "action[t] and observation.state[t] written at same loop index t",
        },
        "episode_start": {
            "n": len(episode_start_samples),
            "mean_raw_hand_frame0": float(
                np.mean([s["raw_action_hand_mean"] for s in episode_start_samples])
            ),
            "mean_gt_vs_hold_abs_xyz_m": float(
                np.mean([s["gt_vs_hold_abs_xyz_m"] for s in episode_start_samples])
            ),
            "mean_gt_vs_hold_pol_xyz_m": float(
                np.mean([s["gt_vs_hold_pol_xyz_m"] for s in episode_start_samples])
            ),
            "pct_ep_start_hand_frame0_lt_0.05": float(
                np.mean([s["raw_action_hand_mean"] < 0.05 for s in episode_start_samples])
            ),
            "pct_ep_start_hand_frame1_gt_0.3": float(
                np.mean([s.get("frame1_hand_mean", 0.0) > 0.3 for s in episode_start_samples])
            ),
            "mean_frame0_to_frame1_xyz_m": float(
                np.mean([s.get("frame0_to_frame1_xyz_m", 0.0) for s in episode_start_samples])
            ),
            "mean_state0_to_state1_xyz_m": float(
                np.mean([s.get("state0_to_state1_xyz_m", 0.0) for s in episode_start_samples])
            ),
            "examples_worst_hold_mismatch": sorted(
                episode_start_samples,
                key=lambda x: x["gt_vs_hold_abs_xyz_m"],
                reverse=True,
            )[:5],
            "examples_large_f0_f1_action_jump": sorted(
                [s for s in episode_start_samples if "frame0_to_frame1_xyz_m" in s],
                key=lambda x: x["frame0_to_frame1_xyz_m"],
                reverse=True,
            )[:5],
        },
        "phase_buckets_gt": {k: summarize_bucket(v) for k, v in bucket_rows.items()},
        "off_by_one": {
            "mean_abs_delta_aligned_vs_action0_state1_m": float(np.mean(off_by_one_scores))
            if off_by_one_scores
            else None,
            "interpretation": (
                "If action[t] should match state[t] (hold), compare gt_vs_state_abs_xyz. "
                "If action[t] is next command, action[t] may align better with state[t+1]."
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
