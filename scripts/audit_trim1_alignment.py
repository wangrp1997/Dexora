#!/usr/bin/env python3
"""Post-trim1 alignment audit for DexJoCo LeRobot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RIGHT_HAND = slice(6, 22)
LEFT_HAND = slice(28, 44)


def hand_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    return float(max(np.abs(a[RIGHT_HAND]).mean(), np.abs(a[LEFT_HAND]).mean()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_trim1_h264")
    p.add_argument("--baseline-dir", default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264")
    p.add_argument("--out", default="/mnt/hdd/dexora/audit/cursor/trim1_alignment_audit.json")
    args = p.parse_args()

    repo = Path(args.repo_dir)
    table = pa.concat_tables(
        [pq.read_table(path) for path in sorted((repo / "data").rglob("file-*.parquet"))]
    )
    order = np.argsort(np.asarray(table.column("index").to_pylist(), dtype=np.int64))
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)[order]
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[order]
    frames = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)[order]
    eps = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)[order]
    starts = np.where(frames == 0)[0]

    # Compare new f0 to old f1 via baseline parquet
    base = Path(args.baseline_dir)
    # baseline data may be symlink to ssd
    btable = pa.concat_tables(
        [pq.read_table(path) for path in sorted((base / "data").rglob("file-*.parquet"))]
    )
    border = np.argsort(np.asarray(btable.column("index").to_pylist(), dtype=np.int64))
    bactions = np.asarray(btable.column("action").to_pylist(), dtype=np.float32)[border]
    bstates = np.asarray(btable.column("observation.state").to_pylist(), dtype=np.float32)[border]
    bframes = np.asarray(btable.column("frame_index").to_pylist(), dtype=np.int64)[border]
    beps = np.asarray(btable.column("episode_index").to_pylist(), dtype=np.int64)[border]
    bstarts = np.where(bframes == 0)[0]

    action_match = []
    state_match = []
    for i, (ns, bs) in enumerate(zip(starts, bstarts)):
        # new ep i frame0 should equal old ep i frame1
        action_match.append(np.allclose(actions[ns], bactions[bs + 1], atol=1e-6))
        state_match.append(np.allclose(states[ns], bstates[bs + 1], atol=1e-6))

    # Episode boundary no bleed: last of ep i and first of ep i+1 different episodes
    boundary_ok = True
    for i in range(len(starts) - 1):
        last = starts[i + 1] - 1
        if eps[last] != eps[starts[i]] or eps[starts[i + 1]] != eps[starts[i]] + 1:
            boundary_ok = False
            break

    # Loader + video smoke via Dexora dataset
    import sys

    sys.path.insert(0, "/home/wangrenpeng/Dexora")
    from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset

    ds = DexJoCoLeRobotVLADataset(
        repo_dir=str(repo), normalize_mode=None, load_imgs=True, state_dim_keep=44
    )
    img_ok = True
    img_notes = []
    for gi in [int(starts[0]), int(starts[1]), int(starts[-1]), len(ds) // 2]:
        sample = ds.get_item(gi)
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            img = np.asarray(sample[key])[-1]
            if img.std() < 1.0 or img.max() == 0:
                img_ok = False
                img_notes.append(f"bad {key} at {gi}")

    # state response after action: hand state should rise after grasp command within a few frames
    response = []
    for ns in starts[:20]:
        ep = int(eps[ns])
        # within episode look ahead
        hand_cmd = hand_mean(actions[ns])
        # find when state hand exceeds 0.1
        rose = None
        for k in range(1, min(30, len(actions) - ns)):
            if int(eps[ns + k]) != ep:
                break
            sh = float(
                max(
                    np.abs(states[ns + k][14:30]).mean(),
                    np.abs(states[ns + k][30:46]).mean(),
                )
            )
            if sh > 0.1:
                rose = k
                break
        response.append({"hand_cmd": hand_cmd, "state_hand_rise_at": rose})

    info = json.loads((repo / "meta" / "info.json").read_text())
    out = {
        "repo_dir": str(repo),
        "total_frames": int(len(actions)),
        "info_total_frames": info.get("total_frames"),
        "n_episodes": int(len(starts)),
        "new_frame0_hand_mean": float(np.mean([hand_mean(actions[i]) for i in starts])),
        "pct_new_frame0_hand_lt_0.05": float(np.mean([hand_mean(actions[i]) < 0.05 for i in starts])),
        "pct_new_frame0_equals_old_frame1_action": float(np.mean(action_match)),
        "pct_new_frame0_equals_old_frame1_state": float(np.mean(state_match)),
        "episode_boundary_ok": boundary_ok,
        "loader_len": len(ds),
        "images_decode_ok": img_ok,
        "image_notes": img_notes,
        "state_response_examples": response[:5],
        "pass": (
            int(len(actions)) == 54070
            and float(np.mean([hand_mean(actions[i]) < 0.05 for i in starts])) == 0.0
            and float(np.mean(action_match)) == 1.0
            and float(np.mean(state_match)) == 1.0
            and boundary_ok
            and img_ok
            and len(ds) == 54070
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
