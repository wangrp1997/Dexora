#!/usr/bin/env python3
"""Build DexJoCo trim1 LeRobot dataset by dropping init hold-hand frame0.

Does NOT overwrite the untrimmed baseline. Writes a new root with:
  - rewritten data/*.parquet (drop sync frame0 per episode when needed)
  - rewritten meta/episodes + info.json
  - videos symlinked from the H.264 root (from_timestamp advanced by 1/fps)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RIGHT_HAND = slice(6, 22)
LEFT_HAND = slice(28, 44)


def hand_mean(action: np.ndarray) -> float:
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    return float(max(np.abs(a[RIGHT_HAND]).mean(), np.abs(a[LEFT_HAND]).mean()))


def should_trim_frame0(action0: np.ndarray, action1: np.ndarray) -> bool:
    return hand_mean(action0) <= 0.05 and hand_mean(action1) >= 0.3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-data",
        default="/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly",
    )
    p.add_argument(
        "--source-videos",
        default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264/videos",
    )
    p.add_argument(
        "--output",
        default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_trim1_h264",
    )
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    src = Path(args.source_data)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output exists and is non-empty: {out}")

    data_src = sorted((src / "data").rglob("file-*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in data_src])
    order = np.argsort(np.asarray(table.column("index").to_pylist(), dtype=np.int64))
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)[order]
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[order]
    task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)[order]

    ep_files = sorted((src / "meta" / "episodes").rglob("file-*.parquet"))
    ep_table = pa.concat_tables([pq.read_table(path) for path in ep_files])
    ep_from = np.asarray(ep_table.column("dataset_from_index").to_pylist(), dtype=np.int64)
    ep_to = np.asarray(ep_table.column("dataset_to_index").to_pylist(), dtype=np.int64)
    n_ep = len(ep_from)

    kept_rows: list[int] = []
    trim_flags: list[bool] = []
    new_lengths = np.zeros(n_ep, dtype=np.int64)
    for ep_i in range(n_ep):
        a = int(ep_from[ep_i])
        b = int(ep_to[ep_i])
        if b - a < 2:
            raise RuntimeError(f"episode {ep_i} too short: {b - a}")
        trim = should_trim_frame0(actions[a], actions[a + 1])
        trim_flags.append(trim)
        start = a + (1 if trim else 0)
        rows = list(range(start, b))
        new_lengths[ep_i] = len(rows)
        kept_rows.extend(rows)

    n_trim = int(sum(trim_flags))
    if n_trim != n_ep:
        raise RuntimeError(f"expected trim all {n_ep} episodes, got {n_trim}")

    kept = np.asarray(kept_rows, dtype=np.int64)
    new_actions = actions[kept]
    new_states = states[kept]
    new_task = task_index[kept]

    new_ep = np.zeros(len(kept), dtype=np.int64)
    new_frame = np.zeros(len(kept), dtype=np.int64)
    new_ts = np.zeros(len(kept), dtype=np.float64)
    new_index = np.arange(len(kept), dtype=np.int64)
    new_ep_from = np.zeros(n_ep, dtype=np.int64)
    new_ep_to = np.zeros(n_ep, dtype=np.int64)
    cursor = 0
    for ep_i in range(n_ep):
        length = int(new_lengths[ep_i])
        new_ep_from[ep_i] = cursor
        new_ep_to[ep_i] = cursor + length
        new_ep[cursor : cursor + length] = ep_i
        new_frame[cursor : cursor + length] = np.arange(length)
        new_ts[cursor : cursor + length] = np.arange(length) / float(args.fps)
        cursor += length

    out_data = pa.table(
        {
            "action": new_actions.tolist(),
            "observation.state": new_states.tolist(),
            "timestamp": new_ts.tolist(),
            "frame_index": new_frame.tolist(),
            "episode_index": new_ep.tolist(),
            "index": new_index.tolist(),
            "task_index": new_task.tolist(),
        }
    )

    ep_cols = {name: ep_table.column(name).to_pylist() for name in ep_table.column_names}
    dt = 1.0 / float(args.fps)
    for ep_i in range(n_ep):
        length = int(new_lengths[ep_i])
        ep_cols["length"][ep_i] = length
        ep_cols["dataset_from_index"][ep_i] = int(new_ep_from[ep_i])
        ep_cols["dataset_to_index"][ep_i] = int(new_ep_to[ep_i])
        if trim_flags[ep_i]:
            for cam in ("ego", "wrist_left", "wrist_right"):
                key = f"videos/observation.images.{cam}/from_timestamp"
                ep_cols[key][ep_i] = float(ep_cols[key][ep_i]) + dt
        ep_cols["stats/frame_index/min"][ep_i] = [0]
        ep_cols["stats/frame_index/max"][ep_i] = [length - 1]
        ep_cols["stats/frame_index/count"][ep_i] = [length]
        ep_cols["stats/index/min"][ep_i] = [int(new_ep_from[ep_i])]
        ep_cols["stats/index/max"][ep_i] = [int(new_ep_to[ep_i]) - 1]
        ep_cols["stats/index/count"][ep_i] = [length]
        ep_cols["stats/timestamp/min"][ep_i] = [0.0]
        ep_cols["stats/timestamp/max"][ep_i] = [(length - 1) / float(args.fps)]
        ep_cols["stats/timestamp/count"][ep_i] = [length]
        ep_cols["stats/action/count"][ep_i] = [length]
        ep_cols["stats/observation.state/count"][ep_i] = [length]

    out.mkdir(parents=True, exist_ok=False)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(out_data, out / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table(ep_cols), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    shutil.copy2(src / "meta" / "tasks.parquet", out / "meta" / "tasks.parquet")
    if (src / "meta" / "stats.json").is_file():
        shutil.copy2(src / "meta" / "stats.json", out / "meta" / "stats.json")
    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_frames"] = int(len(kept))
    info["total_episodes"] = int(n_ep)
    info["trim1"] = {
        "source_data": str(src),
        "source_videos": str(args.source_videos),
        "dropped_frames": int(n_trim),
        "rule": "drop frame0 when hand_mean<=0.05 and next hand_mean>=0.3",
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    videos_src = Path(args.source_videos)
    if not videos_src.is_dir():
        raise FileNotFoundError(videos_src)
    (out / "videos").symlink_to(videos_src.resolve())

    report = {
        "output": str(out),
        "n_episodes": n_ep,
        "n_trim": n_trim,
        "frames_before": int(len(actions)),
        "frames_after": int(len(kept)),
        "expected_after": int(len(actions) - n_ep),
        "new_frame0_hand_mean": float(
            np.mean([hand_mean(new_actions[int(new_ep_from[i])]) for i in range(n_ep)])
        ),
        "new_frame0_hand_lt_0.05_pct": float(
            np.mean([hand_mean(new_actions[int(new_ep_from[i])]) < 0.05 for i in range(n_ep)])
        ),
    }
    (out / "TRIM1_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
