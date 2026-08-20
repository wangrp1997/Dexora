#!/usr/bin/env python3
"""Compute full-pass Dexora relative-rot 44D min-max stats for a DexJoCo root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset
from data.dexjoco_remap import action44_as_policy, state46_to_policy44


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = DexJoCoLeRobotVLADataset(
        repo_dir=args.repo_dir,
        normalize_mode=None,
        load_imgs=False,
        state_dim_keep=44,
    )
    states = state46_to_policy44(ds.states46)
    actions = action44_as_policy(ds.actions44)
    assert states.shape[1] == 44 and actions.shape[1] == 44

    def block(x: np.ndarray) -> dict:
        return {
            "mean": x.mean(axis=0).astype(np.float64).tolist(),
            "std": x.std(axis=0).astype(np.float64).tolist(),
            "min": x.min(axis=0).astype(np.float64).tolist(),
            "max": x.max(axis=0).astype(np.float64).tolist(),
            "percentile_1": np.percentile(x, 1, axis=0).astype(np.float64).tolist(),
            "percentile_99": np.percentile(x, 99, axis=0).astype(np.float64).tolist(),
            "q01": np.percentile(x, 1, axis=0).astype(np.float64).tolist(),
            "q99": np.percentile(x, 99, axis=0).astype(np.float64).tolist(),
            "count": [int(x.shape[0])],
        }

    stats = {
        "state": block(states),
        "action": block(actions),
        "meta": {
            "repo_dir": args.repo_dir,
            "n_frames": int(states.shape[0]),
            "dim": 44,
            "relative_rot": True,
            "seed": args.seed,
            "mode": "full_pass",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out} n={states.shape[0]}")


if __name__ == "__main__":
    main()
