#!/usr/bin/env python3
"""Teacher-forced open-loop by episode phase bucket (Track A diagnostic).

Writes to /mnt/hdd/dexora/audit/cursor/ by default. Run only when GPU is free:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/audit_openloop_phase_buckets.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig
from eval_sim.obs_action import TRAIN_CAMERA_ORDER
from eval_sim.sample_health import (
    CTRL_FREQ_HZ,
    _images_from_sample,
    group_mse,
    summarize_first_action,
    summarize_per_horizon_mse,
)
from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset


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


def _load_samples(repo_dir: str, stats_file: str, per_bucket: int, seed: int) -> dict[str, list[dict]]:
    ds = DexJoCoLeRobotVLADataset(
        repo_dir=repo_dir,
        stats_file=stats_file,
        load_imgs=True,
        config_path="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml",
    )
    rng = np.random.RandomState(seed)
    buckets: dict[str, list[int]] = {k: [] for k in ("ep0", "ep1_7", "ep8_31", "ep32_128", "ep129p")}
    for gi in range(len(ds)):
        fi = int(ds.frame_index[gi])
        b = _phase_bucket(fi)
        if len(buckets[b]) < per_bucket * 3:
            buckets[b].append(gi)
    out: dict[str, list[dict]] = {}
    for name, indices in buckets.items():
        if name == "ep0":
            chosen = [i for i in indices if int(ds.frame_index[i]) == 0][:per_bucket]
        else:
            chosen = indices
            rng.shuffle(chosen)
            chosen = chosen[:per_bucket]
        rows = []
        for gi in chosen:
            item = ds.get_item(index=gi)
            meta = item["meta"]
            rows.append(
                {
                    "global_index": gi,
                    "episode_index": meta["episode_idx"],
                    "frame_index": meta["step_id"],
                    "state": np.asarray(item["state"]).reshape(-1).astype(np.float32),
                    "actions": np.asarray(item["actions"]).astype(np.float32),
                    "images": _images_from_sample(item),
                    "instruction": meta["instruction"],
                }
            )
        out[name] = rows
    return out


def _eval_bucket(policy: DexoraPolicy, rows: list[dict], stats_file: str) -> dict[str, Any]:
    preds, gts, states = [], [], []
    for row in rows:
        pred = policy.get_action(
            {
                "state": row["state"],
                "images": row["images"],
                "instruction": row["instruction"],
                "ctrl_freq": CTRL_FREQ_HZ,
            }
        )
        preds.append(pred)
        gts.append(row["actions"])
        states.append(row["state"])
    pred_a = np.stack(preds, axis=0)
    gt_a = np.stack(gts, axis=0)
    st_a = np.stack(states, axis=0)
    metrics = summarize_per_horizon_mse(pred_a, gt_a)
    metrics["chunk_mse"] = float(np.mean((pred_a - gt_a) ** 2))
    metrics["horizon_00_mse"] = metrics.get("horizon_00_mse", float("nan"))
    metrics.update(summarize_first_action(pred_a, st_a, stats_file=stats_file))
    metrics.update(group_mse(pred_a, gt_a))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/mnt/hdd/dexora/checkpoints/dexora-dexjoco-bimanual-assembly-relative-rot-vpred-50k/checkpoint-50000",
    )
    p.add_argument("--repo-dir", default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264")
    p.add_argument(
        "--stats-file",
        default="/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json",
    )
    p.add_argument("--model-config", default="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml")
    p.add_argument("--per-bucket", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out",
        default="/mnt/hdd/dexora/audit/cursor/openloop_phase_buckets_50k.json",
    )
    args = p.parse_args()

    buckets = _load_samples(args.repo_dir, args.stats_file, args.per_bucket, args.seed)
    policy_cfg = DexoraPolicyConfig(
        model_config_path=args.model_config,
        text_encoder_path="google/t5-v1_1-xxl",
        vision_encoder_path="google/siglip-so400m-patch14-384",
        state_dim=44,
        cameras=TRAIN_CAMERA_ORDER,
        device=args.device,
        dtype=torch.bfloat16,
    )
    policy = DexoraPolicy(args.ckpt, cfg=policy_cfg)
    report: dict[str, Any] = {
        "agent": "cursor",
        "ckpt": args.ckpt,
        "per_bucket": args.per_bucket,
        "buckets": {},
    }
    for name, rows in buckets.items():
        print(f"==> bucket {name} n={len(rows)}", flush=True)
        report["buckets"][name] = {
            "n": len(rows),
            "metrics": _eval_bucket(policy, rows, args.stats_file),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
