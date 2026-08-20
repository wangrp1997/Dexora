#!/usr/bin/env python3
"""Dexora first-plan probe at episode starts (read-only).

Same metric family as scripts/probe_pi05_first_plan.py and eval_sim.evaluate:
pred chunk[0] (absolute rotvec env action) vs hold(state46).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_DEXORA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DEXORA_ROOT))
os.chdir(_DEXORA_ROOT)

from data.dexjoco_remap import policy44_to_action44, state46_to_policy44  # noqa: E402
from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset  # noqa: E402
from deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig  # noqa: E402
from eval_sim.action_exec import measure_first_plan_jump, state46_to_action44  # noqa: E402
from eval_sim.norm import load_minmax_stats, minmax_denormalize, minmax_normalize  # noqa: E402
from eval_sim.obs_action import TRAIN_CAMERA_ORDER  # noqa: E402

CTRL_FREQ_HZ = 30.0
SUMMARY_METRICS = (
    "first_plan_xyz_jump_m",
    "first_plan_rot_jump_rad",
    "first_plan_hand_mean",
    "pred_minus_gt_hand_mae",
    "pred_xyz_minus_gt_m",
)


def _images_from_sample(sample: dict) -> dict[str, np.ndarray]:
    out = {}
    for key in TRAIN_CAMERA_ORDER:
        if key == "cam_third_view":
            continue
        frames = np.asarray(sample[key])
        out[key] = frames[-1]
    return out


def _parse_noise_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one comma-separated noise seed")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("noise seeds must be non-negative")
    return seeds


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _metric_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        metric: _distribution([float(row[metric]) for row in rows])
        for metric in SUMMARY_METRICS
    }


def _set_sampling_seed(noise_seed: int, episode_index: int) -> int:
    sampling_seed = noise_seed * 1_000_003 + episode_index
    np.random.seed(sampling_seed % (2**32))
    torch.manual_seed(sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sampling_seed)
    return sampling_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="checkpoint-* dir or root with weights")
    p.add_argument(
        "--repo-dir",
        default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264",
    )
    p.add_argument(
        "--stats-file",
        default="/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json",
    )
    p.add_argument(
        "--model-config",
        default="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml",
    )
    p.add_argument("--text-encoder", default="google/t5-v1_1-xxl")
    p.add_argument("--vision-encoder", default="google/siglip-so400m-patch14-384")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument(
        "--noise-seeds",
        type=_parse_noise_seeds,
        default=_parse_noise_seeds("0"),
        help="Comma-separated diffusion noise seeds, reset independently per episode.",
    )
    p.add_argument("--tag", default="")
    p.add_argument(
        "--residual-action",
        action="store_true",
        help="Interpret model output as residual normalized action and add current hold anchor.",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    stats = load_minmax_stats(args.stats_file)
    act_lo, act_hi = stats["action"]

    ds = DexJoCoLeRobotVLADataset(
        repo_dir=args.repo_dir,
        stats_file=args.stats_file,
        load_imgs=True,
        state_dim_keep=44,
    )
    starts = [i for i in range(len(ds)) if int(ds.frame_index[i]) == 0][: args.n_episodes]
    print(f"Loading Dexora from {args.checkpoint} ({len(starts)} episode starts) ...", flush=True)

    cfg = DexoraPolicyConfig(
        model_config_path=args.model_config,
        text_encoder_path=args.text_encoder,
        vision_encoder_path=args.vision_encoder,
        state_dim=44,
        cameras=TRAIN_CAMERA_ORDER,
        device=args.device,
        dtype=dtype,
    )
    policy = DexoraPolicy(str(args.checkpoint), cfg=cfg)

    episode_inputs = []
    for gi in starts:
        sample = ds.get_item(gi)
        state46 = ds.states46[gi]
        hold = state46_to_action44(state46)
        gt_raw = ds.actions44[gi]
        obs = {
            "state": np.asarray(sample["state"]).reshape(-1).astype(np.float32),
            "images": _images_from_sample(sample),
            "instruction": sample["meta"]["instruction"],
            "ctrl_freq": CTRL_FREQ_HZ,
        }
        if args.residual_action:
            state_policy = state46_to_policy44(state46)
            obs["action_anchor"] = minmax_normalize(state_policy, act_lo, act_hi)
        episode_inputs.append(
            {
                "global_index": int(gi),
                "episode_index": int(ds.episode_index[gi]),
                "obs": obs,
                "hold": hold,
                "gt_raw": gt_raw,
            }
        )

    rows = []
    for noise_seed in args.noise_seeds:
        print(f"Sampling noise seed {noise_seed} ...", flush=True)
        for item in episode_inputs:
            sampling_seed = _set_sampling_seed(noise_seed, item["episode_index"])
            chunk_n = policy.get_action(item["obs"])
            chunk_pol = minmax_denormalize(chunk_n, act_lo, act_hi)
            first = policy44_to_action44(chunk_pol[0])
            hold = item["hold"]
            gt_raw = item["gt_raw"]
            jump = measure_first_plan_jump(hold, first)
            m = jump.as_dict()
            m.update(
                {
                    "global_index": item["global_index"],
                    "episode_index": item["episode_index"],
                    "noise_seed": noise_seed,
                    "sampling_seed": sampling_seed,
                    "hold_hand_mean": float(
                        max(np.abs(hold[6:22]).mean(), np.abs(hold[28:44]).mean())
                    ),
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
                f"seed{noise_seed:03d} ep{m['episode_index']:03d}: "
                f"xyz={m['first_plan_xyz_jump_m']*100:.1f}cm "
                f"rot={m['first_plan_rot_jump_rad']:.3f} "
                f"hand={m['first_plan_hand_mean']:.3f} "
                f"gt_hand={m['gt_action_hand_mean']:.3f}",
                flush=True,
            )

    by_noise_seed = {}
    for noise_seed in args.noise_seeds:
        seed_rows = [row for row in rows if row["noise_seed"] == noise_seed]
        by_noise_seed[str(noise_seed)] = {
            "n": len(seed_rows),
            "metrics": _metric_summary(seed_rows),
        }

    summary = {
        "tag": args.tag or Path(args.checkpoint).name,
        "checkpoint": str(args.checkpoint),
        "repo_dir": args.repo_dir,
        "stats_file": args.stats_file,
        "n_episodes": len(episode_inputs),
        "noise_seeds": args.noise_seeds,
        "n": len(rows),
        "sampling_seed_rule": "noise_seed * 1000003 + episode_index",
        "metrics": _metric_summary(rows),
        "by_noise_seed": by_noise_seed,
        "mean_first_plan_xyz_jump_m": float(np.mean([r["first_plan_xyz_jump_m"] for r in rows])),
        "mean_first_plan_rot_jump_rad": float(np.mean([r["first_plan_rot_jump_rad"] for r in rows])),
        "mean_first_plan_hand_mean": float(np.mean([r["first_plan_hand_mean"] for r in rows])),
        "mean_gt_action_hand_mean": float(np.mean([r["gt_action_hand_mean"] for r in rows])),
        "mean_hold_hand_mean": float(np.mean([r["hold_hand_mean"] for r in rows])),
        "mean_pred_minus_gt_hand_mae": float(np.mean([r["pred_minus_gt_hand_mae"] for r in rows])),
        "mean_pred_xyz_minus_gt_m": float(np.mean([r["pred_xyz_minus_gt_m"] for r in rows])),
        "per_episode": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
