"""Fixed-batch raw diffusion health gate for DexJoCo 44-D finetune.

Does **not** apply scheduler thresholding. Step 0 may explode; hard fail is
the last requested step (default 500).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

_DEXORA_ROOT = Path(__file__).resolve().parents[1]
if str(_DEXORA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXORA_ROOT))

from deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig  # noqa: E402
from eval_sim.action_exec import rot_geodesic_rad  # noqa: E402
from eval_sim.obs_action import TRAIN_CAMERA_ORDER  # noqa: E402

ARM_SLICES = ((0, 6), (22, 28))
RIGHT_HAND = (6, 22)
LEFT_HAND = (28, 44)
RIGHT_XYZ = slice(0, 3)
LEFT_XYZ = slice(22, 25)
RIGHT_ROT = slice(3, 6)
LEFT_ROT = slice(25, 28)
HARD_FAIL_MAX_ABS = 10.0
BOUNDED_LO, BOUNDED_HI = -1.0, 2.0
BOUNDED_FRAC = 0.999
FIRST_ACTION_XYZ_JUMP_WARN_M = 0.02
FIRST_ACTION_HAND_MEAN_WARN = 0.05
PROBE_TIMESTEP = 999
CTRL_FREQ_HZ = 30.0
DEFAULT_INDICES = (0, 1000, 5000, 20000, 40000)


def _slice_mse(pred: np.ndarray, gt: np.ndarray, sl: tuple[int, int]) -> float:
    a, b = sl
    return float(np.mean((pred[..., a:b] - gt[..., a:b]) ** 2))


def group_mse(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    arm = 0.5 * (
        _slice_mse(pred, gt, ARM_SLICES[0]) + _slice_mse(pred, gt, ARM_SLICES[1])
    )
    return {
        "mse_all": float(np.mean((pred - gt) ** 2)),
        "mse_arm": arm,
        "mse_right_hand": _slice_mse(pred, gt, RIGHT_HAND),
        "mse_left_hand": _slice_mse(pred, gt, LEFT_HAND),
    }


def summarize_normalized(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    finite = bool(np.isfinite(x).all())
    peak = float(np.abs(x).max()) if x.size else float("nan")
    std = float(x.std()) if x.size else float("nan")
    in_band = float(np.mean((x >= BOUNDED_LO) & (x <= BOUNDED_HI))) if x.size else 0.0
    ood01 = float(np.mean((x < 0.0) | (x > 1.0))) if x.size else 1.0
    return {
        "finite": finite,
        "max_abs": peak,
        "std": std,
        "frac_in_m1_2": in_band,
        "ood_vs_01": ood01,
    }


def summarize_tail(arr: np.ndarray) -> dict[str, Any]:
    """Extra diagnostics for pilot: rotvec norms and bound violations."""
    x = np.asarray(arr, dtype=np.float64)
    abs_x = np.abs(x)
    outside = (x < BOUNDED_LO) | (x > BOUNDED_HI)
    margin = np.maximum(BOUNDED_LO - x, x - BOUNDED_HI)
    margin = np.where(outside, margin, 0.0)
    r_norm = np.linalg.norm(x[..., 3:6], axis=-1).reshape(-1)
    l_norm = np.linalg.norm(x[..., 25:28], axis=-1).reshape(-1)
    both = np.concatenate([r_norm, l_norm], axis=0) if r_norm.size else np.array([])

    def _pct(v: np.ndarray, q: float) -> float:
        return float(np.percentile(v, q)) if v.size else float("nan")

    dim_out = outside.mean(axis=tuple(range(outside.ndim - 1))) if outside.size else np.zeros(0)
    top_dims = (
        np.argsort(-dim_out)[:8].astype(int).tolist() if dim_out.size else []
    )
    return {
        "n_out_of_bound": int(outside.sum()),
        "frac_out_of_bound": float(outside.mean()) if outside.size else 0.0,
        "max_violation_margin": float(margin.max()) if margin.size else 0.0,
        "mean_violation_margin": float(margin[outside].mean()) if outside.any() else 0.0,
        "rotvec_norm_p95": _pct(both, 95),
        "rotvec_norm_p99": _pct(both, 99),
        "rotvec_norm_p99_9": _pct(both, 99.9),
        "rotvec_norm_max": float(both.max()) if both.size else float("nan"),
        "abs_p95": _pct(abs_x.reshape(-1), 95),
        "abs_p99": _pct(abs_x.reshape(-1), 99),
        "abs_p99_9": _pct(abs_x.reshape(-1), 99.9),
        "top_ood_dims": top_dims,
        "top_ood_frac": [float(dim_out[i]) for i in top_dims],
    }


def summarize_first_action(
    pred_n: np.ndarray,
    state_n: np.ndarray,
    *,
    stats_file: str,
) -> dict[str, float]:
    """Compare denormalized chunk[0] to current proprio (policy44 layout)."""
    from eval_sim.norm import load_minmax_stats, minmax_denormalize

    stats = load_minmax_stats(stats_file)
    s_lo, s_hi = stats["state"]
    a_lo, a_hi = stats["action"]
    pred0 = minmax_denormalize(pred_n[:, 0], a_lo, a_hi)
    state = np.asarray(state_n, dtype=np.float64).reshape(-1)
    if state.shape[0] > 44:
        state = state[-44:]
    state44 = minmax_denormalize(state, s_lo, s_hi)
    xyz = float(
        max(
            np.linalg.norm(pred0[:, :3] - state44[:3], axis=-1).max(),
            np.linalg.norm(pred0[:, 22:25] - state44[22:25], axis=-1).max(),
        )
    )
    rot_vals = []
    for b in range(pred0.shape[0]):
        rot_vals.append(
            max(
                rot_geodesic_rad(state44[3:6], pred0[b, 3:6]),
                rot_geodesic_rad(state44[25:28], pred0[b, 25:28]),
            )
        )
    rot = float(max(rot_vals)) if rot_vals else 0.0
    hand = float(
        max(
            np.abs(pred0[:, 6:22]).mean(),
            np.abs(pred0[:, 28:44]).mean(),
        )
    )
    return {
        "first_action_xyz_jump_m": xyz,
        "first_action_rot_jump_rad": rot,
        "first_action_hand_mean": hand,
    }


def summarize_per_horizon_mse(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    horizon = min(pred.shape[1], gt.shape[1])
    out: dict[str, float] = {}
    for h in range(horizon):
        out[f"horizon_{h:02d}_mse"] = float(np.mean((pred[:, h] - gt[:, h]) ** 2))
    return out


def _episode_start_indices(repo_dir: str, limit: int = 20) -> tuple[int, ...]:
    from pathlib import Path

    import pyarrow.parquet as pq

    root = Path(repo_dir) / "data" / "chunk-000"
    if not root.is_dir():
        return (0,)
    indices: list[int] = []
    offset = 0
    for path in sorted(root.glob("*.parquet")):
        table = pq.read_table(str(path), columns=["frame_index"])
        frames = table.column("frame_index").to_pylist()
        for local_i, frame_idx in enumerate(frames):
            if int(frame_idx) == 0:
                indices.append(offset + local_i)
        offset += len(frames)
        if len(indices) >= limit:
            break
    return tuple(indices[:limit]) if indices else (0,)


def gate_verdict(per_step: dict[int, dict[str, Any]], *, last_step: int = 500) -> dict[str, Any]:
    """Step 0 explosion is diagnostic. Hard fail looks at ``last_step`` only."""
    last = per_step[last_step]
    solver = last["solver"]
    hard_fail = (not solver["finite"]) or (solver["max_abs"] > HARD_FAIL_MAX_ABS)
    bounded = solver["frac_in_m1_2"] >= BOUNDED_FRAC
    learning = False
    earlier_steps = sorted(step for step in per_step if step < last_step)
    mid = earlier_steps[-1] if earlier_steps else None
    if mid is not None:
        learning = float(last["epsilon"]["mse"]) < float(per_step[mid]["epsilon"]["mse"])
    passed = (not hard_fail) and bounded
    reasons = []
    if hard_fail:
        reasons.append(
            f"step {last_step} hard-fail: finite={solver['finite']} "
            f"max_abs={solver['max_abs']:.4g} (limit {HARD_FAIL_MAX_ABS})"
        )
    if not bounded:
        reasons.append(
            f"step {last_step} not basically bounded: "
            f"frac_in[-1,2]={solver['frac_in_m1_2']:.4f} < {BOUNDED_FRAC}"
        )
    if not learning:
        reasons.append("epsilon MSE did not drop from 250 to last step (warning)")
    start = last.get("episode_start", {})
    start_xyz = float(start.get("first_action_xyz_jump_m", 0.0))
    start_hand = float(start.get("first_action_hand_mean", 0.0))
    start_align = (
        start_xyz <= FIRST_ACTION_XYZ_JUMP_WARN_M
        and start_hand <= FIRST_ACTION_HAND_MEAN_WARN
    )
    if start and not start_align:
        reasons.append(
            f"episode-start misaligned: xyz_jump={start_xyz:.3f}m "
            f"(warn>{FIRST_ACTION_XYZ_JUMP_WARN_M}), hand_mean={start_hand:.3f} "
            f"(warn>{FIRST_ACTION_HAND_MEAN_WARN})"
        )
    return {
        "pass": passed,
        "hard_fail": hard_fail,
        "bounded": bounded,
        "learning": learning,
        "start_align": start_align if start else None,
        "reasons": reasons,
        "note": "step 0 may explode; that is not a hard fail",
    }


def _images_from_sample(sample: dict) -> dict[str, np.ndarray]:
    out = {}
    for key in TRAIN_CAMERA_ORDER:
        if key == "cam_third_view":
            continue
        frames = np.asarray(sample[key])
        out[key] = frames[-1]
    return out


def _load_fixed_batch(
    repo_dir: str,
    stats_file: str,
    indices: tuple[int, ...],
) -> list[dict]:
    from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset

    ds = DexJoCoLeRobotVLADataset(
        repo_dir=repo_dir, stats_file=stats_file, load_imgs=True, state_dim_keep=44
    )
    batch = []
    n = len(ds)
    for idx in indices:
        gi = min(max(int(idx), 0), n - 1)
        sample = ds.get_item(gi)
        batch.append(
            {
                "index": gi,
                "state": np.asarray(sample["state"]).reshape(-1).astype(np.float32),
                "actions": np.asarray(sample["actions"]).astype(np.float32),
                "images": _images_from_sample(sample),
                "instruction": sample["meta"]["instruction"],
            }
        )
    return batch


def _eval_checkpoint(
    ckpt: Path,
    batch: list[dict],
    *,
    model_config: str,
    text_encoder: str,
    vision_encoder: str,
    device: str,
    noise_seed: int,
    dtype: torch.dtype,
    lambda_min_clipped: float | None,
    stats_file: str,
) -> dict[str, Any]:
    cfg = DexoraPolicyConfig(
        model_config_path=model_config,
        text_encoder_path=text_encoder,
        vision_encoder_path=vision_encoder,
        state_dim=44,
        cameras=TRAIN_CAMERA_ORDER,
        device=device,
        dtype=dtype,
    )
    policy = DexoraPolicy(str(ckpt), cfg=cfg)
    runner = policy.policy
    if lambda_min_clipped is not None:
        runner.noise_scheduler_sample = DPMSolverMultistepScheduler.from_config(
            runner.noise_scheduler_sample.config,
            lambda_min_clipped=lambda_min_clipped,
        )
    preds, gts = [], []
    eps_preds, eps_tgts = [], []
    torch.manual_seed(noise_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(noise_seed)

    for item in batch:
        obs = {
            "state": item["state"],
            "images": item["images"],
            "instruction": item["instruction"],
            "ctrl_freq": CTRL_FREQ_HZ,
        }
        pred = policy.get_action(obs)
        preds.append(pred)
        gts.append(item["actions"])

        lang, mask = policy._encode_language(item["instruction"])
        img = policy._encode_images(item["images"])
        # Encodings from DexoraPolicy sit in inference_mode; clone for no_grad loss.
        lang = lang.clone()
        mask = mask.clone()
        img = img.clone()
        state = torch.from_numpy(item["state"][None, None, :]).to(
            policy.device, dtype=dtype
        )
        action_gt = torch.from_numpy(item["actions"][None]).to(policy.device, dtype=dtype)
        action_mask = torch.ones((1, 1, 44), device=policy.device, dtype=dtype)
        ctrl = torch.tensor([CTRL_FREQ_HZ], device=policy.device, dtype=dtype)
        gen = torch.Generator(device=policy.device)
        gen.manual_seed(noise_seed + int(item["index"]))
        noise = torch.randn(
            action_gt.shape, generator=gen, device=policy.device, dtype=torch.float32
        ).to(dtype)
        timesteps = torch.full((1,), PROBE_TIMESTEP, device=policy.device, dtype=torch.long)
        with torch.no_grad():
            _, info = runner.compute_loss(
                lang_tokens=lang,
                lang_attn_mask=mask,
                img_tokens=img,
                state_tokens=state,
                action_gt=action_gt,
                action_mask=action_mask,
                ctrl_freqs=ctrl,
                return_dict=True,
                noise=noise,
                timesteps=timesteps,
            )
        eps_preds.append(info["pred"].detach().float().cpu().numpy())
        eps_tgts.append(info["target"].detach().float().cpu().numpy())

    pred_a = np.stack(preds, axis=0)
    gt_a = np.stack(gts, axis=0)
    states = np.stack([np.asarray(item["state"], dtype=np.float32) for item in batch], axis=0)
    solver = summarize_normalized(pred_a)
    solver.update(group_mse(pred_a, gt_a))
    solver.update(summarize_tail(pred_a))
    solver.update(summarize_first_action(pred_a, states, stats_file=stats_file))
    solver.update(summarize_per_horizon_mse(pred_a, gt_a))
    eps_p = np.concatenate(eps_preds, axis=0)
    eps_t = np.concatenate(eps_tgts, axis=0)
    ground_truth = summarize_normalized(gt_a)
    ground_truth.update(summarize_tail(gt_a))
    dim_ood = np.mean((pred_a < 0.0) | (pred_a > 1.0), axis=(0, 1)).astype(np.float64)
    prediction = summarize_normalized(eps_p)
    prediction["mse"] = float(np.mean((eps_p - eps_t) ** 2))
    prediction["pred_std"] = float(eps_p.std())
    prediction["prediction_type"] = runner.prediction_type
    del policy
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "solver": solver,
        "ground_truth": ground_truth,
        "epsilon": prediction,
        "per_dim_ood_vs_01": dim_ood.tolist(),
        "n_samples": len(batch),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--steps", default="0,100,250,500")
    p.add_argument("--repo-dir", default="/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264")
    p.add_argument("--stats-file", default="/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json")
    p.add_argument("--model-config", default="configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml")
    p.add_argument("--text-encoder", default="google/t5-v1_1-xxl")
    p.add_argument("--vision-encoder", default="google/siglip-so400m-patch14-384")
    p.add_argument("--device", default="cuda")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument(
        "--lambda-min-clipped",
        type=float,
        default=None,
        help="Override DPMSolver++ minimum log-SNR for cosine-schedule stability.",
    )
    args = p.parse_args()

    os.chdir(_DEXORA_ROOT)
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    batch = _load_fixed_batch(args.repo_dir, args.stats_file, DEFAULT_INDICES)
    ep_indices = _episode_start_indices(args.repo_dir, limit=10)
    start_batch = _load_fixed_batch(args.repo_dir, args.stats_file, ep_indices)
    per_step: dict[int, dict[str, Any]] = {}
    root = Path(args.ckpt_root)
    for step in steps:
        ckpt = root / f"checkpoint-{step}"
        if not ckpt.is_dir():
            raise FileNotFoundError(ckpt)
        print(f"==> health step {step} raw {ckpt}", flush=True)
        per_step[step] = _eval_checkpoint(
            ckpt,
            batch,
            model_config=args.model_config,
            text_encoder=args.text_encoder,
            vision_encoder=args.vision_encoder,
            device=args.device,
            noise_seed=args.noise_seed,
            dtype=dtype,
            lambda_min_clipped=args.lambda_min_clipped,
            stats_file=args.stats_file,
        )
        if step == steps[-1]:
            start_eval = _eval_checkpoint(
                ckpt,
                start_batch,
                model_config=args.model_config,
                text_encoder=args.text_encoder,
                vision_encoder=args.vision_encoder,
                device=args.device,
                noise_seed=args.noise_seed,
                dtype=dtype,
                lambda_min_clipped=args.lambda_min_clipped,
                stats_file=args.stats_file,
            )
            per_step[step]["episode_start"] = {
                k: start_eval["solver"][k]
                for k in (
                    "first_action_xyz_jump_m",
                    "first_action_rot_jump_rad",
                    "first_action_hand_mean",
                    "horizon_00_mse",
                )
            }
        s = per_step[step]["solver"]
        e = per_step[step]["epsilon"]
        print(
            f"    solver max_abs={s['max_abs']:.4g} std={s['std']:.4g} "
            f"frac[-1,2]={s['frac_in_m1_2']:.4f} "
            f"first_xyz={s.get('first_action_xyz_jump_m', float('nan')):.3f}m "
            f"eps_mse={e['mse']:.4g} eps_std={e['pred_std']:.4g}",
            flush=True,
        )
        if step == steps[-1]:
            es = per_step[step]["episode_start"]
            print(
                f"    episode_start xyz={es['first_action_xyz_jump_m']:.3f}m "
                f"hand={es['first_action_hand_mean']:.3f} h0_mse={es['horizon_00_mse']:.4f}",
                flush=True,
            )

    last = steps[-1]
    verdict = gate_verdict(per_step, last_step=last)
    payload = {
        "ckpt_root": str(root),
        "steps": steps,
        "indices": list(DEFAULT_INDICES),
        "noise_seed": args.noise_seed,
        "use_ema": False,
        "thresholding": False,
        "lambda_min_clipped": args.lambda_min_clipped,
        "per_step": {str(k): v for k, v in per_step.items()},
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    if verdict["hard_fail"]:
        sys.exit(2)
    if not verdict["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
