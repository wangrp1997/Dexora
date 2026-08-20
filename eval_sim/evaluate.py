"""In-process Dexora closed-loop eval on DexJoCo insert (bimanual_assembly).

Run with conda env ``dexjoco`` (MuJoCo + imageio) and ``PYTHONPATH`` covering
Dexora + ``dexjoco/dexjoco``. Video protocol matches pi0.5 / BotYard
(imageio, 30 fps, 1500-frame cap, success/failure rename).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import imageio
import numpy as np
import torch
import yaml

_DEXORA_ROOT = Path(__file__).resolve().parents[1]
if str(_DEXORA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXORA_ROOT))

from deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig  # noqa: E402
from deploy.dexjoco_action import policy44_to_action44  # noqa: E402
from eval_sim.action_exec import (  # noqa: E402
    TimedAction,
    measure_first_plan_jump,
    merge_chunk_into_buffer,
    rate_limit_dual_arm_action44,
    state46_to_action44,
)
from eval_sim.norm import load_minmax_stats, minmax_denormalize, minmax_normalize  # noqa: E402
from eval_sim.obs_action import (  # noqa: E402
    TRAIN_CAMERA_ORDER,
    env_raw_to_dexora_images,
    env_state_to_policy44,
)

EVAL_MAX_VIDEO_FRAMES = 1500
CTRL_FREQ_HZ = 30.0
NORMALIZED_ABS_HARD_FAIL = 10.0
DEFAULT_REPLAN_STEPS = 8
DEFAULT_SETTLE_STEPS = 5
DEFAULT_MAX_WRIST_STEP_M = 0.02
DEFAULT_MAX_WRIST_ROT_STEP_RAD = 0.06

__all__ = ["main", "assert_normalized_action_chunk", "TimedAction"]


def assert_normalized_action_chunk(
    chunk: np.ndarray,
    *,
    max_abs: float = NORMALIZED_ABS_HARD_FAIL,
) -> float:
    """Fail closed on non-finite or exploding *normalized* actions (before denorm)."""
    if not np.isfinite(chunk).all():
        raise FloatingPointError("Policy produced non-finite normalized actions")
    peak = float(np.abs(chunk).max())
    if peak > max_abs:
        raise FloatingPointError(
            "Policy produced implausible normalized actions "
            f"(max_abs={peak:.3f} > {max_abs}). Do not execute this checkpoint."
        )
    return peak


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _should_replan(timestamp: int, last_plan_timestamp: Optional[int], replan_steps: int) -> bool:
    return last_plan_timestamp is None or timestamp - last_plan_timestamp >= replan_steps


def _append_video_frames(video_writers: dict, raw_images: dict, frame_count: list[int]) -> None:
    if frame_count[0] >= EVAL_MAX_VIDEO_FRAMES:
        return
    for cam_name, writer in video_writers.items():
        writer.append_data(raw_images[cam_name])
    frame_count[0] += 1


def _resolve_dexjoco_root() -> Path:
    env = os.environ.get("DEXJOCO_ROOT", "").strip()
    if env:
        return Path(env)
    sibling = _DEXORA_ROOT.parent / "dexjoco"
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError("Set DEXJOCO_ROOT to the DexJoCo repo root.")


def main(
    config: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    model_config: Optional[Path] = None,
    stats_file: Optional[Path] = None,
    seed: int = 0,
    output: Optional[Path] = None,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    episodes: int = 1,
    overwrite: bool = False,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    replan_steps: int = DEFAULT_REPLAN_STEPS,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    max_steps: int = 1500,
    action_smooth: bool = True,
    chunk_blend: bool = True,
    max_wrist_step_m: float = DEFAULT_MAX_WRIST_STEP_M,
    max_wrist_rot_step_rad: float = DEFAULT_MAX_WRIST_ROT_STEP_RAD,
    legacy_exec: bool = False,
    faithful_exec: bool = False,
    residual_action: bool = False,
    device: str = "cuda",
    inference_dtype: Literal["bf16", "fp32"] = "bf16",
    text_encoder: str = "google/t5-v1_1-xxl",
    vision_encoder: str = "google/siglip-so400m-patch14-384",
) -> None:
    dex_root = _resolve_dexjoco_root()
    for p in (str(dex_root), str(dex_root / "dexjoco")):
        if p not in sys.path:
            sys.path.insert(0, p)

    if faithful_exec:
        action_smooth = False
        chunk_blend = False
        settle_steps = 0
    elif legacy_exec:
        action_smooth = False
        chunk_blend = False
        settle_steps = 0

    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    _set_seed(seed)

    if config is None:
        config = dex_root / "configs" / "rand_obj" / "bimanual_assembly.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    with open(config) as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    prompt = cfg["prompt"]
    if cfg["robot_type"] != "dual_arm":
        raise ValueError("dual_arm only")

    if model_config is None:
        model_config = _DEXORA_ROOT / "configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml"
    if stats_file is None:
        stats_file = Path(
            os.environ.get(
                "DEXORA_STATS",
                "/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json",
            )
        )
    if checkpoint is None:
        ckpt_env = os.environ.get("DEXORA_CKPT", "").strip()
        checkpoint = Path(ckpt_env) if ckpt_env else None
    if checkpoint is None or not Path(checkpoint).exists():
        raise FileNotFoundError("Pass --checkpoint to a Dexora FT dir (or set DEXORA_CKPT)")

    out_root = os.environ.get("DEXORA_EVAL_DIR", "").strip()
    if output is not None:
        output_dir = Path(output)
    elif out_root:
        output_dir = Path(out_root) / f"{env_name}_seed{seed}_dexora"
    else:
        output_dir = Path("outputs") / "dexora" / f"{env_name}_seed{seed}_dexora"
        if not output_dir.is_absolute():
            output_dir = (dex_root / output_dir).resolve()

    print(f"Eval output: {output_dir}", flush=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_minmax_stats(stats_file)
    state_lo, state_hi = stats["state"]
    act_lo, act_hi = stats["action"]
    if state_lo.shape != (44,) or state_hi.shape != (44,):
        raise ValueError(f"Expected 44-D state stats, got {state_lo.shape} / {state_hi.shape}")
    if act_lo.shape != (44,) or act_hi.shape != (44,):
        raise ValueError(f"Expected 44-D action stats, got {act_lo.shape} / {act_hi.shape}")
    if replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {replan_steps}")
    print(f"Loaded min-max stats from {stats_file} (dim={state_lo.shape[0]})", flush=True)
    print(
        f"Exec: replan={replan_steps} settle={settle_steps} "
        f"smooth={action_smooth} blend={chunk_blend} "
        f"max_step={max_wrist_step_m}m max_rot={max_wrist_rot_step_rad}rad",
        flush=True,
    )

    os.chdir(_DEXORA_ROOT)
    policy_cfg = DexoraPolicyConfig(
        model_config_path=str(model_config),
        text_encoder_path=text_encoder,
        vision_encoder_path=vision_encoder,
        state_dim=44,
        cameras=TRAIN_CAMERA_ORDER,
        device=device,
        dtype=torch.bfloat16 if inference_dtype == "bf16" else torch.float32,
    )
    print(f"Loading DexoraPolicy from {checkpoint} ...", flush=True)
    policy = DexoraPolicy(str(checkpoint), cfg=policy_cfg)
    if faithful_exec:
        replan_steps = int(policy.cfg.chunk_size)
        print(
            f"Faithful Dexora exec: replan_steps={replan_steps} "
            f"(full chunk, no settle/smooth/blend)",
            flush=True,
        )
    if replan_steps > policy.cfg.chunk_size:
        raise ValueError(
            f"replan_steps={replan_steps} exceeds action chunk size={policy.cfg.chunk_size}"
        )

    eval_metadata = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "model_config": str(Path(model_config).resolve()),
        "stats_file": str(Path(stats_file).resolve()),
        "dexjoco_config": str(Path(config).resolve()),
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "rand_full": rand_full,
        "randomize_dynamics": randomize_dynamics,
        "replan_steps": replan_steps,
        "settle_steps": settle_steps,
        "action_smooth": action_smooth,
        "chunk_blend": chunk_blend,
        "max_wrist_step_m": max_wrist_step_m,
        "max_wrist_rot_step_rad": max_wrist_rot_step_rad,
        "legacy_exec": legacy_exec,
        "faithful_exec": faithful_exec,
        "residual_action": residual_action,
        "action_chunk_size": policy.cfg.chunk_size,
        "state_dim": policy.cfg.state_dim,
        "inference_dtype": inference_dtype,
        "camera_order": list(TRAIN_CAMERA_ORDER),
        "text_encoder": text_encoder,
        "vision_encoder": vision_encoder,
    }
    (output_dir / "eval_config.json").write_text(json.dumps(eval_metadata, indent=2) + "\n")

    os.chdir(dex_root)

    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv

    env = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=camera_mapping,
        seed=seed,
        rand_full=rand_full,
        randomize_dynamics=randomize_dynamics,
        dual_arm=True,
        prompt=prompt,
        render_mode=render_mode,
    )
    env.start()

    video_writers = None
    try:
        num_success = 0
        action_min = np.full(44, np.inf, dtype=np.float64)
        action_max = np.full(44, -np.inf, dtype=np.float64)
        action_outside_stats = 0
        action_value_count = 0
        first_plan_stats: list[dict[str, float]] = []
        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}", flush=True)
            video_dir = output_dir / f"episode_{ep:02d}_temp"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writers = {
                cam_name: imageio.get_writer(video_dir / f"{cam_name}.mp4", fps=30)
                for cam_name in camera_mapping.values()
            }

            env.reset()
            timestamp = 0
            last_plan_timestamp: Optional[int] = None
            chunk_origin_timestamp = 0
            actions_buffer: deque[TimedAction] = deque()
            video_frame_count = [0]
            last_executed: Optional[np.ndarray] = None
            t0 = time.time()

            raw_images = env.get_raw_images()
            _append_video_frames(video_writers, raw_images, video_frame_count)

            for _ in range(settle_steps):
                env.stay(continue_stay=_ > 0)
                timestamp += 1
                raw_images = env.get_raw_images()
                _append_video_frames(video_writers, raw_images, video_frame_count)
            last_executed = state46_to_action44(env.obs["state"])

            while not env._done and timestamp < max_steps:
                if not actions_buffer or _should_replan(
                    timestamp, last_plan_timestamp, replan_steps
                ):
                    state44 = env_state_to_policy44(env.obs["state"])
                    state_n = minmax_normalize(state44, state_lo, state_hi)
                    images = env_raw_to_dexora_images(env.get_raw_images())
                    policy_obs = {
                        "state": state_n,
                        "images": images,
                        "instruction": prompt,
                        "ctrl_freq": CTRL_FREQ_HZ,
                    }
                    if residual_action:
                        policy_obs["action_anchor"] = state_n
                    chunk = policy.get_action(policy_obs)
                    assert_normalized_action_chunk(chunk)
                    chunk_policy = minmax_denormalize(chunk, act_lo, act_hi)
                    if chunk_policy.shape != (policy.cfg.chunk_size, 44):
                        raise ValueError(
                            f"Expected action chunk {(policy.cfg.chunk_size, 44)}, "
                            f"got {chunk_policy.shape}"
                        )
                    action_min = np.minimum(action_min, chunk_policy.min(axis=0))
                    action_max = np.maximum(action_max, chunk_policy.max(axis=0))
                    action_outside_stats += int(
                        ((chunk_policy < act_lo) | (chunk_policy > act_hi)).sum()
                    )
                    action_value_count += int(chunk_policy.size)
                    chunk_env = policy44_to_action44(chunk_policy)

                    if last_plan_timestamp is None:
                        hold = state46_to_action44(env.obs["state"])
                        jump = measure_first_plan_jump(hold, chunk_env[0])
                        first_plan_stats.append(jump.as_dict())
                        print(
                            f"  first-plan jump: xyz={jump.first_plan_xyz_jump_m*100:.1f}cm "
                            f"rot={jump.first_plan_rot_jump_rad:.3f}rad "
                            f"hand={jump.first_plan_hand_mean:.3f}",
                            flush=True,
                        )

                    chunk_origin_timestamp = timestamp
                    last_plan_timestamp = timestamp
                    if chunk_blend:
                        merge_chunk_into_buffer(
                            actions_buffer,
                            chunk_env,
                            now_timestamp=timestamp,
                            chunk_origin_timestamp=chunk_origin_timestamp,
                        )
                    else:
                        actions_buffer.clear()
                        for i, act in enumerate(chunk_env):
                            actions_buffer.append(
                                TimedAction(
                                    action=act.astype(np.float32), timestamp=timestamp + i
                                )
                            )

                timed = actions_buffer.popleft()
                action = timed.action.astype(np.float32, copy=False)
                if action_smooth and last_executed is not None:
                    action = rate_limit_dual_arm_action44(
                        last_executed,
                        action,
                        max_xyz_step_m=max_wrist_step_m,
                        max_rot_step_rad=max_wrist_rot_step_rad,
                    )
                env.step(action)
                last_executed = action.copy()
                timestamp += 1
                raw_images = env.get_raw_images()
                _append_video_frames(video_writers, raw_images, video_frame_count)
                if video_frame_count[0] >= EVAL_MAX_VIDEO_FRAMES:
                    env._done = True
                if timestamp % 50 == 0:
                    print(
                        f"  t={timestamp} success={env._success} frames={video_frame_count[0]}",
                        flush=True,
                    )

            for w in video_writers.values():
                w.close()
            video_writers = None
            tag = "success" if env._success else "failure"
            if env._success:
                num_success += 1
            final_dir = output_dir / f"episode_{ep:02d}_{tag}"
            video_dir.rename(final_dir)
            print(
                f"  done in {time.time()-t0:.1f}s -> {final_dir.name} "
                f"(steps={timestamp})",
                flush=True,
            )

        summary = {
            "successes": num_success,
            "episodes": episodes,
            "success_rate": num_success / episodes if episodes else 0.0,
            "action_min": action_min.tolist() if action_value_count else [],
            "action_max": action_max.tolist() if action_value_count else [],
            "action_outside_stats_fraction": (
                action_outside_stats / action_value_count if action_value_count else 0.0
            ),
            "first_plan_stats": first_plan_stats,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Success {num_success}/{episodes}", flush=True)
    finally:
        if video_writers is not None:
            for w in video_writers.values():
                try:
                    w.close()
                except Exception:
                    pass
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
