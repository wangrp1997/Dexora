"""LeRobot v3 loader for DexJoCo bimanual_assembly.

Dexora's pinned ``lerobot<0.4`` only speaks v2.1 (episodes.jsonl). The DexJoCo
insert set on SSD is v3.0 (parquet episode meta + split mp4s), so this class
reads parquet/mp4 directly and exposes the same ``get_item`` interface as
``LeRobotVLADataset``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

from data.dexjoco_remap import (
    DEFAULT_PROMPT,
    POLICY_DIM,
    action44_as_policy,
    is_dexjoco_lerobot,
    state46_to_policy44,
)


class DexJoCoLeRobotVLADataset:
    DATASET_NAME = "ours"
    CAMERAS = ("ego", "wrist_left", "wrist_right")
    CAM_OUT = {
        "ego": "cam_high",
        "wrist_left": "cam_left_wrist",
        "wrist_right": "cam_right_wrist",
    }

    def __init__(
        self,
        repo_dir: str,
        normalize_mode: str = "min_max",
        stats_file: str = "new_lerobot_stats/dataset_statistics.json",
        load_imgs: bool = True,
        config_path: Optional[str] = None,
        chunk_size: int = 32,
        img_history_size: int = 1,
        state_dim_keep: Optional[int] = 44,
        early_window_prob: float = 0.0,
        early_window_frames: int = 32,
        action_target: str = "absolute",
    ) -> None:
        del state_dim_keep  # always remap to native policy 44-D
        self.early_window_prob = float(max(0.0, min(1.0, early_window_prob)))
        self.early_window_frames = int(max(1, early_window_frames))
        if action_target not in ("absolute", "residual_from_state"):
            raise ValueError(f"unsupported DexJoCo action_target: {action_target}")
        self.action_target = action_target
        if not is_dexjoco_lerobot(repo_dir):
            raise ValueError(f"not a DexJoCo insert LeRobot root: {repo_dir}")

        self.repo_dir = Path(repo_dir)
        self.load_imgs = load_imgs
        self.normalize_mode = normalize_mode
        self.stats = None
        if normalize_mode and stats_file and Path(stats_file).is_file():
            with open(stats_file, "r") as f:
                self.stats = json.load(f)

        if config_path is None:
            for candidate in (
                "configs/cross_embodiment/ec4_dexjoco_bimanual_assembly.yaml",
                "configs/base_400m.yaml",
            ):
                if Path(candidate).is_file():
                    config_path = candidate
                    break
        if config_path is not None and Path(config_path).is_file():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.CHUNK_SIZE = int(config["common"].get("action_chunk_size", chunk_size))
            self.IMG_HISTORY_SIZE = int(config["common"].get("img_history_size", img_history_size))
        else:
            self.CHUNK_SIZE = int(chunk_size)
            self.IMG_HISTORY_SIZE = int(img_history_size)

        info = json.loads((self.repo_dir / "meta" / "info.json").read_text())
        self.fps = float(info.get("fps", 30))

        self._load_tables()
        self._video_caps: dict[str, cv2.VideoCapture] = {}

    def _load_tables(self) -> None:
        data_files = sorted((self.repo_dir / "data").rglob("file-*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"no parquet under {self.repo_dir / 'data'}")
        tables = [pq.read_table(p, columns=["action", "observation.state", "episode_index", "frame_index", "index"]) for p in data_files]
        table = tables[0]
        for extra in tables[1:]:
            import pyarrow as pa

            table = pa.concat_tables([table, extra], promote_options="default")

        n = table.num_rows
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        order = np.argsort(np.asarray(table.column("index").to_pylist(), dtype=np.int64))
        self.actions44 = actions[order]
        self.states46 = states[order]
        self.episode_index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)[order]
        self.frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)[order]
        self.n = int(n)

        ep_files = sorted((self.repo_dir / "meta" / "episodes").rglob("file-*.parquet"))
        ep_tables = [pq.read_table(p) for p in ep_files]
        ep = ep_tables[0]
        for extra in ep_tables[1:]:
            import pyarrow as pa

            ep = pa.concat_tables([ep, extra], promote_options="default")

        self.ep_from = np.asarray(ep.column("dataset_from_index").to_pylist(), dtype=np.int64)
        self.ep_to = np.asarray(ep.column("dataset_to_index").to_pylist(), dtype=np.int64)
        tasks = ep.column("tasks").to_pylist()
        self.tasks: dict[int, str] = {}
        for i, task in enumerate(tasks):
            if isinstance(task, (list, tuple)) and task:
                self.tasks[i] = str(task[0])
            elif isinstance(task, str) and task.strip():
                self.tasks[i] = task.strip()
            else:
                self.tasks[i] = DEFAULT_PROMPT

        self.video_meta = {}
        for cam in self.CAMERAS:
            prefix = f"videos/observation.images.{cam}"
            self.video_meta[cam] = {
                "chunk": np.asarray(ep.column(f"{prefix}/chunk_index").to_pylist(), dtype=np.int64),
                "file": np.asarray(ep.column(f"{prefix}/file_index").to_pylist(), dtype=np.int64),
                "from_ts": np.asarray(ep.column(f"{prefix}/from_timestamp").to_pylist(), dtype=np.float64),
            }

    def __len__(self) -> int:
        return self.n

    def get_dataset_name(self) -> str:
        return self.DATASET_NAME

    def _instruction(self, ep_idx: int) -> str:
        return self.tasks.get(int(ep_idx), DEFAULT_PROMPT)

    def _normalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        if self.stats is None or self.normalize_mode is None:
            return data
        if data_type not in self.stats:
            return data
        stats_data = self.stats[data_type]
        if self.normalize_mode == "mean_std":
            mean = np.array(stats_data["mean"], dtype=np.float32)
            std = np.array(stats_data["std"], dtype=np.float32)
            std = np.where(std == 0, 1, std)
            return (data - mean) / std
        if self.normalize_mode == "min_max":
            min_val = np.array(stats_data["percentile_1"], dtype=np.float32)
            max_val = np.array(stats_data["percentile_99"], dtype=np.float32)
            span = np.where(max_val - min_val == 0, 1, max_val - min_val)
            return (data - min_val) / span
        return data

    def _read_frame(self, cam: str, global_index: int) -> np.ndarray:
        ep_idx = int(self.episode_index[global_index])
        local = int(self.frame_index[global_index])
        meta = self.video_meta[cam]
        chunk = int(meta["chunk"][ep_idx])
        file_i = int(meta["file"][ep_idx])
        from_ts = float(meta["from_ts"][ep_idx])
        frame_i = int(round(from_ts * self.fps)) + local
        path = str(
            self.repo_dir
            / "videos"
            / f"observation.images.{cam}"
            / f"chunk-{chunk:03d}"
            / f"file-{file_i:03d}.mp4"
        )
        cap = self._video_caps.get(path)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(path)
            self._video_caps[path] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_i, 0))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(
                "Failed to decode DexJoCo video frame. The released videos are AV1, "
                "which this OpenCV build cannot decode reliably. Transcode the dataset "
                "with scripts/transcode_dexjoco_videos.sh and train from the H.264 root. "
                f"video={path}, frame={frame_i}"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _sample_global_index(self) -> int:
        """Uniform frame index, or early-episode window with ``early_window_prob``."""
        if self.early_window_prob > 0.0 and np.random.rand() < self.early_window_prob:
            ep_idx = int(np.random.randint(0, len(self.ep_from)))
            ep_start = int(self.ep_from[ep_idx])
            ep_len = int(self.ep_to[ep_idx] - ep_start)
            max_local = min(self.early_window_frames, ep_len) - 1
            local = int(np.random.randint(0, max(1, max_local + 1)))
            return ep_start + local
        return int(np.random.randint(0, self.n))

    def get_item(self, index: int = None, frame_index: int = None, state_only=False):
        if frame_index is not None:
            if index is None:
                raise ValueError("When providing frame_index, 'index' must be the episode index.")
            global_index = int(self.ep_from[index]) + int(frame_index)
            global_index = min(max(global_index, int(self.ep_from[index])), int(self.ep_to[index]) - 1)
        else:
            if index is None:
                global_index = self._sample_global_index()
            else:
                global_index = int(index)

        if state_only:
            state = state46_to_policy44(self.states46[global_index : global_index + 1])
            action = action44_as_policy(self.actions44[global_index : global_index + 1])
            if self.normalize_mode:
                action = self._normalize(action, "action")
                if self.action_target == "residual_from_state":
                    action = action - self._normalize(state, "action")
                state = self._normalize(state, "state")
            return {"state": state, "action": action}

        ep_idx = int(self.episode_index[global_index])
        step_id = int(self.frame_index[global_index])
        ep_end = int(self.ep_to[ep_idx])
        chunk_end = min(global_index + self.CHUNK_SIZE, ep_end)
        actions44 = self.actions44[global_index:chunk_end]
        if actions44.shape[0] < self.CHUNK_SIZE:
            pad = np.repeat(actions44[-1:], self.CHUNK_SIZE - actions44.shape[0], axis=0)
            actions44 = np.concatenate([actions44, pad], axis=0)

        state = state46_to_policy44(self.states46[global_index])
        actions = action44_as_policy(actions44)
        if self.normalize_mode:
            actions = self._normalize(actions, "action")
            if self.action_target == "residual_from_state":
                actions = actions - self._normalize(state, "action")
            state = self._normalize(state, "state")

        sample = {
            "meta": {
                "dataset_name": self.DATASET_NAME,
                "episode_idx": ep_idx,
                "step_id": step_id,
                "instruction": self._instruction(ep_idx),
            },
            "state": state[None, ...] if state.ndim == 1 else state,
            "actions": actions,
            "state_indicator": np.ones(POLICY_DIM, dtype=bool),
        }
        st = sample["state"]
        sample["state_std"] = np.std(st, axis=0)
        sample["state_mean"] = np.mean(st, axis=0)
        sample["state_norm"] = np.sqrt(np.mean(st**2, axis=0))

        h = self.IMG_HISTORY_SIZE
        if self.load_imgs:
            for cam, out_key in self.CAM_OUT.items():
                frames = []
                for dt in range(1 - h, 1):
                    gi = min(max(global_index + dt, int(self.ep_from[ep_idx])), ep_end - 1)
                    frames.append(self._read_frame(cam, gi))
                sample[out_key] = np.stack(frames, axis=0)
                sample[out_key + "_mask"] = np.ones(h, dtype=bool)
        else:
            dummy = np.zeros((h, 640, 640, 3), dtype=np.uint8)
            for out_key in self.CAM_OUT.values():
                sample[out_key] = dummy
                sample[out_key + "_mask"] = np.ones(h, dtype=bool)

        sample["cam_third_view"] = np.zeros((h, 640, 640, 3), dtype=np.uint8)
        sample["cam_third_view_mask"] = np.zeros(h, dtype=bool)
        return sample


def maybe_make_lerobot_dataset(**kwargs):
    repo_dir = kwargs.get("repo_dir")
    if repo_dir is not None and is_dexjoco_lerobot(repo_dir):
        return DexJoCoLeRobotVLADataset(**kwargs)
    from data.lerobot_vla_dataset import LeRobotVLADataset

    return LeRobotVLADataset(**kwargs)
