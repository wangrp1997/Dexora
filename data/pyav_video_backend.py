"""Pure-PyAV video decode for lerobot under torchvision>=0.26.

torchvision 0.26 dropped ``torchvision.io.VideoReader``. lerobot<0.4 still
routes its ``pyav`` backend through that class, so training dies on
Blackwell stacks that need torch 2.11 / torchvision 0.26.

Monkey-patch ``decode_video_frames_torchvision`` to decode with PyAV only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import torch


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    del backend  # kept for call-signature compatibility with lerobot
    video_path = str(video_path)
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    container = av.open(video_path)
    stream = container.streams.video[0]
    # seek near first query; pyav seeks keyframes only
    container.seek(int(first_ts / stream.time_base), stream=stream)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    for frame in container.decode(stream):
        current_ts = float(frame.pts * stream.time_base)
        if log_loaded_timestamps:
            logging.info("frame loaded at timestamp=%s", f"{current_ts:.4f}")
        # CHW uint8, matching former VideoReader layout
        rgb = frame.to_ndarray(format="rgb24")
        loaded_frames.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break
    container.close()

    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path} for {timestamps=}")

    query_ts = torch.tensor(timestamps, dtype=torch.float64)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance "
        f"({min_[~is_within_tol]} > {tolerance_s=})."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts_t}"
        f"\nvideo: {video_path}"
        f"\nbackend: pyav-native"
    )
    closest_frames = torch.stack([loaded_frames[int(i)] for i in argmin_])
    closest_frames = closest_frames.type(torch.float32) / 255.0
    assert len(timestamps) == len(closest_frames)
    return closest_frames


def install_pyav_video_backend() -> None:
    import lerobot.datasets.video_utils as vu

    vu.decode_video_frames_torchvision = decode_video_frames_pyav
    # Force the torchvision-named path; torchcodec often fails to load here.
    def _decode(video_path, timestamps, tolerance_s, backend=None):
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s)

    vu.decode_video_frames = _decode
