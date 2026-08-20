"""Min-max normalize / denormalize using Dexora dataset_statistics.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_minmax_stats(stats_file: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with open(stats_file) as f:
        raw = json.load(f)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("state", "action"):
        if key not in raw:
            continue
        lo = np.asarray(raw[key]["percentile_1"], dtype=np.float32)
        hi = np.asarray(raw[key]["percentile_99"], dtype=np.float32)
        out[key] = (lo, hi)
    return out


def minmax_normalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (x - lo) / span


def minmax_denormalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * (hi - lo) + lo
