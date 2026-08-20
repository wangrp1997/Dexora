from pathlib import Path

import cv2
import pytest

from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset


def test_decode_failure_is_not_silently_replaced_with_black(monkeypatch, tmp_path: Path) -> None:
    dataset = object.__new__(DexJoCoLeRobotVLADataset)
    dataset.repo_dir = tmp_path
    dataset.fps = 30.0
    dataset.episode_index = [0]
    dataset.frame_index = [0]
    dataset.video_meta = {
        "ego": {"chunk": [0], "file": [0], "from_ts": [0.0]},
    }
    dataset._video_caps = {}

    class BrokenCapture:
        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: BrokenCapture())
    with pytest.raises(RuntimeError, match="Transcode the dataset"):
        dataset._read_frame("ego", 0)
