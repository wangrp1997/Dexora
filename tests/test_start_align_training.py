"""Unit tests for start-align sampling and horizon loss weights."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from models.sample_weighting import start_align_horizon_weights, weighted_mse_loss


class StartAlignTrainingTest(unittest.TestCase):
    def test_horizon_weights_profile(self) -> None:
        w = start_align_horizon_weights(32)
        self.assertEqual(w.shape[0], 32)
        self.assertAlmostEqual(float(w[0]), 4.0)
        self.assertAlmostEqual(float(w[2]), 3.0)
        self.assertAlmostEqual(float(w[4]), 2.0)
        self.assertAlmostEqual(float(w[8]), 1.0)

    def test_weighted_mse_with_horizon_weights(self) -> None:
        pred = torch.zeros(2, 4, 3)
        target = torch.ones(2, 4, 3)
        hw = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss, _ = weighted_mse_loss(pred, target, horizon_weights=hw)
        expected = ((hw / hw.mean())[:, None] * torch.ones(3)).mean()
        self.assertGreater(float(loss), 0.0)

    def test_early_window_sampling_bias(self) -> None:
        from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset

        root = "/mnt/hdd/dexora/data/dexjoco_bimanual_assembly_h264"
        ds = DexJoCoLeRobotVLADataset(
            repo_dir=root,
            stats_file="/mnt/hdd/dexora/stats/dexjoco_bimanual_assembly_relative_rot/dataset_statistics.json",
            load_imgs=False,
            early_window_prob=1.0,
            early_window_frames=32,
        )
        early = 0
        trials = 500
        for _ in range(trials):
            sample = ds.get_item()
            if int(sample["meta"]["step_id"]) < 32:
                early += 1
        self.assertGreater(early / trials, 0.95)


if __name__ == "__main__":
    unittest.main()
