import numpy as np

from data.dexjoco_lerobot_dataset import DexJoCoLeRobotVLADataset


def test_residual_action_zero_at_hold() -> None:
    dataset = object.__new__(DexJoCoLeRobotVLADataset)
    dataset.stats = {
        "action": {
            "percentile_1": [0.0, -2.0],
            "percentile_99": [2.0, 2.0],
        }
    }
    dataset.normalize_mode = "min_max"

    hold = np.array([1.0, 0.0], dtype=np.float32)
    action_n = dataset._normalize(hold, "action")
    anchor_n = dataset._normalize(hold, "action")

    np.testing.assert_allclose(action_n - anchor_n, np.zeros(2, dtype=np.float32))
