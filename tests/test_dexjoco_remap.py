"""DexJoCo policy44 remap tests. No dataset I/O."""

import numpy as np

from data.dexjoco_remap import (
    action44_as_policy,
    policy44_to_action44,
    quat_wxyz_to_rotvec,
    state46_to_policy44,
)


def test_action_identity() -> None:
    rng = np.random.default_rng(0)
    a44 = rng.normal(scale=0.5, size=(4, 44)).astype(np.float32)
    out = action44_as_policy(a44)
    back = policy44_to_action44(out)
    assert out.shape == (4, 44)
    np.testing.assert_allclose(back[..., :3], a44[..., :3], atol=1e-6)
    np.testing.assert_allclose(back[..., 6:25], a44[..., 6:25], atol=1e-6)
    np.testing.assert_allclose(back[..., 28:], a44[..., 28:], atol=1e-6)
    from scipy.spatial.transform import Rotation as R

    for sl in (slice(3, 6), slice(25, 28)):
        delta = (R.from_rotvec(back[..., sl]) * R.from_rotvec(a44[..., sl]).inv()).magnitude()
        np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_state46_to_policy44_hands_and_rot() -> None:
    state = np.zeros(46, dtype=np.float32)
    state[3] = 1.0  # right w
    state[10] = 1.0  # left w
    state[14:30] = np.arange(16, dtype=np.float32)
    state[30:46] = np.arange(16, 32, dtype=np.float32)
    v = state46_to_policy44(state)
    assert v.shape == (44,)
    np.testing.assert_allclose(v[6:22], state[14:30])
    np.testing.assert_allclose(v[28:44], state[30:46])


def test_quat_wxyz_to_rotvec_180_x() -> None:
    rot = quat_wxyz_to_rotvec(np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(rot, [np.pi, 0.0, 0.0], atol=1e-5)
