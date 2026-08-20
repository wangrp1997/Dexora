import numpy as np

from eval_sim.action_exec import (
    interp_dual_arm_action44,
    merge_chunk_into_buffer,
    rate_limit_dual_arm_action44,
    state46_to_action44,
    TimedAction,
)


def test_rate_limit_caps_wrist_step() -> None:
    prev = np.zeros(44, dtype=np.float32)
    target = prev.copy()
    target[:3] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    out = rate_limit_dual_arm_action44(
        prev, target, max_xyz_step_m=0.02, max_rot_step_rad=0.06
    )
    assert np.linalg.norm(out[:3] - prev[:3]) <= 0.0201


def test_merge_chunk_blends_overlap() -> None:
    buf = [
        TimedAction(action=np.zeros(44, dtype=np.float32), timestamp=0),
        TimedAction(action=np.zeros(44, dtype=np.float32), timestamp=1),
    ]
    chunk = np.ones((4, 44), dtype=np.float32)
    merge_chunk_into_buffer(buf, chunk, now_timestamp=1, chunk_origin_timestamp=1)
    assert len(buf) >= 3
    assert 0.0 < float(buf[1].action[0]) < 1.0


def test_interp_rotvec_midpoint_finite() -> None:
    a = np.zeros(44, dtype=np.float32)
    b = np.zeros(44, dtype=np.float32)
    b[3:6] = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    mid = interp_dual_arm_action44(a, b, 0.5)
    assert np.isfinite(mid).all()


def test_state46_to_action44_shape() -> None:
    s = np.arange(46, dtype=np.float32)
    s[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s[10:14] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    a = state46_to_action44(s)
    assert a.shape == (44,)


def test_rot_geodesic_ignores_rotvec_wrap() -> None:
    from eval_sim.action_exec import measure_first_plan_jump, rot_geodesic_rad

    r0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r1 = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    r_wrap = np.array([0.0, 0.0, 2 * np.pi + 0.1], dtype=np.float64)
    assert rot_geodesic_rad(r0, r1) < 0.2
    assert rot_geodesic_rad(r0, r_wrap) < 0.2
    hold = np.zeros(44, dtype=np.float32)
    first = hold.copy()
    first[3:6] = np.array([0.0, 0.0, 0.1], dtype=np.float32)
    jump = measure_first_plan_jump(hold, first)
    assert jump.first_plan_rot_jump_rad < 0.2
