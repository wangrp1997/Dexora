import numpy as np
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from eval_sim.sample_health import gate_verdict, group_mse, summarize_normalized


def test_cosine_solver_lambda_clip_changes_start_timestep() -> None:
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        lambda_min_clipped=-5.1,
    )
    scheduler.set_timesteps(5)
    assert scheduler.timesteps[0].item() < 999


def test_group_mse_slices() -> None:
    gt = np.zeros((2, 32, 44), dtype=np.float32)
    pred = np.zeros_like(gt)
    pred[..., 0:6] = 1.0
    pred[..., 6:22] = 2.0
    out = group_mse(pred, gt)
    assert out["mse_right_hand"] == 4.0
    assert out["mse_left_hand"] == 0.0
    assert abs(out["mse_arm"] - 0.5) < 1e-6


def test_gate_step0_explosion_is_not_hard_fail() -> None:
    boom = summarize_normalized(np.full((2, 32, 44), 3e4))
    ok = summarize_normalized(np.full((2, 32, 44), 0.4))
    per_step = {
        0: {"solver": boom, "epsilon": {"mse": 10.0}},
        250: {"solver": ok, "epsilon": {"mse": 0.4}},
        500: {"solver": ok, "epsilon": {"mse": 0.2}},
    }
    v = gate_verdict(per_step, last_step=500)
    assert v["hard_fail"] is False
    assert v["pass"] is True
    assert v["learning"] is True


def test_gate_hard_fail_at_500() -> None:
    boom = summarize_normalized(np.full((2, 32, 44), 3e4))
    per_step = {
        0: {"solver": boom, "epsilon": {"mse": 10.0}},
        250: {"solver": boom, "epsilon": {"mse": 8.0}},
        500: {"solver": boom, "epsilon": {"mse": 7.0}},
    }
    v = gate_verdict(per_step, last_step=500)
    assert v["hard_fail"] is True
    assert v["pass"] is False


def test_gate_learning_uses_latest_earlier_checkpoint() -> None:
    ok = summarize_normalized(np.full((2, 32, 44), 0.4))
    per_step = {
        1000: {"solver": ok, "epsilon": {"mse": 0.4}},
        2500: {"solver": ok, "epsilon": {"mse": 0.3}},
        5000: {"solver": ok, "epsilon": {"mse": 0.2}},
    }
    assert gate_verdict(per_step, last_step=5000)["learning"] is True
