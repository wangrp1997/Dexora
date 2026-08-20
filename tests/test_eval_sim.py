from eval_sim.evaluate import _should_replan, assert_normalized_action_chunk
import numpy as np
import pytest


def test_should_replan_on_schedule() -> None:
    assert _should_replan(timestamp=0, last_plan_timestamp=None, replan_steps=24)
    assert not _should_replan(timestamp=23, last_plan_timestamp=0, replan_steps=24)
    assert _should_replan(timestamp=24, last_plan_timestamp=0, replan_steps=24)
    assert not _should_replan(timestamp=47, last_plan_timestamp=24, replan_steps=24)
    assert _should_replan(timestamp=48, last_plan_timestamp=24, replan_steps=24)


def test_assert_normalized_before_denorm() -> None:
    ok = np.clip(np.linspace(0.0, 1.0, 32 * 44), 0, 1).reshape(32, 44)
    assert assert_normalized_action_chunk(ok) <= 1.0
    with pytest.raises(FloatingPointError, match="non-finite"):
        assert_normalized_action_chunk(np.array([[np.nan, 0.0]]))
    boom = np.ones((2, 44), dtype=np.float32) * 3e4
    with pytest.raises(FloatingPointError, match="normalized"):
        assert_normalized_action_chunk(boom)
