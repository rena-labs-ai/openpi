import numpy as np
import pandas as pd

from openpi.training import rena_stepped


def _ramp(n=200):
    t = np.linspace(0, 1, n)[:, None]
    return np.concatenate([np.repeat(t, 6, axis=1) * [1, -1, 2, 0.5, 1, 1], np.linspace(0, 1, n)[:, None]], axis=1)


def _steps(action, hold=(2, 3, 1)):
    """The joint commands held for hold[i % 3] frames at a time, the gripper as it was."""
    out, i, k = action.copy(), 0, 0
    while i < len(action):
        h = hold[k % 3]
        out[i : i + h, :6] = action[i, :6]
        i, k = i + h, k + 1
    return out


def test_a_held_ramp_steps_and_the_ramp_does_not():
    assert not rena_stepped.stepped(_ramp())
    assert rena_stepped.stepped(_steps(_ramp()))


def test_interpolation_runs_straight_between_new_values_and_keeps_them_the_gripper_and_long_holds():
    held = _steps(_ramp())
    held[96:120, :6] = held[96, :6]  # the arm pauses for 24 frames, from a new value on
    out = rena_stepped.interpolate(held)
    np.testing.assert_allclose(out[:96, :6], _ramp()[:96, :6], atol=1e-9)  # a ramp, as before it was held
    np.testing.assert_array_equal(out[:, 6], held[:, 6])
    np.testing.assert_array_equal(out[96:120, :6], held[96:120, :6])
    starts = np.flatnonzero(np.any(np.diff(held[:, :6], axis=0) != 0, axis=1)) + 1
    np.testing.assert_array_equal(out[starts], held[starts])
    assert not rena_stepped.stepped(out)


def test_stepped_episodes_reads_each_episodes_actions(tmp_path):
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    for i, action in enumerate((_ramp(), _steps(_ramp()), _ramp(10))):
        pd.DataFrame({"action": list(action.astype(np.float32))}).to_parquet(
            tmp_path / f"data/chunk-000/episode_{i:06d}.parquet"
        )
    assert rena_stepped.stepped_episodes(tmp_path) == {1}
