import numpy as np

from openpi.training import holdout


def chunk(gripper_switch_at: int | None, joints: float = 0.0) -> np.ndarray:
    c = np.zeros((50, 7))
    c[..., :6] = joints
    if gripper_switch_at is not None:
        c[gripper_switch_at:, 6] = 1.0
    return c


def test_a_gripper_switch_counts_as_caught_within_five_steps_of_the_commanded_one():
    want = np.stack([chunk(10), chunk(10), chunk(None), chunk(None)])
    got = np.stack([chunk(15), chunk(16), chunk(None), chunk(30)])
    s = holdout.summary(got, want)
    assert s["gripper_events"] == 2
    assert s["gripper_event_recall"] == 0.5
    assert s["gripper_false_switch"] == 0.5


def test_joint_errors_cover_the_whole_chunk():
    want = np.stack([chunk(None)])
    got = np.stack([chunk(None, joints=0.1)])
    got[0, 49, :6] = 1.1
    s = holdout.summary(got, want)
    assert s["joint_mae_by_step"][0] == 0.1
    assert s["joint_mae_by_step"][49] == 1.1
    np.testing.assert_allclose(s["joint_mae"], 0.12)
