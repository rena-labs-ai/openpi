import json

import numpy as np

from openpi import transforms as _transforms
from openpi.policies import rena_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

MASK = _transforms.make_bool_mask(6, -1)


def test_command_anchor_makes_joint_actions_relative_to_the_last_command_and_back():
    state = np.array([0.0, -1.0, 1.0, 0.5, 0.0, 0.0, 0.3], np.float32)
    prev = np.array([0.1, -0.9, 1.1, 0.6, 0.1, 0.1, 0.9], np.float32)
    chunk = np.stack([prev, prev + 0.2, prev + 0.4]).astype(np.float32)
    data = rena_policy.AnchorOnCommand()(rena_policy.SplitPrevAction()({"observation.state": state, "actions": chunk}))
    np.testing.assert_allclose(data["observation.state"], [0.1, -0.9, 1.1, 0.6, 0.1, 0.1, 0.3])
    rel = _transforms.DeltaActions(MASK)({"state": data["observation.state"], "actions": np.array(data["actions"])})
    np.testing.assert_allclose(rel["actions"][:, :6], [[0.2] * 6, [0.4] * 6], atol=1e-6)
    np.testing.assert_allclose(rel["actions"][:, 6], [1.1, 1.3], atol=1e-6)
    back = _transforms.AbsoluteActions(MASK)(rel)
    np.testing.assert_allclose(back["actions"], chunk[1:], atol=1e-6)


def test_per_timestep_stats_map_each_steps_quantiles_to_minus_one_and_one_and_round_trip():
    q01 = np.stack([np.full(7, -0.1 * (k + 1)) for k in range(3)])
    q99 = -q01
    stats = {"actions": _normalize.NormStats(mean=np.zeros((3, 7)), std=np.ones((3, 7)), q01=q01, q99=q99)}
    normed = _transforms.Normalize(stats, use_quantiles=True)({"actions": q99.copy()})["actions"]
    np.testing.assert_allclose(normed, np.ones((3, 7)), atol=1e-4)
    padded = np.concatenate([normed, np.full((3, 25), 7.0)], axis=-1)
    back = _transforms.Unnormalize(stats, use_quantiles=True)({"actions": padded})["actions"]
    np.testing.assert_allclose(back[:, :7], q99, atol=1e-4)
    np.testing.assert_allclose(back[:, 7:], 7.0)


def test_rena_split_holds_out_about_one_in_twenty_recordings_by_source_key(tmp_path):
    labels = tmp_path / "autolabel.jsonl"
    labels.write_text(
        "\n".join(json.dumps({"episode_index": i, "source_key": f"robot/task/ep_{i}"}) for i in range(2000))
    )
    split = _config.rena_split(labels)
    assert split["train"].isdisjoint(split["holdout"])
    assert split["train"] | split["holdout"] == set(range(2000))
    assert 60 <= len(split["holdout"]) <= 140
    labels.write_text(
        "\n".join(json.dumps({"episode_index": 5000 + i, "source_key": f"robot/task/ep_{i}"}) for i in range(2000))
    )
    assert {i - 5000 for i in _config.rena_split(labels)["holdout"]} == split["holdout"]
