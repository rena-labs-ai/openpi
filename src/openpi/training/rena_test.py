import json

from openpi import transforms as _transforms
from openpi.training import config as _config

MASK = _transforms.make_bool_mask(6, -1)


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
