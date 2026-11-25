import json

import numpy as np
from PIL import Image

from openpi.training import vl_data


def test_vl_samples_fill_the_base_camera_and_learn_no_actions(tmp_path):
    (tmp_path / "images").mkdir()
    Image.fromarray(np.full((30, 40, 3), 200, np.uint8)).save(tmp_path / "images" / "a.jpg")
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({"image": "images/a.jpg", "prompt": "detect cup", "target": "<loc0001>" * 4}) + "\n"
    )
    sample = vl_data.VLInputs(action_horizon=50, action_dim=7)(vl_data.VLDataset(tmp_path)[0])
    assert sample["image"]["base_0_rgb"].shape == (30, 40, 3)
    assert not sample["image"]["left_wrist_0_rgb"].any()
    assert sample["image_mask"]["base_0_rgb"] and not sample["image_mask"]["right_wrist_0_rgb"]
    assert not sample["action_loss_mask"]
    assert sample["actions"].shape == (50, 7)
    assert (sample["prompt"], sample["annotation"]) == ("detect cup", "<loc0001>" * 4)


def test_mixed_dataset_draws_about_its_share_from_the_vl_data():
    robot, vl = [("robot", i) for i in range(5000)], [("vl", i) for i in range(10)]
    mixed = vl_data.MixedDataset(robot, [vl], share=0.1, seed=0)
    drawn = [mixed[i] for i in range(len(mixed))]
    assert len(mixed) == 5000
    assert 0.08 < sum(d[0] == "vl" for d in drawn) / len(drawn) < 0.12
    assert drawn == [mixed[i] for i in range(len(mixed))]  # fixed per index
