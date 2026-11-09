import numpy as np

from openpi.policies import rena_policy


def test_target_box_is_outlined_in_magenta_on_its_camera_only():
    grey = np.full((480, 640, 3), 128, np.uint8)
    out = rena_policy.RenaInputs()(
        {
            "images": {"image_0": grey, "image_1": grey, "image_2": grey},
            "state": np.zeros(7),
            "target_boxes": {"image_0": [0.25, 0.25, 0.75, 0.75]},
        }
    )
    base = out["image"]["base_0_rgb"]
    assert base[120, 320].tolist() == [255, 0, 255]
    assert base[240, 160].tolist() == [255, 0, 255]
    assert base[240, 320].tolist() == [128, 128, 128]
    assert base[100, 320].tolist() == [128, 128, 128]
    assert (out["image"]["left_wrist_0_rgb"] == 128).all()
    assert "target_boxes" not in out
