import numpy as np

from openpi.policies import rena_policy


def test_target_box_is_outlined_in_magenta_on_its_camera_only():
    grey = np.full((480, 640, 3), 128, np.uint8)
    out = rena_policy.RenaInputs()(
        {
            **{f"observation.images.{cam}": grey for cam in ("image_0", "image_1", "image_2")},
            "observation.state": np.zeros(7),
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


def test_client_request_fills_the_three_slots_and_state():
    out = rena_policy.RenaInputs()(rena_policy.make_rena_example())
    assert {slot: im.shape for slot, im in out["image"].items()} == {
        "base_0_rgb": (480, 640, 3),
        "left_wrist_0_rgb": (480, 640, 3),
        "right_wrist_0_rgb": (640, 480, 3),
    }
    assert out["state"].tolist() == [0.0] * 7
    assert out["prompt"] == "pick toy fruit"


def test_outputs_give_the_stage_as_rena_serving_does():
    out = rena_policy.RenaOutputs()({"actions": np.ones((50, 32)), "stage_logits": np.log([0.1, 0.7, 0.2])})
    assert out["actions"].shape == (50, 7)
    assert out["task_stage"] == 1
    np.testing.assert_allclose(out["stage_probs"], [0.1, 0.7, 0.2], rtol=1e-6)
