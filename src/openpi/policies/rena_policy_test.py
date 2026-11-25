import json

import numpy as np
import pytest

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


def test_client_request_fills_the_three_camera_slots_and_masks_out_the_grounded_one():
    out = rena_policy.RenaInputs()(rena_policy.make_rena_example())
    assert {slot: im.shape for slot, im in out["image"].items()} == {
        "base_0_rgb": (480, 640, 3),
        "left_wrist_0_rgb": (480, 640, 3),
        "right_wrist_0_rgb": (640, 480, 3),
        "grounded_rgb": (480, 640, 3),
    }
    assert not out["image"]["grounded_rgb"].any()
    assert {slot: bool(m) for slot, m in out["image_mask"].items()} == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
        "grounded_rgb": False,
    }
    assert out["state"].tolist() == [0.0] * 7
    assert out["prompt"] == "pick toy fruit"


def test_outputs_give_the_stage_as_rena_serving_does():
    out = rena_policy.RenaOutputs()({"actions": np.ones((50, 32)), "stage_logits": np.log([0.1, 0.7, 0.2])})
    assert out["actions"].shape == (50, 7)
    assert out["task_stage"] == 1
    np.testing.assert_allclose(out["stage_probs"], [0.1, 0.7, 0.2], rtol=1e-6)


def _labels():
    track = {
        "image_0": {"frames": [0, 50], "labels": [{"box": [0.25, 0.25, 0.5, 0.5]}, None]},
        "image_1": {"frames": [0, 50], "labels": [{"box": [0.0, 0.5, 0.5, 1.0]}, {"box": [0.1, 0.1, 0.2, 0.2]}]},
    }
    return {
        "episode_index": 3,
        "source_key": "renav2_pnp_1:1",
        "instruction": "pick plush toy and place in box",
        "fps": 25,
        "image_sizes": {"image_0": [640, 480], "image_1": [640, 480], "image_2": [480, 640]},
        "skills": [
            {"name": "pick", "start": 0, "object": "white plush lamb toy", "track": track, "deploy": track},
            {
                "name": "place",
                "start": 60,
                "object": "white plush lamb toy",
                "container": "cardboard box",
                "track": track,
                "deploy": track,
            },
        ],
        "argus_labels": {
            "timeline": [
                {
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "action": "reach toward",
                    "object": "white plush lamb toy (left)",
                    "destination": None,
                    "spatial_relation": None,
                },
                {
                    "start_s": 2.0,
                    "end_s": 4.0,
                    "action": "lower and release",
                    "object": "white plush lamb toy",
                    "destination": "cardboard box",
                    "spatial_relation": "into the",
                },
            ]
        },
    }


def test_annotation_gives_the_skill_its_tracked_boxes_and_the_argus_phase():
    pytest.importorskip("autolabel_prompts")
    from autolabel_prompts import loc_box

    rec = _labels()
    right, wrist = loc_box([0.25, 0.25, 0.5, 0.5], (640, 480), 224), loc_box([0.0, 0.5, 0.5, 1.0], (640, 480), 224)
    assert rena_policy.annotation(rec, 10, boxes=True) == (
        f"Right: {right} Wrist: {wrist}; pick up the white plush lamb toy; reach toward: white plush lamb toy"
    )
    assert rena_policy.annotation(rec, 30, boxes=False) == "pick up the white plush lamb toy"
    assert rena_policy.annotation(rec, 60, boxes=False) == (
        "place the white plush lamb toy in the cardboard box; lower and release: white plush lamb toy, into the "
        "cardboard box"
    )
    del rec["argus_labels"]
    assert rena_policy.annotation(rec, 10, boxes=False) == "pick up the white plush lamb toy"


@pytest.mark.parametrize("boxes_in", ["text", "image"])
def test_annotation_has_boxes_only_when_the_prompt_gives_none(tmp_path, boxes_in):
    pytest.importorskip("autolabel_prompts")
    labels = tmp_path / "autolabel.jsonl"
    labels.write_text(json.dumps(_labels()) + "\n")
    prompter = rena_policy.AutolabelPrompt(str(labels), boxes_in=boxes_in, annotate=True)
    given = set()
    for _ in range(40):
        out = prompter({"episode_index": 3, "frame_index": 10, "prompt": "task"})
        has_box = "<loc" in out["prompt"] if boxes_in == "text" else bool(out.get("target_boxes"))
        assert has_box != ("<loc" in out["annotation"])
        given.add(has_box)
    assert given == {True, False}


def test_annotation_names_the_object_as_the_instruction_does_about_half_the_time(tmp_path):
    pytest.importorskip("autolabel_prompts")
    rec = _labels() | {"argus": {"matches_given": True, "issues": []}}
    labels = tmp_path / "autolabel.jsonl"
    labels.write_text(json.dumps(rec) + "\n")
    prompter = rena_policy.AutolabelPrompt(str(labels), boxes_in="image", annotate=True)
    annotations = [
        prompter({"episode_index": 3, "frame_index": 30, "prompt": "task"})["annotation"] for _ in range(200)
    ]
    skills = [a.split("; ")[-1] for a in annotations]
    assert set(skills) == {"pick up the plush toy", "pick up the white plush lamb toy"}
    assert 70 < skills.count("pick up the plush toy") < 130
