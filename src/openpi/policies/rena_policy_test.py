import json
from typing import ClassVar

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


def test_client_request_fills_the_three_slots_and_state():
    out = rena_policy.RenaInputs()(rena_policy.make_rena_example())
    assert {slot: im.shape for slot, im in out["image"].items()} == {
        "base_0_rgb": (480, 640, 3),
        "left_wrist_0_rgb": (480, 640, 3),
        "right_wrist_0_rgb": (640, 480, 3),
    }
    assert out["state"].tolist() == [0.0] * 7
    assert out["prompt"] == "pick toy fruit"


class _Seen:
    """A policy that keeps what it was asked, and a locate that counts its calls."""

    metadata: ClassVar[dict] = {}

    def __init__(self):
        self.asked, self.located = [], []

    def infer(self, obs):
        self.asked.append(obs)
        return {}

    def locate(self, images, command, what):
        self.located.append((command, what))
        return {"image_0": [0.25, 0.5, 0.5, 0.75], "image_2": [0.5, 0.25, 0.75, 0.5], "image_1": [0, 0, 1, 1]}


def _ask(policy, prompt):
    policy.infer({**rena_policy.make_rena_example(), "prompt": prompt})


def test_static_boxes_ask_once_per_trial_and_write_base_camera_locs():
    pytest.importorskip("autolabel_prompts")
    seen = _Seen()
    boxed = rena_policy.StaticBoxes(seen, seen.locate, "text")
    _ask(boxed, "pick plush toy")
    _ask(boxed, "pick plush toy")
    # Right 640x480 pads 28 px above and below in 224; Left 480x640 pads 28 px each side. No wrist box.
    assert seen.asked[1]["prompt"] == (
        "pick plush toy at Right: <loc0512><loc0256><loc0704><loc0512> Left: <loc0256><loc0512><loc0512><loc0704>"
    )
    _ask(boxed, "place plush toy in bin")
    assert seen.located == [("pick up the plush toy", "the plush toy"), ("place the plush toy in the bin", "the bin")]


def test_static_boxes_for_an_image_model_are_drawn_and_tasks_pass_through():
    seen = _Seen()
    boxed = rena_policy.StaticBoxes(seen, seen.locate, "image")
    _ask(boxed, "pick plush toy")
    assert seen.asked[0]["prompt"] == "pick plush toy"
    assert seen.asked[0]["target_boxes"] == {"image_0": [0.25, 0.5, 0.5, 0.75], "image_2": [0.5, 0.25, 0.75, 0.5]}
    _ask(boxed, "pick plush toy and place in bin")
    assert "target_boxes" not in seen.asked[1]
    assert len(seen.located) == 1


def test_outputs_give_the_stage_as_rena_serving_does():
    out = rena_policy.RenaOutputs()({"actions": np.ones((50, 32)), "stage_logits": np.log([0.1, 0.7, 0.2])})
    assert out["actions"].shape == (50, 7)
    assert out["task_stage"] == 1
    np.testing.assert_allclose(out["stage_probs"], [0.1, 0.7, 0.2], rtol=1e-6)


def test_wrist_first_gives_the_wrist_box_on_the_first_request_only():
    pytest.importorskip("autolabel_prompts")
    seen = _Seen()
    boxed = rena_policy.StaticBoxes(seen, seen.locate, "text", wrist_first=True)
    _ask(boxed, "pick plush toy")
    _ask(boxed, "pick plush toy")
    base = "pick plush toy at Right: <loc0512><loc0256><loc0704><loc0512> Left: <loc0256><loc0512><loc0512><loc0704>"
    assert seen.asked[0]["prompt"] == base + " Wrist: <loc0128><loc0000><loc0896><loc1023>"
    assert seen.asked[1]["prompt"] == base


def test_wrist_first_draws_the_wrist_box_on_the_first_request_only():
    seen = _Seen()
    boxed = rena_policy.StaticBoxes(seen, seen.locate, "image", wrist_first=True)
    _ask(boxed, "pick plush toy")
    _ask(boxed, "pick plush toy")
    assert seen.asked[0]["target_boxes"]["image_1"] == [0, 0, 1, 1]
    assert "image_1" not in seen.asked[1]["target_boxes"]


def test_a_recorder_under_static_boxes_keeps_each_request_as_the_model_got_it(tmp_path):
    pytest.importorskip("autolabel_prompts")
    from openpi.policies import policy as _policy

    seen = _Seen()
    for _ in range(2):  # a restarted server records into the same folder
        recorder = _policy.PolicyRecorder(seen, str(tmp_path))
        _ask(rena_policy.StaticBoxes(recorder, seen.locate, "text"), "pick plush toy")
        recorder._writer.shutdown(wait=True)  # noqa: SLF001
    records = [np.load(f) for f in sorted(tmp_path.glob("*.npz"))]
    assert len(records) == 2
    assert all(str(r["inputs/prompt"]).startswith("pick plush toy at Right: <loc0512>") for r in records)
    assert records[0]["inputs/observation.images.image_2"].shape == (640, 480, 3)
    assert float(records[1]["time"]) >= float(records[0]["time"])


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
        f"pick white plush lamb toy at Right: {right} Wrist: {wrist}; reach toward: white plush lamb toy"
    )
    assert rena_policy.annotation(rec, 30, boxes=False) == "pick white plush lamb toy"
    assert rena_policy.annotation(rec, 60, boxes=False) == (
        "place white plush lamb toy in cardboard box; lower and release: white plush lamb toy, into the cardboard box"
    )
    del rec["argus_labels"]
    assert rena_policy.annotation(rec, 10, boxes=False) == "pick white plush lamb toy"


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
