"""Rena (renav2: one 6-joint arm with a parallel gripper on a mobile base, three cameras) for pi0.5.

Inputs, keyed as the dataset and rena_vla_client's RenaV2InferenceClient both key them:
- observation.images.image_0 (right base camera, 480x640), .image_1 (wrist, 480x640), .image_2 (left base camera,
  640x480), RGB
- observation.state: [7], the six arm joints (rad) then the gripper
- prompt
- actions: [action_horizon, 7], absolute joint targets then the gripper (training only)
- task_stage: the frame's stage index in rena_task_grammar.STAGES, for the stage head (training only)
- target_boxes: optional {camera: [x0, y0, x1, y1]} as fractions of that camera's image, the target outlined in
  magenta on each camera that sees it (AutolabelPrompt with boxes_in="image" in training, the client at serving)

The cameras fill pi0.5's three slots as rena-training's renav2 does: image_0 the base slot, the wrist the left wrist
slot, image_2 the right wrist slot.

Outputs: actions [action_horizon, 7], and with the stage head's logits, stage_probs over STAGES and task_stage, their
argmax, as rena-training's serving returns them.
"""

import dataclasses
import functools
import json
import pathlib
from typing import Literal

import einops
import numpy as np

from openpi import transforms

CAMERAS = {"base_0_rgb": "image_0", "left_wrist_0_rgb": "image_1", "right_wrist_0_rgb": "image_2"}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def draw_box(image: np.ndarray, box) -> np.ndarray:
    """The box outlined on a copy of the image, about 2 px wide once pi0.5 scales the image to 224; magenta, as the
    rena arm is red."""
    h, w = image.shape[:2]
    t = max(2, round(max(h, w) / 112))
    x0, x1 = (int(np.clip(round(v * w), 0, w)) for v in (box[0], box[2]))
    y0, y1 = (int(np.clip(round(v * h), 0, h)) for v in (box[1], box[3]))
    out = image.copy()
    for rows, cols in (
        (slice(y0, y0 + t), slice(x0, x1)),
        (slice(max(y0, y1 - t), y1), slice(x0, x1)),
        (slice(y0, y1), slice(x0, x0 + t)),
        (slice(y0, y1), slice(max(x0, x1 - t), x1)),
    ):
        out[rows, cols] = (255, 0, 255)
    return out


@dataclasses.dataclass(frozen=True)
class RenaInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        boxes = data.get("target_boxes") or {}
        images = {cam: _parse_image(data[f"observation.images.{cam}"]) for cam in CAMERAS.values()}
        images = {cam: draw_box(im, boxes[cam]) if cam in boxes else im for cam, im in images.items()}
        # The model's grounded_rgb input carries no rena camera: blank and masked out, so it changes no output.
        inputs = {
            "image": {slot: images[cam] for slot, cam in CAMERAS.items()}
            | {"grounded_rgb": np.zeros_like(images[CAMERAS["base_0_rgb"]])},
            "image_mask": dict.fromkeys(CAMERAS, np.True_) | {"grounded_rgb": np.False_},
            "state": np.asarray(data["observation.state"], dtype=np.float32),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "task_stage" in data:
            inputs["task_stage"] = np.asarray(data["task_stage"], dtype=np.int32)
        for key in ("prompt", "drop_state"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class RenaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        out = {"actions": np.asarray(data["actions"][:, :7])}
        if "stage_logits" in data:
            logits = np.asarray(data["stage_logits"], dtype=np.float64)
            probs = np.exp(logits - logits.max())
            out["stage_probs"] = (probs / probs.sum()).astype(np.float32)
            out["task_stage"] = int(probs.argmax())
        return out


def make_rena_example() -> dict:
    """A request as rena_vla_client sends it, once the server has decoded its JPEG images."""
    return {
        "observation.images.image_0": np.zeros((480, 640, 3), np.uint8),
        "observation.images.image_1": np.zeros((480, 640, 3), np.uint8),
        "observation.images.image_2": np.zeros((640, 480, 3), np.uint8),
        "observation.state": np.zeros(7, np.float32),
        "prompt": "pick toy fruit",
    }


@dataclasses.dataclass(frozen=True)
class AutolabelPrompt(transforms.DataTransformFn):
    """Each training frame's prompt, sampled afresh every time the frame is drawn from rena-autolabel's labels
    (meta/autolabel.jsonl beside the dataset, autolabel_prompts.sample: the task, the skill, or the skill with the
    target's box per camera), and with probability state_dropout the state left out of the prompt (TokenizePrompt's
    drop_state). boxes_in "text" writes the boxes into the prompt as PaliGemma <loc> tokens, "image" passes them as
    target_boxes for RenaInputs to draw, "none" never draws a box prompt: the skill prompt takes its share. It is a
    repack transform, so it never runs at serving time.

    autolabel_prompts (rena-autolabel/packages/autolabel-prompts) and rena_task_grammar must be importable."""

    labels_path: str
    state_dropout: float = 0.0
    boxes_in: Literal["text", "image", "none"] = "text"
    # Fixes each frame's draw, for evaluations that must give every model the same prompts.
    seed: int | None = None

    @functools.cached_property
    def _records(self) -> dict[int, dict]:
        lines = pathlib.Path(self.labels_path).read_text().splitlines()
        return {(rec := json.loads(line))["episode_index"]: rec for line in lines}

    @property
    def mix(self):
        from autolabel_prompts import Mix

        return Mix(skill=Mix.skill + Mix.boxes, boxes=0.0) if self.boxes_in == "none" else Mix()

    def __call__(self, data: dict) -> dict:
        from autolabel_prompts import prompt
        from autolabel_prompts import sample

        data = dict(data)
        episode, frame = int(data.pop("episode_index")), int(data.pop("frame_index"))
        # fresh entropy by default: DataLoader workers share numpy's global seed
        rng = np.random.default_rng(None if self.seed is None else (self.seed, episode, frame))
        if (rec := self._records.get(episode)) is not None:
            if self.boxes_in == "text":
                data["prompt"] = prompt(rec, frame, rng, self.mix)
            else:
                data["prompt"], data["target_boxes"] = sample(rec, frame, rng, self.mix)
        data["drop_state"] = bool(rng.random() < self.state_dropout)
        return data


@dataclasses.dataclass(frozen=True)
class SplitPrevAction(transforms.DataTransformFn):
    """Training's action sequence carries one step before the frame (DataConfig.action_lookback 1): that step's
    command becomes observation.prev_action, as a client sends it, and the rest the actions. A repack transform."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {**data, "observation.prev_action": actions[..., 0, :], "actions": actions[..., 1:, :]}


@dataclasses.dataclass(frozen=True)
class AnchorOnCommand(transforms.DataTransformFn):
    """The joint state replaced by the joint targets commanded the step before (observation.prev_action), so actions
    are relative to the last command, not to where the arm sagged or lagged to; the gripper stays measured. The
    policy's state is that command from here on, also in the outputs, where AbsoluteActions adds it back."""

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        prev = np.asarray(data.pop("observation.prev_action"), dtype=np.float32)
        state = np.array(data["observation.state"], dtype=np.float32)
        state[..., :6] = prev[..., :6]
        data["observation.state"] = state
        return data
