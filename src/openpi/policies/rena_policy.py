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

from collections.abc import Callable
import dataclasses
import json
import logging
import pathlib
import re
import time
from typing import Literal

import einops
import numpy as np
from openpi_client import base_policy as _base_policy

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
        inputs = {
            "image": {slot: images[cam] for slot, cam in CAMERAS.items()},
            "image_mask": dict.fromkeys(CAMERAS, np.True_),
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
    target_boxes for RenaInputs to draw. It is a repack transform, so it never runs at serving time.

    autolabel_prompts (rena-autolabel/packages/autolabel-prompts) and rena_task_grammar must be importable."""

    labels_path: str
    state_dropout: float = 0.0
    boxes_in: Literal["text", "image"] = "text"
    # Fixes each frame's draw, for evaluations that must give every model the same prompts.
    seed: int | None = None

    def __post_init__(self):
        lines = pathlib.Path(self.labels_path).read_text().splitlines()
        object.__setattr__(self, "_records", {(rec := json.loads(line))["episode_index"]: rec for line in lines})

    def __call__(self, data: dict) -> dict:
        from autolabel_prompts import prompt
        from autolabel_prompts import sample

        data = dict(data)
        episode, frame = int(data.pop("episode_index")), int(data.pop("frame_index"))
        # fresh entropy by default: DataLoader workers share numpy's global seed
        rng = np.random.default_rng(None if self.seed is None else (self.seed, episode, frame))
        if (rec := self._records.get(episode)) is not None:
            if self.boxes_in == "text":
                data["prompt"] = prompt(rec, frame, rng)
            else:
                data["prompt"], data["target_boxes"] = sample(rec, frame, rng)
        data["drop_state"] = bool(rng.random() < self.state_dropout)
        return data


# The base cameras: fixed to the body, so a box from a trial's first frame holds until the object moves. The wrist
# camera moves with the arm, so its box holds only for the frame it was found on.
STATIC_BOX_CAMERAS = ("image_0", "image_2")
WRIST_CAMERA = "image_1"
_PICK = re.compile(r"^pick (?P<obj>.+)$")
_PLACE = re.compile(r"^place (?P<obj>.+) (?P<prep>in|on) (?P<dest>.+)$")
_HAS_BOXES = re.compile(r" at (?:Right|Left|Wrist): <loc\d{4}>")


def box_question(prompt: str) -> tuple[str, str] | None:
    """(command, what) to ask a pointer for a skill prompt's target, as rena-autolabel's deploy boxes were asked: the
    object for "pick X", the container for "place X in Y"; None for a whole task or a prompt that has its boxes."""
    if " and place " in prompt or _HAS_BOXES.search(prompt):
        return None
    if m := _PLACE.match(prompt):
        return f"place the {m['obj']} {m['prep']} the {m['dest']}", f"the {m['dest']}"
    if m := _PICK.match(prompt):
        return f"pick up the {m['obj']}", f"the {m['obj']}"
    return None


class StaticBoxes(_base_policy.BasePolicy):
    """A rena policy whose pick and place skill prompts get the target's box on the base cameras, as training's box
    prompts gave it: <loc> tokens after the prompt (boxes_in "text") or outlines RenaInputs draws (boxes_in "image").
    The boxes come from `locate` (images by camera, command, what -> box or None by camera) once per trial, on its
    first request, and stay for the rest of it. With wrist_first, that first request also gets the wrist camera's box:
    it is found on the very frame the policy sees, and training's boxes were up to 2 s old, while the next request
    comes a chunk (2 s) later with the arm moved. A trial starts when the prompt changes or after a pause in requests
    longer than `new_trial_after_s`; other prompts pass through."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        locate: Callable[[dict[str, np.ndarray], str, str], dict[str, list[float] | None]],
        boxes_in: Literal["text", "image"],
        new_trial_after_s: float = 5.0,
        *,
        wrist_first: bool = False,
    ):
        self._policy, self._locate, self._boxes_in = policy, locate, boxes_in
        self._new_trial_after_s = new_trial_after_s
        self._cameras = (*STATIC_BOX_CAMERAS, WRIST_CAMERA) if wrist_first else STATIC_BOX_CAMERAS
        self._prompt: str | None = None
        self._boxes: dict[str, list[float]] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._last = -np.inf

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        prompt = obs.get("prompt", "")
        if (question := box_question(prompt)) is None:
            return self._policy.infer(obs)
        first = prompt != self._prompt or time.monotonic() - self._last > self._new_trial_after_s
        if first:
            images = {cam: np.asarray(obs[f"observation.images.{cam}"]) for cam in CAMERAS.values()}
            found = self._locate(images, *question)
            self._boxes = {cam: found[cam] for cam in self._cameras if found.get(cam)}
            self._sizes = {cam: (im.shape[1], im.shape[0]) for cam, im in images.items()}
            self._prompt = prompt
            logging.info("New trial %r: boxes %s", prompt, self._boxes)
        boxes = {cam: box for cam, box in self._boxes.items() if first or cam in STATIC_BOX_CAMERAS}
        obs = dict(obs)
        if self._boxes_in == "text":
            from autolabel_prompts import with_boxes

            obs["prompt"] = with_boxes(prompt, boxes, self._sizes)
        else:
            obs["target_boxes"] = boxes
        try:
            return self._policy.infer(obs)
        finally:
            self._last = time.monotonic()
