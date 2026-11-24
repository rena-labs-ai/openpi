"""Training prompts from a recording's labels (labels.json, see join_argus.py): one sampled per frame each time the frame
is drawn, as steerable-policies-bridge samples its commands, so a frame trains under every level of the same skill.

    task         "pick money roll plush toy and place in black bin", or the recording's instruction
    skill        "pick money roll plush toy"
    skill+boxes  "pick money roll plush toy at Right: <loc..><loc..><loc..><loc..> Wrist: <loc..>..."

The text is rena-training's canonical task grammar (rena_task_grammar.compose), with argus's names for the object and
container, so serving reads these prompts as it reads every other task string.

A box is PaliGemma's own detection format, the backbone pi05 shares: y_min x_min y_max x_max, each a <locNNNN> token
on a 1024-step grid over the image the model sees, which pi05 makes by scaling the recorded image to fit a square and
padding the rest. A camera where the target is not seen is left out, and with none left the prompt is the skill alone.
Some box prompts name the target by its box only ("pick object at ..."), so the policy learns to trust the box
over the name. Each skill prompt also names its object and container either as argus does ("plush unicorn") or by
the class the recording's instruction gives ("plush toy"), as a high-level policy might say either; the class only
where argus found the footage matches the instruction and raised no instruction mismatch, so the class names the
object actually moved. The boxes come from a moment up to requery_s before the frame, as a high-level policy that re-points
every few seconds would have sent them, and some carry the pointing noise of Gemini asked without thinking (deploy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rena_task_grammar import compose, parse, to_class

CAMS = (("Right", "image_0"), ("Left", "image_2"), ("Wrist", "image_1"))
TARGETS = re.compile(r" at((?: (?:Right|Left|Wrist): (?:<loc\d{4}>){4})+)$")


@dataclass(frozen=True)
class Mix:
    task: float = 0.1
    skill: float = 0.3
    boxes: float = 0.6
    nameless: float = 1 / 3  # of box prompts
    class_names: float = 0.5  # of skill and box prompts
    deploy: float = 0.5  # of box prompts
    requery_s: float = 2.0
    input_px: int = 224


def loc_box(box: list[float], size: tuple[int, int], input_px: int) -> str:
    """A box as fractions of the recorded image (x0, y0, x1, y1) in PaliGemma's tokens over the padded square input."""
    w, h = size
    scale = input_px / max(w, h)
    pad_x, pad_y = (input_px - w * scale) / 2, (input_px - h * scale) / 2
    x0, x1 = ((pad_x + v * w * scale) / input_px for v in (box[0], box[2]))
    y0, y1 = ((pad_y + v * h * scale) / input_px for v in (box[1], box[3]))
    return "".join(f"<loc{min(1023, max(0, int(v * 1024))):04d}>" for v in (y0, x0, y1, x1))


def skill_at(rec: dict, frame: int) -> dict | None:
    return next((sk for sk in reversed(rec["skills"]) if sk["start"] <= frame), None)


def task_text(rec: dict) -> str:
    by = {sk["name"]: sk for sk in rec["skills"]}
    if "pick" in by and "place" in by:
        return compose("pick_and_place", by["pick"]["object"], by["place"]["container"])
    return skill_text(by.get("pick") or by["place"])


def class_names(rec: dict) -> dict[str, str]:
    """The object and container by class, from the recording's instruction ("pick Plush Toys and place in Basket" ->
    plush toy, basket); none where argus doubts the instruction or it does not parse."""
    argus = rec.get("argus") or {}
    if argus.get("matches_given") is not True or any(i.get("category") == "instruction_mismatch"
                                                     for i in argus.get("issues") or []):
        return {}
    parts = parse(rec["instruction"])
    if parts is None:
        return {}
    return {k: to_class(v) for k, v in (("object", parts.obj), ("container", parts.location)) if v}


def skill_text(sk: dict, nameless: bool = False, names: dict[str, str] | None = None) -> str:
    """The skill in the task grammar, with argus's names unless `names` gives others."""
    obj, container = ((names or {}).get(k, sk.get(k)) for k in ("object", "container"))
    if sk["name"] == "pick":
        return compose("pick", "object" if nameless else obj)
    return compose("place", obj, "container" if nameless else container)


def boxes_at(sk: dict, frame: int, copy: str) -> dict[str, dict | None]:
    """Each camera's label at the 10 fps track frame nearest `frame`, from the clean track or the deploy copy."""
    out = {}
    for c, tr in sk[copy].items():
        i = int(np.argmin(np.abs(np.asarray(tr["frames"]) - frame)))
        out[c] = tr["labels"][i]
    return out


def sample(rec: dict, frame: int, rng: np.random.Generator, mix: Mix = Mix()) -> tuple[str, dict[str, list[float]]]:
    """The frame's prompt text and, for a box prompt, the target's box in each camera that sees it (x0, y0, x1, y1 as
    fractions of the recorded image), for prompt() to write as <loc> tokens or a policy to draw on its images."""
    sk = skill_at(rec, frame)
    if sk is None:
        return rec["instruction"], {}
    kind = rng.choice(["task", "skill", "boxes"], p=[mix.task, mix.skill, mix.boxes])
    if kind == "task":
        return (task_text(rec) if rng.random() < 0.5 else rec["instruction"]), {}
    names = class_names(rec) if rng.random() < mix.class_names else None
    if kind == "skill":
        return skill_text(sk, names=names), {}
    query = int(rng.integers(max(sk["start"], frame - round(mix.requery_s * rec["fps"])), frame + 1))
    labels = boxes_at(sk, query, "deploy" if rng.random() < mix.deploy else "track")
    boxes = {c: b["box"] for _, c in CAMS if (b := labels.get(c))}
    if not boxes:
        return skill_text(sk, names=names), {}
    return skill_text(sk, nameless=rng.random() < mix.nameless, names=names), boxes


def prompt(rec: dict, frame: int, rng: np.random.Generator, mix: Mix = Mix()) -> str:
    text, boxes = sample(rec, frame, rng, mix)
    return with_boxes(text, boxes, rec["image_sizes"], mix.input_px)


def with_boxes(text: str, boxes: dict[str, list[float]], sizes: dict, input_px: int = Mix.input_px) -> str:
    """The text with each camera's box (x0, y0, x1, y1 as fractions of that camera's (width, height) image in `sizes`)
    as <loc> tokens, "... at Right: <loc..>x4 Left: ...", in CAMS order; the text alone when there are none."""
    views = " ".join(f"{name}: {loc_box(boxes[c], sizes[c], input_px)}" for name, c in CAMS if c in boxes)
    return f"{text} at {views}" if views else text


def split_targets(text: str) -> tuple[str, str]:
    """A prompt's task and its " at Right: <loc..>..." part, which is "" when it has none."""
    m = TARGETS.search(text)
    return (text[: m.start()], m.group(0)) if m else (text, "")
