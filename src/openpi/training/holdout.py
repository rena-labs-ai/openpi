"""A fixed set of frames from a rena config's holdout (config.rena_split) and how a model does on them: the absolute joint
targets it samples against the commanded ones, through the policy's own input and output transforms. Every model gets
the same frames and the same prompt for each frame (AutolabelPrompt seeded), so their scores compare.
scripts/eval_rena_holdout.py scores checkpoints with it, and training scores itself every holdout_interval steps."""

import collections
from collections.abc import Callable
import dataclasses
import json
import pathlib
from typing import Literal

import jax
import numpy as np

import openpi.policies.rena_policy as rena_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


@dataclasses.dataclass
class Holdout:
    batches: list[dict]  # model inputs as training's data loader makes them, before Observation.from_dict
    want: np.ndarray  # [n, horizon, 7] the commanded joint targets and gripper
    kinds: list[str]  # the prompt each frame got: task, skill or boxes
    sources: list[str]  # the recording's dataset
    outputs: Callable[[dict], dict]  # one sample's model output -> absolute actions, as the policy's output transforms

    def absolute(self, predicted: list[np.ndarray]) -> np.ndarray:
        """Each batch's normalized model actions [b, horizon, dim] -> absolute joint targets and gripper [n, horizon, 7]."""
        states = [s for b in self.batches for s in b["state"]]
        actions = [a for p in predicted for a in p]
        return np.stack(
            [self.outputs({"state": s, "actions": a})["actions"] for s, a in zip(states, actions, strict=True)]
        )

    def report(self, got: np.ndarray) -> dict:
        kinds, sources = np.array(self.kinds), np.array(self.sources)
        return {
            "all": summary(got, self.want),
            "by_prompt": {k: summary(got[kinds == k], self.want[kinds == k]) for k in sorted(set(self.kinds))},
            "by_source": {
                k: summary(got[sources == k], self.want[sources == k])
                for k, _ in collections.Counter(self.sources).most_common()
            },
        }

    def showcase(self, k: int = 6) -> list[int]:
        """k frames to plot, the same every time: half whose commanded gripper opens or closes within the chunk, the
        rest those whose joints travel furthest."""
        switching = np.flatnonzero(gripper_switch(self.want[..., 6]) >= 0)
        events = switching[np.linspace(0, len(switching) - 1, min(k // 2, len(switching))).round().astype(int)]
        travel = np.abs(np.diff(self.want[..., :6], axis=1)).sum((1, 2))
        rest = [int(i) for i in np.argsort(-travel) if i not in set(events)][: k - len(events)]
        return [int(i) for i in events] + rest


GRIPPER_TOLERANCE = 5  # steps either way a predicted gripper switch may be off by and still count


def gripper_switch(gripper: np.ndarray) -> np.ndarray:
    """[n, horizon] gripper in 0-1 -> [n] the first step it is open where it was closed at step 0 or the reverse; -1
    where it never switches."""
    closed = gripper > 0.5
    switched = closed != closed[:, :1]
    return np.where(switched.any(1), switched.argmax(1), -1)


def summary(got: np.ndarray, want: np.ndarray) -> dict:
    """got, want [n, horizon, 7]: joints in rad, the gripper in its 0-1 range. A gripper event is a commanded switch
    within the chunk; the model catches it with a switch of its own within GRIPPER_TOLERANCE steps."""
    err = np.abs(got - want)
    t_want, t_got = gripper_switch(want[..., 6]), gripper_switch(got[..., 6])
    events = t_want >= 0
    caught = events & (t_got >= 0) & (np.abs(t_got - t_want) <= GRIPPER_TOLERANCE)
    return {
        "n": len(err),
        "joint_mae": float(err[..., :6].mean()),
        "joint_mae_per_joint": err[..., :6].mean((0, 1)).round(4).tolist(),
        "gripper_mae": float(err[..., 6].mean()),
        "joint_mae_by_step": err[:, :, :6].mean((0, 2)).round(4).tolist(),
        "gripper_events": int(events.sum()),
        "gripper_event_recall": float(caught.sum() / events.sum()) if events.any() else float("nan"),
        "gripper_false_switch": float((t_got[~events] >= 0).mean()) if (~events).any() else float("nan"),
    }


def pyplot():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_image(fig) -> np.ndarray:
    """The figure as an RGB image, closed."""
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    pyplot().close(fig)
    return image


def plot_chunk(got: np.ndarray, want: np.ndarray, title: str) -> np.ndarray:
    """One frame's predicted (red) and commanded (black) chunk [horizon, 7], a panel per joint and the gripper, as an
    RGB image."""
    plt = pyplot()
    fig, axes = plt.subplots(1, 7, figsize=(14, 2), dpi=80)
    for j, ax in enumerate(axes):
        ax.plot(want[:, j], "k", lw=1.5)
        ax.plot(got[:, j], "r", lw=1.2)
        ax.set_title("gripper" if j == 6 else f"joint {j}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=8)
    return figure_image(fig)


def plot_by_step(by_step: dict[str, list[float]]) -> np.ndarray:
    """Joint MAE per chunk step, a line per split, as an RGB image."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    for name, values in by_step.items():
        ax.plot(values, label=name)
    ax.set_xlabel("chunk step")
    ax.set_ylabel("joint MAE (rad)")
    ax.legend(fontsize=8)
    return figure_image(fig)


def prompt_kind(prompter: rena_policy.AutolabelPrompt, rec: dict, frame: int) -> str:
    """The kind of prompt the seeded prompter gives the frame."""
    from autolabel_prompts import sample
    from autolabel_prompts import task_text

    rng = np.random.default_rng((prompter.seed, rec["episode_index"], frame))
    text, boxes = sample(rec, frame, rng, prompter.mix)
    if boxes:
        return "boxes"
    return "task" if text in (rec["instruction"], task_text(rec)) else "skill"


def load(
    config: _config.TrainConfig,
    norm_stats: dict,
    *,
    batch_size: int,
    max_frames: int | None = None,
    seed: int = 0,
    split: Literal["train", "holdout", "site"] = "holdout",
) -> Holdout | None:
    """Evenly spread frames of the config's holdout (or of its train part), at most max_frames, in whole batches."""
    factory = dataclasses.replace(config.data, split=split)
    data_config = factory.create(config.assets_dirs, config.model)
    repack = [
        # the state in every prompt, as a robot sends it
        dataclasses.replace(t, seed=seed, state_dropout=0.0) if isinstance(t, rena_policy.AutolabelPrompt) else t
        for t in data_config.repack_transforms.inputs
    ]
    prompter = next(t for t in repack if isinstance(t, rena_policy.AutolabelPrompt))
    lines = pathlib.Path(prompter.labels_path).read_text().splitlines()
    records = {(rec := json.loads(line))["episode_index"]: rec for line in lines}
    inputs = _transforms.compose(
        [
            *repack,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    outputs = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    n = len(dataset) if max_frames is None else min(len(dataset), max_frames)
    n -= n % batch_size
    if n == 0:  # fewer frames than one batch
        return None
    picks = np.linspace(0, len(dataset) - 1, n).round().astype(int)
    samples, want, kinds, sources = [], [], [], []
    for i in picks:
        raw = dataset[int(i)]
        # before the inputs run: DeltaActions subtracts in place, on memory np.asarray shares with the tensor
        want.append(np.array(raw["action"], dtype=np.float32)[data_config.action_lookback :])
        rec = records[int(raw["episode_index"])]
        kinds.append(prompt_kind(prompter, rec, int(raw["frame_index"])))
        sources.append(rec["dataset"])
        samples.append(inputs(dict(raw)))
    batches = [jax.tree.map(lambda *x: np.stack(x), *samples[s : s + batch_size]) for s in range(0, n, batch_size)]
    return Holdout(batches, np.stack(want), kinds, sources, outputs)


@dataclasses.dataclass
class Detect:
    """Held-out "detect <name>" frames (a vision-language dataset such as vl/rena_inventory_toronto) as model inputs,
    with the box each one's target gives, in <loc> bins."""

    batches: list[dict]
    want: np.ndarray  # [n, 4] y0 x0 y1 x1, bins 0-1023 over the padded square input
    prompts: list[str]


def load_detect(name: str, data_config: _config.DataConfig, action_horizon: int, *, batch_size: int) -> Detect | None:
    """All of the dataset's frames that fill whole batches."""
    from lerobot.common.constants import HF_LEROBOT_HOME

    from openpi.models.pi0 import LOC0
    from openpi.training import vl_data

    raw = vl_data.VLDataset(HF_LEROBOT_HOME / name)
    dataset = _data_loader.vl_dataset(name, data_config, action_horizon)
    n = len(dataset) - len(dataset) % batch_size
    if n == 0:
        return None
    samples = [dataset[i] for i in range(n)]
    want = np.stack([s["target_tokens"][:4] - LOC0 for s in samples])
    batches = [jax.tree.map(lambda *x: np.stack(x), *samples[s : s + batch_size]) for s in range(0, n, batch_size)]
    return Detect(batches, want, [s["prompt"] for s in raw.samples[:n]])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[n] intersection over union of boxes [n, 4] given as y0 x0 y1 x1; 0 for a box with no area."""
    y0, x0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    y1, x1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(y1 - y0, 0, None) * np.clip(x1 - x0, 0, None)

    def area(c):
        return np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)

    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def draw_boxes(image: np.ndarray, want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """The model's input image (uint8, or float in [-1, 1]) with the target box in green and the predicted in red."""
    from PIL import Image
    from PIL import ImageDraw

    if image.dtype != np.uint8:
        image = ((np.asarray(image, np.float32) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(image)
    h, w = image.shape[:2]
    draw = ImageDraw.Draw(pil)
    for box, colour in ((want, (0, 255, 0)), (got, (255, 0, 0))):
        y0, x0, y1, x1 = (np.asarray(box, np.float32) / 1024) * np.array([h, w, h, w])
        draw.rectangle([x0, y0, max(x0, x1), max(y0, y1)], outline=colour, width=2)
    return np.asarray(pil)
