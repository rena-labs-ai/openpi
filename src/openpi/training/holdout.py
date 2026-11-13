"""A fixed set of frames from a rena config's holdout (config.rena_split) and how a model does on them: the absolute joint
targets it samples against the commanded ones, through the policy's own input and output transforms. Every model gets
the same frames and the same prompt for each frame (AutolabelPrompt seeded), so their scores compare.
scripts/eval_rena_holdout.py scores checkpoints with it, and training scores itself every holdout_interval steps."""

import collections
from collections.abc import Callable
import dataclasses
import json
import pathlib

import jax
import numpy as np

import openpi.policies.rena_policy as rena_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

EXECUTED = 10  # the steps of a chunk a robot runs before it asks again


@dataclasses.dataclass
class Holdout:
    batches: list[dict]  # model inputs as training's data loader makes them, before Observation.from_dict
    want: np.ndarray  # [n, horizon, 7] the commanded joint targets and gripper
    kinds: list[str]  # the prompt each frame got: task, skill or boxes
    sources: list[str]  # the recording's dataset
    outputs: Callable[[dict], dict]  # one sample's model output -> absolute actions, as the policy's output transforms

    def errors(self, predicted: list[np.ndarray]) -> np.ndarray:
        """Each batch's normalized model actions [b, horizon, dim] -> absolute errors [n, horizon, 7]."""
        states = [s for b in self.batches for s in b["state"]]
        actions = [a for p in predicted for a in p]
        got = np.stack(
            [self.outputs({"state": s, "actions": a})["actions"] for s, a in zip(states, actions, strict=True)]
        )
        return np.abs(got - self.want)

    def report(self, err: np.ndarray) -> dict:
        kinds, sources = np.array(self.kinds), np.array(self.sources)
        return {
            "all": summary(err),
            "by_prompt": {k: summary(err[kinds == k]) for k in sorted(set(self.kinds))},
            "by_source": {k: summary(err[sources == k]) for k, _ in collections.Counter(self.sources).most_common()},
        }


def summary(err: np.ndarray) -> dict:
    """err [n, horizon, 7]: joints in rad, the gripper in its 0-1 range."""
    return {
        "n": len(err),
        "joint_mae_executed": float(err[:, :EXECUTED, :6].mean()),
        "joint_mae_chunk": float(err[:, :, :6].mean()),
        "joint_mae_executed_per_joint": err[:, :EXECUTED, :6].mean((0, 1)).round(4).tolist(),
        "gripper_mae_executed": float(err[:, :EXECUTED, 6].mean()),
        "joint_mae_by_step": err[:, :, :6].mean((0, 2)).round(4).tolist(),
    }


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
    config: _config.TrainConfig, norm_stats: dict, *, batch_size: int, max_frames: int | None = None, seed: int = 0
) -> Holdout:
    """Evenly spread frames of the config's holdout, at most max_frames, in whole batches."""
    factory = dataclasses.replace(config.data, split="holdout")
    data_config = factory.create(config.assets_dirs, config.model)
    repack = [
        dataclasses.replace(t, seed=seed) if isinstance(t, rena_policy.AutolabelPrompt) else t
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
