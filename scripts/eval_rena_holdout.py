"""Score a rena checkpoint on rena_split's holdout: the absolute joint targets it predicts against the commanded ones,
per step of the chunk, as the robot would execute them (the policy's own input and output transforms, the checkpoint's
norm stats). Every model gets the same frames, the same prompt for each frame (AutolabelPrompt seeded) and the same
flow noise, so their scores compare.

    uv run scripts/eval_rena_holdout.py --config-name pi05_rena_nostate_ho \
        --checkpoint checkpoints/pi05_rena_nostate_ho/ho_v1/29999 --out ho_v1.json
"""

import collections
import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.policies.rena_policy as rena_policy
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

EXECUTED = 10  # the steps of a chunk a robot runs before it asks again


def prompt_kind(prompter: rena_policy.AutolabelPrompt, rec: dict, frame: int) -> str:
    """The kind of prompt the seeded prompter gives the frame."""
    from autolabel_prompts import sample
    from autolabel_prompts import task_text

    rng = np.random.default_rng((prompter.seed, rec["episode_index"], frame))
    text, boxes = sample(rec, frame, rng, prompter.mix)
    if boxes:
        return "boxes"
    return "task" if text in (rec["instruction"], task_text(rec)) else "skill"


def summary(err: np.ndarray) -> dict:
    """err [n, horizon, 7] absolute errors: joints in rad, gripper in its 0-1 range."""
    return {
        "n": len(err),
        "joint_mae_executed": float(err[:, :EXECUTED, :6].mean()),
        "joint_mae_chunk": float(err[:, :, :6].mean()),
        "joint_mae_executed_per_joint": err[:, :EXECUTED, :6].mean((0, 1)).round(4).tolist(),
        "gripper_mae_executed": float(err[:, :EXECUTED, 6].mean()),
        "joint_mae_by_step": err[:, :, :6].mean((0, 2)).round(4).tolist(),
    }


def main(
    config_name: str,
    checkpoint: pathlib.Path,
    out: pathlib.Path,
    every: int = 10,
    seed: int = 0,
    batch_size: int = 32,
    num_steps: int = 10,
) -> None:
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, split="holdout"))
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    repack = [
        dataclasses.replace(t, seed=seed) if isinstance(t, rena_policy.AutolabelPrompt) else t
        for t in data_config.repack_transforms.inputs
    ]
    prompter = next(t for t in repack if isinstance(t, rena_policy.AutolabelPrompt))
    records = {
        (rec := json.loads(line))["episode_index"]: rec
        for line in pathlib.Path(prompter.labels_path).read_text().splitlines()
    }
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
    model = config.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    sample_actions = nnx_utils.module_jit(model.sample_actions)

    errs, kinds, sources = [], [], []
    picks = range(0, len(dataset), every)
    rng = jax.random.key(seed)
    for start in range(0, len(picks), batch_size):
        raws = [dataset[i] for i in picks[start : start + batch_size]]
        # before the inputs run: DeltaActions subtracts in place, on memory np.asarray shares with the tensor
        wants = [np.array(r["action"], dtype=np.float32)[data_config.action_lookback :] for r in raws]
        samples = [inputs(dict(r)) for r in raws]
        batch = jax.tree.map(lambda *x: np.stack(x), *samples)
        rng, step_rng = jax.random.split(rng)
        pred = np.asarray(sample_actions(step_rng, _model.Observation.from_dict(batch), num_steps=num_steps))
        for raw, want, s, p in zip(raws, wants, samples, pred, strict=True):
            got = outputs({"state": s["state"], "actions": p})["actions"]
            errs.append(np.abs(got - want))
            rec = records[int(raw["episode_index"])]
            kinds.append(prompt_kind(prompter, rec, int(raw["frame_index"])))
            sources.append(rec["dataset"])
        print(f"{min(start + batch_size, len(picks))}/{len(picks)}", flush=True)

    err = np.stack(errs)
    res = {
        "config": config_name,
        "checkpoint": str(checkpoint),
        "episodes": len(data_config.episodes),
        "every": every,
        "seed": seed,
        "num_steps": num_steps,
        "all": summary(err),
        "by_prompt": {k: summary(err[np.array(kinds) == k]) for k in sorted(set(kinds))},
        "by_source": {k: summary(err[np.array(sources) == k]) for k, _ in collections.Counter(sources).most_common()},
    }
    out.write_text(json.dumps(res, indent=1))
    a = res["all"]
    print(
        f"{a['n']} frames: joint MAE {a['joint_mae_executed']:.4f} rad over the first {EXECUTED} steps,"
        f" {a['joint_mae_chunk']:.4f} over the chunk; gripper {a['gripper_mae_executed']:.3f} -> {out}"
    )


if __name__ == "__main__":
    tyro.cli(main)
