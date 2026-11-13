"""Score a rena checkpoint on rena_split's holdout (training.holdout): the absolute joint targets it predicts against
the commanded ones, per step of the chunk, with the checkpoint's norm stats and the same flow noise for every model.

    uv run scripts/eval_rena_holdout.py --config-name pi05_rena_ho_nobox \
        --checkpoint checkpoints/pi05_rena_ho_nobox/v1/29999 --out v1.json
"""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.holdout as _holdout


def main(
    config_name: str,
    checkpoint: pathlib.Path,
    out: pathlib.Path,
    max_frames: int | None = None,
    seed: int = 0,
    batch_size: int = 32,
    num_steps: int = 10,
) -> None:
    config = _config.get_config(config_name)
    asset_id = config.data.create(config.assets_dirs, config.model).asset_id
    norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", asset_id)
    holdout = _holdout.load(config, norm_stats, batch_size=batch_size, max_frames=max_frames, seed=seed)
    model = config.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rngs = jax.random.split(jax.random.key(seed), len(holdout.batches))
    predicted = [
        np.asarray(sample_actions(r, _model.Observation.from_dict(b), num_steps=num_steps))
        for r, b in zip(rngs, holdout.batches, strict=True)
    ]
    res = {
        "config": config_name,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "num_steps": num_steps,
        **holdout.report(holdout.errors(predicted)),
    }
    out.write_text(json.dumps(res, indent=1))
    a = res["all"]
    print(
        f"{a['n']} frames: joint MAE {a['joint_mae_executed']:.4f} rad over the first {_holdout.EXECUTED} steps,"
        f" {a['joint_mae_chunk']:.4f} over the chunk; gripper {a['gripper_mae_executed']:.3f} -> {out}"
    )


if __name__ == "__main__":
    tyro.cli(main)
