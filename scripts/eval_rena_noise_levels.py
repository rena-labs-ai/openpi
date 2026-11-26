"""A run's checkpoints scored by their flow error at each noise level, on the same frames of rena_split's train part
and of its holdout: the check of "Beyond Validation Loss" (appendix D) that a later checkpoint has memorized its
training demonstrations when its holdout error rises at the low-noise end while its train error keeps falling.

pi0.5 noises the action chunk as x_t = t * eps + (1 - t) * a, t = 1 pure noise and t = 0 the actions, and predicts
v = eps - a. At one t the eps-prediction error is (1 - t)^2 times the v error and the clean-action error t^2 times it,
so checkpoints rank the same at each t in every unit. Each frame keeps one noise draw for every t and every
checkpoint, so two checkpoints' errors differ by their weights alone. Errors are over the 7 real action dims;
"all32" also counts the zero-padded ones, as training's flow loss does.

    uv run scripts/eval_rena_noise_levels.py --config-name pi05_rena_ho_nobox \
        --checkpoints ckpts/v3/5000 ckpts/v3/29999 --out-dir noise_levels/v3
"""

import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.holdout as _holdout

TIMES = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SPLITS = ("train", "holdout")


def frame_errors(model, batch: dict, noise: jax.Array, t: jax.Array, dims: int) -> jax.Array:
    """[b, 2]: each frame's squared v error at time t, over the real dims and over all dims."""
    observation = _model.preprocess_observation(None, _model.Observation.from_dict(batch), train=False)
    actions = batch["actions"]
    x_t = t * noise + (1 - t) * actions
    *_, suffix_out = model._forward(observation, x_t, jnp.full(actions.shape[:1], t))  # noqa: SLF001
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    err = jnp.square(v_t - (noise - actions))
    return jnp.stack(
        [
            err[..., :dims].mean((1, 2)),
            err.mean((1, 2)),
        ],
        axis=-1,
    )


def main(
    config_name: str,
    checkpoints: list[pathlib.Path],
    out_dir: pathlib.Path,
    frames: int = 512,
    batch_size: int = 32,
    seed: int = 0,
) -> None:
    config = _config.get_config(config_name)
    asset_id = config.data.create(config.assets_dirs, config.model).asset_id
    norm_stats = _checkpoints.load_norm_stats(checkpoints[0] / "assets", asset_id)
    for c in checkpoints[1:]:
        other = _checkpoints.load_norm_stats(c / "assets", asset_id)
        assert all(
            np.allclose(norm_stats[k].q01, other[k].q01) and np.allclose(norm_stats[k].q99, other[k].q99)
            for k in norm_stats
        ), f"{c} has other norm stats than {checkpoints[0]}"
    data = {
        s: _holdout.load(config, norm_stats, batch_size=batch_size, max_frames=frames, seed=seed, split=s)
        for s in SPLITS
    }
    if empty := [s for s, d in data.items() if d is None]:
        raise SystemExit(f"{', '.join(empty)} has fewer frames than one batch of {batch_size}")
    dims = data["holdout"].want.shape[-1]
    noise = {
        s: [
            jax.random.normal(jax.random.fold_in(jax.random.key(seed), i * 1000 + b), batch["actions"].shape)
            for b, batch in enumerate(data[s].batches)
        ]
        for i, s in enumerate(SPLITS)
    }
    graphdef = nnx.graphdef(nnx.eval_shape(config.model.create, jax.random.key(0)))

    @jax.jit
    def errors(state: nnx.State, batch: dict, noise: jax.Array, t: jax.Array) -> jax.Array:
        return frame_errors(nnx.merge(graphdef, state), batch, noise, t, dims)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in checkpoints:
        state = nnx.state(config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16)))
        out = {"times": np.array(TIMES)}
        for s in SPLITS:
            got = np.stack(
                [
                    np.concatenate(
                        [np.asarray(errors(state, b, n, t)) for b, n in zip(data[s].batches, noise[s], strict=True)]
                    )
                    for t in TIMES
                ],
                axis=1,
            )  # [frames, times, 2]
            out |= {f"{s}_v": got[..., 0], f"{s}_v_all32": got[..., 1]}
            out[f"{s}_kinds"] = np.array(data[s].kinds)
        np.savez(out_dir / f"{ckpt.name}.npz", **out)
        print(ckpt, "mean v error per t:")
        print("  t       ", " ".join(f"{t:<6}" for t in TIMES))
        for s in SPLITS:
            print(f"  {s:<8}", " ".join(f"{v:.4f}" for v in out[f"{s}_v"].mean(0)))
        del state


if __name__ == "__main__":
    tyro.cli(main)
