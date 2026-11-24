"""Norm stats for a rena config (config.LeRobotRenaDataConfig) from the dataset's parquet files alone, without the
videos compute_norm_stats.py decodes: the state, and the actions as the model trains on them (relative to the
config's anchor, the gripper absolute), over the config's split. The actions get one set of stats over all steps of a
chunk, or one per step (action_norm "per_timestep"). Quantiles are exact.

    uv run scripts/compute_rena_norm_stats.py --config-name pi05_rena_ho_nobox
"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import tyro

import openpi.models.pi0_config as pi0_config
import openpi.policies.rena_policy as rena_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as _transforms


def chunks(state: np.ndarray, action: np.ndarray, horizon: int, lookback: int) -> np.ndarray:
    """Each frame's action sequence as LeRobotDataset gives it: clamped to the episode, so it repeats the last step."""
    idx = np.clip(np.arange(len(state))[:, None] + np.arange(-lookback, horizon)[None], 0, len(state) - 1)
    return action[idx]


def model_view(state: np.ndarray, actions: np.ndarray, anchor: str) -> tuple[np.ndarray, np.ndarray]:
    """The state and actions the model sees, through the config's own transforms."""
    data = {"observation.state": state, "actions": actions}
    if anchor == "command":
        data = rena_policy.AnchorOnCommand()(rena_policy.SplitPrevAction()(data))
    out = _transforms.DeltaActions(_transforms.make_bool_mask(6, -1))(
        {"state": data["observation.state"], "actions": np.array(data["actions"], dtype=np.float32)}
    )
    return out["state"], out["actions"]


def stats(x: np.ndarray, axis) -> normalize.NormStats:
    return normalize.NormStats(
        mean=x.mean(axis), std=x.std(axis), q01=np.quantile(x, 0.01, axis=axis), q99=np.quantile(x, 0.99, axis=axis)
    )


def main(config_name: str, out_dir: pathlib.Path | None = None) -> None:
    config = _config.get_config(config_name)
    factory = config.data
    # Built without the assets dir: the stats it holds may be the stale ones this script replaces.
    data_config = factory.create(pathlib.Path(tempfile.mkdtemp()), config.model)
    root = factory.dataset_dir
    files = sorted((root / "data").rglob("episode_*.parquet"))
    keep = set(data_config.episodes) if data_config.episodes is not None else None
    states, actions = [], []
    for f in files:
        if keep is not None and int(f.stem.split("_")[1]) not in keep:
            continue
        episode = pd.read_parquet(f, columns=["observation.state", "action"])
        s, a = np.stack(episode["observation.state"]), np.stack(episode["action"])
        st, ac = model_view(s, chunks(s, a, config.model.action_horizon, data_config.action_lookback), factory.anchor)
        states.append(st)
        actions.append(ac)
    state, action = np.concatenate(states), np.concatenate(actions)
    per_step = factory.action_norm == "per_timestep"
    norm = {
        "state": stats(state, 0),
        "actions": stats(action, 0) if per_step else stats(action.reshape(-1, action.shape[-1]), 0),
    }
    if isinstance(config.model, pi0_config.Pi0Config) and config.model.knowledge_insulation:
        # the FAST target's, over every step of a chunk whatever action_norm is (transforms.FASTActions)
        norm["fast_actions"] = stats(action.reshape(-1, action.shape[-1]), 0)
    out = out_dir or config.assets_dirs / data_config.asset_id
    normalize.save(out, norm)
    print(f"{len(states)} episodes, {len(state)} frames, actions {action.shape[1:]} ({factory.action_norm}) -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
