import dataclasses
import enum
import json
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class ModelSet:
    """Load several checkpoints of one training config from a models.json
    roster: {"models": [{"id", "label", "dir"}, ...], "default": "<id>"}.
    The model is selected per connection (ws path /m/<id>)."""

    # Training config name shared by every checkpoint in the set.
    config: str
    # Path to the models.json roster.
    roster: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | ModelSet | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def checkpoint_norm_stats(ckpt_dir: str) -> dict:
    """Norm stats from the checkpoint's own assets dir. The shared config's
    asset_id names only ONE model's assets, so each sibling checkpoint must
    resolve its own (each self-contained prefix holds exactly one asset id)."""
    assets = pathlib.Path(ckpt_dir) / "assets"
    ids = [p.name for p in assets.iterdir() if p.is_dir()]
    if len(ids) != 1:
        raise ValueError(f"expected exactly one asset id under {assets}, got {ids}")
    return _checkpoints.load_norm_stats(assets, ids[0])


def create_model_set(args: Args) -> tuple[dict[str, _policy.Policy], str, dict[str, str]]:
    """(policies by id, default id, labels by id) from a models.json roster."""
    roster = json.loads(pathlib.Path(args.policy.roster).read_text())
    train_config = _config.get_config(args.policy.config)
    policies: dict[str, _policy.Policy] = {}
    labels: dict[str, str] = {}
    for m in roster["models"]:
        logging.info("Loading model %s from %s", m["id"], m["dir"])
        policies[m["id"]] = _policy_config.create_trained_policy(
            train_config,
            m["dir"],
            default_prompt=args.default_prompt,
            norm_stats=checkpoint_norm_stats(m["dir"]),
        )
        labels[m["id"]] = m.get("label") or m["id"]
    return policies, roster["default"], labels


def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    if isinstance(args.policy, ModelSet):
        policies, default, labels = create_model_set(args)
        server = websocket_policy_server.WebsocketPolicyServer(
            policies=policies,
            default=default,
            labels=labels,
            host="0.0.0.0",
            port=args.port,
        )
        server.serve_forever()
        return

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
