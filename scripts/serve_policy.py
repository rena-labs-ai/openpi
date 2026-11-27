import dataclasses
import enum
import json
import logging
import pathlib
import socket
import time

import jax
from openpi_client import base_policy as _base_policy
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import rena_policy
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
class Roster:
    """Serve several rena policies at once, each built from the training config it was trained with, for a client to
    choose by id (see WebsocketPolicyServer). A JSON file:
    {"default": id, "models": [{"id": "<exp>/<step>", "config": "pi05_rena_full", "dir": "<checkpoint dir>"}, ...]}
    A model with "boxes": true is also served as "<id>+boxes": its pick and place prompts get the target's box from
    Gemini once per trial (rena_policy.StaticBoxes, gemini_locate), which needs rena-autolabel's autolabel folder
    importable and GEMINI_API_KEY set; with "wristboxes": true it is also served as "<id>+wristboxes", whose first
    request of a trial gets the wrist camera's box too, and with "trackedboxes": true as "<id>+trackedboxes", whose
    later requests get the wrist box SAM tracks through the robot's wrist stream (Roster.tracker)."""

    path: str
    # Where each Gemini answer is kept, with the frames it saw.
    boxes_dir: str = "boxes"
    # If set, every request each id answers is recorded in <record_dir>/<id>/ as the model got it, boxes included.
    record_dir: str | None = None
    # host:port of rena-autolabel's track_server, for models with "trackedboxes".
    tracker: str | None = None


@dataclasses.dataclass
class ModelSet:
    """Load several checkpoints of one training config from a models.json
    roster: {"models": [{"id", "label", "delivered_at", "dir"}, ...],
    "default": "<id>"}.
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
    policy: Checkpoint | ModelSet | Roster | Default = dataclasses.field(default_factory=Default)


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


def create_model_set(
    args: Args,
) -> tuple[dict[str, _policy.Policy], str, dict[str, str], dict[str, str | None]]:
    """(policies, default id, labels, delivered_at) from a models.json roster."""
    roster = json.loads(pathlib.Path(args.policy.roster).read_text())
    train_config = _config.get_config(args.policy.config)
    policies: dict[str, _policy.Policy] = {}
    labels: dict[str, str] = {}
    delivered_at: dict[str, str | None] = {}
    for m in roster["models"]:
        logging.info("Loading model %s from %s", m["id"], m["dir"])
        policies[m["id"]] = _policy_config.create_trained_policy(
            train_config,
            m["dir"],
            default_prompt=args.default_prompt,
            norm_stats=checkpoint_norm_stats(m["dir"]),
        )
        labels[m["id"]] = m.get("label") or m["id"]
        delivered_at[m["id"]] = m.get("delivered_at")
    default = roster["default"]
    warm_default(policies[default], default, train_config)
    return policies, default, labels, delivered_at


def warm_default(policy: _policy.Policy, model_id: str, train_config) -> None:
    """Compile the default model before the port binds.

    Only the default: warming every model would add each compile to a restart's
    serving gap, and the rest compile on first use without blocking the loop.
    Best-effort, since raising here would leave nothing serving at all.
    """
    started = time.monotonic()
    try:
        policy.warm(train_config.model.fake_obs())
    except Exception:
        logging.exception("Warm-up of %s failed; it compiles on first request", model_id)
    else:
        logging.info("Warmed %s in %.0fs", model_id, time.monotonic() - started)


def create_roster(
    roster: Roster, *, default_prompt: str | None = None
) -> tuple[dict[str, _base_policy.BasePolicy], str]:
    """The roster's policies by id, each compiled on a client-shaped request before serving starts (a first request
    that compiled would outlast the client's ping timeout), and the default id."""
    doc = json.loads(pathlib.Path(roster.path).read_text())
    policies = {}
    locate = None
    tracker = None

    def recorded(policy: _policy.Policy, mid: str) -> _base_policy.BasePolicy:
        return _policy.PolicyRecorder(policy, f"{roster.record_dir}/{mid}") if roster.record_dir else policy

    for m in doc["models"]:
        train_config = _config.get_config(m["config"])
        policy = _policy_config.create_trained_policy(train_config, m["dir"], default_prompt=default_prompt)

        def served(mid: str, policy=policy, anchor=train_config.data.anchor) -> _base_policy.BasePolicy:
            # each id its own anchor: a robot's trial runs on one id
            return rena_policy.CommandAnchor(recorded(policy, mid)) if anchor == "command" else recorded(policy, mid)

        served(m["id"]).infer(rena_policy.make_rena_example())
        logging.info("Loaded %s (%s) from %s", m["id"], m["config"], m["dir"])
        policies[m["id"]] = served(m["id"])
        for kind, wrist_first in (("boxes", False), ("wristboxes", True), ("trackedboxes", True)):
            if m.get(kind):
                locate = locate or gemini_locate(pathlib.Path(roster.boxes_dir))
                if kind == "trackedboxes" and tracker is None:
                    if roster.tracker is None:
                        raise ValueError(f"{m['id']} asks for trackedboxes, but the roster has no tracker address")
                    from track_server import TrackerClient

                    tracker = TrackerClient(roster.tracker)
                policies[f"{m['id']}+{kind}"] = rena_policy.StaticBoxes(
                    served(f"{m['id']}+{kind}"),
                    locate,
                    train_config.data.boxes_in,
                    wrist_first=wrist_first,
                    wrist_tracker=tracker if kind == "trackedboxes" else None,
                )
    return policies, doc["default"]


# A Gemini box smaller than this share of its image is a speck, not the target: a replay got an 8 px wrist box on an
# empty view, and training's labels kept Gemini's noise only where it landed on the object. A far lego brick is ~0.5%.
MIN_BOX_AREA = 0.002


def box_area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def gemini_locate(boxes_dir: pathlib.Path):
    """A StaticBoxes locate: rena-autolabel's deploy question to Gemini (pointing.ask_frames on all three cameras, no
    thinking), each answer saved in boxes_dir as the frames with the boxes drawn (look.sheet) and as JSON. A box
    smaller than MIN_BOX_AREA counts as no box."""
    import cv2
    import look
    import pointing

    boxes_dir.mkdir(parents=True, exist_ok=True)

    def locate(images: dict, command: str, what: str) -> dict:
        t0 = time.monotonic()
        bgr = {cam: cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for cam, im in images.items()}
        answer, usage = pointing.ask_frames(bgr, command, what, thinking=False)
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{what.removeprefix('the ').replace(' ', '_')}"
        cv2.imwrite(str(boxes_dir / f"{name}.jpg"), look.sheet(images, answer))
        record = {"command": command, "what": what, "answer": answer, "usage": usage, "seconds": time.monotonic() - t0}
        (boxes_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
        boxes = {cam: a and a["box"] for cam, a in answer.items()}
        if specks := {cam: box for cam, box in boxes.items() if box and box_area(box) < MIN_BOX_AREA}:
            logging.info("Dropped Gemini's speck boxes %s", specks)
        return {cam: None if cam in specks else box for cam, box in boxes.items()}

    return locate


def main(args: Args) -> None:
    # A restart's compiles run while robots wait; cached, each is paid once per box.
    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    if isinstance(args.policy, Roster):
        policies, default = create_roster(args.policy, default_prompt=args.default_prompt)
        server = websocket_policy_server.WebsocketPolicyServer(
            policies=policies, default=default, host="0.0.0.0", port=args.port
        )
        server.serve_forever()
        return

    if isinstance(args.policy, ModelSet):
        policies, default, labels, delivered_at = create_model_set(args)
        server = websocket_policy_server.WebsocketPolicyServer(
            policies=policies,
            default=default,
            labels=labels,
            delivered_at=delivered_at,
            host="0.0.0.0",
            port=args.port,
        )
        server.serve_forever()
        return

    policy = create_policy(args)
    policy_metadata = policy.metadata

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
