import dataclasses
import enum
import json
import logging
import pathlib
import socket
import time

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import rena_policy
from openpi.serving import websocket_policy_server
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
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Roster:
    """Serve several rena policies at once, each built from the training config it was trained with, for a client to
    choose by id (see WebsocketPolicyServer). A JSON file:
    {"default": id, "models": [{"id": "<exp>/<step>", "config": "pi05_rena_full", "dir": "<checkpoint dir>"}, ...]}
    A model with "boxes": true is also served as "<id>+boxes": its pick and place prompts get the target's box from
    Gemini once per trial (rena_policy.StaticBoxes, gemini_locate), which needs rena-autolabel's autolabel folder
    importable and GEMINI_API_KEY set; with "wristboxes": true it is also served as "<id>+wristboxes", whose first
    request of a trial gets the wrist camera's box too."""

    path: str
    # Where each Gemini answer is kept, with the frames it saw.
    boxes_dir: str = "boxes"


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
    policy: Checkpoint | Default | Roster = dataclasses.field(default_factory=Default)


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


def create_roster(roster: Roster, *, default_prompt: str | None = None) -> tuple[dict[str, _policy.Policy], str]:
    """The roster's policies by id, each compiled on a client-shaped request before serving starts (a first request
    that compiled would outlast the client's ping timeout), and the default id."""
    doc = json.loads(pathlib.Path(roster.path).read_text())
    policies = {}
    locate = None
    for m in doc["models"]:
        train_config = _config.get_config(m["config"])
        policy = _policy_config.create_trained_policy(train_config, m["dir"], default_prompt=default_prompt)
        policy.infer(rena_policy.make_rena_example())
        logging.info("Loaded %s (%s) from %s", m["id"], m["config"], m["dir"])
        policies[m["id"]] = policy
        for kind, wrist_first in (("boxes", False), ("wristboxes", True)):
            if m.get(kind):
                locate = locate or gemini_locate(pathlib.Path(roster.boxes_dir))
                policies[f"{m['id']}+{kind}"] = rena_policy.StaticBoxes(
                    policy, locate, train_config.data.boxes_in, wrist_first=wrist_first
                )
    return policies, doc["default"]


def gemini_locate(boxes_dir: pathlib.Path):
    """A StaticBoxes locate: rena-autolabel's deploy question to Gemini (pointing.ask_frames on all three cameras, no
    thinking), each answer saved in boxes_dir as the frames with the boxes drawn (look.sheet) and as JSON."""
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
        return {cam: a and a["box"] for cam, a in answer.items()}

    return locate


def main(args: Args) -> None:
    if isinstance(args.policy, Roster):
        policies, default = create_roster(args.policy, default_prompt=args.default_prompt)
    else:
        policies, default = {"default": create_policy(args)}, "default"

    # Record the policy's behavior.
    if args.record:
        policies = {mid: _policy.PolicyRecorder(p, f"policy_records/{mid}") for mid, p in policies.items()}

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(policies, default, host="0.0.0.0", port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
