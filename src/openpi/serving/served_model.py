"""One model of a multi-model catalog, and how the picker labels it.

The robot's model picker listed "v21" and "v22" with nothing to say which one
came out of last night's training run. The day is read from the checkpoint
itself, so it cannot drift from what is actually being served.

`ServedModel` rather than `Model`: in this repo "model" is the network
(`openpi.models.model.BaseModel`), and `ServedModel` is the name the rest of the
stack already gives this entry — rena_msgs, rena-control's catalog and
rena-training's promotion all call it that.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib

# orbax writes this into every step directory it commits
# (orbax-checkpoint 0.11.13, _src/metadata/checkpoint.py).
STEP_METADATA_FILENAME = "_CHECKPOINT_METADATA"


def checkpoint_trained_on(
    ckpt_dir: str | pathlib.Path,
    tz: datetime.tzinfo | None = None,
) -> datetime.date | None:
    """The day orbax committed the checkpoint in `ckpt_dir`.

    In `tz`, or the serving host's local zone when None — so a host that sets
    TZ shows the day its team would call it, and a checkpoint committed at
    11 pm is not reported as the next morning.

    Read from orbax's step metadata, never the directory's mtime. Roster
    checkpoints are synced onto the serving host, so mtime is when they were
    copied rather than when they were trained: the wrong answer to exactly the
    question this exists to answer. Anything missing or unreadable is None, so
    a label never carries a guessed date.
    """
    try:
        meta = json.loads((pathlib.Path(ckpt_dir) / STEP_METADATA_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    # commit is when the step became durable; init, when it started writing.
    # Both name the same training run, so init stands in when commit is absent
    # — but only then. A commit field that is present and malformed means the
    # file is not trustworthy, and its other field would be a guess.
    commit = meta.get("commit_timestamp_nsecs")
    nsecs = meta.get("init_timestamp_nsecs") if commit is None else commit
    if not isinstance(nsecs, int) or isinstance(nsecs, bool) or nsecs <= 0:
        return None
    try:
        moment = datetime.datetime.fromtimestamp(nsecs / 1e9, tz=datetime.UTC)
        return moment.astimezone(tz).date()
    except (OverflowError, OSError, ValueError):
        # A positive int datetime cannot represent. Raising here would stop the
        # server from starting over a label.
        return None


def dated_label(label: str, trained_on: datetime.date | None) -> str:
    """ "v22 · Sep 24" — the date and not the time, which is all the picker needs
    to tell one night's model from the next. The label unchanged without one."""
    if trained_on is None:
        return label
    return f"{label} · {trained_on:%b} {trained_on.day}"


@dataclasses.dataclass(frozen=True)
class ServedModel:
    """One entry of a models.json roster: {"id", "label", "dir"}."""

    # "<exp_name>/<step>", the selection key the robot and the app hold.
    id: str
    # The trained checkpoint directory this model is loaded from.
    dir: str
    # The roster's own name for it, usually the version ("v22"); None when the
    # roster gave none.
    name: str | None = None

    @classmethod
    def from_roster(cls, entry: dict) -> ServedModel:
        """From one roster entry. A missing id or dir is a broken roster and
        raises; a missing or empty label only costs the version name."""
        return cls(id=entry["id"], dir=entry["dir"], name=entry.get("label") or None)

    def trained_on(self, tz: datetime.tzinfo | None = None) -> datetime.date | None:
        """The day this model's checkpoint was committed; see checkpoint_trained_on."""
        return checkpoint_trained_on(self.dir, tz)

    def label(self, tz: datetime.tzinfo | None = None) -> str:
        """What the picker shows: "v22 · Sep 24", or the id when the roster
        named nothing, with no date when the checkpoint recorded none."""
        return dated_label(self.name or self.id, self.trained_on(tz))
