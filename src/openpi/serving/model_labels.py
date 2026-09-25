"""Display labels for the multi-model catalog: the roster's label and the day
the checkpoint was trained.

The robot's model picker listed "v21" and "v22" with nothing to say which one
came out of last night's training run. The day is read from the checkpoint
itself, so it cannot drift from what is actually being served.
"""

from __future__ import annotations

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
    # Either names the same training run, so init stands in when commit is absent.
    for key in ("commit_timestamp_nsecs", "init_timestamp_nsecs"):
        nsecs = meta.get(key)
        if isinstance(nsecs, int) and not isinstance(nsecs, bool) and nsecs > 0:
            moment = datetime.datetime.fromtimestamp(nsecs / 1e9, tz=datetime.UTC)
            return moment.astimezone(tz).date()
    return None


def dated_label(label: str, trained_on: datetime.date | None) -> str:
    """ "v22 · Sep 24" — the date and not the time, which is all the picker needs
    to tell one night's model from the next. The label unchanged without one."""
    if trained_on is None:
        return label
    return f"{label} · {trained_on:%b} {trained_on.day}"
