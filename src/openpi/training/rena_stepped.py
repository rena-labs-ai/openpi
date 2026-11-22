"""Recordings whose arm commands step: the joint targets reach the follower about 12 times a second on a 25 fps
recording, so each value holds for 1-3 frames and then jumps (renav2 robots recorded since 2026-10-29, as the leader
reaches the follower over WiFi). The gripper's command is unaffected, and the follower's measured joints stay smooth."""

import pathlib

import numpy as np
import pandas as pd

# A recording steps when its joint command changes on fewer than this share of the frames where the arm moves: stepped
# recordings sit at 0.4-0.6, the others at 0.8-1.0.
THRESHOLD = 0.75
# Holds up to this many frames are steps to interpolate across; longer ones may be the arm pausing. Stepped gaps are
# 1-3 frames 97% of the time.
MAX_GAP = 4


def _changes(action: np.ndarray) -> np.ndarray:
    """[n-1] whether the joint command changes between consecutive frames."""
    return np.any(np.abs(np.diff(action[:, :6], axis=0)) > 1e-6, axis=1)


def stepped(action: np.ndarray) -> bool:
    """Whether a recording's [n, 7] joint commands step."""
    changes = _changes(action)
    if len(changes) < 25:
        return False
    moving = np.convolve(changes, np.ones(25), "same") > 5  # within a second-long window where the arm moves
    return bool(moving.any() and changes[moving].mean() < THRESHOLD)


def interpolate(action: np.ndarray) -> np.ndarray:
    """The joint commands linear between the frames where a new value arrives, across holds of at most MAX_GAP frames;
    the gripper and longer holds as they were. Each interpolated frame leads the command it has as recorded by less
    than a hold."""
    out = np.array(action, copy=True)
    starts = np.concatenate([[0], np.flatnonzero(_changes(action)) + 1])
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        if 1 < b - a <= MAX_GAP:
            w = (np.arange(b - a) / (b - a))[:, None]
            out[a:b, :6] = (1 - w) * action[a, :6] + w * action[b, :6]
    return out


def stepped_episodes(root: pathlib.Path) -> set[int]:
    """The episode indices of a LeRobot dataset whose joint commands step."""
    out = set()
    for f in sorted((root / "data").rglob("episode_*.parquet")):
        if stepped(np.stack(pd.read_parquet(f, columns=["action"])["action"])):
            out.add(int(f.stem.split("_")[1]))
    return out
