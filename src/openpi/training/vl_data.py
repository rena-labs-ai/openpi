"""Vision-language data for co-training rena's pi0.5: images with a prompt and the answer the backbone learns as its
target, mixed into the robot frames. A dataset is a folder under HF_LEROBOT_HOME holding samples.jsonl, one
{"image": path relative to the folder, "prompt": ..., "target": ...} per line, and the images
(scripts/prepare_vl_data.py writes them)."""

import dataclasses
import json
import pathlib

import numpy as np
from PIL import Image
import torch.utils.data

from openpi import transforms
from openpi.policies import rena_policy


class VLDataset(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path):
        self._root = root
        self.samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines() if line]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = np.asarray(Image.open(self._root / sample["image"]).convert("RGB"))
        return {"image": image, "prompt": sample["prompt"], "annotation": sample["target"]}


@dataclasses.dataclass(frozen=True)
class VLInputs(transforms.DataTransformFn):
    """A vision-language sample as rena's model inputs: its image in the base camera's slot, the other cameras blank and
    masked out, zero state and actions that action_loss_mask keeps out of the flow and stage losses; the state is left
    out of the prompt, as it is of the robot frames state dropout drops."""

    action_horizon: int
    action_dim: int

    def __call__(self, data: dict) -> dict:
        image = data["image"]
        blank = np.zeros_like(image)
        slots = list(rena_policy.CAMERAS)
        return {
            "image": {slot: image if slot == slots[0] else blank for slot in slots} | {"grounded_rgb": blank},
            "image_mask": {slot: np.bool_(slot == slots[0]) for slot in slots} | {"grounded_rgb": np.False_},
            "state": np.zeros(self.action_dim, np.float32),
            "actions": np.zeros((self.action_horizon, self.action_dim), np.float32),
            "task_stage": np.int32(0),
            "action_loss_mask": np.False_,
            "drop_state": True,
            "prompt": data["prompt"],
            "annotation": data["annotation"],
        }


class MixedDataset(torch.utils.data.Dataset):
    """The robot dataset's indices, each one drawn instead, with probability share, from one of the vision-language
    datasets (picked by weight, evenly by default); fixed per index and seed, as the loader's shuffling expects."""

    def __init__(
        self,
        robot: torch.utils.data.Dataset,
        vl: list[torch.utils.data.Dataset],
        share: float,
        seed: int,
        weights: list[float] | None = None,
    ):
        self._robot, self._vl, self._share, self._seed = robot, vl, share, seed
        w = np.ones(len(vl)) if not weights else np.asarray(weights, float)
        self._p = w / w.sum()

    def __len__(self) -> int:
        return len(self._robot)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng((self._seed, index))
        if rng.random() < self._share:
            source = self._vl[rng.choice(len(self._vl), p=self._p)]
            return source[int(rng.integers(len(source)))]
        return self._robot[index]
