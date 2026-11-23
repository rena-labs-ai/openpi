import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


# PaliGemma pt-224. Google's bucket no longer serves it anonymously, so a machine without it in its openpi cache needs the
# same file from Hugging Face (google/paligemma-3b-pt-224, paligemma-3b-pt-224.npz) copied to its cached path.
PALIGEMMA_PARAMS = "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
# The vocabulary's 1024 <loc> and 128 <seg> tokens start here.
PALIGEMMA_LOCS = 256_000


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Put PaliGemma pt-224's <loc>/<seg> embedding rows back, which pi05_base rewrote, so <loc> tokens mean image
    # coordinates again for box targets.
    paligemma_locs: bool = False

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.paligemma_locs:
            embedder = loaded_params["PaliGemma"]["llm"]["embedder"]
            path = download.maybe_download(PALIGEMMA_PARAMS, gs={"token": "anon"})
            with path.open("rb") as f:
                rows = np.load(f, allow_pickle=False)["params/llm/embedder/input_embedding"][PALIGEMMA_LOCS:]
            table = embedder["input_embedding"]
            embedder["input_embedding"] = np.concatenate([table[:PALIGEMMA_LOCS], rows.astype(table.dtype)])
        # Add all missing LoRA + stage-head weights (left at fresh init via the strip in _load_weights_and_validate).
        return _merge_params(loaded_params, params, missing_regex=".*lora.*|.*stage_head.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(PALIGEMMA_PARAMS, gs={"token": "anon"})
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
