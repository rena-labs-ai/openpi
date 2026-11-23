"""Tests for weight_loaders._merge_params and CheckpointWeightLoader."""

import flax.traverse_util as tu
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.download as download
from openpi.training.weight_loaders import PALIGEMMA_PARAMS
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.training.weight_loaders import _merge_params


def _ref():
    return {
        "PaliGemma": {"layer": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32)}},
        "stage_head_in": {"kernel": jax.ShapeDtypeStruct((3, 4), np.float32)},
        "llm": {"lora": {"a": jax.ShapeDtypeStruct((1, 1), np.float32)}},
    }


def _loaded():
    # A base checkpoint: only the PaliGemma weight, no lora, no stage_head.
    return {"PaliGemma": {"layer": {"kernel": np.ones((2, 2), np.float32)}}}


def test_old_regex_drops_stage_head():
    """Documents the bug: stage_head keys are dropped when only '.*lora.*' is used."""
    flat = tu.flatten_dict(_merge_params(_loaded(), _ref(), missing_regex=".*lora.*"), sep="/")
    assert "stage_head_in/kernel" not in flat


def test_new_regex_backfills_stage_head_and_lora():
    """Validates that the fix regex keeps the loaded weight, backfills lora and stage_head."""
    flat = tu.flatten_dict(_merge_params(_loaded(), _ref(), missing_regex=".*lora.*|.*stage_head.*"), sep="/")
    # Base weight kept as the real loaded array (not a ShapeDtypeStruct).
    assert "PaliGemma/layer/kernel" in flat
    assert not isinstance(flat["PaliGemma/layer/kernel"], jax.ShapeDtypeStruct)
    # lora and stage_head backfilled from reference as ShapeDtypeStruct (later stripped → left at init).
    assert isinstance(flat["llm/lora/a"], jax.ShapeDtypeStruct)
    assert isinstance(flat["stage_head_in/kernel"], jax.ShapeDtypeStruct)


def test_paligemma_locs_puts_back_the_loc_and_seg_rows_only(tmp_path, monkeypatch):
    pi05 = np.zeros((257_152, 2), np.float32)
    paligemma = np.arange(257_152 * 2, dtype=np.float32).reshape(-1, 2)
    np.savez(tmp_path / "pt_224.npz", **{"params/llm/embedder/input_embedding": paligemma})
    monkeypatch.setattr(
        _model, "restore_params", lambda *_, **__: {"PaliGemma": {"llm": {"embedder": {"input_embedding": pi05}}}}
    )
    monkeypatch.setattr(
        download, "maybe_download", lambda url, **_: tmp_path / "pt_224.npz" if url == PALIGEMMA_PARAMS else url
    )
    ref = {"PaliGemma": {"llm": {"embedder": {"input_embedding": jax.ShapeDtypeStruct(pi05.shape, np.float32)}}}}
    table = CheckpointWeightLoader("ckpt", paligemma_locs=True).load(ref)["PaliGemma"]["llm"]["embedder"][
        "input_embedding"
    ]
    assert (table[:256_000] == 0).all()
    assert (table[256_000:] == paligemma[256_000:]).all()
