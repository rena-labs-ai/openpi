"""Score rena checkpoints on point evaluation sets made by scripts/prepare_vl_data.py (vl/where2place, free space, and
vl/roborefit_testb, objects): the points each greedily writes (Pi0.decode_points), scored as RoboPoint scores them, by
the share that land inside the answer's mask. Writes OUT/<checkpoint>/<set>.json with each sample's points and score,
and <set>.png with the first frames drawn: the mask in green, the points in red.

    uv run scripts/eval_rena_points.py --config-name pi05_rena_tor_cotrain --out points \\
        --checkpoints CKPT_DIR/5000 CKPT_DIR/10000 [--checkpoints base]

"base" is the config's starting weights (pi05_base with PaliGemma's <loc> rows put back).
"""

import dataclasses
import json
import pathlib

from autolabel_prompts import Mix
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tyro

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import vl_data
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def image_xy(bins: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(y, x) <loc> bins [p, 2] over the padded square input -> (x, y) pixels [p, 2] of an image of size (w, h), as
    autolabel_prompts.loc_box maps them the other way."""
    w, h = size
    scale = Mix.input_px / max(w, h)
    pad_x, pad_y = (Mix.input_px - w * scale) / 2, (Mix.input_px - h * scale) / 2
    v = (bins + 0.5) / 1024 * Mix.input_px
    return np.stack([(v[:, 1] - pad_x) / scale, (v[:, 0] - pad_y) / scale], axis=-1)


def inside(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[p] whether each (x, y) pixel point falls on the mask [h, w]; off the image counts as outside."""
    x, y = np.floor(points[:, 0]).astype(int), np.floor(points[:, 1]).astype(int)
    on = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    hit = np.zeros(len(points), bool)
    hit[on] = mask[y[on], x[on]]
    return hit


def draw(image: Image.Image, mask: np.ndarray, points: np.ndarray, caption: str) -> Image.Image:
    shown = np.asarray(image.convert("RGB")).astype(np.float32)
    shown[mask] = 0.6 * shown[mask] + 0.4 * np.array([0, 255, 0])
    pil = Image.fromarray(shown.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    r = max(2, round(max(pil.size) / 120))
    for x, y in points:
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255))
    d.rectangle([0, 0, pil.width, 14], fill=(0, 0, 0))
    d.text((2, 1), caption[:90], fill=(255, 255, 255))
    return pil


def grid(tiles: list[Image.Image], cols: int = 4, side: int = 320) -> Image.Image:
    tiles = [t.resize((side, round(side * t.height / t.width))) for t in tiles]
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    heights = [max(t.height for t in row) for row in rows]
    sheet = Image.new("RGB", (cols * side, sum(heights)), "white")
    y = 0
    for row, height in zip(rows, heights, strict=True):
        for j, t in enumerate(row):
            sheet.paste(t, (j * side, y))
        y += height
    return sheet


def load_model(config: _config.TrainConfig, checkpoint: str):
    if checkpoint == "base":
        model = config.model.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(config.weight_loader.load(state.to_pure_dict()))
        return nnx.merge(graphdef, state)
    return config.model.load(_model.restore_params(pathlib.Path(checkpoint) / "params", dtype=jnp.bfloat16))


def main(
    config_name: str,
    checkpoints: list[str],
    out: pathlib.Path,
    sets: tuple[str, ...] = ("vl/where2place", "vl/roborefit_testb"),
    batch_size: int = 16,
    max_points: int = 8,
    drawn: int = 16,
) -> None:
    from lerobot.common.constants import HF_LEROBOT_HOME

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    for checkpoint in checkpoints:
        if checkpoint != "base":
            data_config = dataclasses.replace(
                data_config,
                norm_stats=_checkpoints.load_norm_stats(pathlib.Path(checkpoint) / "assets", data_config.asset_id),
            )
        model = load_model(config, checkpoint)
        decode = nnx_utils.module_jit(model.decode_points, static_argnames=("max_points",))
        name = "base" if checkpoint == "base" else pathlib.Path(checkpoint).name
        (out / name).mkdir(parents=True, exist_ok=True)
        for vl_set in sets:
            root = HF_LEROBOT_HOME / vl_set
            samples = vl_data.VLDataset(root).samples
            dataset = _data_loader.vl_dataset(vl_set, data_config, config.model.action_horizon)
            inputs = [dataset[i] for i in range(len(dataset))]
            inputs += [inputs[-1]] * (-len(inputs) % batch_size)  # whole batches; the copies are dropped below
            got_bins, got_written = [], []
            for s in range(0, len(inputs), batch_size):
                batch = jax.tree.map(lambda *x: np.stack(x), *inputs[s : s + batch_size])
                bins, written = decode(_model.Observation.from_dict(batch), max_points=max_points)
                got_bins.append(np.asarray(bins))
                got_written.append(np.asarray(written))
            bins, written = np.concatenate(got_bins)[: len(samples)], np.concatenate(got_written)[: len(samples)]
            results, tiles = [], []
            for i, sample in enumerate(samples):
                image = Image.open(root / sample["image"])
                mask = np.asarray(Image.open(root / sample["mask"])) > 0
                points = image_xy(bins[i][written[i]], image.size)
                hits = inside(points, mask)
                results.append(
                    {
                        "image": sample["image"],
                        "prompt": sample["prompt"],
                        "category": sample.get("category"),
                        "points": points.round(1).tolist(),
                        "accuracy": float(hits.mean()) if len(hits) else 0.0,
                    }
                )
                if i < drawn:
                    tiles.append(draw(image, mask, points, f"{results[-1]['accuracy']:.2f} {sample['prompt']}"))
            accuracy = np.array([r["accuracy"] for r in results])
            summary = {
                "checkpoint": checkpoint,
                "set": vl_set,
                "n": len(results),
                "accuracy": float(accuracy.mean()),
                "answered": float(np.mean([len(r["points"]) > 0 for r in results])),
                "points_per_answer": float(np.mean([len(r["points"]) for r in results])),
                "by_category": {
                    c: float(np.mean([r["accuracy"] for r in results if r["category"] == c]))
                    for c in sorted({str(r["category"]) for r in results})
                    if vl_set.endswith("where2place")
                },
            }
            short = vl_set.split("/")[-1]
            (out / name / f"{short}.json").write_text(json.dumps({"summary": summary, "samples": results}, indent=1))
            grid(tiles).save(out / name / f"{short}.png")
            print(json.dumps(summary))
        del model


if __name__ == "__main__":
    tyro.cli(main)
