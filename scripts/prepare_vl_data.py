"""Vision-language datasets for co-training (training.vl_data), from RefCOCO's referring expressions and RoboPoint's
object and free-space references: an image, a prompt, and a target of PaliGemma <loc> tokens over the image padded to
the model's square input (autolabel_prompts.loc_box), four per box and two (y, x) per point.

    uv run scripts/prepare_vl_data.py --source refcoco --annotations ANNOTATIONS.parquet --images COCO_DIR --out OUT
    uv run scripts/prepare_vl_data.py --source robopoint --annotations robopoint_1432k.json --images IMAGES_DIR \
        --out OUT --kind object_ref [--limit N]

Two point evaluation sets with the answer's mask (samples' "mask", saved beside the image at its size) and no target:
RoboPoint's Where2Place (free space) and RoboRefIt (objects, its testB parquet shards):

    uv run scripts/prepare_vl_data.py --source where2place --annotations WHERE2PLACE_DIR --images WHERE2PLACE_DIR --out OUT
    uv run scripts/prepare_vl_data.py --source roborefit --annotations TESTB_SHARDS_DIR --images TESTB_SHARDS_DIR \
        --out OUT --limit 400

Images are saved with their longest side at most --max-side pixels; boxes and points are fractions, so they hold.
"""

import json
import pathlib
import re

from autolabel_prompts import Mix
from autolabel_prompts import loc_box
import numpy as np
import pandas as pd
from PIL import Image
import tqdm
import tyro


def save_image(src: pathlib.Path, out: pathlib.Path, name: str, max_side: int) -> tuple[int, int]:
    """The image resized to max_side and saved as out/images/name; its (width, height) as saved."""
    image = Image.open(src).convert("RGB")
    image.thumbnail((max_side, max_side))
    (out / "images").mkdir(parents=True, exist_ok=True)
    image.save(out / "images" / name, quality=90)
    return image.size


def save_mask(mask: Image.Image, out: pathlib.Path, name: str, size: tuple[int, int]) -> str:
    """The mask as a 0/255 PNG at the saved image's size, under out/masks."""
    (out / "masks").mkdir(parents=True, exist_ok=True)
    mask.resize(size, Image.NEAREST).save(out / "masks" / name)
    return f"masks/{name}"


def where2place(root: pathlib.Path, out: pathlib.Path, max_side: int) -> list[dict]:
    """Its 100 free-space questions, without the answer-format sentence, as RoboPoint's training prompts are."""
    samples = []
    for line in (root / "point_questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        name = q["image"]
        size = save_image(root / "images" / name, out, name, max_side)
        stem = root / "masks" / pathlib.Path(name).stem
        # most masks are PNGs, a few only JPEGs, whose compression blurs the edge
        src = stem.with_suffix(".png") if stem.with_suffix(".png").exists() else stem.with_suffix(".jpg")
        mask = Image.fromarray((np.asarray(Image.open(src).convert("L")) > 127).astype(np.uint8) * 255)
        samples.append(
            {
                "image": f"images/{name}",
                "prompt": q["text"].split(" Your answer should be formatted")[0],
                "target": "",
                "mask": save_mask(mask, out, pathlib.Path(name).with_suffix(".png").name, size),
                "category": q["category"],
            }
        )
    return samples


def roborefit(shards: pathlib.Path, out: pathlib.Path, limit: int, max_side: int, seed: int) -> list[dict]:
    """Evenly spread expressions from the testB shards, each asking for points on the object it means; the mask is any
    nonzero channel of RoboRefIt's."""
    import io

    rows = pd.concat([pd.read_parquet(f) for f in sorted(shards.glob("testB-*.parquet"))], ignore_index=True)
    rows = rows.sample(n=min(limit, len(rows)), random_state=seed).sort_values("sample_id")
    samples = []
    for _, r in tqdm.tqdm(rows.iterrows(), total=len(rows)):
        name = f"{r['sample_id']}.jpg"
        image = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        image.thumbnail((max_side, max_side))
        (out / "images").mkdir(parents=True, exist_ok=True)
        image.save(out / "images" / name, quality=90)
        mask = Image.fromarray(
            (
                np.asarray(Image.open(io.BytesIO(r["mask"]["bytes"]))).reshape(r["height"], r["width"], -1).max(-1) > 0
            ).astype(np.uint8)
            * 255
        )
        samples.append(
            {
                "image": f"images/{name}",
                "prompt": f'Locate several points on the item meant by "{r["expression"]}".',
                "target": "",
                "mask": save_mask(mask, out, f"{r['sample_id']}.png", image.size),
                "category": r["category"],
            }
        )
    return samples


def loc_point(x: float, y: float, size: tuple[int, int]) -> str:
    return loc_box([x, y, x, y], size, Mix.input_px)[: len("<loc0000><loc0000>")]


def refcoco(annotations: pathlib.Path, images: pathlib.Path, out: pathlib.Path, limit: int, max_side: int, seed: int):
    """One sample per referring expression: "detect <expression>", the referred object's box."""
    rows = pd.read_parquet(annotations)
    samples = []
    rng = np.random.default_rng(seed)
    for _, r in tqdm.tqdm(rows.sample(frac=1, random_state=seed).iterrows(), total=len(rows)):
        info = json.loads(r["raw_image_info"])
        x, y, w, h = json.loads(r["raw_anns"])["bbox"]  # COCO's [x, y, width, height] in pixels
        src = images / info["file_name"]
        if not src.exists():  # train2017 names the same image by its id alone
            src = images / f"{int(r['image_id']):012d}.jpg"
        name = f"{r['image_id']}.jpg"
        size = (
            Image.open(out / "images" / name).size
            if (out / "images" / name).exists()
            else save_image(src, out, name, max_side)
        )
        box = [x / info["width"], y / info["height"], (x + w) / info["width"], (y + h) / info["height"]]
        sentence = r["sentences"][rng.integers(len(r["sentences"]))]["sent"]
        samples.append(
            {"image": f"images/{name}", "prompt": f"detect {sentence}", "target": loc_box(box, size, Mix.input_px)}
        )
        if len(samples) >= limit:
            break
    return samples


def robopoint(
    questions: pathlib.Path, images: pathlib.Path, out: pathlib.Path, kind: str, limit: int, max_side: int, seed: int
):
    """RoboPoint's questions of one kind, without the answer-format sentence, and its points (at most eight)."""
    rows = [r for r in json.loads(questions.read_text()) if r["id"].startswith(f"{kind}/")]
    np.random.default_rng(seed).shuffle(rows)
    samples = []
    for r in tqdm.tqdm(rows[:limit]):
        question = r["conversations"][0]["value"].replace("<image>\n", "").split(" Your answer should be formatted")[0]
        points = [
            (float(a), float(b)) for a, b in re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", r["conversations"][1]["value"])
        ]
        if not points:
            continue
        name = (
            pathlib.Path(r["image"].replace("/", "_")).with_suffix(".jpg").name
        )  # JPEG: RoboPoint's PNGs are 5x larger
        size = save_image(images / r["image"], out, name, max_side)
        samples.append(
            {
                "image": f"images/{name}",
                "prompt": question,
                "target": "".join(loc_point(x, y, size) for x, y in points[:8]),
            }
        )
    return samples


def main(
    source: str,
    annotations: pathlib.Path,
    images: pathlib.Path,
    out: pathlib.Path,
    kind: str = "object_ref",
    limit: int = 50_000,
    max_side: int = 448,
    seed: int = 0,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if source == "refcoco":
        samples = refcoco(annotations, images, out, limit, max_side, seed)
    elif source == "robopoint":
        samples = robopoint(annotations, images, out, kind, limit, max_side, seed)
    elif source == "where2place":
        samples = where2place(annotations, out, max_side)
    elif source == "roborefit":
        samples = roborefit(annotations, out, limit, max_side, seed)
    else:
        raise SystemExit(f"unknown source {source}: refcoco, robopoint, where2place or roborefit")
    (out / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(f"{len(samples)} samples -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
