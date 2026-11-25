"""Vision-language datasets for co-training (training.vl_data), from RefCOCO's referring expressions and RoboPoint's
object and free-space references: an image, a prompt, and a target of PaliGemma <loc> tokens over the image padded to
the model's square input (autolabel_prompts.loc_box), four per box and two (y, x) per point.

    uv run scripts/prepare_vl_data.py refcoco ANNOTATIONS.parquet COCO_IMAGES_DIR OUT [--limit N]
    uv run scripts/prepare_vl_data.py robopoint robopoint_1432k.json IMAGES_DIR OUT --kind object_ref [--limit N]

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
        samples.append({"image": f"images/{name}", "prompt": f"detect {sentence}", "target": loc_box(box, size, Mix.input_px)})
        if len(samples) >= limit:
            break
    return samples


def robopoint(questions: pathlib.Path, images: pathlib.Path, out: pathlib.Path, kind: str, limit: int, max_side: int,
              seed: int):
    """RoboPoint's questions of one kind, without the answer-format sentence, and its points (at most eight)."""
    rows = [r for r in json.loads(questions.read_text()) if r["id"].startswith(f"{kind}/")]
    np.random.default_rng(seed).shuffle(rows)
    samples = []
    for r in tqdm.tqdm(rows[:limit]):
        question = r["conversations"][0]["value"].replace("<image>\n", "").split(" Your answer should be formatted")[0]
        points = [(float(a), float(b)) for a, b in re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", r["conversations"][1]["value"])]
        if not points:
            continue
        name = r["image"].replace("/", "_")
        size = save_image(images / r["image"], out, name, max_side)
        samples.append({"image": f"images/{name}", "prompt": question,
                        "target": "".join(loc_point(x, y, size) for x, y in points[:8])})
    return samples


def main(source: str, annotations: pathlib.Path, images: pathlib.Path, out: pathlib.Path, kind: str = "object_ref",
         limit: int = 50_000, max_side: int = 448, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if source == "refcoco":
        samples = refcoco(annotations, images, out, limit, max_side, seed)
    elif source == "robopoint":
        samples = robopoint(annotations, images, out, kind, limit, max_side, seed)
    else:
        raise SystemExit(f"unknown source {source}: refcoco or robopoint")
    (out / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(f"{len(samples)} samples -> {out}")


if __name__ == "__main__":
    tyro.cli(main)
