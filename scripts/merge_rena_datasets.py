"""One rena LeRobot v2.1 dataset from several rena-autolabel datasets of the same schema, episodes renumbered in order:
the parquet files rewritten, the videos hard-linked (copied across filesystems), tasks merged by text. Each episode's
labels in meta/autolabel.jsonl gain argus's timeline (argus_labels, which knowledge insulation's annotation reads) from
--timelines, one {"source_key", "timeline"} per line, for the recordings it has. Run it again to redo the output.

    uv run scripts/merge_rena_datasets.py --srcs rena/rena_autolabel_v3 rena/rena_autolabel_dep1 \
        --dst rena/rena_autolabel_v4 --timelines timelines.jsonl
"""

import json
import os
import pathlib
import shutil

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


def jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def link(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def column_stats(v: np.ndarray) -> dict:
    v = v.astype(np.float64).reshape(len(v), -1)
    return {
        "min": v.min(0).tolist(),
        "max": v.max(0).tolist(),
        "mean": v.mean(0).tolist(),
        "std": v.std(0).tolist(),
        "count": [len(v)],
    }


def main(srcs: list[str], dst: str, timelines: pathlib.Path | None = None) -> None:
    out = (HF_LEROBOT_HOME / dst).resolve()
    for src in ((HF_LEROBOT_HOME / s).resolve() for s in srcs):
        if out == src or out in src.parents or src in out.parents:
            raise SystemExit(f"{dst} overlaps the source {src}: redoing the output would delete it")
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    timeline = {} if timelines is None else {r["source_key"]: r["timeline"] for r in jsonl(timelines)}
    infos = [json.loads((HF_LEROBOT_HOME / s / "meta/info.json").read_text()) for s in srcs]
    layout = ("features", "chunks_size", "data_path", "video_path", "fps")
    assert all(i[k] == infos[0][k] for i in infos for k in layout), f"the datasets differ in one of {layout}"
    info = infos[0]
    data_path, video_path, chunk = info["data_path"], info["video_path"], info["chunks_size"]
    videos = [k for k, f in info["features"].items() if f["dtype"] == "video"]

    tasks: dict[str, int] = {}
    episodes, stats, labels = [], [], []
    n_ep = n_frames = 0
    for src_name in srcs:
        src = HF_LEROBOT_HOME / src_name
        src_tasks = {t["task_index"]: t["task"] for t in jsonl(src / "meta/tasks.jsonl")}
        src_stats = {s["episode_index"]: s["stats"] for s in jsonl(src / "meta/episodes_stats.jsonl")}
        src_labels = {r["episode_index"]: r for r in jsonl(src / "meta/autolabel.jsonl")}
        for ep in jsonl(src / "meta/episodes.jsonl"):
            old, new = ep["episode_index"], n_ep
            fmt = {"episode_chunk": old // chunk, "episode_index": old}
            new_fmt = {"episode_chunk": new // chunk, "episode_index": new}
            table = pq.read_table(src / data_path.format(**fmt))
            remap = np.array([tasks.setdefault(src_tasks[i], len(tasks)) for i in range(max(src_tasks) + 1)])
            cols = {
                "episode_index": np.full(table.num_rows, new, np.int64),
                "index": np.arange(n_frames, n_frames + table.num_rows, dtype=np.int64),
                "task_index": remap[table.column("task_index").to_numpy()].astype(np.int64),
            }
            ep_stats = dict(src_stats[old])
            for name, values in cols.items():
                i = table.schema.get_field_index(name)
                table = table.set_column(i, table.schema.field(i), pa.array(values, table.schema.field(i).type))
                ep_stats[name] = column_stats(values)
            dst_file = out / data_path.format(**new_fmt)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst_file)
            for key in videos:
                link(src / video_path.format(video_key=key, **fmt), out / video_path.format(video_key=key, **new_fmt))
            episodes.append({**ep, "episode_index": new})
            stats.append({"episode_index": new, "stats": ep_stats})
            rec = {**src_labels[old], "episode_index": new}
            if rec["source_key"] in timeline:
                rec["argus_labels"] = {"timeline": timeline[rec["source_key"]]}
            labels.append(rec)
            n_ep += 1
            n_frames += table.num_rows

    def write(name: str, rows: list[dict]) -> None:
        (out / "meta" / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    write("episodes.jsonl", episodes)
    write("episodes_stats.jsonl", stats)
    write("autolabel.jsonl", labels)
    write("tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])
    info |= {
        "total_episodes": n_ep,
        "total_frames": n_frames,
        "total_tasks": len(tasks),
        "total_videos": n_ep * len(videos),
        "total_chunks": (n_ep - 1) // chunk + 1,
        "splits": {"train": f"0:{n_ep}"},
    }
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(
        f"{n_ep} episodes, {n_frames} frames, {len(tasks)} tasks, {sum(r.get('argus_labels') is not None for r in labels)}"
        f" with argus's timeline -> {out}"
    )


if __name__ == "__main__":
    tyro.cli(main)
