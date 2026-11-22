"""A copy of a rena LeRobot dataset whose stepped recordings have their joint commands interpolated
(training.rena_stepped): the parquet files rewritten, the meta and videos linked to the source's. Idempotent.

    uv run scripts/write_rena_interp_dataset.py --src-repo rena/rena_autolabel_v3 --dst-repo rena/rena_autolabel_v3_interp
"""

import os

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from openpi.training import rena_stepped


def main(src_repo: str, dst_repo: str) -> None:
    src, dst = HF_LEROBOT_HOME / src_repo, HF_LEROBOT_HOME / dst_repo
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("meta", "videos"):
        link = dst / name
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(src / name, dst))
    files = sorted((src / "data").rglob("episode_*.parquet"))
    n = 0
    for f in files:
        table = pq.read_table(f)
        col = table.schema.get_field_index("action")
        action = np.stack(table.column(col).to_numpy(zero_copy_only=False))
        if rena_stepped.stepped(action):
            n += 1
            action = rena_stepped.interpolate(action)
            table = table.set_column(col, table.schema.field(col), pa.array(list(action), table.schema.field(col).type))
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
    print(f"{n} of {len(files)} episodes interpolated -> {dst}")


if __name__ == "__main__":
    tyro.cli(main)
