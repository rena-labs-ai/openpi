---
name: cluv-training
description: Launch, smoke-test or debug rena pi0.5 training on Mila and the DRAC clusters with cluv. Use when asked to queue, race, resume or check a training run, or to add a cluster.
---

# Training with cluv

cluv submits the same job to every configured cluster and allocation, keeps the first one that starts, and cancels
the rest. The config is `[tool.cluv]` in `pyproject.toml`. The job script is `scripts/cluv_train.sh`.

## Before anything

- Work from the `rena-pi05-cluv` branch, which tracks the `rena-labs` remote (`git@github.com:rena-labs-ai/openpi`).
  Never set it to track `origin` (montrealrobotics).
- `cluv sync` and `cluv submit` run `git push` and then clone that commit on every cluster. Commit first, because a
  dirty tree is refused.
- Clusters need a live SSH session. Logging in takes multi-factor authentication, so ask the user to run `cluv login`.
  Check sessions with `cluv status`.

## Commands

```
cluv submit first --chunking 12 --time=24:00:00 -- CONFIG EXP_NAME [train.py flags]   # race all clusters
cluv submit nibi -- CONFIG EXP_NAME ...       # one cluster, racing its accounts
cluv status                                    # GPUs free, cluv jobs
```

- A smoke test is: `--time=1:00:00 -- CONFIG smoke_X --num-train-steps 30 --log-interval 5 --no-wandb-enabled`.
- `--chunking N` splits the run into consecutive jobs of N hours. The script resumes from its own checkpoints, so
  short jobs reach DRAC's faster partitions. Resuming across chunks has not been tested yet.

## On each cluster

| What | Where |
|---|---|
| Clone and venv | `$SCRATCH/repos/openpi` |
| Data | `$SCRATCH/rena-data/{lerobot_home, openpi_cache, python}` |
| Checkpoints | `$SCRATCH/openpi-checkpoints/CONFIG/EXP_NAME` |
| Logs | `$SCRATCH/cluv/openpi/<cluster>_<jobid>/slurm-<jobid>.out` |

- `openpi_cache` holds pi05_base and the tokenizer.
- `python` holds `autolabel_prompts` and `rena_task_grammar`, which are on `PYTHONPATH`.
- The script computes a config's norm stats on first use.

## GPUs

- Full fine-tuning needs 4 GPUs of 48 GB or 2 of 80 GB. The script sets `--fsdp-devices` to the GPUs it got.
- Allocations raced:
  - Mila: 4× L40S or 2× A100L.
  - Nibi and Fir: 2× H100 under def-kdarvish, rrg-bengioy-ad and def-gberseth.
  - Rorqual: 2× H100 under rrg-bengioy-ad, rrg-gberseth, def-gberseth and def-kdarvish.
- Speed: about 0.9 s/step on 2× H100 and 2.1 s/step on 4× L40S. A 30k-step run takes about 7.5 h or 17.5 h.
- W&B runs online on Mila, Nibi and Fir. Compute nodes elsewhere have no internet, so the script logs offline there.

## Known problems

- **Rebuilding a venv.** This happens after a lock change. The `evdev` package will not compile with DRAC's gcc
  unless it gets matching kernel headers. Mila's and Fir's home quotas are full. Run `uv sync` on the login node:
  `C_INCLUDE_PATH=$SCRATCH/evdev_inc` on Nibi and Rorqual, `$SCRATCH/evdev_inc_sys` on Fir, plus
  `UV_CACHE_DIR=$SCRATCH/.cache/uv`. cluv's own sync sets neither, and exits 0 even when it fails. Check that
  `.venv/bin/python -c "import jax, openpi"` works.
- **Data for a new cluster.** Stream it through the local machine so nothing lands on its disk:
  `ssh mila 'tar -chf - -C $SCRATCH rena-data' | ssh HOST 'bash -lc "tar -xf - -C \$SCRATCH"'`. It is 22 GB and takes
  about 18 min. Do not set cluv's `data_source` to a cluster, because cluv pulls that to the local machine first.
- **Clusters not in the race.**
  - Tamia: its SSH key is not on GitHub.
  - Killarney: SSH is refused.
  - Narval: its A100s have 40 GB, and full fine-tuning on them is untested.
