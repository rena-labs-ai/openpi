#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# pi0.5 training through cluv, on whichever cluster and allocation starts it first:
#   cluv submit first scripts/cluv_train.sh [sbatch flags] -- CONFIG EXP_NAME [train.py flags]
# Each cluster holds the data in $SCRATCH/rena-data (the LeRobot dataset in lerobot_home, openpi's cache with pi05_base
# and the tokenizer in openpi_cache, the pure-Python autolabel_prompts and rena_task_grammar in python) and the
# checkpoints in $SCRATCH/openpi-checkpoints. The job shards the model over all the GPUs it got, resumes its own
# checkpoints, so `cluv submit --chunking` can split it into short jobs, and writes the config's norm stats first, from
# whichever dataset the config names.
set -euo pipefail
config=$1 exp=$2
shift 2
data=$SCRATCH/rena-data
ckpt=$SCRATCH/openpi-checkpoints
export HF_LEROBOT_HOME=$data/lerobot_home OPENPI_DATA_HOME=$data/openpi_cache HF_HOME=$data/hf_cache PYTHONPATH=$data/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1
module load ffmpeg 2>/dev/null || true # the video decoder's libraries, on the DRAC clusters
py=(uv run --no-sync python)

"${py[@]}" scripts/compute_rena_norm_stats.py --config-name "$config"
resume=()
if ls "$ckpt/$config/$exp" 2>/dev/null | grep -qE '^[0-9]+$'; then resume=(--resume); fi
gpus=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${py[@]}" scripts/train.py "$config" --exp-name="$exp" --fsdp-devices="$gpus" --checkpoint-base-dir="$ckpt" \
    "${resume[@]}" "$@"
