#!/bin/bash

# usage: bash launch_ar.sh <num_gpus> <config_file> [extra args for train_autoregressive.py]

set -euo pipefail
# Four trainer processes plus two loader workers per rank share the Slurm CPU
# allocation. Avoid each process creating a full-node OpenMP/MKL thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Use the activated cluster environment's Python, including uv-created venvs.
# AR_PYTHON can explicitly select /absolute/path/to/d3/bin/python.
python_bin="${AR_PYTHON:-python}"

if [[ $# -lt 2 ]]; then
  echo "usage: bash launch_ar.sh <num_gpus> <config_file> [extra args]" >&2
  exit 2
fi

args=()
for arg in "$@"; do
  args+=("$arg")
done

config="accelerate_configs/multigpu.yaml"

nproc=${args[0]}
arg2=${args[1]}

# add prefix if not present
prefix="training_configs/"
if [[ "$arg2" != $prefix* ]]; then
    arg2="${prefix}${arg2}"
fi

# add .json suffix if not present
suffix=".json"
if [[ "$arg2" != *$suffix ]]; then
    arg2="${arg2}${suffix}"
fi

extra_args=("${args[@]:2}")

# generate random port
RND_PORT=$(($RANDOM % 1000 + 12000))
echo "assigned random port: $RND_PORT"

"$python_bin" -m accelerate.commands.launch --num_processes "$nproc" \
  --main_process_port "$RND_PORT" \
  --config_file "$config" \
  train_autoregressive.py --config "$arg2" "${extra_args[@]}"
