#!/bin/bash

# usage: bash launch_fid3d.sh <num_gpus> <config_file> [extra args for train_fid3d.py]

if [[ $# -lt 2 ]]; then
  echo "usage: bash launch_fid3d.sh <num_gpus> <config_file> [extra args]" >&2
  exit 2
fi

args=()
for arg in "$@"; do
  args+=("$arg")
done

config="accelerate_configs/multigpu.yaml"

nproc=${args[0]}
arg2=${args[1]}

prefix="training_configs/"
if [[ "$arg2" != $prefix* ]]; then
    arg2="${prefix}${arg2}"
fi

suffix=".json"
if [[ "$arg2" != *$suffix ]]; then
    arg2="${arg2}${suffix}"
fi

extra_args=("${args[@]:2}")

RND_PORT=$(($RANDOM % 1000 + 12000))
echo "assigned random port: $RND_PORT"

accelerate launch --num_processes "$nproc" \
  --main_process_port "$RND_PORT" \
  --config_file "$config" \
  train_fid3d.py --config "$arg2" "${extra_args[@]}"
