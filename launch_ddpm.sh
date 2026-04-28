#!/bin/bash


# usage: bash launch.sh <num_gpus> <config_file> [extra args for train_discrete_conditional.py]

#!/bin/bash
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

accelerate launch --num_processes "$nproc" \
  --main_process_port "$RND_PORT" \
  --config_file "$config" \
  train_conditional_ddpm.py --config "$arg2" "${extra_args[@]}"

