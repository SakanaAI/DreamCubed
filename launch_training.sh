#!/usr/bin/env bash
# Shared launcher. One GPU runs directly; multiple GPUs use Accelerate.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: launch_training.sh <trainer.py> <num_gpus> <config.json> [trainer arguments]" >&2
  exit 2
fi
trainer=$1
nproc=$2
requested=$3
shift 3
if [[ ! "$nproc" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_gpus must be a positive integer" >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
resolved=""
for candidate in "$requested" "$requested.json" "$repo_dir/$requested" "$repo_dir/training_configs/$requested" "$repo_dir/training_configs/$requested.json"; do
  if [[ -f "$candidate" ]]; then
    resolved=$(cd -- "$(dirname -- "$candidate")" && pwd)/$(basename -- "$candidate")
    break
  fi
done
if [[ -z "$resolved" ]]; then
  echo "Configuration file not found: $requested" >&2
  exit 2
fi
cd -- "$repo_dir"
python_bin="${DREAMCUBED_PYTHON:-${AR_PYTHON:-python}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ "$nproc" == 1 ]]; then
  exec "$python_bin" "$trainer" --config "$resolved" "$@"
fi
exec "$python_bin" -m accelerate.commands.launch --num_processes "$nproc" \
  --main_process_port "$((RANDOM % 1000 + 12000))" \
  --config_file "$repo_dir/accelerate_configs/multigpu.yaml" \
  "$trainer" --config "$resolved" "$@"
