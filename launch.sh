#!/usr/bin/env bash
# Usage: bash launch.sh <num_gpus> <config.json> [trainer arguments]
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch_training.sh" train_discrete.py "$@"
