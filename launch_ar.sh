#!/usr/bin/env bash
# Usage: bash launch_ar.sh <num_gpus> <config.json> [trainer arguments]
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch_training.sh" train_autoregressive.py "$@"
