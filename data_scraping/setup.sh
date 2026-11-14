#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -m pip install -r requirements.txt
python -m pip install -e Pyubiomes
