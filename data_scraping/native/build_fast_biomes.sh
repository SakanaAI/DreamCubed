#!/usr/bin/env bash
set -euo pipefail

# Build the fast cubiomes backend used by seed_utils.py
#
# This produces: native/libdream_cubiomes.so
#
# Notes:
# - We intentionally avoid -O3: cubiomes has UB that can miscompile under -O3.
# - We also disable strict-aliasing and enable wrapv to preserve deterministic results.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_SO="${ROOT_DIR}/native/libdream_cubiomes.so"
SRC_C="${ROOT_DIR}/native/dream_cubiomes.c"

echo "Building ${OUT_SO} ..."
gcc -O2 -std=c99 -D_DEFAULT_SOURCE -fno-strict-aliasing -fwrapv -fPIC -shared \
  -o "${OUT_SO}" "${SRC_C}" -lm

echo "Done."

