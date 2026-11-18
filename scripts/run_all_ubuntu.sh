#!/usr/bin/env bash
# Reproducible one-command setup & experiments on Ubuntu / WSL Ubuntu.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROFILE="${1:-full}"
case "$PROFILE" in full|quick) ;; *) echo 'Usage: bash scripts/run_all_ubuntu.sh [full|quick]' >&2; exit 2;; esac
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plots]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q
python scripts/run_paper_protocol.py --profile "$PROFILE" --include-baselines --fail-fast
