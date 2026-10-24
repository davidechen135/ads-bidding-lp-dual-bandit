#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Locate python venv
if [ -f "${PROJECT_DIR}/../.venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/../.venv/bin/python"
    VENV_PYTEST="${PROJECT_DIR}/../.venv/bin/pytest"
elif [ -f "${PROJECT_DIR}/../../.venv/bin/python" ]; then
    VENV_PYTHON="${PROJECT_DIR}/../../.venv/bin/python"
    VENV_PYTEST="${PROJECT_DIR}/../../.venv/bin/pytest"
else
    VENV_PYTHON="$(which python3)"
    VENV_PYTEST="$(which pytest)"
fi

echo "=== Running: Ads Bidding LP Dual & Bandits ==="
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "[1/2] Running Unit & Integration Tests..."
"${VENV_PYTEST}" tests -v

echo "[2/2] Running Benchmark Experiments..."
"${VENV_PYTHON}" src/experiments.py

echo "=== Ads Bidding LP Dual Completed Successfully! ==="
