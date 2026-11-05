#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in gears|scgpt) mode=$1;; *) exit 2;; esac
mkdir -p .venvs
venv="$PWD/.venvs/$mode"
if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv --copies "$venv"
fi
export PIP_CACHE_DIR="$PWD/.venvs/wheel-cache"
"$venv/bin/python" -m pip install --disable-pip-version-check 'torch==2.5.1+cu124' --extra-index-url https://download.pytorch.org/whl/cu124
"$venv/bin/python" -m pip install --disable-pip-version-check -r envs/requirements.txt
"$venv/bin/python" -m pip check
"$venv/bin/python" -m pip freeze > ".venvs/$mode.freeze.txt"
