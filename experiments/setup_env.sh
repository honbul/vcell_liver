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
if [[ "$mode" == gears ]]; then
  "$venv/bin/python" -m pip install --disable-pip-version-check 'torch==2.5.1+cu124' --extra-index-url https://download.pytorch.org/whl/cu124
else
  "$venv/bin/python" -m pip install --disable-pip-version-check 'torch==1.13.1+cu117' 'torchtext==0.14.1' --extra-index-url https://download.pytorch.org/whl/cu117
  "$venv/bin/python" -m pip install 'transformers==4.38.2' 'ipython==8.26.0' 'wandb==0.17.9'
fi
"$venv/bin/python" -m pip install --disable-pip-version-check -r envs/requirements.txt
"$venv/bin/python" -m pip check
"$venv/bin/python" -m pip freeze > ".venvs/$mode.freeze.txt"
