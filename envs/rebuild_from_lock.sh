#!/usr/bin/env bash
# Obtain pinned official third_party sources and apply the recorded patch first.
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
model="${1:?Pass scgpt or geneformer}"
case "$model" in
  scgpt) source_path=third_party/scGPT ;;
  geneformer) source_path=third_party/Geneformer ;;
  *) exit 2 ;;
esac
test -f "$source_path/setup.py" || test -f "$source_path/pyproject.toml"
/usr/local/anaconda3/bin/python -m venv ".venvs/vcell-$model"
P=".venvs/vcell-$model/bin/python"
export PIP_CACHE_DIR=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/pip
# --no-deps preserves the fully enumerated tested inference set and excludes optional training extras.
"$P" -m pip install --no-deps -r "envs/vcell-$model.requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu124
"$P" -m pip install --no-deps -e "$source_path"
