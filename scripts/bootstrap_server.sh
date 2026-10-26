#!/usr/bin/env bash
# Rebuild only this project's isolated environments and assets, at the documented paths.
set -euo pipefail
ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
DATA=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver
cd "$ROOT"
mkdir -p logs third_party "$DATA"/{annotation,checkpoints,cache,processed,subsets,features}
if [ ! -d third_party/scGPT/.git ]; then
    git clone https://github.com/bowang-lab/scGPT.git third_party/scGPT
    git -C third_party/scGPT checkout --detach cebd6fae655b9c585a4807daa3ac31bb764f06b4
fi
test "$(git -C third_party/scGPT rev-parse HEAD)" = cebd6fae655b9c585a4807daa3ac31bb764f06b4
if git -C third_party/scGPT apply --check "$ROOT/patches/scgpt-linear-vocab-insertion.patch"; then
    git -C third_party/scGPT apply "$ROOT/patches/scgpt-linear-vocab-insertion.patch"
else
    git -C third_party/scGPT apply --reverse --check "$ROOT/patches/scgpt-linear-vocab-insertion.patch"
fi
/usr/local/anaconda3/bin/python scripts/download_assets.py
if [ ! -d .venvs/vcell-scgpt ]; then
    /usr/local/anaconda3/bin/python -m venv .venvs/vcell-scgpt
fi
if [ -s envs/vcell-scgpt.requirements.txt ]; then
    bash envs/rebuild_from_lock.sh scgpt
else
    bash envs/install_scgpt.sh
fi
source envs/activate.sh scgpt
python scripts/download_scgpt.py
python -m ipykernel install --prefix "$ROOT/.jupyter" --name vcell-scgpt --display-name vcell-scgpt
if [ -s envs/vcell-geneformer.requirements.txt ]; then
    bash envs/rebuild_from_lock.sh geneformer
else
    bash envs/install_geneformer.sh
fi
if [ ! -e "$ROOT/data" ]; then ln -s "$DATA" "$ROOT/data"; fi
if [ ! -e "$ROOT/checkpoints" ]; then ln -s "$DATA/checkpoints" "$ROOT/checkpoints"; fi
if [ ! -e "$DATA/raw" ]; then ln -s /mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614 "$DATA/raw"; fi
bash scripts/download_hcc.sh
python scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml
python scripts/00_inspect_environment.py --out reports/environment.json
