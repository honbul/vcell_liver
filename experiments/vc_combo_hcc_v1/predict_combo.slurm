#!/usr/bin/env bash
#SBATCH --job-name=combo-v1-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
work=$root/prepared/combo-v1
out=$root/results/combo-v1
emb=$root/fm_store_v1/combo-v1
hallmark=/mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/h.all.v2026.1.Hs.symbols.gmt
python=$root/runtime/bin/python
cd "$root/code/vc_combo_hcc_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export PYTHONPATH="$root/code/vc_fm_bench_v1:$root/code/vc_combo_hcc_v1"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

for model in ecfp4 molformer; do
    if [[ -f "$emb/$model/receipt.json" ]]; then
        continue
    fi
    device=cpu
    args=()
    if [[ $model == molformer ]]; then
        device=cuda
        args=(--checkpoint "$root/checkpoints/molformer-transformers4")
    fi
    "$python" -m vc_fm_bench.embed --model "$model" --input "$work/context/molecules.tsv" \
        --output "$emb/$model" --device "$device" "${args[@]}"
done

sources=$root/prepared/scrna-v1/source-panels-v1
limore=${root%/nextgen_5track}/vc_build_v1/function/limore-v1
if [[ ! -f "$work/virtual-screen/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.screen register --context "$work/context" \
        --dili "$root/prepared/track6-v2" --organoids "$sources" \
        --limore-rna "$root/prepared/limore-rna-v1/rna.parquet" --limore-assays "$limore/rows.parquet" \
        --limore-receipt "$limore/receipt.json" --output "$work/virtual-screen"
fi

for axis in lco_hcc lpo ldo; do
    panels=("$work/panels/$axis"-*)
    for panel in "${panels[@]}"; do
        name=${panel##*/}
        if [[ ! -f "$out/predictions/$name/receipt.json" ]]; then
            extra=()
            if [[ $axis == lco_hcc ]]; then
                extra=(--screen "$work/virtual-screen")
            fi
            "$python" -m vc_combo_hcc.predict --panel "$panel" --context "$work/context" \
                --embeddings "$emb" --hallmark "$hallmark" --output "$out/predictions/$name" "${extra[@]}"
        fi
    done
    if [[ $axis == lco_hcc && ! -f "$out/screen/receipt.json" ]]; then
        "$python" -m vc_combo_hcc.screen rank --predictions "$out/predictions/lco_hcc-0" \
            --query "$work/virtual-screen" --organoids "$sources" --output "$out/screen"
    fi
    if [[ ! -f "$out/scores/$axis/receipt.json" ]]; then
        "$python" -m vc_combo_hcc.score --axes "$axis" --panels "$work/panels" \
            --predictions "$out/predictions" --output "$out/scores/$axis"
    fi
done
if [[ ! -f "$out/scores/all/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.score --panels "$work/panels" \
        --predictions "$out/predictions" --output "$out/scores/all"
fi
