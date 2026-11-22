"""Run a frozen V0+ v2 research sensitivity ensemble on local RNA profiles."""

import argparse
from pathlib import Path

import pandas as pd

from vc_build.v0plus_cli import read_expression, requested_panel

from .lean import CELL_MODALITIES, fm_paths, fm_read
from .release import load_release


def run(args):
    release, manifest = load_release(args.package)
    raw = read_expression(args.rna, args.scale, args.gene_map)
    expression, coverage, valid = release.align(raw)
    drugs = requested_panel(release.compounds, args.compounds)
    q = pd.DataFrame({"cell_id": raw.index}).merge(drugs, how="cross")
    q = q.join(coverage, on="cell_id")
    q = q.assign(status="predicted")
    q.loc[~q.cell_id.isin(valid[valid].index), "status"] = "insufficient_rna_coverage"
    q.loc[q.isomeric_smiles.isna(), "status"] = "compound_outside_training_panel"
    fm = {}
    if args.limore_fm:
        fm, _ = fm_paths(args.limore_fm, limore=True)
    if args.cell_bank:
        for name in CELL_MODALITIES:
            fm[name], _ = fm_read(args.cell_bank / name)
    mask = q.status.eq("predicted")
    for column in ("sensitivity_score", "drug_mean_anchor", "personalization_delta"):
        q[column] = float("nan")
    if mask.any():
        forecasts, anchor = release.predict(q.loc[mask], expression, fm)
        name = args.recipe or manifest["default"]
        q.loc[mask, "sensitivity_score"] = forecasts[name]
        q.loc[mask, "drug_mean_anchor"] = anchor
        q.loc[mask, "personalization_delta"] = forecasts[name]-anchor
    q["version"] = "V0+ v2"
    q["model"] = args.recipe or manifest["default"]
    q["cells_fm"] = q.cell_id.map({cell: sum(cell in fm[v].index for v in CELL_MODALITIES if v in fm)
                                 for cell in raw.index})
    q = q.rename(columns={"cell_id": "sample_id"})
    with args.out.open("x") as f:
        q.to_csv(f, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("package", "rna", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("compounds", "gene-map", "limore-fm", "cell-bank"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--recipe")
    parser.add_argument("--scale", choices=("bulk_tpm", "bulk_counts", "bulk_log2_tpm_plus_1"), required=True)
    run(parser.parse_args())
