"""Refit CellHit and Chemprop model families on registered Track 6 panels."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_record, new_directory, verify_record, write_json
from .benchmark import by_source_scale, choose_validation, load_profiles, panel_input, validation_score
from .features import rank_block, read_vectors


def cellhit_fit(fit, query, rna, depth, trees, seed):
    from xgboost import XGBRegressor

    cells = rank_block(rna, fit.cell_id, "rank")
    fit_x = cells.loc[fit.cell_id].to_numpy()
    test_x = cells.loc[query.cell_id].to_numpy()
    y = by_source_scale(fit)
    predictions = np.full(len(query), y.mean())
    conditions = fit.groupby("drug_id").indices
    for drug, external in query.groupby("drug_id").indices.items():
        train = conditions.get(drug)
        if train is None:
            continue
        if len(train) < 20:
            predictions[external] = y[train].mean()
            continue
        model = XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=.05,
                             min_child_weight=5, colsample_bytree=.65, subsample=.85,
                             reg_lambda=5, n_jobs=2, tree_method="hist", random_state=seed)
        model.fit(fit_x[train], y[train])
        predictions[external] = model.predict(test_x[external])
    return predictions


def predict(a):
    panel_path = a.panel / "receipt.json"
    contract = json.loads(panel_path.read_text())
    fit, query = [panel_input(a.panel, name, contract) for name in ("train", "query")]
    if "outcome" in query or not query.id.is_unique or not fit.id.is_unique:
        raise ValueError("Unexpected fit or held-out schema")
    covered = set.intersection(*(set(read_vectors(a.chem / name).index)
                                 for name in ("ecfp4", "chemberta2", "molformer", "unimol", "clamp")))
    fit = fit[fit.drug_id.isin(covered)].copy()
    original = len(query)
    query = query[query.drug_id.isin(covered)].copy()
    classification = contract["task"] == "classification"
    feature_paths = [a.chem / name / "receipt.json" for name in
                     ("ecfp4", "chemberta2", "molformer", "unimol", "clamp")]
    if classification:
        from .dmpnn import chemistry_predictor

        manifest = json.loads((a.prepared / "receipt.json").read_text())
        structures = a.prepared / "molecules.tsv"
        verify_record(structures, manifest["files"]["molecules.tsv"])
        model_fit = chemistry_predictor(pd.read_csv(structures, sep="\t"))
        configs = [(64, 3, 18), (96, 4, 28)]
        name = "chemprop_dmpnn"
        feature_paths.append(structures)
        libraries = {"torch": version("torch"), "rdkit": version("rdkit")}
    else:
        if not a.rna or not a.cell or "cell-cold" not in a.panel.name:
            raise ValueError("The CellHit family estimates responses for known drugs in unseen cells")
        rna = load_profiles(a.rna)
        matrices = [read_vectors(a.cell / name) for name in ("scgpt", "geneformer")]
        cells = set(rna.index) & set(matrices[0].index) & set(matrices[1].index)
        fit = fit[fit.cell_id.isin(cells)].copy()
        query = query[query.cell_id.isin(cells)].copy()
        configs = [(3, 100), (5, 160)]
        name = "cellhit_xgboost"

        def model_fit(rows, queries, config, seed):
            return cellhit_fit(rows, queries, rna, *config, seed)

        feature_paths.extend([*a.rna, *(a.cell / m / "receipt.json" for m in ("scgpt", "geneformer"))])
        libraries = {"xgboost-cpu": version("xgboost-cpu")}
    inner, valid = choose_validation(fit, contract["split_group"])
    selection = []
    for config in configs:
        prediction = model_fit(inner, valid, config, a.seed)
        value = validation_score(valid.outcome.to_numpy(), prediction, valid, classification, a.panel.name)
        selection.append({"config": config, "validation": value})
        print(f"{a.panel.name} {name} {config} val={value:.5f}", flush=True)
    chosen = max(selection, key=lambda row: row["validation"])["config"]
    query[name] = model_fit(fit, query, chosen, a.seed)
    if not np.isfinite(query[name]).all():
        raise ValueError(f"Nonfinite {name} predictions")
    with new_directory(a.output) as stage:
        file = stage / "predictions.parquet"
        query.to_parquet(file, index=False)
        write_json(stage / "receipt.json", {
            "predictions": file_record(file), "panel": file_record(panel_path), "columns": [name],
            "seed": a.seed, "targets": len(query), "original_targets": original, "fit_rows": len(fit),
            "method": name, "chosen": chosen, "candidates": selection, "runtime": libraries,
            "feature_inputs": {str(p): file_record(p) for p in feature_paths},
        })


def main():
    q = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "chem", "prepared", "output"):
        q.add_argument("--" + name, type=Path, required=True)
    q.add_argument("--seed", type=int, default=20261122)
    q.add_argument("--cell", type=Path)
    q.add_argument("--rna", type=Path, nargs="+")
    predict(q.parse_args())


if __name__ == "__main__":
    main()
