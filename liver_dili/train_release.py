"""Train a reproducible preclinical CLI release with group cross-fit calibration."""

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .benchmark import Dataset, SEED, chemistry, choose, inner_views, permitted
from .splits import digest, nearest, receipt


def train(args):
    plan = args.splits / "murcko-0"
    train_rows, held = [pd.read_parquet(plan / f"{name}.parquet") for name in ("train", "query")]
    ds = Dataset(args.embeddings, args.assays)
    ds.assays = permitted(ds.assays, held)
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(
        train_rows, train_rows.label, train_rows.group))
    candidates = []
    path_head = None
    for mode in ("structure", "predicted_pathways"):
        parts = inner_views(ds, train_rows, mode, "logistic", folds)
        for method in ("logistic", "lightgbm"):
            _, fit = choose(ds, train_rows, held, mode, method, folds, parts)
            if mode == "predicted_pathways":
                path_head = fit["path_head"]
            candidates.append((fit["inner_ap"], mode, fit))
    ap, mode, selection = max(candidates, key=lambda item: item[0])
    refs = [chemistry(s)[0] for s in train_rows.smiles]
    group = train_rows.group.to_numpy()
    out_group = []
    for i, current in enumerate(refs):
        out_group.append(nearest([reference for j, reference in enumerate(refs) if group[j] != group[i]], [current])[0])
    cutoff = float(np.quantile(out_group, .05))
    if path_head is None:
        raise ValueError("Missing preregistered HepaRG/PHH pathway predictor")
    bundle = {"classifier": selection["model"], "method": selection["method"], "mode": mode,
              "path_head": path_head, "calibration": selection["calibration"],
              "threshold": selection["threshold"], "coefficients": selection["coefficients"],
              "references": refs, "ood_threshold": cutoff}
    args.out.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, args.out / "checkpoint.joblib", compress=3)
    modelfiles = ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "pytorch_model.bin")
    for name in modelfiles:
        target = args.out / "chemberta" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(args.encoder / name, target)
    receipt(args.out / "contract.json", dict(version="liver-dili-v1", input="canonicalizable small-molecule SMILES",
        label="FDA DILIrank 2.0 vMost or vLess concern vs vNo concern",
        calibration="group cross-fit Platt probability and 90% group bootstrap calibration-parameter interval",
        prevalence=float(train_rows.label.mean()), fit_n=len(train_rows), mode=mode, method=selection["method"],
        threshold=selection["threshold"], ood_threshold=cutoff, selected_inner_auprc=ap,
        pathway_platforms=sorted(path_head.models), clinical_exposure_inputs=False,
        release_selection="Murcko fold 0 fit roster, model selected on internal group OOF average precision",
        external_holdout_labels_read=False))
    paths = sorted(p for p in args.out.rglob("*") if p.is_file())
    receipt(args.out / "receipt.json", dict(assets={str(p.relative_to(args.out)): digest(p) for p in paths},
        split_receipt=digest(args.splits / "receipt.json"), assay_receipt=digest(args.assays / "receipt.json"),
        embedding_file=digest(args.embeddings), code_sha256=digest(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("splits", "assays", "embeddings", "encoder", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    train(parser.parse_args())
