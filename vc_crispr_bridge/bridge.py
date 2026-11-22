"""Compare chemical-only and genetic bridges on frozen LIMORE drug-cold panels."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .bridge_features import BridgeFeatures
from .bridge_panels import verify_roles
from .perturbseq import record
from .predict import checked, write_json
from .regression import regression_metrics, train_scalar, validation_value


PANEL_NAMES = tuple(f"{label}-{i}" for label in ("scaffold", "unseen_target") for i in range(5))
DATA_ARGUMENTS = ("panel", "output", "chemistry", "context", "hgnc", "chronos", "depmap_models",
                  "cells", "hepg2", "hepg2_predictions", "gene_panels")


def load_roles(root, name, features):
    path = root / name
    receipt = json.loads((path / "receipt.json").read_text())
    fit, valid, query = [pd.read_parquet(checked(path, f"{role}.parquet", receipt))
                         for role in ("fit", "validation", "query")]
    if (set(query) != {"cell_id", "cell_group", "source_id", "drug_id", "scaffold", "connectivity", "id"}
            or any(not r.id.is_unique for r in (fit, valid, query))):
        raise ValueError(f"Unexpected query or duplicate identity in {name}")
    for a, b in ((fit, valid), (fit, query), (valid, query)):
        verify_roles(a, b, features.targets, receipt["strict_target_cold"])
    return fit, valid, query


def predict_panel(fit, valid, query, a, features):
    columns = ["id", "cell_id", "cell_group", "drug_id"]
    outputs = query[columns].copy()
    blocks, has_measured = features.arrays([fit, valid, query])
    outputs["chronos_measured"] = has_measured[2]
    y = fit.outcome.to_numpy()
    vy = valid.outcome.to_numpy()
    units = valid.drug_id.to_numpy()
    general = float(y.mean())
    marker = "ABCB1"
    if marker not in features.context:
        raise ValueError("Missing preregistered ABCB1 marker")
    outputs["train_mean"] = general
    outputs["ABCB1"] = -features.context.loc[query.cell_id, marker].to_numpy()
    outputs["max_loss"] = blocks["dep"][2][:, 1]
    settings = {
        "train_mean": validation_value(np.full(len(vy), general), vy, units),
        "ABCB1": validation_value(-features.context.loc[valid.cell_id, marker].to_numpy(), vy, units),
        "max_loss": validation_value(blocks["dep"][1][:, 1], vy, units),
    }
    # Limit influence from sibling/derivative cultures within one parent group.
    counts = fit.groupby(["drug_id", "cell_group"]).id.transform("size").to_numpy()
    weights = (1 / counts).astype(np.float32)
    settings = {key: {"source_val_spearman": value[0], "source_val_rmse": -value[1]}
                for key, value in settings.items()}
    for method, representation in (
        ("ecfp4", ["basal", "ecfp4"]),
        ("molformer", ["basal", "molformer"]),
        ("genetic_observed", ["basal", "dep", "observed"]),
        ("joint_observed", ["basal", "molformer", "dep", "observed"]),
        ("joint_predicted", ["basal", "molformer", "dep", "predicted"]),
        ("joint_no_chronos", ["basal", "molformer", "observed"]),
        ("joint_no_expression_signature", ["basal", "molformer", "dep"]),
    ):
        data = [np.column_stack(xs).astype(np.float32)
                for xs in zip(*(blocks[key] for key in representation))]
        for estimator in ("ridge", "elasticnet", "lightgbm", "mlp"):
            model = f"{method}/{estimator}"
            best = train_scalar(estimator, data, y, vy, units, a.device, a.seed, weight=weights)
            outputs[model] = best.predictions[0].astype(np.float32)
            settings[model] = {
                "source_val_spearman": best.score[0], "source_val_rmse": -best.score[1],
                "hyperparameters": best.settings,
            }
    return outputs, settings


def predict(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {arg: getattr(a, arg) for arg in DATA_ARGUMENTS if arg not in ("panel", "output", "chemistry")}
    fingerprints = {k: record(v / "receipt.json" if v.is_dir() else v) for k, v in paths.items()}
    manifest = a.panel / "receipt.json"
    receipt = json.loads(manifest.read_text())
    claims = pd.read_csv(checked(a.panel, "targets.tsv", receipt), sep="\t")
    features = BridgeFeatures(a, claims)
    registration, scores = {}, {}
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        for name in PANEL_NAMES:
            print(f"Fit {name}", flush=True)
            manifest_key = f"{name}/receipt.json"
            registration[name] = record(checked(a.panel, manifest_key, receipt))
            predictions, settings = predict_panel(*load_roles(a.panel, name, features), a, features)
            predictions.to_parquet(staged / f"{name}.parquet", index=False)
            scores[name] = settings
        write_json(staged / "receipt.json", {
            "panel": record(manifest), "panel_receipts": registration, "models": scores,
            "inputs": fingerprints, "target_registry": receipt["target_registry"],
            "training_only_selector": "within-drug Spearman and RMSE on source validation",
            "measured_chronos": "exact registry DepMap model join; otherwise non-Liver genetic assay median",
            "unseen_target_definition": "None of the registered test drug targets occur in the pharmacological fit or validation label sets.",
            "chemistry_store": {key: record(a.chemistry / key / "receipt.json") for key in ("ecfp4", "molformer")},
            "files": {x.name: record(x) for x in sorted(staged.iterdir())},
        })
        staged.rename(a.output)


def measures(frame, name):
    grouped = frame.groupby(["cell_group", "drug_id"], as_index=False)[["outcome", name]].mean()
    out = regression_metrics(grouped[name].to_numpy(), grouped.outcome.to_numpy(),
                             grouped.drug_id.to_numpy())
    cell_ndcg = []
    for _, cell in grouped.groupby("cell_group"):
        if len(cell) >= 5 and cell.outcome.nunique() > 1:
            y = cell.outcome.to_numpy()
            cell_ndcg.append(ndcg_score((y - min(0, y.min()))[None, :],
                                        cell[name].to_numpy()[None, :], k=5))
    return {**out, "cell_ndcg5": float(np.mean(cell_ndcg)) if cell_ndcg else None,
            "lineage_groups": grouped.cell_group.nunique(), "drugs": grouped.drug_id.nunique(),
            "native_rows": len(frame)}


def score(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = a.panel / "receipt.json"
    prediction_manifest = a.predictions / "receipt.json"
    receipt = json.loads(prediction_manifest.read_text())
    if receipt["panel"] != record(manifest):
        raise ValueError("Bridge registration differs from prediction freeze")
    table = []
    for name in PANEL_NAMES:
        local = a.panel / name
        path = local / "receipt.json"
        if receipt["panel_receipts"][name] != record(path):
            raise ValueError(f"Frozen role registration mismatch for {name}")
        panel_receipt = json.loads(path.read_text())
        labels = pd.read_parquet(checked(local, "labels.parquet", panel_receipt))
        frozen = pd.read_parquet(checked(a.predictions, f"{name}.parquet", receipt))
        joined = frozen.merge(labels, on="id", validate="1:1", how="inner")
        if len(joined) != len(frozen) or len(joined) != len(labels):
            raise ValueError(f"Missing scores or outcomes for {name}")
        for mode, group in (("all", joined), ("measured_chronos", joined[joined.chronos_measured])):
            if len(group):
                for method in receipt["models"][name]:
                    table.append({"panel": name, "support": mode, "method": method, **measures(group, method)})
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        metric_file = staged / "bridge_metrics.tsv"
        pd.DataFrame(table).to_csv(metric_file, sep="\t", index=False, float_format="%.8g")
        write_json(staged / "receipt.json", {
            "panel": record(manifest), "predictions": record(prediction_manifest),
            "sibling_cultures": "average assay and predictions per lineage group before evaluation",
            "score_rows": len(table), "files": {metric_file.name: record(metric_file)},
        })
        staged.rename(a.output)


def arguments(score_mode=False):
    p = argparse.ArgumentParser(description=__doc__)
    for name in (("panel", "predictions", "output") if score_mode else DATA_ARGUMENTS):
        p.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    if not score_mode:
        p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        p.add_argument("--seed", type=int, default=41026)
    return p.parse_args()
