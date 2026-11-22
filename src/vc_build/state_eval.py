"""Fit, freeze and score the public CMap baseline-RNA state development arms."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from .state_data import ADULT, CONTEXT, open_scope, pinned, response, table
from .state_model import ARMS, PAIR, StateModel, conditions

SLICES = ("C-JHH5", "C-JHH7", "C-both")
SIMPLE = ARMS[:5]


def partitions(rows, name):
    held = set(ADULT) if name == "C-both" else {name.removeprefix("C-")}
    fit = rows.loc[~rows.cell_id.isin(held)]
    query = rows.loc[rows.cell_id.isin(held) & rows.compound_id.isin(fit.compound_id)]
    if len(held) == 2:
        counts = query.groupby("compound_id").cell_id.nunique()
        query = query.loc[query.compound_id.isin(counts[counts.eq(2)].index)]
    if (len(query) == 0 or set(query.cell_id) != held or set(fit.lineage_group) & set(query.lineage_group)
            or set(fit.rna_plate) & set(query.rna_plate)):
        raise ValueError(f"Held cell or plate boundaries are not valid for {name}")
    return fit, query


def directory_manifest(root):
    return {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}


def predict_slice(run, name, rows, controls, basal, genes, stamp, index_hash):
    fit, query = partitions(rows, name)
    out = run / name
    if out.exists():
        raise FileExistsError(out)
    y, fit_controls = response(fit, controls, genes, stamp)
    model = StateModel.fit(fit, y, basal, genes)
    query_x = basal.loc[query.cell_id].to_numpy()
    predictions = model.predict(query.smiles, query_x)
    with TemporaryDirectory(dir=run, prefix=".prediction-") as temp:
        stage = Path(temp)
        query.to_csv(stage / "query.tsv", sep="\t", index=False)
        fit[["inst_id", *CONTEXT, "connectivity_group"]].to_csv(stage / "fit.tsv", sep="\t", index=False)
        fit_controls[["inst_id", *CONTEXT]].to_csv(stage / "fit_controls.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecast.npz", **{k: p.astype(np.float32) for k, p in predictions.items()})
        joblib.dump(model, stage / "checkpoint.joblib", compress=3)
        test_request = {
            "input_mode": "baseline_rna", "assay_id": "CMap2020_L1000_Level3", "dose_uM": 10, "time_h": 24,
            "baseline_scale": stamp["basal_units"], "smiles": query.smiles.iloc[0],
            "basal_expression": dict(zip(genes.astype(str), query_x[0].tolist(), strict=True)),
        }
        direct = model.predict_request(test_request)
        if not np.allclose(direct["programme_delta"], model.project(predictions["rna_interaction"][:1])[0]):
            raise ValueError("Standalone state inference differs from frozen prediction")
        write_json(stage / "demo_request.json", test_request)
        write_json(stage / "demo_response.json", direct)
        write_json(stage / "receipt.json", {
            "index_sha256": index_hash, "protocol_sha256": stamp["protocol_sha256"],
            "fit_cells": sorted(set(fit.cell_id)), "query_cells": sorted(set(query.cell_id)),
            "fit_wells": len(fit), "query_wells": len(query), "fit_vehicle_wells": len(fit_controls),
            "fit_compounds": fit.compound_id.nunique(), "query_compounds": query.compound_id.nunique(),
            "query_expression_read": False, "prediction_code_sha256": sha256(__file__),
            "files": directory_manifest(stage),
        })
        stage.rename(out)


def load_forecast_slice(run, name, index_hash):
    root = run / name
    receipt = read_json(root / "receipt.json")
    content = directory_manifest(root)
    if (receipt["files"] != {k: v for k, v in content.items() if k != "receipt.json"}
            or receipt["query_expression_read"] or receipt["index_sha256"] != index_hash):
        raise ValueError(f"Forecast {name} changed or has invalid heldout access")
    query = table(root / "query.tsv")
    with np.load(root / "forecast.npz", allow_pickle=False) as arrays:
        p = {k: arrays[k] for k in ARMS}
        if set(arrays.files) != set(ARMS) or any(
            a.shape != (len(query), 978) or a.dtype != np.float32 or not np.isfinite(a).all() for a in p.values()
        ):
            raise ValueError(f"Malformed forecast arrays in {name}")
    return receipt, query, p


def freeze(run, index_hash):
    locks = {}
    for name in SLICES:
        load_forecast_slice(run, name, index_hash)
        locks[name] = sha256(run / name / "receipt.json")
    with (run / "freeze.json").open("x") as handle:
        json.dump({"index_sha256": index_hash, "slices": locks}, handle, sort_keys=True, indent=2)
        handle.write("\n")


def grouped_squares(error, groups):
    return pd.DataFrame(np.square(error)).assign(group=groups).groupby("group", sort=True).mean().to_numpy()


def macro_rmse(squares):
    return np.sqrt(squares.mean(axis=0)).mean()


def bootstrap_gain(squares, reference):
    rng = np.random.default_rng(22026)
    weights = rng.multinomial(len(squares), np.full(len(squares), 1 / len(squares)), size=2000) / len(squares)
    base = macro_rmse(reference)
    baseline_samples = np.sqrt(weights @ reference).mean(axis=1)
    if base <= 0 or np.any(baseline_samples <= 0):
        return np.nan, np.nan, np.nan
    values = 1 - np.sqrt(weights @ squares).mean(axis=1) / baseline_samples
    return 1 - macro_rmse(squares) / base, *np.quantile(values, [.025, .975])


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-10)


def aggregate_scope(query, y, preds, model, across_cells):
    grouped_y = conditions(query, y)
    grouped_pred = {k: conditions(query, p) for k, p in preds.items()}
    groups = query.set_index("compound_id").connectivity_group.to_dict()
    if across_cells:
        def diff(z):
            return z.xs("JHH5", level=0).sort_index() - z.xs("JHH7", level=0).sort_index()

        grouped_y = diff(grouped_y)
        grouped_pred = {k: diff(z) for k, z in grouped_pred.items()}
        grouping = [groups[c] for c in grouped_y.index]
    else:
        grouping = [groups[c] for _, c in grouped_y.index]

    y = grouped_y.to_numpy()
    zy = model.project(y)
    metrics = []
    arm_programme = {k: model.project(z.to_numpy()) for k, z in grouped_pred.items()}
    squares = {k: grouped_squares(p - zy, grouping) for k, p in arm_programme.items()}
    best = min(SIMPLE, key=lambda k: macro_rmse(squares[k]))
    for k, p in arm_programme.items():
        rel, lo, hi = bootstrap_gain(squares[k], squares[best])
        metrics.append({
            "arm": k, "reference_macro": best, "conditions": len(y), "groups": len(set(grouping)),
            "macro_programme_rmse": macro_rmse(squares[k]), "pooled_programme_rmse": np.sqrt(squares[k].mean()),
            "relative_macro_gain": rel, "relative_macro_ci95_low": lo, "relative_macro_ci95_high": hi,
            "landmark_rmse": np.sqrt(np.square(grouped_pred[k].to_numpy() - y).mean()),
            "programme_sign_agreement": (np.sign(p) == np.sign(zy)).mean(),
            "programme_cosine": np.sum(unit(p) * unit(zy), axis=1).mean(),
        })
    return metrics


def repeat_summary(rows, outcomes):
    reports = []
    for line, frame in rows.groupby("cell_id", sort=True):
        pairs = []
        for compound, samples in frame.groupby("compound_id", sort=True):
            indices = samples.index.to_numpy()
            for i in indices:
                comparable = samples.loc[samples.rna_plate.ne(rows.at[i, "rna_plate"])
                                         & samples.rna_well.ne(rows.at[i, "rna_well"])]
                if len(comparable):
                    means = {
                        plate: outcomes[group.index.to_numpy()].mean(axis=0)
                        for plate, group in comparable.groupby("rna_plate")
                    }
                    donor = np.mean(list(means.values()), axis=0)
                    pairs.append((compound, i, donor))
        n = len(pairs)
        if n:
            y = outcomes[[i for _, i, _ in pairs]]
            p = np.stack([d for _, _, d in pairs])
            original_rmse = np.sqrt(np.square(y).mean())
            observed = np.sqrt(np.square(y - p).mean())
            cosine = np.sum(unit(y)*unit(p), axis=1).mean()
        else:
            original_rmse = observed = cosine = None
        reports.append({
            "cell": line, "repeat_exact_compounds": len({q[0] for q in pairs}), "target_wells": n,
            "different_position_repeat_rmse": observed, "zero_change_rmse": original_rmse,
            "landmark_direction_cosine": cosine,
        })
    return reports


def score(index, run, index_hash, stamp, rows, controls, genes):
    freeze_digest = sha256(run / "freeze.json")
    manifest = read_json(run / "freeze.json")
    if manifest["index_sha256"] != index_hash or set(manifest["slices"]) != set(SLICES):
        raise ValueError("Frozen state predictions refer to another state index")
    outputs = {}
    for name, digest in manifest["slices"].items():
        pinned(run / name / "receipt.json", digest)
        _, forecast_rows, pred = load_forecast_slice(run, name, index_hash)
        fit, query = partitions(rows, name)
        if forecast_rows.inst_id.to_list() != query.inst_id.to_list():
            raise ValueError(f"Forecast order changed on {name}")
        fit_control = table(run / name / "fit_controls.tsv")
        query_keys = pd.MultiIndex.from_frame(query[CONTEXT])
        if pd.MultiIndex.from_frame(fit_control[CONTEXT]).isin(query_keys).any() or set(fit.inst_id) & set(query.inst_id):
            raise ValueError("Frozen model could access a subsequently scored well")
        outputs[name] = (query, pred, joblib.load(run / name / "checkpoint.joblib"))
    # The two C slices have held-context labels, and the same cells are queried in C-both.
    # Read each adult physical outcome once in this scoring process.
    adult = rows.loc[rows.cell_id.isin(ADULT)].copy().reset_index(drop=True)
    values, vehicle = response(adult, controls, genes, stamp)
    lookup = pd.Series(np.arange(len(adult)), index=adult.inst_id, dtype=int)
    metric = []
    for name, (query, pred, model) in outputs.items():
        y = values[lookup.loc[query.inst_id].to_numpy()]
        for arm in aggregate_scope(query, y, pred, model, name == "C-both"):
            metric.append(dict(slice=name, **arm))
    out = run / "scored"
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=run, prefix=".score-") as temp:
        stage = Path(temp)
        pd.DataFrame(metric).to_csv(stage / "metrics.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(repeat_summary(adult, values)).to_csv(stage / "repeat.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", {
            "freeze_sha256": freeze_digest, "index_sha256": index_hash, "held_adult_wells": len(adult),
            "held_vehicle_wells": len(vehicle), "code_sha256": sha256(__file__),
            "files": directory_manifest(stage),
        })
        stage.rename(out)


def export_auxiliary(run, index_hash, stamp, rows, controls, basal, genes):
    score_stamp = read_json(run / "scored/receipt.json")
    pinned(run / "freeze.json", score_stamp["freeze_sha256"])
    if score_stamp["index_sha256"] != index_hash:
        raise ValueError("Auxiliary export is only allowed for an already frozen and scored state index")
    out = run / "auxiliary"
    if out.exists():
        raise FileExistsError(out)
    delta, used = response(rows, controls, genes, stamp)
    key = used.groupby(CONTEXT).inst_id.agg("|".join)
    rows = rows.assign(control_ids=key.reindex(pd.MultiIndex.from_frame(rows[CONTEXT])).to_numpy())
    with TemporaryDirectory(dir=run, prefix=".auxiliary-") as temp:
        stage = Path(temp)
        rows.drop(columns="level3_row_index").to_csv(stage / "rows.tsv", sep="\t", index=False)
        pd.DataFrame({"gene_id": genes}).to_csv(stage / "genes.tsv", sep="\t", index=False)
        basal.to_csv(stage / "basal.tsv", sep="\t")
        np.save(stage / "delta.npy", delta, allow_pickle=False)
        write_json(stage / "receipt.json", {
            "index_sha256": index_hash, "freeze_sha256": score_stamp["freeze_sha256"],
            "score_sha256": sha256(run / "scored/receipt.json"), "source_version": stamp["source_version"],
            "delta_units": stamp["delta_units"], "basal_units": stamp["basal_units"], "dose_uM": 10,
            "time_h": 24, "baseline_is_proxy": True, "shape": list(delta.shape),
            "usage": "previously exposed development state labels only; use auxiliary outcomes only for permitted fold-fit lineages",
            "files": directory_manifest(stage),
        })
        stage.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("predict", "freeze", "score", "auxiliary", "apply"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.step == "apply":
        print(json.dumps(joblib.load(args.checkpoint).predict_request(read_json(args.request)), sort_keys=True))
        return
    stamp, rows, controls, basal, genes = open_scope(args.index)
    index_hash = sha256(args.index / "receipt.json")
    if args.step == "predict":
        args.run.mkdir(parents=True, exist_ok=True)
        for name in SLICES:
            predict_slice(args.run, name, rows, controls, basal, genes, stamp, index_hash)
    elif args.step == "freeze":
        freeze(args.run, index_hash)
    elif args.step == "score":
        score(args.index, args.run, index_hash, stamp, rows, controls, genes)
    elif args.step == "auxiliary":
        export_auxiliary(args.run, index_hash, stamp, rows, controls, basal, genes)


if __name__ == "__main__":
    main()
