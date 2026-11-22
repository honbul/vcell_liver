"""Seal all nine corrected LIMORE forecasts, then score them jointly once."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vc_build.evaluate import join_observations
from vc_build.external import KEY
from vc_build.io import freeze_table, publish_json, sha256, verify_forecast
from vc_build.v0plus_cv import attach_native
from vc_build.v0plus_score import axis_metrics

from .base_models import CellHitFit, LinearPanel, PathElasticNet, as_forecast, predict_part
from .contract import EXPOSURE, MODELS, OBSERVATIONS, ORIGINAL, TRACK, TRAIN_SHA, RNA_SHA
from .contract import corrected, fold_indices, inputs, read_json, verified
from .metrics import AXES, RESIDUAL, association, bootstrap, interval, metrics, pivot

BASE_PARTS = ("linear", "cellhit", "structured", "elasticnet")
BASELINES = ("drug_mean", "additive", "drug_ridge", "cellhit", "structured")
TKI = {"Sorafenib", "Regorafenib", "Lenvatinib", "Cabozantinib"}


def checkpoint(root, part, fold=3):
    path = root / "base" / part / f"fold{fold}"
    stamp = read_json(path / "receipt.json")
    if (stamp["train_data_sha256"], stamp["part"], stamp["fold"]) != (TRAIN_SHA, part, fold):
        raise ValueError("Model fit receipt changed")
    return verified(path / "model.joblib", stamp["checkpoint_sha256"])


def load_base(path):
    """Read immutable early worker artifacts written by module-mode joblib."""
    import __main__

    types = (LinearPanel, CellHitFit, PathElasticNet)
    old = {cls.__name__: getattr(__main__, cls.__name__, None) for cls in types}
    try:
        for cls in types:
            setattr(__main__, cls.__name__, cls)
        return joblib.load(path)
    finally:
        for cls in types:
            name = cls.__name__
            if old[name] is None:
                delattr(__main__, name)
            else:
                setattr(__main__, name, old[name])


def members(root):
    for fold in range(3):
        path = root / "pairwise" / f"fold{fold}"
        _, stamp = verify_forecast(path / "oof.parquet")
        meta = stamp["provenance"]
        if meta["train_data_sha256"] != TRAIN_SHA or meta["refit_rows"] != 65678:
            raise ValueError("Pairwise fit receipt changed")
        yield verified(path / "member.joblib", meta["member_sha256"])


def frozen_pairwise(paths, query, rna, banks):
    results = {name: np.zeros(len(query)) for name in ("pairwise_rank", "pairwise_top5")}
    for path in paths:
        model = joblib.load(path)
        values = model.predict(query, rna, banks, device="cpu")
        base = query.compound_id.map(model.adaptive.anchor).to_numpy()
        for name, head in model.meta.items():
            results[name] += head.predict(values, base, query.compound_id) / len(paths)
    return results


def blend(values):
    return values | {"ridge_structured_50_50": .5 * (values["drug_ridge"] + values["structured"])}


def check_keys(frame, query, names):
    expected = query[KEY].set_index("row_id").sort_index()
    if set(frame.method) != set(names):
        raise ValueError("Joint prediction must contain all and only the registered arms")
    for _, arm in frame.groupby("method"):
        if not arm[KEY].set_index("row_id").sort_index().equals(expected):
            raise ValueError("Forecast panel support differs between models")


def forecast(args):
    from .pairwise import neural_banks

    _, _, query, qrna = inputs()
    artifacts = {name: checkpoint(args.root, name) for name in BASE_PARTS}
    paths = list(members(args.root))
    banks, embed_hash = neural_banks(limore=True)
    values = {}
    for part, path in artifacts.items():
        values.update(predict_part(part, load_base(path), query, qrna))
    values.update(frozen_pairwise(paths, query, qrna, banks))
    frame = as_forecast(query, blend(values), fold=0, track=TRACK)
    check_keys(frame, query, MODELS)
    model_hashes = {part: sha256(path) for part, path in artifacts.items()}
    model_hashes["pairwise_members"] = [sha256(path) for path in paths]
    freeze_table(frame, args.out, provenance={
        "training_sha256": TRAIN_SHA, "training_rna_sha256": RNA_SHA, "training_rows": 65678,
        "observation_sha256": OBSERVATIONS, "limore_registration_sha256": sha256(ORIGINAL / "receipt.json"),
        "code_contract_sha256": sha256(args.contract), "model_sha256": model_hashes,
        "embedding_sha256": embed_hash, "scope": EXPOSURE,
    })


def save(path, frame):
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, sep="\t", index=False, float_format="%.10g", na_rep="NA")


def oof(root, out):
    train, _, _ = corrected()
    native, fold_support, covered = [], [], set()
    for fold, (fit, held) in enumerate(fold_indices(train)):
        group_fit, group_held = set(fit.lineage_group), set(held.lineage_group)
        if group_fit & group_held or group_held & covered or group_fit | group_held != set(train.lineage_group):
            raise ValueError("The source OOF lineage assignments are not mutually held out")
        covered |= group_held
        fold_support.append({"fold": fold, "fit_rows": len(fit), "held_rows": len(held),
                             "fit_lineages": len(group_fit), "held_lineages": len(group_held),
                             "fit_held_lineage_overlap": 0})
        frames = []
        files = [root / "base" / part / f"fold{fold}" / "oof.parquet" for part in BASE_PARTS]
        files.append(root / "pairwise" / f"fold{fold}" / "oof.parquet")
        for path in files:
            table, stamp = verify_forecast(path)
            provenance = stamp["provenance"]
            if provenance["train_data_sha256"] != TRAIN_SHA or provenance["fit_rows"] != len(fit):
                raise ValueError("OOF forecast does not match the held split")
            frames.append(table)
        f = pd.concat(frames)
        expected = tuple(k for k in MODELS if k != "ridge_structured_50_50")
        from vc_build.v0plus_cv import validation_roster
        check_keys(f, validation_roster(held), expected)
        a = f[f.method.isin(("drug_ridge", "structured"))].groupby([*KEY, "fold", "track"], as_index=False)
        combo = a.predicted_value.mean().assign(method="ridge_structured_50_50")
        evaluated = attach_native(pd.concat([f, combo], ignore_index=True), held, fold)
        summary, drugs, _ = axis_metrics(evaluated, ())
        cells = pd.DataFrame(summary["scores"]).drop(columns="confusion", errors="ignore")
        by_drug = drugs.groupby(["track", "method"], as_index=False)[["spearman", "pearson"]].mean()
        by_drug = by_drug.rename(columns={k: "within_drug_" + k for k in ("spearman", "pearson")})
        section = cells.merge(by_drug, on=["track", "method"], validate="one_to_one")
        section.insert(0, "outer_fold", fold)
        native.append(section)
    if covered != set(train.lineage_group) or sum(r["held_rows"] for r in fold_support) != len(train):
        raise ValueError("The OOF score did not include the complete corrected training roster once")
    save(out / "SOURCE_OOF_SUPPORT.tsv", pd.DataFrame(fold_support))
    frame = pd.concat(native, ignore_index=True)
    save(out / "SOURCE_OOF_BY_FOLD.tsv", frame)
    columns = ["within_drug_spearman", "within_drug_pearson", "macro_ndcg_at_5", "macro_top5_overlap",
               "macro_spearman"]
    mean = frame.groupby("method", as_index=False)[columns].mean().rename(columns=dict(zip(columns, AXES)))
    save(out / "SOURCE_OOF_MACRO.tsv", mean.rename(columns={"method": "model_id"}))


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    frames, stamp = verify_forecast(args.forecast)
    provenance = stamp["provenance"]
    if (provenance["training_sha256"], provenance["observation_sha256"], provenance["code_contract_sha256"]) != (
            TRAIN_SHA, OBSERVATIONS, sha256(args.contract)):
        raise ValueError("Registered comparison changed between prediction and scoring")
    # Opening the measured LIMORE file happens only here, after verification and OOF scoring.
    train, train_rna, query, _ = inputs()
    check_keys(frames, query, MODELS)
    args.out.mkdir(parents=True)
    oof(args.root, args.out)
    labels = pd.read_parquet(verified(ORIGINAL / "test_labels/outcomes.parquet", OBSERVATIONS))
    joined = join_observations(frames, labels)
    y = pivot(joined[joined.method.eq("drug_mean")], "endpoint_value")
    models = {name: pivot(block, "predicted_value").reindex_like(y).to_numpy()
              for name, block in joined.groupby("method")}
    if y.shape != (50, 68) or not np.isfinite(y).all().all() or any(not np.isfinite(m).all() for m in models.values()):
        raise ValueError("Evaluation contrast is not a complete matched 50×68 rectangle")
    estimates = {model: metrics(y.to_numpy(), p) for model, p in models.items()}
    draws = bootstrap(y.to_numpy(), models)
    contrasts, rows, residuals, tkis = [], [], [], []
    registry = query[["compound_id", "source_compound_label"]].drop_duplicates().set_index("compound_id")
    for model in MODELS:
        m = {"model_id": model, "train_data_sha256": TRAIN_SHA, "split": "LIMORE_adult_HCC_cell_cold",
             "n_rows": y.size, **{key: estimates[model][key] for key in AXES},
             "inference_checkpoint": ("fixed_50_50_component_checkpoints" if model == "ridge_structured_50_50"
                                      else "available"),
             "forecast_sha256": stamp["sha256"], "evaluation_exposure": EXPOSURE,
             "within_drug_math_status": "undefined_constant_convention_0" if model == "drug_mean" else "defined"}
        for baseline in BASELINES:
            if model == baseline:
                continue
            for metric in AXES:
                delta = estimates[model][metric] - estimates[baseline][metric]
                lo, hi = interval(draws[model][metric] - draws[baseline][metric])
                contrasts.append({"model_id": model, "baseline": baseline, "metric": metric,
                                  "delta": delta, "ci95_low": lo, "ci95_high": hi})
                m[f"vs_{baseline}_{metric}_delta"] = delta
                m[f"vs_{baseline}_{metric}_ci95"] = f"[{lo:.8f}, {hi:.8f}]"
        rows.append(m)
        for metric in RESIDUAL:
            lo, hi = interval(draws[model][metric])
            residuals.append({"model_id": model, "metric": metric, "estimate": estimates[model][metric],
                              "ci95_low": lo, "ci95_high": hi})
        for drug, label in registry.source_compound_label.items():
            if label not in TKI:
                continue
            j = y.columns.get_loc(drug)
            values = [association(y.to_numpy()[:, j], models[model][:, j], axis=0, rank=rank,
                                  constant=np.nan).item() for rank in (True, False)]
            tkis.append({"model_id": model, "compound_id": drug, "drug": label, "n_lineages": len(y),
                         "spearman": values[0], "pearson": values[1]})
    # Directly compare the fast implementation with the prior study's public reference on this exact panel.
    reference, _, _ = axis_metrics(joined, ())
    native_cell = {row["method"]: row for row in reference["scores"]}
    native_drug = {row["method"]: row for row in reference["across_cell_scores"]}
    errors = []
    for method in MODELS:
        for ours, old in zip(AXES, (*AXES[:2], "macro_ndcg_at_5", "macro_top5_overlap", "macro_spearman")):
            ref = native_drug[method] if ours in AXES[:2] else native_cell[method]
            errors.append(abs(ref[old] - estimates[method][ours]))
    if max(errors) > 1e-9:
        raise ValueError(f"Canonical metric reproduction differs: {max(errors):.5g}")
    save(args.out / "UNIFIED_2D_LEDGER.tsv", pd.DataFrame(rows).rename(columns={"precision_at_k": "precision_at_5"}))
    save(args.out / "LIMORE_PAIRED_95CI.tsv", pd.DataFrame(contrasts))
    save(args.out / "LIMORE_TKI.tsv", pd.DataFrame(tkis))
    save(args.out / "LIMORE_RESIDUALS.tsv", pd.DataFrame(residuals))
    publish_json(args.out / "score_receipt.json", {
        "prediction_sha256": stamp["sha256"], "labels_sha256": OBSERVATIONS, "training_sha256": TRAIN_SHA,
        "actual_n": len(labels), "n_lineages": len(y), "n_compounds": y.shape[1], "score_passes": 1,
        "training_n": len(train), "training_lineages": int(train.lineage_group.nunique()),
        "training_models": len(train_rna), "training_by_source": train.source_id.value_counts().to_dict(),
        "contract_sha256": sha256(args.contract), "models": list(MODELS), "crossed_bootstrap_draws": 2400,
        "legacy_2d_metric_max_abs_difference": float(max(errors)), "drug_mean_within_drug_math_value": None,
        "drug_mean_compared_within_drug_value": 0.0, "files": {p.name: sha256(p) for p in args.out.glob("*.tsv")},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast", "score"))
    for field in ("root", "out", "contract", "forecast"):
        parser.add_argument("--" + field, type=Path, required=field != "forecast")
    args = parser.parse_args()
    {"forecast": forecast, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
