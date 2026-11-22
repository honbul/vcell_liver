"""LIMORE RNA intervention, double-centered interaction and Top-5 diversity audit."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.evaluate import join_observations
from vc_build.io import freeze_table, publish_json, sha256, verify_forecast

from .base_models import as_forecast, predict_part
from .contract import OBSERVATIONS, ORIGINAL, TRACK, TRAIN_SHA, inputs, read_json, verified
from .ledger import BASE_PARTS, checkpoint, check_keys, frozen_pairwise, load_base, members, save
from .metrics import AXES, RESIDUAL, metrics, pivot, top5
from .pairwise import neural_banks

AUDIT = ("additive", "drug_ridge", "cellhit", "structured", "pairwise_rank", "ecfp4_path_elasticnet")
REPEATS = 100


def derangements(n, seed=74821):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    for _ in range(REPEATS):
        while True:
            order = rng.permutation(n)
            if np.all(order != indices):
                yield order
                break


def apply(args):
    _, fit_rna, query, rna = inputs()
    raw, seal = verify_forecast(args.forecast)
    raw = raw[raw.method.isin(AUDIT)]
    check_keys(raw, query, AUDIT)
    fit_banks, fit_hash = neural_banks()
    banks, query_hash = neural_banks(limore=True)
    paths = list(members(args.root))
    models = {part: load_base(checkpoint(args.root, part)) for part in BASE_PARTS}

    def predict(x, bank):
        available = {}
        for part, model in models.items():
            available.update(predict_part(part, model, query, x))
        available.update(frozen_pairwise(paths, query, x, bank))
        return {name: available[name] for name in AUDIT}

    reference = {name: rows.set_index("row_id").reindex(query.row_id).predicted_value.to_numpy()
                 for name, rows in raw.groupby("method")}
    fit_mean = fit_rna.mean(axis=0).to_numpy()
    average_rna = pd.DataFrame(np.tile(fit_mean, (len(rna), 1)), index=rna.index, columns=rna.columns)
    average_banks = banks.copy()
    for feature in ("scgpt", "geneformer"):
        table = fit_banks[feature].reindex(fit_rna.index).dropna()
        if len(table) < 0.85 * len(fit_rna):
            raise ValueError("Insufficient source-only transcriptome embedding support")
        average_banks[feature] = pd.DataFrame(np.tile(table.mean().to_numpy(), (len(rna), 1)), index=rna.index)
    outputs = [as_forecast(query, predict(average_rna, average_banks), fold=0, track="mean_rna")]
    assignment = query[["cell_id", "compound_id"]].set_index(["cell_id", "compound_id"])
    reference = pd.DataFrame(reference, index=assignment.index)
    max_replay = {}
    signatures = []
    for i, order in enumerate(derangements(len(rna))):
        donor = pd.Series(rna.index[order], index=rna.index)
        target_keys = pd.MultiIndex.from_arrays([query.cell_id.map(donor), query.compound_id])
        shuffled = {name: reference[name].reindex(target_keys).to_numpy() for name in AUDIT}
        signatures.append("|".join(donor.to_list()))
        if i in (0, 49, 99):
            permuted = rna.iloc[order].set_axis(rna.index)
            permuted_banks = banks.copy()
            for feature in ("scgpt", "geneformer"):
                permuted_banks[feature] = banks[feature].reindex(rna.index[order]).set_axis(rna.index)
            replay = predict(permuted, permuted_banks)
            max_replay[str(i)] = {name: float(np.max(np.abs(replay[name] - shuffled[name]))) for name in AUDIT}
            if max(max_replay[str(i)].values()) > 2e-6:
                raise ValueError("RNA-swap projection does not reproduce actual perturbed inference")
        outputs.append(as_forecast(query, shuffled, fold=i, track="rna_shuffle"))
    freeze_table(pd.concat(outputs, ignore_index=True), args.out, provenance={
        "parent_forecast_sha256": seal["sha256"], "train_sha256": TRAIN_SHA, "labels_sha256": OBSERVATIONS,
        "repeats": REPEATS, "settings_sha256": sha256(args.contract), "exact_expanded_swap_replay": max_replay,
        "ordered_query_derangements": signatures, "source_fm_hashes": fit_hash, "query_fm_hashes": query_hash,
    })


def spread(p, reference):
    sets, control = top5(p), top5(reference)
    overlap = [len(a & b) / 5 for a, b in zip(sets, control)]
    return {"unique_top5_sets": len({frozenset(a) for a in sets}), "lineages": len(sets),
            "mean_pairwise_top5_jaccard": float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)])),
            "mean_overlap_with_drug_mean_top5": float(np.mean(overlap))}


def decision(official, out):
    result = pd.read_csv(official / "UNIFIED_2D_LEDGER.tsv", sep="\t").set_index("model_id")
    source = pd.read_csv(official / "SOURCE_OOF_MACRO.tsv", sep="\t").set_index("model_id")
    deltas = pd.read_csv(official / "LIMORE_PAIRED_95CI.tsv", sep="\t")
    residual = pd.read_csv(official / "LIMORE_RESIDUALS.tsv", sep="\t")
    axis = ["ndcg_at_5", "within_drug_spearman"]
    diversity_table = pd.read_csv(out / "TOP5_DIVERSITY.tsv", sep="\t").set_index("model_id")
    rows = []
    for name in result.index:
        diff = deltas[deltas.model_id.eq(name)]
        r = residual[residual.model_id.eq(name) & residual.metric.eq("residual_spearman")].iloc[0]
        oof_gain = source.loc[name, axis] - source.loc["additive", axis]
        oof_alt = source.loc[name, axis] - source.loc["drug_mean", axis]
        lcb = {}
        for control in ("additive", "drug_mean"):
            section = diff[diff.baseline.eq(control)].set_index("metric").reindex(axis).ci95_low
            lcb[control] = {key: float(value) if pd.notna(value) else None for key, value in section.items()}
        joint = all(value is not None and value > 0 for c in lcb.values() for value in c.values())
        diversity = int(diversity_table.loc[name, "unique_top5_sets"])
        pass_joint = bool(joint and (oof_gain > 0).all() and (oof_alt > 0).all() and r.ci95_low > 0 and diversity > 1)
        rows.append({"model_id": name, "qualified_joint": pass_joint,
                     "oof_gain_vs_additive_avg": float(oof_gain.mean()),
                     "oof_gain_vs_additive": dict(oof_gain), "limore_lower_bounds": lcb,
                     "residual_lower_95ci": float(r.ci95_low), "top5_sets": diversity})
    eligible = [r for r in rows if r["qualified_joint"]]
    if eligible:
        choice = max(eligible, key=lambda r: r["oof_gain_vs_additive_avg"])["model_id"]
        scope = "joint research reference"
    else:
        sensitivity = next(r for r in rows if r["model_id"] == "structured")
        signal = all(sensitivity["limore_lower_bounds"][b]["within_drug_spearman"] > 0
                     for b in ("additive", "drug_mean"))
        if signal and sensitivity["residual_lower_95ci"] > 0:
            choice, scope = "structured", "limited within-drug research reference"
        else:
            choice, scope = "drug_mean", "nonpersonalized ranking reference"
    publish_json(out / "reference_decision.json", {"designated_model": choice, "scope": scope,
                 "joint_promotion": bool(eligible), "model_specific_evidence": rows})


def score(args):
    original, reg = verify_forecast(args.forecast)
    intervention, seal = verify_forecast(args.intervention)
    prov = seal["provenance"]
    if (prov["parent_forecast_sha256"], prov["train_sha256"], prov["labels_sha256"], prov["settings_sha256"]) != (
            reg["sha256"], TRAIN_SHA, OBSERVATIONS, sha256(args.contract)):
        raise ValueError("Intervention did not originate from the verified Task 1 fit")
    if read_json(args.official / "score_receipt.json")["prediction_sha256"] != reg["sha256"]:
        raise ValueError("Task 1 must finish before the personalization score")
    args.out.mkdir(parents=True, exist_ok=False)
    truth = pd.read_parquet(verified(ORIGINAL / "test_labels/outcomes.parquet", OBSERVATIONS))
    observed = join_observations(original, truth)
    y = pivot(observed[observed.method.eq("drug_mean")], "endpoint_value")
    reference = pivot(observed[observed.method.eq("drug_mean")], "predicted_value").reindex_like(y).to_numpy()
    original_values, diversity = [], []
    for name, rows in observed.groupby("method"):
        p = pivot(rows, "predicted_value").reindex_like(y).to_numpy()
        original_values.append({"model_id": name, **metrics(y.to_numpy(), p)})
        diversity.append({"model_id": name, **spread(p, reference)})
    results = []
    for (model, condition, repeat), block in intervention.groupby(["method", "track", "fold"]):
        p = pivot(block, "predicted_value").reindex_like(y).to_numpy()
        results.append({"model_id": model, "condition": condition, "repeat": repeat, **metrics(y.to_numpy(), p)})
    base = pd.DataFrame(original_values).set_index("model_id")
    raw = pd.DataFrame(results)
    stats = []
    for model in AUDIT:
        arm = raw[raw.model_id.eq(model)]
        mean = arm.loc[arm.condition.eq("mean_rna")].iloc[0]
        shuffle = arm.loc[arm.condition.eq("rna_shuffle")]
        if len(shuffle) != REPEATS:
            raise ValueError("Expected all 100 RNA shuffles")
        for key in (*AXES, *RESIDUAL):
            real = base.loc[model, key]
            s = shuffle[key]
            rg = s.quantile([.025, .975])
            percentile = (s <= real).sum() if "rmse" in key else (s >= real).sum()
            stats.append({"model_id": model, "metric": key, "native": real, "train_mean_rna": mean[key],
                          "native_minus_mean": real - mean[key], "shuffle_average": s.mean(),
                          "shuffle_p025": rg.iloc[0], "shuffle_p975": rg.iloc[1],
                          "native_minus_shuffle_mean": real - s.mean(),
                          "permutation_p_one_sided": (1 + percentile) / (REPEATS + 1)})
    save(args.out / "RNA_INTERVENTIONS.tsv", pd.DataFrame(stats))
    save(args.out / "RNA_SHUFFLE_100.tsv", raw)
    save(args.out / "TOP5_DIVERSITY.tsv", pd.DataFrame(diversity))
    decision(args.official, args.out)
    publish_json(args.out / "personalization_receipt.json", {
        "intervention_sha256": seal["sha256"], "source_forecast_sha256": reg["sha256"], "observations_sha256": OBSERVATIONS,
        "repeats": REPEATS, "replayed_perturbation_max_abs_diff": prov["exact_expanded_swap_replay"],
        "files": {f.name: sha256(f) for f in args.out.iterdir()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score"))
    for field in ("root", "out", "contract", "forecast", "intervention", "official"):
        parser.add_argument("--" + field, type=Path, required=field in ("out", "contract", "forecast"))
    args = parser.parse_args()
    {"freeze": apply, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
