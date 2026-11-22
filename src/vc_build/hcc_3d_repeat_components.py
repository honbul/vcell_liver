"""Measure repeat concordance and score frozen differences between same-specimen PDOs."""

import argparse
from contextlib import closing
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import rankdata

from .evaluate import rank_metrics
from .external import KEY
from .hcc_3d_public_response import public_inputs
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_full_axis import validated
from .response_components import safe_corr
from .v0plus_licob import ASSET_HASHES, paired_axis, sample_metadata
from .v0plus_score import correlations
from .yang_hcc_external import require_hash

REFERENCE = "723575cabf92e6f3b985fdda88e16836b59e4fff77eca4a3cf94090ac10f75e8"
METHODS = ("structured", "drug_ridge", "cellhit", "codeae_inductive", "velodrome_inductive",
           "v0plus_inductive", "interaction_neural_blend", "source_gated_affine")
TRACK = "LICOB_same_specimen"


def interaction(values):
    return (values - values.mean(axis=-1, keepdims=True) - values.mean(axis=-2, keepdims=True)
            + values.mean(axis=(-1, -2), keepdims=True))


def concordance(first, second):
    drug = np.nanmean(correlations(first, second), axis=1)
    ranked = np.nanmean(correlations(rankdata(first, axis=1), rankdata(second, axis=1)), axis=1)
    f, s = interaction(first), interaction(second)
    magnitude = np.linalg.norm(f, axis=(1, 2)) * np.linalg.norm(s, axis=(1, 2))
    residual = np.divide((f * s).sum(axis=(1, 2)), magnitude,
                         out=np.full(len(f), np.nan), where=magnitude > 1e-12)
    return dict(spearman=ranked, pearson=drug, interaction_pearson=residual)


def repeat_stats(first, second, *, reciprocal):
    if (not first.index.equals(second.index) or not first.columns.equals(second.columns)
            or not all(np.isfinite(x).all(axis=None) for x in (first, second))):
        raise ValueError("Repeat measurements require the same complete donor and drug panel")
    a, b = first.to_numpy(), second.to_numpy()
    estimate = {k: float(v[0]) for k, v in concordance(a[None], b[None]).items()}
    rng = np.random.default_rng(11942)
    row = rng.integers(len(a), size=(2400, len(a), 1))
    drug = rng.integers(a.shape[1], size=(2400, 1, a.shape[1]))
    interval = {k: np.nanquantile(v, (.025, .975)).tolist() for k, v in concordance(a[row, drug], b[row, drug]).items()}
    fields = ("spearman", "ndcg_at_5", "top5_overlap")
    directions = ((a, b), (b, a)) if reciprocal else ((b, a),)
    rankings = pd.DataFrame([rank_metrics(y, p) for y_mat, p_mat in directions for y, p in zip(y_mat, p_mat)])
    ranking_fields = fields if a.shape[1] >= 6 else fields[:1]
    top_1 = [float((np.equal(y, y.max()) & np.equal(p, p.max())).sum() /
                   (np.equal(y, y.max()).sum() * np.equal(p, p.max()).sum())) for y, p in zip(a, b)]
    return {"groups": len(a), "drugs": a.shape[1], "point": estimate, "ci95": interval,
            "drug_marginal_pearson": safe_corr(a.mean(0), b.mean(0)),
            "sample_marginal_pearson": safe_corr(a.mean(1), b.mean(1)),
            "interaction_energy_shares": [float(np.square(interaction(x)).sum()/np.square(x-x.mean()).sum()) for x in (a,b)],
            "within_sample": {field: float(rankings[field].mean()) for field in ranking_fields},
            "top1": float(np.mean(top_1)), "positive_interaction": interval["interaction_pearson"][0] > 0,
            "drugwise": {name: {"spearman": safe_corr(rankdata(a[:, i]), rankdata(b[:, i])),
                                "pearson": safe_corr(a[:, i], b[:, i])} for i, name in enumerate(first.columns)}}


def repeats(args):
    _, (mono, caf, kim) = public_inputs(args.public)
    matrices = {i: rows.pivot(index="lineage_group", columns="drug", values="endpoint_value")
                for i, rows in kim.groupby("repeat", sort=True)}
    if len(matrices) != 3 or any(m.shape != (18, 8) for m in matrices.values()):
        raise ValueError("Expected three complete Kim repeat panels")
    pairs = {f"{a+1}_vs_{b+1}": repeat_stats(matrices[a], matrices[b], reciprocal=True)
             for a, b in combinations(matrices, 2)}
    mono = mono.pivot(index="lineage_group", columns="drug", values="endpoint_value")
    caf = caf.pivot(index="lineage_group", columns="drug", values="endpoint_value")
    if caf.shape != (33, 4) or not set(caf.index) <= set(mono.index):
        raise ValueError("Expected 33 same-donor Zhang CAF panels")
    publish_json(args.out, {
        "Kim": pairs, "Zhang": repeat_stats(mono.loc[caf.index, caf.columns], caf, reciprocal=False),
        "Kim_distinct_passage_strings_per_cell_drug": {
            str(i): int(n) for i, n in kim.groupby(["lineage_group", "drug"]).passage.nunique().value_counts().items()
        }, "input_sha256": {p: sha256(args.public / p) for p in ("kim-hcc.zip", "zhang-3dplc.zip")},
    })


def licob_panel(args):
    registration = validated(args.licob)
    require_hash(args.licob_labels, ASSET_HASHES["supplement"])
    query = pd.read_parquet(args.licob / "query.parquet")
    samples = query[["cell_id", "donor_id", "adult_age_confirmed"]].drop_duplicates().set_index("cell_id")
    with closing(load_workbook(args.licob_labels, read_only=True, data_only=True)) as book:
        metadata = sample_metadata(book)[["donor_id", "adult_age_confirmed"]]
        _, rows = paired_axis(book)
    if not samples.sort_index().equals(metadata.loc[samples.index].sort_index()) or not set(samples.index) <= set(rows):
        raise ValueError("LICOB matched specimen metadata changed")
    groups = [sorted(part.index) for _, part in samples[samples.adult_age_confirmed].groupby("donor_id") if len(part) >= 2]
    if sorted(map(len, groups)) != [2, 2, 2, 3] or query.compound_id.nunique() != 53:
        raise ValueError("Expected four paired adult LICOB donor groups")
    pairs = [(samples.loc[left, "donor_id"], left, right) for group in groups for left, right in combinations(group, 2)]
    return registration, query, groups, pairs, rows


def pair_rows(pairs, drugs):
    rows = [(donor, a, b, name) for donor, a, b in pairs for name in drugs]
    df = pd.DataFrame(rows, columns=["lineage_group", "model_a", "model_b", "compound_id"])
    df["cell_id"] = df.model_a + "|" + df.model_b
    return df.assign(row_id=df.cell_id + "|" + df.compound_id, track=TRACK, fold=0)


def predict(args):
    registration, query, _, pairs, _ = licob_panel(args)
    require_hash(args.reference, REFERENCE)
    refs = json.loads(args.reference.read_text())["studies"]["LICOB"]
    old, seal = verify_forecast(args.forecast)
    if seal["sha256"] != refs["forecast_sha256"] or seal["provenance"]["inputs_sha256"] != registration:
        raise ValueError("Wrong previously frozen LICOB forecast")
    roster = pair_rows(pairs, sorted(query.compound_id.unique()))
    scores = old[old.method.isin(METHODS)].pivot(index=["cell_id", "compound_id"], columns="method", values="predicted_value")
    a = scores.loc[pd.MultiIndex.from_frame(roster[["model_a", "compound_id"]])]
    b = scores.loc[pd.MultiIndex.from_frame(roster[["model_b", "compound_id"]])]
    if set(a) != set(METHODS) or not np.isfinite(a).all(axis=None) or not np.isfinite(b).all(axis=None):
        raise ValueError("Incomplete paired RNA model support")
    freeze_table(pd.concat([roster[[*KEY, "track", "fold"]].assign(
        method=m, predicted_value=a[m].to_numpy()-b[m].to_numpy()) for m in METHODS]), args.out,
        provenance={"inputs_sha256": registration, "labels_sha256": ASSET_HASHES["supplement"], "original_sha256": seal["sha256"]})


def measured_pair_summary(values, groups):
    def group_mse(pairs):
        return float(np.mean([np.square(values.loc[a].to_numpy()-values.loc[b].to_numpy()).mean() for a,b in pairs]))

    within = np.sqrt(np.mean([group_mse(combinations(group, 2)) for group in groups]))
    between = np.sqrt(np.mean([group_mse(product(a, b)) for a, b in combinations(groups, 2)]))
    return {"drugs": values.shape[1], "within_specimen_rms": float(within), "between_specimen_rms": float(between),
            "within_to_between_ratio": float(within/between)}


def pair_eval(predicted, observed):
    joined = predicted.merge(observed, on=[*KEY, "track", "fold"], how="left", validate="many_to_one")
    if joined.endpoint_value.isna().any() or len(joined) != len(observed)*len(METHODS):
        raise ValueError("Sealed differences do not match the registered measured pairs")
    pieces = []
    for (method, donor, pair), block in joined.groupby(["method","lineage_group","cell_id"]):
        y, p = block.endpoint_value.to_numpy(), block.predicted_value.to_numpy()
        valid = y != 0
        pieces.append({"method": method,"donor":donor,"pair":pair,
                       "pearson":safe_corr(y,p),"spearman":safe_corr(rankdata(y),rankdata(p)),
                       "sign":float(np.mean(np.sign(y[valid])==np.sign(p[valid])))})
    grouped = pd.DataFrame(pieces).groupby(["method","donor"])[["pearson","spearman","sign"]].mean()
    output = {}
    for method, part in grouped.groupby(level="method"):
        output[method] = {name: {"point": float(s.mean()), "leave_one_donor_range": [
            float((s.sum()-s).min()/(len(s)-1)), float((s.sum()-s).max()/(len(s)-1))]}
            for name,s in part.items() if s.notna().all()}
    return output


def score(args):
    registration, query, groups, pairs, row_lookup = licob_panel(args)
    forecast, seal = verify_forecast(args.forecast)
    info = seal["provenance"]
    if (info["inputs_sha256"], info["labels_sha256"]) != (registration, ASSET_HASHES["supplement"]):
        raise ValueError("Different LICOB registration or outcome artefact")
    with closing(load_workbook(args.licob_labels, read_only=True, data_only=True)) as book:
        s6 = list(book["S6. AUC"].values)
    samples = sorted(set().union(*groups))
    if any(s6[row_lookup[m]-1][0] != m for m in samples):
        raise ValueError("Different LICOB S6 model identifiers")
    all_auc = pd.DataFrame([-np.asarray(s6[row_lookup[m]-1][1:], dtype=float) for m in samples], index=samples)
    drugs = query[["compound_id","response_column"]].drop_duplicates().set_index("compound_id").response_column.sort_index()
    matched = all_auc.loc[:, drugs.to_numpy()-1].set_axis(drugs.index, axis=1)
    if all_auc.shape != (9,76) or not np.isfinite(all_auc).all(axis=None):
        raise ValueError("Incomplete S6 measurements for paired PDOs")
    roster = pair_rows(pairs, list(drugs.index))
    observed = roster.assign(endpoint_value=[
        matched.loc[a,d]-matched.loc[b,d] for a,b,d in zip(roster.model_a, roster.model_b, roster.compound_id)
    ])
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "inputs_sha256": registration,
        "supplement_sha256": ASSET_HASHES["supplement"], "donors": len(groups), "samples": len(samples),
        "pairs": len(pairs), "full": measured_pair_summary(all_auc, groups), "matched": measured_pair_summary(matched, groups),
        "delta_models": pair_eval(forecast, observed),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for mode, fields in {"repeats": ("public",), "predict": ("licob","licob-labels","forecast","reference"),
                         "score": ("licob","licob-labels","forecast")}.items():
        stage = stages.add_parser(mode)
        for name in (*fields, "out"):
            stage.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    {"repeats": repeats, "predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
