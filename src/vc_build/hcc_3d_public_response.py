"""Use public organoid response measurements to adjust frozen RNA predictions."""

import argparse
from contextlib import closing
from io import BytesIO
from itertools import combinations
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import rankdata

from .evaluate import join_observations
from .external import KEY
from .hcc_study_pool import DRUGS, STUDIES, data, registered
from .io import freeze_table, publish_json, verify_forecast
from .limore_corrected import shared_support
from .v0plus_score import axis_metrics, correlations, crossed_patient_drug_ci
from .yang_hcc_external import require_hash

INPUT = "bd27132354dcf5537b717227099147514851ec0218d0c96920ec4c8a00a96a4c"
ARCHIVE = "d3cfcde3c4d4e71d29edacf1200176b2abbe5828b5ad0bddc7d266fc466f0616"
SOURCES = {
    "zhang-3dplc.zip": "af264009d89c28e481f180f1330a2131c4a57b84989966cd528783881eac4b06",
    "kim-hcc.zip": "c8fabfa454c453506292cf113a8889f1bf033af34c389e9adf3f21ab3f233598",
}
HOLDOUTS = ("LICOB", "Wong")
REFERENCES = ("2d_mean", "2d_teacher", "2d_ridge", "2d_cellhit", "balanced_residual")
PRIMARY = "public_graph"
ARMS = ("previous_graph", PRIMARY, "shuffled_graph")
CONTROL = ("2d_teacher", "previous_graph", "shuffled_graph")


def table(archive, name, sheet):
    matches = [part for part in archive.namelist() if Path(part).name.startswith(name) and part.endswith(".xlsx")]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous published table: {name}")
    with closing(load_workbook(BytesIO(archive.read(matches[0])), read_only=True, data_only=True)) as book:
        return list(book[sheet].values)


def canonical_name(value):
    clean = str(value).strip().lower()
    return {"5-fu": "fluorouracil", "5-fluorouracil": "fluorouracil"}.get(clean, clean)


def zhang_rows(rows, donors):
    drugs = {j: canonical_name(name) for j, (name, key) in enumerate(zip(rows[1], rows[2])) if key == "AUC"}
    measures = []
    for row in rows[3:]:
        match = re.fullmatch(r"(?:PLC|patient)([0-9]+)", str(row[0]))
        if match is None or int(match[1]) not in donors:
            continue
        donor = f"Zhang:{int(match[1]):03d}"
        measures.extend((donor, name, -float(row[j])) for j, name in drugs.items()
                        if isinstance(row[j], (int, float)) and np.isfinite(row[j]))
    result = pd.DataFrame(measures, columns=["lineage_group", "drug", "endpoint_value"])
    if result.empty or result.duplicated(["lineage_group", "drug"]).any():
        raise ValueError("Unexpected bioprinted organoid response keys")
    return result


def zhang_inputs(path):
    with zipfile.ZipFile(path) as archive:
        patients = table(archive, "Table S1.", "S1. Clinical information")
        if patients[2][2] != "Age" or patients[2][6] != "Neoadjuvant therapy":
            raise ValueError("Unexpected Zhang clinical identity fields")
        treated = {}
        for row in patients[3:]:
            match = re.fullmatch(r"PLC([0-9]+)", str(row[0]))
            if match and row[5] == "HCC" and isinstance(row[2], (int, float)) and row[2] >= 18:
                treated[int(match[1])] = row[6] is not None
        if len(treated) != 51:
            raise ValueError("Unexpected number of adult HCC 3DP models")
        mono = zhang_rows(table(archive, "Table S3.", "TKI drugs"), treated)
        chemo = zhang_rows(table(archive, "Table S3.", "Chemotherapy drugs"), treated)
        co = zhang_rows(table(archive, "Table S13.", "Sheet1"), treated)
        eligible = {f"Zhang:{donor:03d}" for donor, listed in treated.items() if not listed}
        fit = pd.concat([mono, chemo], ignore_index=True)
    return fit[fit.lineage_group.isin(eligible)], mono, co


def kim_inputs(path):
    with zipfile.ZipFile(path) as archive:
        rows = table(archive, "Supplementary Table 1, 4, 7 and 10_Revised", "Table S7")
    drug_columns = {j: canonical_name(name) for j, name in enumerate(rows[3])
                    if j > 0 and name is not None and rows[4][j + 1] == "AUC"}
    if len(drug_columns) != 8:
        raise ValueError("Unexpected Kim pharmacology support")
    records, donor, trial = [], None, 0
    for row in rows[5:]:
        match = re.fullmatch(r"SNU-([0-9]+)-(?:TO|CO)", str(row[0]))
        if match:
            donor, trial = f"Kim:{match[1]}", 0
        elif row[0] is not None:
            continue
        if donor is None or trial >= 3 or str(row[1]).strip().lower() in ("mean", "sd"):
            continue
        for j, name in drug_columns.items():
            if isinstance(row[j + 1], (int, float)) and np.isfinite(row[j + 1]):
                records.append((donor, name, trial, str(row[j]), -float(row[j + 1])))
        trial += 1
    result = pd.DataFrame(records, columns=["lineage_group", "drug", "repeat", "passage", "endpoint_value"])
    if result.lineage_group.nunique() != 18 or result.duplicated(["lineage_group", "drug", "repeat"]).any():
        raise ValueError("Unexpected Kim donor or replicate identities")
    return result


def public_inputs(root):
    for name, digest in SOURCES.items():
        require_hash(root / name, digest)
    zhang, mono, co = zhang_inputs(root / "zhang-3dplc.zip")
    kim = kim_inputs(root / "kim-hcc.zip")
    consensual = kim.groupby(["lineage_group", "drug"], as_index=False).endpoint_value.mean()
    return pd.concat([zhang.assign(study="Zhang"), consensual.assign(study="Kim")]), (mono, co, kim)


def into_panel(public, roster):
    candidates = roster[["compound_id", "source_compound_label"]].drop_duplicates()
    labels = candidates.assign(drug=candidates.source_compound_label.map(canonical_name))[["compound_id", "drug"]]
    labels = labels.drop_duplicates()
    labels = labels[~labels.drug.duplicated(keep=False)]
    selected = public.merge(labels, on="drug", validate="many_to_one")
    if selected.duplicated(["study", "lineage_group", "compound_id"]).any():
        raise ValueError("Published active names did not map to unique registered compounds")
    return selected


def associations(fit, drugs):
    support = {}
    count = {}
    for name, cohort in fit.groupby("study", sort=True):
        wide = cohort.groupby(["lineage_group", "compound_id"]).endpoint_value.mean().unstack().reindex(columns=drugs)
        count[name] = {"groups": int(wide.dropna(how="all").shape[0]), "compounds": int(wide.notna().any().sum())}
        for a, b in combinations(drugs, 2):
            paired = wide[[a, b]].dropna()
            if len(paired) >= 8 and paired.nunique().min() > 1:
                corr = float(paired.corr(method="spearman").iloc[0, 1])
                support.setdefault((a, b), {})[name] = corr * len(paired) / (len(paired) + 10)
    graph = pd.DataFrame(0., index=drugs, columns=drugs)
    retained = []
    for (a, b), estimates in support.items():
        values = np.array(list(estimates.values()))
        if len(values) >= 2 and (np.all(values > 0) or np.all(values < 0)):
            graph.at[a, b] = graph.at[b, a] = values.mean()
            retained.append({"drug_a": a, "drug_b": b, "source_assays": sorted(estimates), "weight": float(values.mean())})
    return graph, {"source_support": count, "retained": retained, "pairs_with_two_studies": sum(len(x) >= 2 for x in support.values())}


def adjust(archived, old, new):
    panel = archived.pivot(index=KEY, columns="method", values="predicted_value").reset_index()
    pure = panel.assign(effect=panel["2d_teacher"] - panel["2d_mean"]).pivot(index="cell_id", columns="compound_id", values="effect")
    pure = pure.loc[:, old.index]
    centered = pure.sub(pure.mean(axis=1), axis=0)
    shuffled = new.to_numpy().copy()
    active = np.flatnonzero(np.any(shuffled != 0, axis=0))
    ordering = np.arange(len(new))
    ordering[active] = np.random.default_rng(1031).permutation(active)
    shuffled = shuffled[np.ix_(ordering, ordering)]
    matrices = (old.to_numpy(), new.to_numpy(), shuffled)
    cols, rows = pure.columns.get_indexer(panel.compound_id), pure.index.get_indexer(panel.cell_id)
    if not np.isfinite(centered).all(axis=None) or min(cols.min(), rows.min()) < 0:
        raise ValueError("The held RNA prediction is not a complete registered panel")
    parts = []
    for arm, graph in zip(ARMS, matrices):
        weight = graph / np.maximum(1., np.abs(graph).sum(axis=1, keepdims=True))
        delta = (centered.to_numpy() @ weight.T)[rows, cols]
        parts.append(panel[KEY].assign(method=arm, predicted_value=panel["2d_teacher"] + .35 * delta))
    return pd.concat(parts, ignore_index=True), int(np.count_nonzero(ordering != np.arange(len(new))))


def predict(args):
    if registered(args.inputs) != INPUT:
        raise ValueError("Unexpected study design")
    depth = args.licob_book if args.holdout == "LICOB" else args.wong_book
    if depth is not None:
        raise ValueError("Predictor received the held response file")
    prior, receipt = verify_forecast(args.archive)
    if (receipt["sha256"], receipt["provenance"]["input_sha256"]) != (ARCHIVE, INPUT):
        raise ValueError("Unexpected frozen RNA predictions")
    query = pd.read_parquet(args.inputs / f"{args.holdout}.parquet")
    comparison = prior[prior.track.eq(args.holdout) & prior.method.isin(REFERENCES)]
    shared_support(comparison, query, REFERENCES)
    old = pd.concat([data(args, study, full=False) for study in STUDIES if study != args.holdout], ignore_index=True)
    public, _ = public_inputs(args.public)
    public = into_panel(public, query)
    fields = ["study", "lineage_group", "compound_id", "endpoint_value"]
    drugs = pd.Index(sorted(query.compound_id.unique()))
    base, original = associations(old, drugs)
    graph, expanded = associations(pd.concat([old[fields], public[fields]], ignore_index=True), drugs)
    result, moved = adjust(comparison, base, graph)
    mixed = pd.concat([comparison, result.assign(track=args.holdout, fold=0)], ignore_index=True)
    shared_support(mixed, query, (*REFERENCES, *ARMS))
    changed = graph.index[np.abs(graph.to_numpy() - base.to_numpy()).max(axis=1) > 1e-12].tolist()
    freeze_table(mixed, args.out, provenance={
        "holdout": args.holdout, "inputs_sha256": INPUT, "archive_sha256": ARCHIVE,
        "public_sha256": SOURCES, "previous": original, "expanded": expanded,
        "moved_drugs": moved, "drugs_with_changed_relations": changed,
    })


def complete_panel_metrics(a, b, *, bootstrap):
    # Correlations compare the same donor blocks across contexts; invalid constant draws are discarded.
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    pair = np.isfinite(x) & np.isfinite(y)
    x, y = (np.where(pair, v, np.nan) for v in (x, y))
    indices = np.random.default_rng(87819).integers(len(x), size=(2400, len(x))) if bootstrap else None
    score = {}
    for name in ("pearson", "spearman"):
        def correlation(u, v):
            if name == "spearman":
                u, v = (rankdata(z, axis=1, nan_policy="omit") for z in (u, v))
            return correlations(u, v)
        observed = correlation(x[None, :, :], y[None, :, :])[0]
        ci = None if indices is None else np.nanquantile(np.nanmean(correlation(x[indices], y[indices]), axis=1), [.025, .975]).tolist()
        score[name] = {"macro": float(np.nanmean(observed)), "ci95_patient": ci,
                       "per_drug": {k: (float(v) if np.isfinite(v) else None) for k, v in zip(a.columns, observed)}}
    keep = pair.all(axis=1)
    first, second = x[keep], y[keep]
    ordinal, top = None, None
    if len(first):
        r1, r2 = (rankdata(v, axis=1) for v in (first, second))
        x1, x2 = (arr - arr.mean(axis=1, keepdims=True) for arr in (r1, r2))
        denom = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)
        ordinal = np.divide((x1 * x2).sum(axis=1), denom, out=np.full(len(denom), np.nan), where=denom > 0)
        a_top, b_top = (v == v.max(axis=1, keepdims=True) for v in (first, second))
        top = (a_top & b_top).sum(axis=1) / (a_top.sum(axis=1) * b_top.sum(axis=1))
        donor_draws = (np.random.default_rng(87819).integers(len(first), size=(2400, len(first)))
                       if bootstrap else None)

        def patient_summary(values):
            ci = None if donor_draws is None else np.nanquantile(
                np.nanmean(values[donor_draws], axis=1), [.025, .975],
            ).tolist()
            return {"value": float(np.nanmean(values)), "ci95_patient": ci}

        ordinal, top = patient_summary(ordinal), patient_summary(top)
    return {"within_drug": score, "per_drug_donors": dict(zip(a.columns, pair.sum(axis=0).tolist())),
            "complete_donors": int(keep.sum()), "within_donor_spearman": ordinal, "top1_agreement": top}


def direct(args):
    _, (mono, co, kim) = public_inputs(args.public)
    a, b = (table.pivot(index="lineage_group", columns="drug", values="endpoint_value") for table in (mono, co))
    a, b = a.align(b, join="inner")
    cafe = complete_panel_metrics(a, b, bootstrap=True)
    values = kim.pivot(index=["lineage_group", "drug"], columns="repeat", values="endpoint_value").dropna()
    papers = []
    for j in values.columns:
        early = values[j].unstack("drug")
        late = values.drop(columns=j).mean(axis=1).unstack("drug")
        papers.append(complete_panel_metrics(early, late, bootstrap=True))
    return {"co_culture": cafe, "kim_leave_one_measurement": papers,
            "kim_complete_donor_drug_pairs": len(values),
            "kim_distinct_passage_pairs": int((kim.groupby(["lineage_group", "drug"]).passage.nunique() > 1).sum())}


def matched_scores(joined, graph_drugs):
    anchor = joined[joined.method.eq("2d_mean")].groupby("compound_id").predicted_value
    if not anchor.nunique().eq(1).all():
        raise ValueError("2D fixed mean lost its within-drug identity")
    block = joined.groupby(["lineage_group", "compound_id", "track", "fold", "method"], as_index=False)[
        ["predicted_value", "endpoint_value"]].mean()
    mean = block.method.eq("2d_mean")
    block.loc[mean, "predicted_value"] = block.loc[mean, "compound_id"].map(anchor.first())
    block = block.assign(cell_id=block.lineage_group, row_id=block.lineage_group + "|" + block.compound_id)
    pairs = [(PRIMARY, b) for b in CONTROL]

    def measure(frame):
        report = axis_metrics(frame, pairs)[0]
        if frame.compound_id.nunique() <= 5:
            for row in report["scores"]:
                row.pop("macro_ndcg_at_5", None)
                row.pop("macro_top5_overlap", None)
            report["paired_contrasts"] = [r for r in report["paired_contrasts"] if r["metric"] == "spearman"]
        return report

    overall = measure(block)
    across = {b: crossed_patient_drug_ci(block, a, b) for a, b in pairs}
    within = {(r["baseline"], r["metric"]): r["ci95"] for r in overall["paired_contrasts"]}
    return {"groups": block.lineage_group.nunique(), "compounds": block.compound_id.nunique(), "overall": overall,
            "changed_drugs": measure(block[block.compound_id.isin(graph_drugs)]) if graph_drugs else None,
            "hcc_tki": measure(block[block.compound_id.isin(DRUGS)]),
            "paired_ci95": across, "pass": {b: all(v[0] > 0 for v in across[b].values())
            and all(within[b, metric][0] > -.005 for metric in ("spearman", "ndcg_at_5")) for b in CONTROL}}


def score(args):
    if registered(args.inputs) != INPUT:
        raise ValueError("Unexpected study data")
    result, frozen = {}, {}
    for study in HOLDOUTS:
        forecast, stamp = verify_forecast(args.forecasts / f"{study}.parquet")
        info = stamp["provenance"]
        if (info["holdout"], info["inputs_sha256"], info["archive_sha256"], info["public_sha256"]) != (
                study, INPUT, ARCHIVE, SOURCES):
            raise ValueError("The frozen study comparison changed")
        truth = data(args, study, full=True)
        shared_support(forecast, truth, (*REFERENCES, *ARMS))
        result[study] = matched_scores(join_observations(forecast, truth), info["drugs_with_changed_relations"])
        frozen[study] = stamp["sha256"]
    publish_json(args.out, {"cohorts": result, "forecast_sha256": frozen, "direct_assay_comparisons": direct(args),
                            "pass": all(all(row["pass"].values()) for row in result.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {"predict": ("archive",), "score": ("forecasts",)}.items():
        sub = stages.add_parser(stage)
        for field in ("inputs", "transfer", "public", "out", *fields):
            sub.add_argument(f"--{field}", type=Path, required=True)
        for field in ("licob-book", "wong-book"):
            sub.add_argument(f"--{field}", type=Path)
        if stage == "predict":
            sub.add_argument("--holdout", choices=HOLDOUTS, required=True)
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
