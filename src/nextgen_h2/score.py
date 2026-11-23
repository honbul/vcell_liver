"""Consume the sealed H2 forecasts once and report paired native-endpoint scores."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from vc_build.external import native_sensitivity
from vc_build.io import publish_json, sha256

from .assets import COLD, GDSC, KEY, LIMORE, SUPPLEMENT, canonical_registry, linked_groups, parquet_lock
from .metrics import CELL, DRUG, Evaluator, contrast, macro, stack
from .run import META, read_json, response, subfold


def split_files(root, assets):
    decision = read_json(root / "r3_decision_receipt.json")
    chosen = sha256(root / "champion.json")
    if decision["selection_sha256"] != chosen or decision["outer_outcomes_opened"]:
        raise ValueError("Outer scoring requires the unmodified inner-selected policy")
    asset = read_json(assets / "data_receipt.json")
    registry = parquet_lock(assets / "folds.parquet", asset["files"]["folds.parquet"])
    forecasts, receipts = [], []
    for part in range(4):
        directory = root / "forecasts" / str(part)
        receipt = read_json(directory / "receipt.json")
        if (receipt["partition"], receipt["champion_sha256"], receipt["query_outcomes_loaded"]) != (part, chosen, 0):
            raise ValueError("Outer forecast predates final selection or exposed query outcomes")
        if sha256(directory / "predictor.pkl") != receipt["predictor_sha256"]:
            raise ValueError("Offline prediction package changed")
        frame = parquet_lock(directory / "predictions.parquet", receipt["prediction_sha256"])
        if (frame.row_id.duplicated().any() or not frame.structure_only.equals(frame.C1)
                or not frame.loc[~frame.champion_observed, "champion"].equals(frame.loc[~frame.champion_observed, "C1"])):
            raise ValueError("Forecast support or missing-profile fallback changed")
        if part < 3:
            _, keys = subfold(registry, part)
            replay = response(keys, labels=False)
            if not frame[META].equals(replay[META]):
                raise ValueError("Source query and forecast rosters differ")
        forecasts.append(frame)
        receipts.append(receipt)
    if sum(len(part) for part in forecasts[:3]) != 670830:
        raise ValueError("Source outer forecasts must cover the full registered response panel once")
    replay = read_json(root / "offline_replay.json")
    if replay["network_connections_allowed"] or len(replay["partitions"]) != 4:
        raise ValueError("The four sealed prediction packages require offline replay")
    for compared, receipt in zip(replay["partitions"], receipts):
        if (compared["partition"] != receipt["partition"] or compared["rows"] != receipt["query_rows"]
                or not compared["max_abs_diff"] < 1e-6 or compared["missing_fallback_max_abs_diff"] != 0.
                or compared["structure_only_fallback_max_abs_diff"] != 0.):
            raise ValueError(f"Offline replay receipt failed for partition {receipt['partition']}")
    return forecasts, receipts, registry, asset


def external_labels(assets, asset):
    lim = read_json(LIMORE / "receipt.json")["artifacts"]
    limore = parquet_lock(LIMORE / "test_inputs/rows.parquet", lim["test_inputs/rows.parquet"]["sha256"])
    limore_y = parquet_lock(LIMORE / "test_labels/outcomes.parquet", lim["test_labels/outcomes.parquet"]["sha256"])
    limore_y = limore_y.merge(limore[["record_id", "endpoint_direction"]], on="record_id", validate="one_to_one")
    limore_y = limore_y.assign(truth=native_sensitivity(limore_y)).rename(columns={"record_id": "row_id"})

    lc = read_json(COLD / "receipt.json")
    licob = parquet_lock(COLD / "query.parquet", lc["files"]["query.parquet"])
    if sha256(SUPPLEMENT) != lc["workbook_sha256"]:
        raise ValueError("LICOB public S6 workbook changed")
    book = load_workbook(SUPPLEMENT, data_only=True, read_only=True)
    try:
        worksheet = list(book["S6. AUC"].values)
    finally:
        book.close()
    values = []
    for row in licob.itertuples(index=False):
        specimen = worksheet[row.response_row - 1]
        v = specimen[row.response_column]
        if specimen[0] != row.cell_id or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError("Unexpected frozen LICOB AUC cell or row")
        values.append(-float(v))
    licob = licob.assign(truth=values)

    gdsc = parquet_lock(assets / "gdsc_query.parquet", asset["files"]["gdsc_query.parquet"])
    if sha256(GDSC) != asset["gdsc"]["raw_sha256"]:
        raise ValueError("GDSC1 held-out release changed")
    identity = ["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID"]
    gdsc_y = pd.read_csv(GDSC, usecols=[*identity, "AUC"], dtype={item: str for item in identity})
    gdsc_y = gdsc.merge(gdsc_y, on=identity, validate="one_to_many")
    gdsc_y = (gdsc_y.assign(truth=-pd.to_numeric(gdsc_y.AUC, errors="coerce"))
              .groupby("row_id", as_index=False).truth.median())
    labels = pd.concat([part[["row_id", "truth"]] for part in (limore_y, licob, gdsc_y)], ignore_index=True)
    if labels.row_id.duplicated().any() or not np.isfinite(pd.concat([limore_y.truth, licob.truth])).all():
        raise ValueError("External outcome rosters changed or contain invalid LIMORE/LICOB measurements")
    external_registry = canonical_registry(pd.concat([limore, licob, gdsc], ignore_index=True))
    return labels, licob, linked_groups(external_registry)


def compare_arms(frame, evaluator, arms, selector, name):
    pairs = []
    for arm in arms:
        pairs.extend(evaluator.score(arm, frame[arm].to_numpy(), selector, name))
    return pairs


def report_scope(frame, clusters, fold, scope, *, interaction):
    frame = frame.reset_index(drop=True)
    ev = Evaluator(frame, clusters, fold, scope=scope, interaction=interaction)
    all_rows = [("all", ("C0", "C1", "champion"), None),
                ("structure_only", ("structure_only", "C1"), None)]
    if frame.champion_observed.any():
        all_rows.append(("champion_observed", ("champion", "C1", "champion_mask", "champion_shuffle"),
                         frame.champion_observed))
    if (~frame.champion_observed).any():
        all_rows.append(("champion_unavailable", ("champion", "C1"), ~frame.champion_observed))
    for view in ("painting", "lincs"):
        available = frame[view + "_observed"]
        controls = ["C0", "C1", *(view + "/" + method for method in ("C2", "C3", "C4"))]
        if available.any():
            all_rows.append((view + "_observed", controls, available))
        all_rows.append((view + "_all", controls, None))
        challenger = [view + "/fusion_candidate", view + "/naive"]
        if all(arm in frame for arm in challenger):
            all_rows.append((view + "_fusion_all", ["C1", *challenger], None))
            if available.any():
                all_rows.append((view + "_fusion_observed", ["C1", *challenger], available))
    pairs = []
    for support, arms, selection in all_rows:
        pairs.extend(compare_arms(frame, ev, arms, selection, support))
    return pairs


def adult_licob(frame, licence):
    link = licence[["row_id", "donor_id", "adult_age_confirmed", "in_source", "remaining_unspecified_stereo"]]
    joint = frame.merge(link, on="row_id", validate="one_to_one")
    adult = joint[joint.adult_age_confirmed].copy()
    cols = [x for x in frame if x not in META]
    checks = [x for x in cols if x.endswith("_observed")]
    if any(adult.groupby(KEY)[col].nunique().max() != 1 for col in checks):
        raise ValueError("Drug-assay support changed within LICOB cases")
    names = ["donor_id", KEY, "source_id", "endpoint_id", "in_source", "remaining_unspecified_stereo"]
    adult = adult.groupby(names, as_index=False)[cols].mean().rename(columns={"donor_id": "cell_id"})
    adult[checks] = adult[checks].astype(bool)
    return adult.assign(lineage_group=adult.cell_id)


def summary(table, metrics):
    rows = []
    group = ["scope", "support", "arm"]
    for parts, block in table.groupby(group, sort=True):
        tag = dict(zip(group, parts))
        point = {metric: macro(block, metric) for metric in metrics if metric in block}
        rows.append(tag | {"screen": "macro", "fold": "all", "units": len(block)} | point)
        for (screen, fold), sub in block.groupby(["screen", "fold"]):
            rows.append(tag | {"screen": screen, "fold": str(fold), "units": len(sub)}
                        | {metric: float(sub[metric].mean()) for metric in metrics if metric in sub})
    return pd.DataFrame(rows)


def prereg_contrasts(d, c):
    rows = []
    for (scope, support), drug in d.groupby(["scope", "support"], sort=True):
        arms = set(drug.arm.unique())
        if support in ("all", "structure_only", "champion_unavailable"):
            trials = [("champion", "C1"), ("structure_only", "C1"), ("C1", "C0")]
        elif support == "champion_observed":
            trials = [("champion", method) for method in ("C1", "champion_mask", "champion_shuffle")]
        elif "_fusion_" in support:
            view = support.split("_")[0]
            trials = [(view + "/fusion_candidate", view + "/naive")]
        else:
            view = support.split("_")[0]
            trials = [(view + "/C4", method) for method in ("C1", view + "/C2", view + "/C3")]
        cell = c[c.scope.eq(scope) & c.support.eq(support)]
        for candidate, control in trials:
            if not {candidate, control} <= arms:
                continue
            for table, stats in ((drug, DRUG), (cell, CELL)):
                if table.empty:
                    continue
                for metric in stats:
                    if metric not in table:
                        continue
                    delta, lower, upper, clusters = contrast(table, candidate, control, metric)
                    rows.append({"scope": scope, "support": support, "candidate": candidate, "control": control,
                                 "metric": metric, "delta": delta, "ci95_lower": lower, "ci95_upper": upper,
                                 "clusters": clusters})
    return pd.DataFrame(rows)


def run(root, assets):
    frames, receipts, registry, asset = split_files(root, assets)
    score = root / "score"
    score.mkdir(exist_ok=False)
    lock = score / "evaluation_consumed.json"
    publish_json(lock, {"score_passes": 1, "pid": os.getpid(),
                       "forecasts": [proof["prediction_sha256"] for proof in receipts]})
    predictions, tissue = [], []
    groups = registry.scaffold_block
    cells = parquet_lock(assets / "cells.parquet", asset["files"]["cells.parquet"])
    for fold, dataset in enumerate(frames[:3]):
        _, test = subfold(registry, fold)
        held = response(test, labels=True)
        if not dataset.row_id.equals(held.row_id):
            raise ValueError("Native outer labels no longer match sealed forecasts")
        data = dataset.assign(truth=native_sensitivity(held), tissue=dataset.cell_id.map(cells.tissue))
        predictions.extend(report_scope(data, groups, fold, "DRP", interaction=True))
        for disease, part in data.groupby("tissue", sort=True):
            obs = part.reset_index(drop=True)
            ev = Evaluator(obs, groups, fold, scope=disease)
            for support, names, flag in (
                ("all", ["C0", "C1", "champion", "structure_only"], None),
                ("painting_observed", ["C1", "painting/C4"], obs.painting_observed),
                ("lincs_observed", ["C1", "lincs/C4"], obs.lincs_observed),
            ):
                tissue.extend(compare_arms(obs, ev, names, flag, support))
        print(f"Scored DRP fold {fold}", flush=True)
    labels, licob, external_groups = external_labels(assets, asset)
    external = frames[3].merge(labels, on="row_id", validate="one_to_one", how="left")
    if set(labels.row_id) != set(frames[3].row_id):
        raise ValueError("External forecast and label keys differ")
    finite = np.isfinite(external.truth)
    if not external.loc[~finite, "source_id"].eq("GDSC1").all():
        raise ValueError("Nonfinite outcome outside the public GDSC1 roster")
    external_unavailable = external.loc[~finite].groupby("source_id").size().to_dict()
    external = external.loc[finite]
    for dataset in ("LIMORE", "LICOB", "GDSC1"):
        subset = external[external.source_id.eq(dataset)]
        if dataset == "LICOB":
            subset = adult_licob(subset, licob)
            novel = subset[~subset.in_source & subset.remaining_unspecified_stereo.eq(0)]
            if (subset.cell_id.nunique(), subset[KEY].nunique(), novel[KEY].nunique()) != (38, 76, 18):
                raise ValueError("LICOB donor and novel parent populations differ from the lock")
            predictions.extend(report_scope(novel, external_groups, 0, "LICOB18", interaction=True))
        predictions.extend(report_scope(subset, external_groups, 0, dataset, interaction=True))
        print(f"Scored {dataset}", flush=True)

    d, c = stack(predictions)
    t_d, t_c = stack(tissue)
    points = pd.concat([summary(d, DRUG).assign(axis="drug"), summary(c, CELL).assign(axis="cell")])
    breakdown = pd.concat([summary(t_d, DRUG[:2]).assign(axis="drug"), summary(t_c, CELL).assign(axis="cell")])
    points.to_csv(score / "H2_METRICS.tsv", sep="\t", index=False, na_rep="NA")
    breakdown.to_csv(score / "H2_PANCANCER_BY_TISSUE.tsv", sep="\t", index=False, na_rep="NA")
    paired = prereg_contrasts(d, c)
    paired.to_csv(score / "H2_PAIRED_95CI.tsv", sep="\t", index=False, na_rep="NA")
    for name, data in [("unit_drug", d), ("unit_cell", c)]:
        data.to_parquet(score / f"{name}.parquet", index=False)
    publish_json(score / "score_receipt.json", {"score_passes": 1, "forecasts": receipts,
        "champion_sha256": sha256(root / "champion.json"), "score_guard_sha256": sha256(lock),
        "offline_replay_sha256": sha256(root / "offline_replay.json"),
        "source_rows": sum(len(part) for part in frames[:3]), "source_unique_parents": len(registry),
        "external_rows": external.groupby("source_id").size().to_dict(),
        "external_rows_missing_native_response": external_unavailable,
        "results": {path.name: sha256(path) for path in sorted(score.iterdir())}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("assets", type=Path)
    arguments = parser.parse_args()
    run(arguments.root, arguments.assets)
