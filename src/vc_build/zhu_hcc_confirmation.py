"""Fit fixed comparators and score the first protected use of the Zhu assays."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, expanded_targets
from .external import KEY, SOURCE, _validate
from .io import freeze_table, publish_json, verify_forecast
from .licob_full_axis import validated
from .licob_icc_bridge import ANCHOR, ARMS, BASE, PRIMARY, SHUFFLE, forecast, icc_inputs
from .limore_corrected import shared_support
from .v0plus_cv import candidate_forecasts, molecular_inputs, training_inputs
from .v0plus_neural import load_auxiliary
from .v0plus_score import CYTOTOXIC, HCC_DRUGS, axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .yang_hcc_external import SETTINGS, require_hash
from .zhu_hcc_public import ASSETS, EXPANDED, SHEETS

ZHU = "107d42c75f152697033c2f9d3763def485c12c237672db389d82316e2af6e4dc"
ICC = "4a338c39f8418cc1b5edd60bdb972155f72fa1a41e8c1a2e068afd541735668e"
SCOPE = ("auc", "single_dose")
PUBLIC = ("cellhit", "codeae_inductive", "velodrome_inductive")
COMPONENTS = {
    "linear": ("drug_mean", "additive", "drug_ridge", "drug_elasticnet"),
    "lightgbm": ("lightgbm",),
    **{f"cellhit-{i}": ("cellhit",) for i in range(3)},
    "catboost": ("catboost",),
    "codeae_inductive": ("codeae_inductive",),
    "velodrome_inductive": ("velodrome_inductive",),
    "v0plus_inductive": ("v0plus_inductive",),
    "icc": ARMS,
}
REFERENCES = tuple(dict.fromkeys(m for block, methods in COMPONENTS.items() if block != "icc" for m in methods))
METHODS = (*REFERENCES, *ARMS, *("icc_offset_" + name for name in REFERENCES))


def read_rosters(path, *, expression=False):
    receipt = path / "receipt.json"
    require_hash(receipt, ZHU)
    manifest = json.loads(receipt.read_text())
    for key in (*[f"{label}_roster.parquet" for label in SCOPE], *(("rna.parquet",) if expression else ())):
        require_hash(path / key, manifest["files"][key])
    rosters = {label: pd.read_parquet(path / f"{label}_roster.parquet") for label in SCOPE}
    if [len(q) for q in rosters.values()] != [288, 777]:
        raise ValueError("The first-use Zhu roster changed")
    return rosters


def training_bundle(args):
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    require_hash(args.expanded / "receipt.json", EXPANDED)
    core, source_rna, _ = training_inputs(args.core)
    extension_meta = json.loads((args.expanded / "receipt.json").read_text())
    path = args.expanded / "train.parquet"
    require_hash(path, extension_meta["files"][path.name])
    rosters = read_rosters(args.zhu, expression=True)
    # Use one forecast per unique specimen–drug across both outcomes.
    pairs = pd.concat([q[[*KEY[1:], "isomeric_smiles"]] for q in rosters.values()]).drop_duplicates()
    probe = pairs.assign(row_id=pairs.cell_id + "|" + pairs.compound_id).sort_values("row_id", ignore_index=True)
    additions = set(probe.compound_id) - set(core.compound_id)
    extra = pd.read_parquet(path, filters=[("compound_id", "in", sorted(additions))])
    if additions != set(extra.compound_id) or not set(extra.cell_id) <= set(core.cell_id):
        raise ValueError("The exact requested additions changed the source identity")
    fit = pd.concat([core, extra], ignore_index=True)
    if fit.duplicated(["cell_id", "compound_id", *SOURCE]).any():
        raise ValueError("Duplicate anchor observations across training releases")
    held = pd.read_parquet(args.zhu / "rna.parquet").loc[sorted(probe.cell_id.unique())]
    require_hash(args.icc / "receipt.json", ICC)
    icc, response, _ = icc_inputs(args.icc, validated(args.licob))
    genes = source_rna.columns.intersection(held.columns).intersection(icc.columns).sort_values()
    rna = pd.concat([source_rna[genes], held[genes]], verify_integrity=True)
    _validate(fit, probe)
    return fit, probe, rna, icc[genes], response, additions


def part_roster(probe, part):
    if part.startswith("cellhit-"):
        group = int(part.split("-")[1])
        selected = np.array_split(sorted(probe.compound_id.unique()), 3)[group]
        return probe[probe.compound_id.isin(selected)]
    return probe


def fit_part(args):
    require_hash(args.settings, SETTINGS)
    fit, probe, rna, icc, measured, additions = training_bundle(args)
    query = part_roster(probe, args.part)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    targets, mapping = expanded_targets(fit[fit.compound_id.isin(additions)], targets, rna.columns)
    aux, aux_meta = None, None
    if args.part in ("icc", "v0plus_inductive"):
        aux, aux_meta = load_auxiliary(args.auxiliary)
    if args.part == "icc":
        rows, offsets, audit = forecast(fit, query, rna, icc, measured, edges, targets, aux)
        audit["offsets"] = {d: float(offsets[d]) for d in sorted(set(query.compound_id))}
    else:
        arm = args.part.split("-")[0]
        if arm in PUBLIC:
            verify_upstream(args.upstream)
        settings = json.loads(args.settings.read_text())["settings"][arm]
        rows, _ = candidate_forecasts(arm, fit, query, rna, edges, targets, gpu=args.gpu,
                                      selected=[tuple(s) for s in settings], upstream=args.upstream, auxiliary=aux)
        rows.method = rows.method.str.split("/").str[0]
        audit = {}
    shared_support(rows, query, COMPONENTS[args.part])
    freeze_table(rows.assign(track="ZHU", fold=0), args.out, provenance={
        "cohort_sha256": ZHU, "core_sha256": RECEIPTS["core"], "expanded_sha256": EXPANDED, "icc_sha256": ICC,
        "settings_sha256": SETTINGS, "part": args.part, "training_rows": len(fit),
        "training_drugs": fit.compound_id.nunique(), "union_rows": len(probe),
        "gene_count": len(rna.columns), "targets_mapped": mapping, "priors": priors, "aux": aux_meta, "audit": audit,
    })


def merge(args):
    rosters = read_rosters(args.zhu)
    probe = (pd.concat([q[KEY[1:]] for q in rosters.values()]).drop_duplicates()
             .assign(row_id=lambda x: x.cell_id + "|" + x.compound_id))
    parts, hashes = [], {}
    for arm, methods in COMPONENTS.items():
        rows, stamp = verify_forecast(args.parts / f"{arm}.parquet")
        info = stamp["provenance"]
        if (info["cohort_sha256"], info["icc_sha256"], info["settings_sha256"], info["part"]) != (ZHU, ICC, SETTINGS, arm):
            raise ValueError("An external comparator does not match its fixed plan")
        shared_support(rows, part_roster(probe, arm), methods)
        parts.append(rows)
        hashes[arm] = stamp["sha256"]
        if arm == "icc":
            offsets = info["audit"]["offsets"]
    all_rows = pd.concat(parts, ignore_index=True)
    controls = all_rows[all_rows.method.isin(REFERENCES)]
    equal = controls.assign(
        method="icc_offset_" + controls.method,
        predicted_value=controls.predicted_value + controls.compound_id.map(offsets),
    )
    joined = pd.concat([all_rows, equal], ignore_index=True)
    shared_support(joined, probe, METHODS)
    per_assay = [q[KEY].merge(joined[[*KEY[1:], "method", "predicted_value"]],
                              on=KEY[1:], validate="one_to_many").assign(track=name, fold=0)
                 for name, q in rosters.items()]
    freeze_table(pd.concat(per_assay, ignore_index=True), args.out, provenance={
        "cohort_sha256": ZHU, "icc_sha256": ICC, "settings_sha256": SETTINGS, "components_sha256": hashes,
        "unseen_assays": list(SCOPE), "offset_positive_support": sum(float(v) != 0 for v in offsets.values()),
    })


def observed(book, label, query):
    sheet = book[SHEETS[SCOPE.index(label)]]
    matrix = list(sheet.values)
    labels = []
    for item in query.itertuples(index=False):
        row, col = item.sheet_row - 1, item.sheet_column
        if label == "auc":
            sample = matrix[row][0]
            key = matrix[1][col]
            if not isinstance(key, str) or key != f"drug{col}":
                raise ValueError("The AUC response header changed")
        else:
            sample = matrix[1][col]
            if str(matrix[row][0]).strip() != item.source_compound_label:
                raise ValueError("The 10 µM response annotation changed")
        if str(sample).strip() != item.cell_id:
            raise ValueError("An endpoint cell is outside its registered donor")
        raw = matrix[row][col]
        if raw is None or raw in ("", "NA", "N/A", "-"):
            labels.append(np.nan)
        elif not isinstance(raw, (int, float)) or not np.isfinite(raw):
            raise ValueError("Non-numeric registered endpoint")
        else:
            labels.append(-float(raw) if label == "auc" else float(raw))
    return query.assign(endpoint_value=labels).dropna(subset=["endpoint_value"])


def annotated_roster(book, label, query):
    if label == "single_dose":
        sheet = book["Chemical Data-301cpds"]
        names = {}
        for row in sheet.iter_rows(min_row=2, max_col=7, values_only=True):
            if row[1] is not None:
                names[str(row[1]).strip()] = str(row[6] or "")
        query = query.assign(drug_target_source_labels=query.source_compound_label.map(names).fillna(""))
    pattern = re.compile(r"^(" + "|".join(sorted(HCC_DRUGS | CYTOTOXIC)) + r")\b", re.I)
    canonical = {name.casefold(): name for name in HCC_DRUGS | CYTOTOXIC}
    values = query.source_compound_label.str.extract(pattern)[0].str.casefold().map(canonical)
    return query.assign(source_compound_label=values.fillna(query.source_compound_label))


def results(joined, query, label):
    metrics_by_rank = ("spearman", "ndcg_at_5", "top5_overlap")
    comparisons = (BASE, ANCHOR, SHUFFLE, *("icc_offset_" + r for r in REFERENCES))
    pairs = [(PRIMARY, c) for c in comparisons] + [(ANCHOR, BASE)]
    report, drug_values, _ = axis_metrics(joined, pairs)
    intervals = {c: crossed_patient_drug_ci(joined, PRIMARY, c, repeats=1600) for c in comparisons}
    contrasts = {(p["candidate"], p["baseline"], p["metric"]): p["ci95"] for p in report["paired_contrasts"]}
    support = {"donors": joined.lineage_group.nunique(), "rows": joined.row_id.nunique(),
               "drugs_at_least_5_donors": joined.drop_duplicates("row_id").groupby("compound_id").size().ge(5).sum().item()}
    registered = support["donors"] >= (12 if label == "auc" else 16) and support["drugs_at_least_5_donors"] >= (
        12 if label == "auc" else 25)
    anchor_gates = {m: contrasts[ANCHOR, BASE, m][0] > (0. if m in ("spearman", "ndcg_at_5") else -.005)
                    for m in metrics_by_rank}
    gates = {c: all(intervals[c][m][0] > 0 for m in ("spearman", "pearson"))
             and all(contrasts[PRIMARY, c, m][0] > -.005 for m in metrics_by_rank) for c in comparisons}
    strata = {name: {"drugs": part.compound_id.nunique(), "metrics": axis_metrics(part, ())[0]}
              for name, drugs in compound_strata(query).items()
              if len(part := joined[joined.compound_id.isin(drugs)])}
    per_drug = drug_values.astype(object).where(pd.notna(drug_values), None).to_dict("records")
    return {"metrics": report, "per_drug": per_drug, "bootstrap": intervals, "support": support, "min_support_pass": registered,
            "anchor_gates": anchor_gates, "anchor_pass": registered and all(anchor_gates.values()),
            "gates": gates, "joint_pass": registered and all(gates.values()), "strata": strata}


def score(args):
    require_hash(args.labels, ASSETS["zhu-s4.xlsx"])
    rosters = read_rosters(args.zhu)
    pred, stamp = verify_forecast(args.forecast)
    info = stamp["provenance"]
    if (info["cohort_sha256"], info["icc_sha256"], info["settings_sha256"]) != (ZHU, ICC, SETTINGS):
        raise ValueError("The Zhu outcome reader requires the complete prediction freeze")
    output = {}
    with closing(load_workbook(args.labels, read_only=True, data_only=True)) as book:
        for label, query in rosters.items():
            subset = pred[pred.track.eq(label)]
            shared_support(subset, query, METHODS)
            truth = observed(book, label, query)
            matched = join_observations(subset[subset.row_id.isin(truth.row_id)], truth)
            output[label] = results(matched, annotated_roster(book, label, query), label)
    publish_json(args.out, {"forecast_sha256": stamp["sha256"], "cohort_sha256": ZHU,
                            "assays": output, "primary_assay": "auc",
                            "label_sha256": ASSETS["zhu-s4.xlsx"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in {
        "predict": ("zhu", "core", "expanded", "icc", "licob", "settings", "priors", "hallmark", "targets", "auxiliary", "upstream"),
        "merge": ("zhu", "parts"),
        "score": ("zhu", "forecast", "labels"),
    }.items():
        p = stages.add_parser(stage)
        for field in ("out", *fields):
            p.add_argument("--" + field, type=Path, required=True)
        if stage == "predict":
            p.add_argument("--part", choices=COMPONENTS, required=True)
            p.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"predict": fit_part, "merge": merge, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
