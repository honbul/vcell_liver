"""Seal source-only forecasts for previously unscored drugs in the public Wong cohort."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS, expanded_targets, registered_inputs
from .external import KEY, SOURCE, _validate
from .function_data import new_output
from .hcc_panel_support import GDSC_RECEIPT
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_components import ARMS, component_scores, forecast, interaction_intervals
from .v0plus_cv import candidate_forecasts, molecular_inputs, panel_targets
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .v0plus_wong import WORKBOOK_HASH, auc_mean, read_registration, workbook
from .yang_hcc_external import SETTINGS, kernel_predictions, require_hash

COHORT = "WONG_EXPANDED_3D_HCC"
ORIGINAL_WONG = "1b7871c110bf61a7dd22d374a95e5bb68e0cdad6831eeb8f6de54c8cd469cc0d"
SUPPORT = "c56df24d431aac8d9a53e31e778eb51c28f22d711bd0cef2f37a401c5868ec07"
PRIMARY = "interaction_only"
COMPARATORS = ("drug_mean", "drug_ridge", "cellhit", "structured")
LINEAR = ("drug_mean", "drug_ridge", "drug_elasticnet", "additive")
PUBLIC = ("cellhit", "codeae_inductive", "velodrome_inductive")
COMPONENTS = {
    "linear": LINEAR, "lightgbm": ("lightgbm",), "catboost": ("catboost",),
    **{f"cellhit-{n}": ("cellhit",) for n in range(3)},
    **{name: (name,) for name in (*PUBLIC[1:], "v0plus_inductive")},
    "kernel": ("structured", "structured_rna_chem", *ARMS[:-1]),
}
METHODS = tuple(dict.fromkeys(name for methods in COMPONENTS.values() for name in methods))
SOURCE_STAMPS = RECEIPTS | {"gdsc": GDSC_RECEIPT, "original_wong": ORIGINAL_WONG, "support": SUPPORT}


def register(args):
    require_hash(args.wong / "receipt.json", ORIGINAL_WONG)
    require_hash(args.support, SUPPORT)
    require_hash(args.gdsc / "receipt.json", GDSC_RECEIPT)
    read_registration(args.wong)
    _, core, rna, extended = registered_inputs(args, include_extension=True)
    manifest = json.loads(args.support.read_text())["cohorts"]["Wong"]
    extra_drugs = set(manifest["extra_unscored_supported_by_1129_panel"])
    gdsc_drugs = set(manifest["extra_unscored_supported_by_gdsc_candidate"])
    gdsc_path = args.gdsc / "additional_rows.parquet"
    require_hash(gdsc_path, json.loads((args.gdsc / "receipt.json").read_text())["artifacts"][gdsc_path.name])
    gdsc = pd.read_parquet(gdsc_path, filters=[("compound_id", "in", sorted(gdsc_drugs))])
    extra = extended[extended.compound_id.isin(extra_drugs)]
    if (set(extra.compound_id), set(gdsc.compound_id)) != (extra_drugs, gdsc_drugs):
        raise ValueError("A registered exact source drug is missing")
    if not set(gdsc.cell_id) <= set(core.cell_id) or not set(gdsc.lineage_group) <= set(core.lineage_group):
        raise ValueError("GDSC extension adds unregistered training models or lineages")
    train = pd.concat([core, extra, gdsc], ignore_index=True)
    if train.duplicated(["cell_id", "compound_id", *SOURCE]).any() or train.row_id.duplicated().any():
        raise ValueError("Repeated training identities in compound augmentation")
    shapes = train[["compound_id", "isomeric_smiles"]].drop_duplicates()
    old = pd.read_parquet(args.wong / "query.parquet")
    held = pd.read_parquet(args.wong / "rna.parquet")
    catalogue = pd.read_parquet(args.wong / "chemistry.parquet")
    known = set(old.compound_id)
    new = extra_drugs | gdsc_drugs
    if known & new or (len(known), len(extra_drugs), len(gdsc_drugs), len(held)) != (27, 32, 3, 10):
        raise ValueError("Unexpected old/new compound or model support")
    drugs = catalogue[catalogue.compound_id.isin(known | new)].merge(shapes, on="compound_id", validate="one_to_one")
    drugs = drugs.assign(source_compound_label=drugs.product_name, drug_target_source_labels=drugs.mechanism,
                         previously_scored=drugs.compound_id.isin(known))
    query = pd.DataFrame({"cell_id": held.index}).merge(drugs, how="cross")
    query = query.assign(lineage_group=query.cell_id,
                         row_id=COHORT + "|" + query.cell_id + "|" + query.compound_id)
    if len(query) != 620:
        raise ValueError("Expanded forecast requires a complete 10-by-62 roster")
    _validate(train, query)
    _, core_anchor, _, _ = panel_targets(core)
    _, anchor, _, calibration = panel_targets(train)
    shift = anchor.reindex(core_anchor.index) - core_anchor
    genes = rna.columns.intersection(held.columns)
    row_support = extra.groupby("compound_id").size().to_dict() | gdsc.groupby("compound_id").size().to_dict()
    with new_output(args.out) as out:
        files = {"train.parquet": train, "query.parquet": query,
                 "rna.parquet": pd.concat([rna.loc[sorted(train.cell_id.unique()), genes], held[genes]],
                                           verify_integrity=True)}
        for name, table in files.items():
            table.to_parquet(out / name, index=name == "rna.parquet")
        publish_json(out / "receipt.json", {
            "cohort": COHORT, "input_sha256": SOURCE_STAMPS, "workbook_sha256": WORKBOOK_HASH,
            "files": {name: sha256(out / name) for name in files},
            "query_drugs": len(drugs), "previous_drugs": len(known), "novel_drugs": len(new),
            "donors": len(held), "genes": len(genes), "source_rows": len(train),
            "source_models": int(train.cell_id.nunique()), "source_drugs": int(train.compound_id.nunique()),
            "source_rows_by_assay": train.source_id.value_counts().to_dict(),
            "calibration": calibration, "maximum_old_anchor_change": float(shift.abs().max()),
            "novel_drug_min_max_source_rows": [min(row_support.values()), max(row_support.values())],
            "new_drug_source_support": row_support,
        })


def checked(path, names):
    receipt = json.loads((path / "receipt.json").read_text())
    if (receipt["input_sha256"], receipt["cohort"], receipt["workbook_sha256"]) != (
        SOURCE_STAMPS, COHORT, WORKBOOK_HASH
    ):
        raise ValueError("Unexpected source and study registration")
    for name in names:
        require_hash(path / name, receipt["files"][name])
    return sha256(path / "receipt.json")


def component_query(query, arm):
    if arm.startswith("cellhit-"):
        drugs = np.array_split(sorted(query.compound_id.unique()), 3)[int(arm[-1])]
        return query[query.compound_id.isin(drugs)]
    return query


def predict(args):
    digest = checked(args.inputs, ("train.parquet", "rna.parquet", "query.parquet"))
    require_hash(args.settings, SETTINGS)
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    train = pd.read_parquet(args.inputs / "train.parquet")
    query = component_query(pd.read_parquet(args.inputs / "query.parquet"), args.component)
    _validate(train, query)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    known_drugs = set(targets)
    new_targets = train[~train.compound_id.isin(known_drugs)]
    targets, mapped = expanded_targets(new_targets, targets, rna.columns)
    auxiliary, meta = None, {}
    if args.component in ("kernel", "v0plus_inductive"):
        from .v0plus_neural import load_auxiliary
        auxiliary, meta = load_auxiliary(args.auxiliary)
    if args.component == "kernel":
        structure = kernel_predictions(train, query, rna, edges, targets, auxiliary)
        response, notes = forecast(train, query, rna, edges, targets, auxiliary, device_cuda())
        rows = pd.concat([structure, response], ignore_index=True)
    else:
        arm = args.component.split("-")[0]
        if arm in PUBLIC:
            verify_upstream(args.upstream)
        configuration = json.loads(args.settings.read_text())["settings"][arm]
        selected = [tuple(setting) for setting in configuration if setting[0] in COMPONENTS[args.component]]
        rows, notes = candidate_forecasts(arm, train, query, rna, edges, targets, selected=selected,
                                          upstream=args.upstream, auxiliary=auxiliary, gpu=args.gpu)
        rows.method = rows.method.str.split("/").str[0]
    shared_support(rows, query, COMPONENTS[args.component])
    freeze_table(rows.assign(track=COHORT, fold=0), args.out, provenance={
        "registration_sha256": digest, "settings_sha256": SETTINGS, "arm": args.component,
        "priors": priors, "mapped_targets_added": mapped, "auxiliary": meta, "model": notes,
    })


def merge(args):
    digest = checked(args.inputs, ("query.parquet",))
    query = pd.read_parquet(args.inputs / "query.parquet")
    parts, hashes = [], {}
    for arm, methods in COMPONENTS.items():
        rows, receipt = verify_forecast(args.parts / f"{arm}.parquet")
        source = receipt["provenance"]
        if (source["registration_sha256"], source["settings_sha256"], source["arm"]) != (digest, SETTINGS, arm):
            raise ValueError("Comparison forecast failed registration")
        shared_support(rows, component_query(query, arm), methods)
        parts.append(rows)
        hashes[arm] = receipt["sha256"]
    rows = pd.concat(parts, ignore_index=True)
    shared_support(rows, query, METHODS)
    freeze_table(rows, args.out, provenance={"registration_sha256": digest, "settings_sha256": SETTINGS,
        "components_sha256": hashes, "workbook_sha256": WORKBOOK_HASH})


def observations(query, path):
    with closing(workbook(path)) as handle:
        values = handle["Supplementary Table 7"].values
        header, matrix = next(values), list(values)
    indices = {row: header.index(row) for row in query.cell_id.unique()}
    observed = [auc_mean(matrix[row.response_row - 2][indices[row.cell_id]]) for row in query.itertuples()]
    return query.assign(endpoint_value=observed).dropna(subset=["endpoint_value"])


def summarize(joined, scope, *, main=False):
    pairs = [(PRIMARY, name) for name in COMPARATORS]
    report, per_drug, _ = axis_metrics(joined, pairs)
    crossed = {name: crossed_patient_drug_ci(joined, PRIMARY, name) for name in COMPARATORS}
    measured = joined.drop_duplicates("row_id")
    donors_per_drug = measured.groupby("compound_id").lineage_group.nunique()
    eligible = sorted(donors_per_drug[donors_per_drug.eq(scope.lineage_group.nunique())].index)
    interaction = joined[joined.compound_id.isin(eligible)]
    if len(eligible) < 5:
        raise ValueError("Fewer than five complete drugs for interaction analysis")
    cf = component_scores(interaction, cell_axis="lineage_group")
    interaction_ci = {name: interaction_intervals(interaction, PRIMARY, name)
                      for name in ("drug_ridge", "structured", "cellhit")}
    ranks = {(row["baseline"], row["metric"]): row["ci95"]
             for row in report["paired_contrasts"] if row["candidate"] == PRIMARY}
    gate = {name: all(crossed[name][metric][0] > 0 for metric in ("pearson", "spearman"))
            and all(ranks[name, metric][0] > 0 for metric in ("spearman", "ndcg_at_5"))
            and (name == "drug_mean" or interaction_ci[name][0] > 0)
            for name in COMPARATORS} if main else None
    strata = {}
    for name, drugs in compound_strata(scope).items():
        actual = per_drug[per_drug.compound_id.isin(drugs)]
        if not actual.empty:
            means = actual.groupby("method")[["spearman", "pearson"]].mean()
            strata[name] = {"drugs": int(actual.compound_id.nunique()),
                            "metrics": means.astype(object).where(pd.notna(means), None).to_dict("index")}
    return {"drugs": len(donors_per_drug), "models": int(measured.cell_id.nunique()), "rows": len(measured),
        "complete_drugs": len(eligible), "registered_rows": len(scope), "components": cf,
        "interaction_ci95": interaction_ci, "report": report, "paired_drug_ci95": crossed, "gate": gate,
        "strata": strata, "compound_labels": scope.drop_duplicates("compound_id").set_index(
            "compound_id").source_compound_label.to_dict(),
        "per_drug": per_drug.astype(object).where(pd.notna(per_drug), None).to_dict("records")}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    digest = checked(args.inputs, ("query.parquet",))
    rows, seal = verify_forecast(args.forecast)
    if (seal["provenance"]["registration_sha256"], seal["provenance"]["workbook_sha256"]) != (digest, WORKBOOK_HASH):
        raise ValueError("Evaluation does not match the frozen registration")
    query = pd.read_parquet(args.inputs / "query.parquet")
    shared_support(rows, query, METHODS)
    truth = observations(query, args.workbook)
    all_scopes = {
        "new_drugs": query[~query.previously_scored],
        "all_matched": query, "previous_drugs": query[query.previously_scored],
    }
    scores = {}
    for name, scope in all_scopes.items():
        labels = truth[truth.row_id.isin(scope.row_id)]
        if labels.lineage_group.nunique() != 10 or labels.groupby("compound_id").size().ge(5).sum() < 20:
            raise ValueError("Measured support below the preregistered comparison")
        joined = join_observations(rows[rows.row_id.isin(labels.row_id)], labels)
        scores[name] = summarize(joined, scope, main=name == "new_drugs")
    publish_json(args.out, {"stage": COHORT, "registration_sha256": digest, "forecast_sha256": seal["sha256"],
        "workbook_sha256": WORKBOOK_HASH, "primary": PRIMARY, "joint_gate": all(scores["new_drugs"]["gate"].values()),
        "results": scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    paths = {
        "register": ("original", "core", "extension", "gdsc", "wong", "support"),
        "predict": ("inputs", "settings", "priors", "hallmark", "targets", "upstream", "auxiliary"),
        "merge": ("inputs", "parts"), "score": ("inputs", "forecast", "workbook"),
    }
    for name, files in paths.items():
        command = commands.add_parser(name)
        for key in (*files, "out"):
            command.add_argument("--" + key, type=Path, required=True)
        if name == "predict":
            command.add_argument("--component", choices=COMPONENTS, required=True)
            command.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"register": register, "predict": predict, "merge": merge, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
