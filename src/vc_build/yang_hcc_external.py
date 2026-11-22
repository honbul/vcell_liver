"""Train source-only forecasts, freeze them jointly, and score the held Yang cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import join_observations, rank_metrics
from .expanded_panel_cv import PENALTY, registered_inputs
from .external import KEY, _validate
from .function_data import canonical_chemistry
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_broutier import writer
from .v0plus_cv import candidate_forecasts, molecular_inputs, panel_targets
from .v0plus_score import axis_metrics, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream
from .yang_hcc_audit import ARTIFACTS, COHORT, REFERENCE_SMILES, worksheet

REGISTRATION = "b1430ab540a30a66ba1c73ad74d5a9e02c6d41e4a409b47065526872a42b3dc3"
SETTINGS = "9fb42665b3d5bce1757bab31f341c256bc2f9bcf5434e6239c48c47dd3be3474"
PRISM_CURVES = "2ac69a21f1d681fe7447689262b82ca6e3dc90bfef0bd96eb5479b96f424e43d"
BROAD = "3324dfac93d19ceaeafbe7d9e466127709328777cb2f6c5a465740cd327088e9"
APATINIB_BATCH = "BRD-K25244359-066-03-4"
PRIMARY = "ridge_structured_average"
EXPOSED_DONORS = {f"{COHORT}:{value}" for value in ("P1", "P2")}
ARMS = {
    "linear": {"drug_mean", "additive", "drug_ridge", "drug_elasticnet"},
    "lightgbm": {"lightgbm"},
    "catboost": {"catboost"},
    "cellhit": {"cellhit"},
    "velodrome_inductive": {"velodrome_inductive"},
    "codeae_inductive": {"codeae_inductive"},
    "v0plus_inductive": {"v0plus_inductive"},
    "structured": {"structured", "structured_rna_chem"},
}
BASELINES = ("drug_mean", "drug_ridge", "cellhit")
CURVE_COLUMNS = ("broad_id", "depmap_id", "screen_id", "name", "smiles", "auc", "passed_str_profiling")
SMALL_MOLECULE_TARGETS = {
    "WPEWQEMJFLWMLV-UHFFFAOYSA-N": ["CSK", "KDR", "KIT", "RET"],
    "WIJZXSAJMHAVGX-DHLKQENFSA-N": ["IDH1"],
}


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def read_inputs(path, expected_hash=None):
    if expected_hash is not None:
        require_hash(path / "receipt.json", expected_hash)
    recorded = json.loads((path / "receipt.json").read_text())
    if recorded["cohort"] != COHORT:
        raise ValueError("Expected Yang registration")
    for filename, digest in recorded["artifacts"].items():
        require_hash(path / filename, digest)
    return recorded


def apatinib_source(core, curves, catalogue):
    require_hash(curves, PRISM_CURVES)
    require_hash(catalogue, BROAD)
    listed = pd.read_csv(catalogue, sep="\t", comment="!", usecols=["broad_id", "smiles"])
    batch = listed.loc[listed.broad_id.eq(APATINIB_BATCH), "smiles"].item()
    smiles, parent, connectivity, scaffold = canonical_chemistry(batch)
    if parent != canonical_chemistry(REFERENCE_SMILES["Apatinib"][1])[1]:
        raise ValueError("Broad Apatinib compound differs from the registered PubChem parent")
    pieces = (block[block.broad_id.eq(APATINIB_BATCH)] for block in
              pd.read_csv(curves, usecols=list(CURVE_COLUMNS), chunksize=100_000))
    raw = pd.concat(pieces, ignore_index=True)
    if (len(raw) != 454 or not raw.name.eq("apatinib").all() or not raw.screen_id.eq("HTS002").all()
            or raw.smiles.isna().any()):
        raise ValueError("Unexpected Apatinib curve metadata")
    for value in raw.smiles.unique():
        parents = {canonical_chemistry(piece.strip())[1] for piece in value.split(",")}
        if parents != {parent}:
            raise ValueError("Apatinib dose curves contain inconsistent molecular parents")
    roster_columns = ["cell_id", "lineage_group", "source_cell_label", "tissue_origin", "lineage_role"]
    roster = core[roster_columns].drop_duplicates().set_index("cell_id", verify_integrity=True)
    eligible = raw[raw.depmap_id.isin(roster.index) & raw.passed_str_profiling.eq(True) & np.isfinite(raw.auc)]
    if len(eligible) != 443 or not eligible.depmap_id.is_unique:
        raise ValueError("Expected 443 registered STR-confirmed source models for Apatinib")
    template = core[core.source_id.eq("PRISM20Q2")].iloc[0]
    rows = eligible.rename(columns={"depmap_id": "cell_id", "auc": "endpoint_value"})[
        ["cell_id", "endpoint_value"]
    ].join(roster, on="cell_id", validate="one_to_one")
    rows = rows.assign(
        compound_id=parent, native_compound_id=APATINIB_BATCH, native_drug_label="apatinib", source_run="HTS002",
        dose_min_uM=np.nan, dose_max_uM=np.nan, dose_uM=None, n_source_records=1,
        source_compound_label="Apatinib", isomeric_smiles=smiles, connectivity_group=connectivity,
        scaffold_group=scaffold, drug_target_source_labels="CSK;KDR;KIT;RET",
    )
    fields = ("source_id", "assay_id", "source_protocol", "endpoint_id", "endpoint_unit", "endpoint_direction",
              "endpoint_contract", "supported_dose_response", "baseline_is_proxy", "time_h", "rna_assay", "culture_format")
    rows = rows.assign(**{field: template[field] for field in fields})
    rows["row_id"] = rows.source_id + "|" + rows.cell_id + "|" + rows.compound_id + "|" + rows.endpoint_id
    return rows[core.columns]


def prepare(args):
    registry = read_inputs(args.registration, REGISTRATION)
    _, core, source_rna, extension = registered_inputs(args, include_extension=True)
    apatinib = apatinib_source(core, args.curves, args.broad)
    ivosidenib = extension[extension.compound_id.eq("WIJZXSAJMHAVGX-DHLKQENFSA-N")]
    if len(ivosidenib) != 557 or not ivosidenib.cell_id.is_unique or not ivosidenib.source_id.eq("PRISM20Q2").all():
        raise ValueError("Unexpected Ivosidenib training support")
    if set(apatinib.compound_id) & set(core.compound_id) or set(ivosidenib.compound_id) & set(core.compound_id):
        raise ValueError("Unexpected duplicate compound in the broader training panel")
    train = pd.concat([core, apatinib, ivosidenib[core.columns]], ignore_index=True)
    if not train.row_id.is_unique or train.compound_id.nunique() != 70:
        raise ValueError("Unexpected training identities or coverage")
    query = pd.read_parquet(args.registration / "query.parquet")
    lookup = [{"compound_id": name["parent"], "response_label": name["response_label"]}
              for name in registry["compound_support"] if "parent" in name]
    additions = apatinib[["compound_id", "isomeric_smiles"]].drop_duplicates().merge(
        pd.DataFrame(lookup), on="compound_id", validate="one_to_one",
    )
    extra = query[["cell_id", "lineage_group", "response_row"]].drop_duplicates().merge(additions, how="cross")
    extra["response_column"] = 18
    extra["in_core"] = False
    extra["row_id"] = extra.cell_id + "|" + extra.compound_id
    query = pd.concat([query, extra], ignore_index=True).sort_values("row_id")
    _validate(train, query)
    if len(query) != 1115 or query.compound_id.nunique() != 5:
        raise ValueError("Unexpected registered five-drug evaluation support")
    rna = pd.read_parquet(args.registration / "rna.parquet")
    shared = source_rna.columns.intersection(rna.columns).sort_values()
    combined_rna = pd.concat([source_rna.loc[sorted(train.cell_id.unique()), shared], rna[shared]], verify_integrity=True)
    files = writer(args.out, {"train.parquet": train, "query.parquet": query, "rna.parquet": combined_rna})
    publish_json(args.out / "receipt.json", {
        "cohort": COHORT, "registration_sha256": REGISTRATION,
        "source_receipts": {name: sha256(getattr(args, name) / "receipt.json")
                            for name in ("original", "core", "extension")},
        "curves_sha256": PRISM_CURVES, "broad_sha256": BROAD,
        "training_rows": len(train), "training_cells": train.cell_id.nunique(),
        "training_compounds": train.compound_id.nunique(), "rna_genes": len(shared),
        "additional_apatinib_rows": len(apatinib), "additional_ivosidenib_rows": len(ivosidenib),
        "additional_apatinib_adult_hcc_cells": apatinib.lineage_role.eq("adult_hcc").sum().item(),
        "training_by_source": train.source_id.value_counts().to_dict(),
        "expected_observations": len(query), "artifacts": files,
    })


def kernel_predictions(train, query, rna, edges, targets, auxiliary):
    import torch

    from .structured_kernel import KernelGrid

    if not torch.cuda.is_available():
        raise RuntimeError("Structured kernels require G1 GPU")
    _, anchor, aligned, _ = panel_targets(train)
    fit = rna.loc[sorted(train.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    grid = KernelGrid.prepare(aligned, fit, held, edges, targets, auxiliary, device=torch.device("cuda"))
    base = query.compound_id.map(anchor).to_numpy()
    parts = []
    for arm, name in (("structured", "structured"), ("rna_chem_only", "structured_rna_chem")):
        residual, _ = grid.forecast(query, arm, PENALTY)
        parts.append(query[KEY].assign(method=name, predicted_value=base + residual))
    return pd.concat(parts, ignore_index=True)


def predict(args):
    read_inputs(args.inputs)
    require_hash(args.settings, SETTINGS)
    train = pd.read_parquet(args.inputs / "train.parquet")
    query = pd.read_parquet(args.inputs / "query.parquet", columns=[*KEY, "isomeric_smiles"])
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    _validate(train, query)
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    targets.update(SMALL_MOLECULE_TARGETS)
    auxiliary = None
    if args.arm in {"structured", "v0plus_inductive"}:
        from .v0plus_neural import load_auxiliary

        auxiliary, _ = load_auxiliary(args.auxiliary)
    if args.arm in {"cellhit", "velodrome_inductive", "codeae_inductive"}:
        verify_upstream(args.upstream)
    if args.arm == "structured":
        predictions = kernel_predictions(train, query, rna, edges, targets, auxiliary)
    else:
        selected = [tuple(entry) for entry in json.loads(args.settings.read_text())["settings"][args.arm]]
        predictions, _ = candidate_forecasts(
            args.arm, train, query, rna, edges, targets, selected=selected,
            gpu=args.gpu, upstream=args.upstream, auxiliary=auxiliary,
        )
        predictions.method = predictions.method.str.split("/").str[0]
    if set(predictions.method) != ARMS[args.arm] or not np.isfinite(predictions.predicted_value).all():
        raise ValueError(f"Incomplete Yang predictions for {args.arm}")
    freeze_table(predictions.assign(track=COHORT, fold=0), args.out, provenance={
        "inputs_sha256": sha256(args.inputs / "receipt.json"), "settings_sha256": SETTINGS, "arm": args.arm,
    })


def merge(args):
    read_inputs(args.inputs)
    input_hash = sha256(args.inputs / "receipt.json")
    roster = pd.read_parquet(args.inputs / "query.parquet", columns=KEY).set_index("row_id").sort_index()
    parts, hashes = [], {}
    for arm, methods in ARMS.items():
        data, seal = verify_forecast(args.parts / f"{arm}.parquet")
        expected = {"inputs_sha256": input_hash, "settings_sha256": SETTINGS, "arm": arm}
        if seal["provenance"] != expected or set(data.method) != methods:
            raise ValueError(f"Unexpected prediction provenance for {arm}")
        for _, part in data.groupby("method"):
            if not part[KEY].set_index("row_id").sort_index().equals(roster):
                raise ValueError(f"Incomplete common support for {arm}")
        hashes[arm] = seal["sha256"]
        parts.append(data)
    data = pd.concat(parts, ignore_index=True)
    combined = (data[data.method.isin({"structured", "drug_ridge"})].groupby(
        [*KEY, "track", "fold"], as_index=False,
    ).predicted_value.mean().assign(method=PRIMARY))
    freeze_table(pd.concat([data, combined], ignore_index=True), args.out, provenance={
        "inputs_sha256": input_hash, "settings_sha256": SETTINGS, "components_sha256": hashes,
    })


def outcomes(path, query):
    require_hash(path, ARTIFACTS["mmc8.xlsx"])
    rows = list(worksheet(path, "Table S6").values)
    measured = []
    for record in query.itertuples(index=False):
        row, col = record.response_row - 1, record.response_column - 1
        panel_label = rows[0][col - 1]
        original_id = record.cell_id.removeprefix(f"{COHORT}:")
        if rows[row][0] != original_id or not panel_label or panel_label.strip() != record.response_label:
            raise ValueError("Frozen row did not map to the registered response identity")
        value = rows[row][col]
        measured.append(-float(value) if isinstance(value, (int, float)) and np.isfinite(value) else np.nan)
    return query.assign(endpoint_value=measured)


def aggregate_patients(frame, aggregation):
    keys = ["lineage_group", "compound_id", "method", "track", "fold"]
    result = frame.groupby(keys, as_index=False)[["endpoint_value", "predicted_value"]].agg(aggregation)
    anchor = frame[frame.method.eq("drug_mean")].groupby("compound_id").predicted_value
    if not anchor.nunique().eq(1).all():
        raise ValueError("Frozen drug mean must be exactly constant within drug")
    fixed = result.method.eq("drug_mean")
    result.loc[fixed, "predicted_value"] = result.loc[fixed, "compound_id"].map(anchor.first())
    return result.assign(cell_id=result.lineage_group, row_id=result.lineage_group + "|" + result.compound_id)


def summarize(frame, comparisons):
    table, drugs, _ = axis_metrics(frame, comparisons)
    metrics = []
    for (method, _, _), panel in frame.groupby(["method", "lineage_group", "cell_id"]):
        if panel.compound_id.nunique() == 5:
            p3 = rank_metrics(panel.endpoint_value, panel.predicted_value, k=3)["top5_overlap"]
            metrics.append({"method": method, "precision_at_3": p3})
    p3 = pd.DataFrame(metrics)
    table["precision_at_3"] = p3.groupby("method").precision_at_3.mean().dropna().to_dict() if len(p3) else {}
    table["sensitivity_ci95"] = [
        {"candidate": candidate, "baseline": baseline,
         "ci95": crossed_patient_drug_ci(frame, candidate, baseline)}
        for candidate, baseline in comparisons
    ]
    return {"metrics": table, "patients": frame.lineage_group.nunique(), "models": frame.cell_id.nunique(),
            "observations": frame.row_id.nunique(), "drugs": frame.compound_id.nunique(),
            "per_drug": drugs.astype(object).where(drugs.notna(), None).to_dict("records")}


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    read_inputs(args.inputs)
    forecast, seal = verify_forecast(args.forecast)
    meta = seal["provenance"]
    if ((meta["inputs_sha256"], meta["settings_sha256"]) != (sha256(args.inputs / "receipt.json"), SETTINGS)
            or set(meta["components_sha256"]) != set(ARMS)
            or set(forecast.method) != set().union(*ARMS.values(), {PRIMARY})):
        raise ValueError("Expected the jointly frozen registered methods")
    labels = outcomes(args.workbook, pd.read_parquet(args.inputs / "query.parquet"))
    observed = labels[labels.endpoint_value.notna()]
    present = join_observations(forecast[forecast.row_id.isin(observed.row_id)], observed)
    primary = present[~present.lineage_group.isin(EXPOSED_DONORS)]
    comparisons = [(PRIMARY, baseline) for baseline in BASELINES]
    sections = {
        "primary_patients": summarize(aggregate_patients(primary, "mean"), comparisons),
        "all_patients": summarize(aggregate_patients(present, "mean"), ()),
        "primary_most_resistant_region": summarize(aggregate_patients(primary, "min"), ()),
        "primary_pdos": summarize(primary, ()),
    }
    scores = sections["primary_patients"]["metrics"]
    gates = {}
    for baseline in BASELINES:
        across = next(row["ci95"] for row in scores["sensitivity_ci95"] if row["baseline"] == baseline)
        within = [row["ci95"] for row in scores["paired_contrasts"] if row["baseline"] == baseline
                  and row["metric"] in {"spearman", "ndcg_at_5"}]
        gates[baseline] = len(across) == len(within) == 2 and all(
            interval[0] > 0 for interval in [*across.values(), *within]
        )
    publish_json(args.out, {
        "cohort": COHORT, "forecast_sha256": seal["sha256"], "excluded_patients": sorted(EXPOSED_DONORS),
        "endpoint": "Negative S6 Normalized AUC", "completeness": labels.groupby("response_label").endpoint_value.count().to_dict(),
        "sections": sections, "compound_labels": labels.set_index("compound_id").response_label.to_dict(),
        "gate": gates, "promoted": all(gates.values()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    options = {
        "prepare": ("registration", "original", "core", "extension", "curves", "broad"),
        "predict": ("inputs", "settings", "priors", "hallmark", "targets", "auxiliary", "upstream"),
        "merge": ("inputs", "parts"),
        "score": ("inputs", "forecast", "workbook"),
    }
    for name, args in options.items():
        child = sub.add_parser(name)
        for field in (*args, "out"):
            child.add_argument("--" + field, type=Path, required=True)
        if name == "predict":
            child.add_argument("--arm", required=True, choices=ARMS)
            child.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "merge": merge, "score": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
