"""Register and score the public Wong HCC organoid cohort against locked source models."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations
from .external import KEY, _validate
from .function_data import alias_token, canonical_chemistry
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_broutier import writer
from .v0plus_cv import candidate_forecasts, molecular_inputs, training_inputs
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .v0plus_sota import verify_upstream

COHORT = "WONG2025_HCC_ORGANOIDS"
WORKBOOK_HASH = "724e0276c474c8afbdacb734c291f53e0399a77a4e473b5ae57751dd637841e7"
BROAD_HASH = "3324dfac93d19ceaeafbe7d9e466127709328777cb2f6c5a465740cd327088e9"
SETTINGS_HASH = "9fb42665b3d5bce1757bab31f341c256bc2f9bcf5434e6239c48c47dd3be3474"
ARMS = {
    "linear": ("drug_mean", "drug_ridge"), "cellhit": ("cellhit",),
    "v0plus_inductive": ("v0plus_inductive",), "velodrome_inductive": ("velodrome_inductive",),
    "codeae_inductive": ("codeae_inductive",),
}
PUBLIC_ARMS = {"cellhit", "velodrome_inductive", "codeae_inductive"}
ALIASES = {
    "AFATINIBBIBW2992": "AFATINIBBIBW2992DIMALEATE",
    "RUCAPARIBAG014699PF01367338": "RUCAPARIBAG014699PF01367338PHOSPHATE",
}
MEAN_AUC = re.compile(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:±|\+/-).+")


def workbook(path):
    if sha256(path) != WORKBOOK_HASH:
        raise ValueError("Wong workbook changed")
    return load_workbook(path, read_only=True, data_only=True)


def canonical_key(text):
    try:
        return canonical_chemistry(text)[1]
    except (ValueError, RuntimeError):
        return None


def chemistry(sheet, broad, cas_dir):
    records = sheet.values
    drugs = pd.DataFrame(records, columns=next(records)).dropna(subset=["Product Name"])
    drugs["key"] = drugs["Product Name"].map(alias_token)
    if not drugs.key.is_unique or len(drugs) != 100:
        raise ValueError("Wong S3 requires 100 unique product identities")
    broad_numbers = drugs["Repurposing IDs"].astype(str).map(
        lambda text: re.findall(r"BRD-[A-Z]\d{8}-\d{3}-\d{2}-\d", text),
    )
    wanted = {code for codes in broad_numbers for code in codes}
    hub = pd.read_csv(broad, sep="\t", comment="!", dtype=str).fillna("")
    hub = hub[hub.broad_id.isin(wanted) | hub.deprecated_broad_id.isin(wanted)].copy()
    hub["parent"] = hub.smiles.map(canonical_key)
    resolved = []
    for (_, row), codes in zip(drugs.iterrows(), broad_numbers):
        result = hub[hub.broad_id.isin(codes) | hub.deprecated_broad_id.isin(codes)]
        keys = set(result.parent.dropna())
        evidence = ";".join(codes)
        source = "Broad full sample ID"
        if not keys:
            cas = str(row["CAS Number"])
            if not re.fullmatch(r"\d+-\d\d-\d", cas):
                raise ValueError(f"Invalid CAS identifier for {row['Product Name']}")
            path = cas_dir / f"{cas}.json"
            properties = json.loads(path.read_text())["PropertyTable"]["Properties"]
            keys = {canonical_key(record["SMILES"]) for record in properties} - {None}
            source = "PubChem CAS lookup"
            evidence = f"{cas}|" + ";".join(str(record["CID"]) for record in properties)
        if len(keys) != 1:
            raise ValueError(f"Ambiguous exact parent for {row['Product Name']}")
        resolved.append({
            "key": row.key, "compound_id": next(iter(keys)), "catalogue": row["Catalog Number"],
            "product_name": row["Product Name"], "mechanism": row["MOA"],
            "metadata_source": source, "metadata_id": evidence,
        })
    return pd.DataFrame(resolved).set_index("key", verify_integrity=True)


def sample_registry(data, fit_rna, score_header):
    records = data["Supplementary Table 6"].values
    header = next(records)
    shared = sorted({name for name in header if re.fullmatch(r"PDTO_\d+", str(name))} & set(score_header))
    diagnoses = {str(row[0]).replace("-", "_"): (row[7], row[9])
                 for row in data["Supplementary Table1"].iter_rows(min_row=3, values_only=True)
                 if row[0] is not None and str(row[0]).replace("-", "_") in shared}
    if len(shared) != 10 or set(shared) != set(diagnoses) or any(
        kind != "HCC" or int(age) <= 18 for kind, age in diagnoses.values()
    ):
        raise ValueError("Wong paired adult-HCC sample support changed")
    indices = [header.index(name) for name in shared]
    rna = pd.DataFrame(
        [(row[0], *(row[index] for index in indices)) for row in records if row[0] in fit_rna.columns],
        columns=["gene", *shared],
    )
    rna = rna[~rna.gene.duplicated(keep=False)].set_index("gene")
    values = rna.apply(pd.to_numeric, errors="raise").sort_index().T
    if (values < 0).any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Invalid adult HCC organoid TPM")
    return np.log2(values + 1).rename_axis("cell_id")


def register(args):
    if sha256(args.broad) != BROAD_HASH:
        raise ValueError("Broad sample structure catalogue changed")
    train, fit_rna, _ = training_inputs(args.bundle)
    fit_drugs = train[["compound_id", "isomeric_smiles", "source_compound_label"]].drop_duplicates()
    fit_drugs = fit_drugs.set_index("compound_id", verify_integrity=True)
    data = workbook(args.workbook)
    responses = data["Supplementary Table 7"]
    rna = sample_registry(data, fit_rna, next(responses.values))
    mapped = chemistry(data["Supplementary Table 3"], args.broad, args.cas)
    response_ids = []
    for index, (label,) in enumerate(responses.iter_rows(min_row=2, max_col=1, values_only=True), start=2):
        key = alias_token(label)
        response_ids.append((ALIASES.get(key, key), index))
    indices = pd.DataFrame(response_ids, columns=["key", "response_row"]).set_index("key", verify_integrity=True)
    mapped = mapped.join(indices, validate="1:1")
    if mapped.response_row.isna().any():
        raise ValueError("Unresolved S3 to S7 product identity")
    mapped["in_training"] = mapped.compound_id.isin(fit_drugs.index)
    panel = mapped[mapped.in_training].reset_index().join(fit_drugs, on="compound_id", validate="1:1")
    if len(panel) < 5:
        raise ValueError("Fewer than five exact training compounds")
    query = pd.DataFrame({"cell_id": rna.index}).merge(panel, how="cross")
    query["lineage_group"] = query.cell_id
    query["row_id"] = COHORT + "|" + query.cell_id + "|" + query.compound_id
    query["drug_target_source_labels"] = query.mechanism
    _validate(train, query)
    hashes = writer(args.out, {"query.parquet": query, "rna.parquet": rna, "chemistry.parquet": mapped.reset_index()})
    publish_json(args.out / "receipt.json", {
        "cohort": COHORT, "workbook_sha256": WORKBOOK_HASH, "broad_catalogue_sha256": BROAD_HASH,
        "cas_response_sha256": {p.stem: sha256(p) for p in sorted(args.cas.glob("*.json"))},
        "lineages": len(rna), "rna_genes": rna.shape[1], "source_panel": len(fit_drugs),
        "compounds": len(panel), "screened_compounds": len(mapped), "forecast_rows_per_method": len(query),
        "files": hashes,
    })


def read_registration(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["cohort"] != COHORT or receipt["workbook_sha256"] != WORKBOOK_HASH:
        raise ValueError("Wrong Wong cohort registration")
    for name, digest in receipt["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Wong input changed: {name}")
    return receipt


def fit(args):
    receipt = read_registration(args.cohort)
    if sha256(args.settings) != SETTINGS_HASH:
        raise ValueError("Use the original V0+ source-only settings")
    train, fit_rna, _ = training_inputs(args.bundle)
    query = pd.read_parquet(args.cohort / "query.parquet")
    rna = pd.read_parquet(args.cohort / "rna.parquet")
    _validate(train, query)
    settings = json.loads(args.settings.read_text())["settings"][args.arm]
    selected = [tuple(item) for item in settings if item[0] in ARMS[args.arm]]
    if args.arm in PUBLIC_ARMS:
        verify_upstream(args.upstream)
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary = None
    if args.arm == "v0plus_inductive":
        from .v0plus_neural import load_auxiliary

        auxiliary, _ = load_auxiliary(args.auxiliary)
    forecasts, _ = candidate_forecasts(
        args.arm, train, query, pd.concat([fit_rna[rna.columns], rna]), edges, targets,
        gpu=args.gpu, selected=selected, upstream=args.upstream, auxiliary=auxiliary,
    )
    forecasts.method = forecasts.method.str.split("/").str[0]
    freeze_table(forecasts.assign(track=COHORT, fold=0), args.out, provenance={
        "arm": args.arm, "registration_sha256": sha256(args.cohort / "receipt.json"),
        "settings_sha256": SETTINGS_HASH, "workbook_sha256": receipt["workbook_sha256"],
    })


def merge(args):
    receipt = read_registration(args.cohort)
    roster_hash = sha256(args.cohort / "receipt.json")
    query = pd.read_parquet(args.cohort / "query.parquet", columns=KEY).set_index("row_id").sort_index()
    parts, hashes = [], {}
    for arm, methods in ARMS.items():
        records, seal = verify_forecast(args.parts / f"{arm}.parquet")
        expected = {"arm": arm, "registration_sha256": roster_hash, "settings_sha256": SETTINGS_HASH,
                    "workbook_sha256": WORKBOOK_HASH}
        if seal["provenance"] != expected or set(records.method) != set(methods):
            raise ValueError(f"Component receipt mismatch for {arm}")
        if not records.groupby("method").size().eq(receipt["forecast_rows_per_method"]).all():
            raise ValueError(f"Incomplete Wong grid for {arm}")
        for _, method in records.groupby("method"):
            if not method[KEY].set_index("row_id").sort_index().equals(query):
                raise ValueError(f"Unexpected Wong forecast identities for {arm}")
        parts.append(records)
        hashes[arm] = seal["sha256"]
    forecast = pd.concat(parts, ignore_index=True)
    pair = forecast[forecast.method.isin(["drug_ridge", "v0plus_inductive"])]
    blend = pair.groupby([*KEY, "track", "fold"]).predicted_value.mean().rename("predicted_value")
    combined = pd.concat([forecast, blend.reset_index().assign(method="ridge_molecular_average")], ignore_index=True)
    if not np.isfinite(combined.predicted_value.to_numpy()).all():
        raise ValueError("A supported Wong forecast is missing")
    freeze_table(combined, args.out, provenance={
        "registration_sha256": roster_hash, "settings_sha256": SETTINGS_HASH,
        "component_sha256": hashes, "workbook_sha256": WORKBOOK_HASH,
    })


def auc_mean(value):
    if value is None or str(value).strip().lower() in {"", "-", "—", "na", "n/a", "nt", "not tested"}:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    match = MEAN_AUC.fullmatch(str(value))
    if not match:
        raise ValueError(f"Unrecognized Wong AUC mean/SE format: {value!r}")
    return float(match[1])


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    predictions, seal = verify_forecast(args.forecast)
    read_registration(args.cohort)
    provenance = seal["provenance"]
    if provenance["registration_sha256"] != sha256(args.cohort / "receipt.json") or (
        provenance["settings_sha256"] != SETTINGS_HASH or provenance["workbook_sha256"] != WORKBOOK_HASH
    ):
        raise ValueError("Wong registration differs from the freeze")
    query = pd.read_parquet(args.cohort / "query.parquet")
    records = workbook(args.workbook)["Supplementary Table 7"].values
    header = next(records)
    panel = list(records)
    query["endpoint_value"] = [
        auc_mean(panel[row.response_row - 2][header.index(row.cell_id)])
        for row in query.itertuples(index=False)
    ]
    observed = query.dropna(subset=["endpoint_value"])
    if observed.groupby("lineage_group").size().min() < 5 or observed.lineage_group.nunique() < 5:
        raise ValueError("Wong measured support is below the registered threshold")
    merged = join_observations(predictions[predictions.row_id.isin(observed.row_id)], observed)
    pairs = [("v0plus_inductive", name) for name in ("drug_mean", "drug_ridge", "cellhit")]
    pairs.extend(("ridge_molecular_average", name) for name in ("drug_mean", "drug_ridge"))
    full, drugs, _ = axis_metrics(merged, pairs)
    full["crossed_patient_drug_ci95"] = [
        {"candidate": a, "baseline": b, "intervals": crossed_patient_drug_ci(merged, a, b)} for a, b in pairs
    ]
    strata = {}
    for name, ids in compound_strata(query).items():
        subset = merged[merged.compound_id.isin(ids)]
        if not subset.empty:
            table, _, _ = axis_metrics(subset, ())
            strata[name] = {"compounds": subset.compound_id.nunique(), "across_cell": table["across_cell_scores"]}
    publish_json(args.out, {
        "cohort": COHORT, "forecast_sha256": seal["sha256"], "workbook_sha256": WORKBOOK_HASH,
        "endpoint": "Higher published mean AUC of growth inhibition", "lineages": observed.cell_id.nunique(),
        "compounds": observed.compound_id.nunique(), "measured_rows": len(observed), "registered_rows": len(query),
        "full": full, "strata": strata, "per_drug": drugs.astype(object).where(pd.notna(drugs), None).to_dict("records"),
        "compound_labels": query.drop_duplicates("compound_id").set_index("compound_id").source_compound_label.to_dict(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    modes = {
        "register": ("workbook", "broad", "cas", "bundle", "out"),
        "predict": ("cohort", "bundle", "settings", "priors", "hallmark", "targets", "upstream", "auxiliary", "out"),
        "merge": ("cohort", "parts", "out"),
        "score": ("cohort", "forecast", "workbook", "out"),
    }
    for mode, arguments in modes.items():
        command = commands.add_parser(mode)
        for name in arguments:
            command.add_argument("--" + name, type=Path, required=True)
        if mode == "predict":
            command.add_argument("--arm", choices=ARMS, required=True)
            command.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"register": register, "predict": fit, "merge": merge, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
