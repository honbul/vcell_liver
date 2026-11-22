"""Register and score the two publicly paired Broutier HCC organoids."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .evaluate import join_observations, score
from .external import KEY, _validate
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_cv import candidate_forecasts, molecular_inputs, training_inputs

MODELS = {"HCC-1": "HCC1_O", "HCC-3": "HCC3_O"}
MAPPING = {
    "1012": ("Vorinostat", "Vorinostat"), "1017": ("Olaparib", "Olaparib"),
    "1049": ("PD-173074", "PD173074"), "1054": ("PD-0332991", "Palbociclib"),
    "1079": ("Dasatanib", "Dasatinib"), "1085": ("Sorafenib", "Sorafenib"),
    "1372": ("Trametinib", "Trametinib"),
}
RNA_HASH = "b7dd92496a24a04728c8419ee3a964ca29dd8e4016dd9a0eec06063d2c7175e4"
RESPONSE_HASH = "6db9960ab42976248839f13b23576bad6a0cedc22da881e9b383f2bc4cecf465"


def writer(out, files):
    out.mkdir(parents=True, exist_ok=False)
    for name, table in files.items():
        table.to_parquet(out / name, index=name == "rna.parquet")
    return {name: sha256(out / name) for name in files}


def register(args):
    if sha256(args.rna) != RNA_HASH or sha256(args.response) != RESPONSE_HASH:
        raise ValueError("Broutier published inputs differ from the audited originals")
    _, fit_rna, training = training_inputs(args.bundle)
    if sha256(args.gdsc_map) != training["inputs"]["gdsc_pug_match"]["sha256"]:
        raise ValueError("Broutier match requires the registered GDSC structural map")
    mapping = pd.read_csv(args.gdsc_map, sep="\t", dtype=str)
    mapping = mapping[mapping.match_status.eq("matched")].set_index("gdsc_drug_id", verify_integrity=True)
    input_rna = load_workbook(args.rna, read_only=True, data_only=True)
    records = input_rna["S1_RPKM_values"].iter_rows(min_row=3, values_only=True)
    headers = list(next(records))
    rpkm = pd.DataFrame(
        [(row[1], *[row[headers.index(label)] for label in MODELS.values()]) for row in records if row[1]],
        columns=["gene", *MODELS],
    )
    rpkm = rpkm[~rpkm.gene.duplicated(keep=False)].set_index("gene").apply(pd.to_numeric, errors="raise").T
    genes = sorted(set(rpkm.columns) & set(fit_rna.columns))
    rna = np.log2(rpkm.loc[:, genes] + 1).rename_axis("cell_id")
    if len(rna) != len(MODELS) or not np.isfinite(rna.to_numpy()).all():
        raise ValueError("Incomplete Broutier pretreatment RPKM")
    workbook = load_workbook(args.response, read_only=True, data_only=True)
    rows = []
    for model, drug_id, short, target, _, _, _, _, sensitivity, _ in workbook["3_Raw data"].iter_rows(
        min_row=2, values_only=True
    ):
        drug_id = str(drug_id)
        if model not in MODELS or drug_id not in MAPPING:
            continue
        label, gdsc_label = MAPPING[drug_id]
        equivalent = mapping.loc[drug_id]
        if short.strip() != label or equivalent.gdsc_drug_name != gdsc_label:
            raise ValueError(f"Published drug metadata conflict for {drug_id}")
        rows.append({"cell_id": model, "lineage_group": model, "compound_id": equivalent.compound_id,
                     "raw_drug_id": drug_id, "isomeric_smiles": equivalent.pubchem_isomeric_smiles,
                     "source_compound_label": gdsc_label, "drug_target_source_labels": target,
                     "endpoint_value": float(sensitivity)})
    panel = pd.DataFrame(rows)
    key = ["cell_id", "compound_id"]
    sub = panel.groupby(key).endpoint_value.agg(["median", "size"])
    registered = panel.drop(columns="endpoint_value").drop_duplicates().join(sub, on=key, validate="1:1")
    if len(registered) != 14 or not registered["size"].eq(4).all():
        raise ValueError("Broutier published replicated response support changed")
    registered["row_id"] = "GSE84073|" + registered.cell_id + "|" + registered.compound_id
    outcomes = registered.rename(columns={"median": "endpoint_value"})[[*KEY, "endpoint_value"]]
    inputs = registered.drop(columns=["median", "size"])
    hashes = writer(args.out, {"query.parquet": inputs, "rna.parquet": rna, "labels.parquet": outcomes})
    publish_json(args.out / "receipt.json", {
        "cohort": "GSE84073_HCC_TECHNICAL", "compounds": 7, "lineages": 2, "registered_rows": 14,
        "measurements_before_median": 56, "rna_genes": len(genes), "files": hashes,
        "source_workbooks": {"rna_sha256": RNA_HASH, "response_sha256": RESPONSE_HASH},
        "scope": "within-organoid drug ranking sensitivity only, no supported across-patient correlation",
    })


def external_3d_forecasts(args, receipt):
    for name in ("query.parquet", "rna.parquet"):
        if sha256(args.cohort / name) != receipt["files"][name]:
            raise ValueError(f"Query differs from registration: {name}")
    train, fit_rna, _ = training_inputs(args.bundle)
    query = pd.read_parquet(args.cohort / "query.parquet")
    query_rna = pd.read_parquet(args.cohort / "rna.parquet")
    fit_rna = fit_rna[query_rna.columns]
    _validate(train, query)
    rna = pd.concat([fit_rna, query_rna])
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    from .v0plus_neural import load_auxiliary

    auxiliary, _ = load_auxiliary(args.auxiliary)
    settings = json.loads(args.settings.read_text())
    arms = {"drug_ridge": "linear", "drug_elasticnet": "linear", "additive": "linear"}
    strongest = settings["best_traditional"]
    traditional = arms.get(strongest, strongest)
    choices = {
        "v0plus_inductive": [tuple(it) for it in settings["settings"]["v0plus_inductive"]],
        traditional: [tuple(it) for it in settings["settings"][traditional] if it[0] == strongest],
    }
    if "linear" not in choices:
        choices["linear"] = []
    choices["linear"] += [tuple(it) for it in settings["settings"]["linear"] if it[0] == "drug_mean"]
    parts = []
    for arm, selected in choices.items():
        prediction, _ = candidate_forecasts(
            arm, train, query, rna, edges, targets, gpu=True, selected=selected, auxiliary=auxiliary,
        )
        prediction.method = prediction.method.str.split("/").str[0]
        parts.append(prediction)
    return pd.concat(parts).assign(track=receipt["cohort"], fold=0)


def fit(args):
    receipt = json.loads((args.cohort / "receipt.json").read_text())
    prediction = external_3d_forecasts(args, receipt)
    freeze_table(prediction, args.out, provenance={
        "settings_sha256": sha256(args.settings),
        "registered_observations_sha256": receipt["files"]["labels.parquet"],
    })


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    table, seal = verify_forecast(args.forecast)
    path = args.cohort / "labels.parquet"
    if sha256(path) != seal["provenance"]["registered_observations_sha256"]:
        raise ValueError("Broutier outcomes differ from the freeze")
    merged = join_observations(table, pd.read_parquet(path))
    results, cells = score(merged, endpoint_direction="higher", rank_only=True)
    publish_json(args.out, {"forecast_sha256": seal["sha256"], "label_sha256": sha256(path), **results,
                           "per_organoid": cells.to_dict("records"), "min_lineages_for_within_drug": 5,
                           "within_drug_eligible": 0})


def main():
    options = argparse.ArgumentParser(description=__doc__)
    sub = options.add_subparsers(dest="action", required=True)
    modes = {
        "register": ("rna", "response", "gdsc-map", "bundle", "out"),
        "predict": ("cohort", "bundle", "settings", "priors", "hallmark", "targets", "auxiliary", "out"),
        "score": ("cohort", "forecast", "out"),
    }
    for command, arguments in modes.items():
        parser = sub.add_parser(command)
        for key in arguments:
            parser.add_argument("--" + key, type=Path, required=True)
    args = options.parse_args()
    {"register": register, "predict": fit, "score": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
