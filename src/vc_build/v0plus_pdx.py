"""Score the GSE90653 sorafenib PDX rank endpoint as a separate technical transfer."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .evaluate import join_observations
from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_broutier import external_3d_forecasts, writer
from .v0plus_cv import training_inputs

COHORT = "GSE90653_HCC_PDX_SORAFENIB_TECHNICAL"
COMPOUND = "MLDQJTXFUGDVEO-UHFFFAOYSA-N"
MATRIX_HASH = "2e233cf691603088b3e7d47e90d084f198fd001a13316ae9e4dd5ccd9d2e9fc4"
ANNOTATION_HASH = "d7cd44352127b1e34f3a720ebea86093ef255a38f1612a85a2962b71bde8f394"
FIGURE_HASH = "997df4039ef27293ec41a116faf1d279a85ad6ca22f1353679afd360a4a7219b"
SAMPLES = {
    "HFP318": "GSM2406837", "HMP136": "GSM2406842", "HMP215": "GSM2406836",
    "HMP308": "GSM2406857", "HMP312": "GSM2406834", "HMP319": "GSM2406825",
    "HMP320": "GSM2406828", "HMP321": "GSM2406829", "HMP322": "GSM2406830",
    "HMP323": "GSM2406832", "HMP325": "GSM2406841", "HMP326": "GSM2406835",
    "HMP333": "GSM2406838", "HMP338": "GSM2406843", "HMP359": "GSM2406844",
    "HMR210": "GSM2406822",
}


def geo_table(path, marker, columns):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == marker:
                return pd.read_csv(stream, sep="\t", comment="!", usecols=columns).set_index(
                    columns[0], verify_integrity=True,
                )
    raise ValueError(f"No {marker} section in {path}")


def register(args):
    if sha256(args.matrix) != MATRIX_HASH or sha256(args.annotation) != ANNOTATION_HASH:
        raise ValueError("GSE90653 inputs differ from the audited GEO downloads")
    train, train_rna, _ = training_inputs(args.bundle)
    sorafenib = train.loc[train.compound_id.eq(COMPOUND), "isomeric_smiles"].unique()
    if len(sorafenib) != 1:
        raise ValueError("Expected a unique registered training structure for sorafenib")
    probes = geo_table(args.annotation, "!platform_table_begin", ["ID", "Gene symbol"])["Gene symbol"]
    probes = probes[probes.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", na=False)]
    values = geo_table(args.matrix, "!series_matrix_table_begin", ["ID_REF", *SAMPLES.values()])
    mapped = values.join(probes.rename("gene"), how="inner", validate="1:1")
    genes = sorted(set(mapped.gene) & set(train_rna.columns))
    rna = mapped[mapped.gene.isin(genes)].groupby("gene").median().loc[genes].T
    rna = rna.rename(index={geo: model for model, geo in SAMPLES.items()}).sort_index()
    rna = rna.rename_axis(index="cell_id", columns=None)
    if len(rna) != len(SAMPLES) or not np.isfinite(rna.to_numpy()).all():
        raise ValueError("Missing GSE90653 RNA observations")
    query = pd.DataFrame({"cell_id": list(SAMPLES), "lineage_group": list(SAMPLES),
                          "compound_id": COMPOUND, "isomeric_smiles": sorafenib[0],
                          "source_compound_label": "Sorafenib"})
    query["row_id"] = COHORT + "|" + query.cell_id + "|" + query.compound_id
    hashes = writer(args.out, {"query.parquet": query, "rna.parquet": rna})
    publish_json(args.out / "receipt.json", {
        "cohort": COHORT, "lineages": len(SAMPLES), "compounds": 1,
        "rna_genes": len(genes), "training_rna_genes": train_rna.shape[1],
        "array_passage": "F2 non-cryopreserved", "array_platform": "GPL570",
        "geo_matrix_sha256": MATRIX_HASH, "geo_annotation_sha256": ANNOTATION_HASH,
        "figure_pdf_sha256": FIGURE_HASH, "files": hashes,
    })


def predict(args):
    receipt_path = args.cohort / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["cohort"] != COHORT or receipt["figure_pdf_sha256"] != FIGURE_HASH:
        raise ValueError("Wrong GSE90653 registration")
    prediction = external_3d_forecasts(args, receipt)
    freeze_table(prediction, args.out, provenance={
        "settings_sha256": sha256(args.settings), "registration_sha256": sha256(receipt_path),
        "figure_pdf_sha256": FIGURE_HASH,
    })


def correlations(actual, forecasts):
    actual = rankdata(actual, axis=-1).astype(float)
    predicted = rankdata(forecasts, axis=-1).astype(float)
    actual -= actual.mean(axis=-1, keepdims=True)
    predicted -= predicted.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(actual, axis=-1) * np.linalg.norm(predicted, axis=-1)
    return np.divide((actual * predicted).sum(axis=-1), denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    predictions, seal = verify_forecast(args.forecast)
    if seal["provenance"]["figure_pdf_sha256"] != FIGURE_HASH or sha256(args.figure) != FIGURE_HASH:
        raise ValueError("Unexpected response figure revision")
    if seal["provenance"]["registration_sha256"] != sha256(args.cohort / "receipt.json"):
        raise ValueError("GSE90653 registration changed")
    order = args.order.read_text().split()
    if len(order) != len(SAMPLES) or set(order) != set(SAMPLES):
        raise ValueError("Bar order must contain exactly the 16 registered model IDs")
    outcomes = predictions[KEY].drop_duplicates().assign(
        endpoint_value=lambda rows: rows.cell_id.map(dict(zip(order, range(1, len(order) + 1)))),
    )
    joined = join_observations(predictions, outcomes)
    table = joined.pivot(index="cell_id", columns="method", values="predicted_value").sort_index()
    target = outcomes.set_index("cell_id").loc[table.index, "endpoint_value"].to_numpy()
    values = table.T.to_numpy()
    if set(table.index) != set(SAMPLES) or not np.isfinite(values).all():
        raise ValueError("All frozen arms must predict all 16 PDX lineages")
    draws = np.random.default_rng(90653).integers(len(target), size=(10000, len(target)))
    estimates = correlations(target, values)
    bootstrap = correlations(target[draws], values[:, draws])
    quantile = lambda draws: np.quantile(draws, [.025, .975]).tolist()
    primary = table.columns.get_loc("v0plus_inductive")
    publish_json(args.out, {
        "cohort": COHORT, "lineages": len(target), "compounds": 1,
        "endpoint": "Sensitivity rank, left to right in Hu et al. Figure 4c",
        "forecast_sha256": seal["sha256"], "figure_pdf_sha256": FIGURE_HASH,
        "figure_bar_order_sha256": sha256(args.order),
        "patient_bootstrap_draws": len(draws),
        "methods": [
            {"method": name, "spearman": float(estimate), "ci95": quantile(boot)}
            for name, estimate, boot in zip(table.columns, estimates, bootstrap)
        ],
        "paired_v0plus_deltas": [
            {"comparator": name, "spearman_delta": float(estimates[primary] - estimates[i]),
             "ci95": quantile(bootstrap[primary] - bootstrap[i])}
            for i, name in enumerate(table.columns) if i != primary
        ],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    modes = {
        "register": ("matrix", "annotation", "bundle", "out"),
        "predict": ("cohort", "bundle", "settings", "priors", "hallmark", "targets", "auxiliary", "out"),
        "score": ("cohort", "forecast", "figure", "order", "out"),
    }
    for mode, arguments in modes.items():
        command = commands.add_parser(mode)
        for name in arguments:
            command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"register": register, "predict": predict, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
