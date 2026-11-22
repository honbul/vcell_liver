"""Train external response rankers on source-specific endpoints, then seal LIMORE forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import OUTCOME_COLUMNS, freeze_table, publish_json, read_table, sha256

KEY = ["row_id", "cell_id", "lineage_group", "compound_id"]


def endpoint_ranks(frame):
    """Convert response only within each source, cell and native endpoint."""
    direction = frame["endpoint_direction"].map({
        "higher": 1, "lower": -1, "higher_is_more_sensitive": 1, "lower_is_more_sensitive": -1,
    })
    value = frame["endpoint_value"].astype(float)
    if direction.isna().any() or not np.isfinite(value).all():
        raise ValueError("Every training observation needs a finite value and a declared endpoint direction")
    frame = frame.assign(sensitivity=value * direction)
    group = frame.groupby(["cell_id", "endpoint_id"], sort=False)["sensitivity"]
    return (group.rank(method="average") - 0.5) / group.transform("size")


def _prepare_expression(train_rna, query_rna, train_cells, query_cells):
    genes = sorted(set(train_rna.columns) & set(query_rna.columns))
    if len(genes) < 50:
        raise ValueError("Insufficient measured shared RNA genes")
    train = train_rna.loc[train_cells, genes].to_numpy(dtype=float)
    query = query_rna.loc[query_cells, genes].to_numpy(dtype=float)
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("Shared RNA axis must contain measured values")
    train, query = (rankdata(x, axis=1) / len(genes) for x in (train, query))
    nonconstant = np.std(train, axis=0) > 1e-7
    scaled = StandardScaler()
    x = scaled.fit_transform(train[:, nonconstant])
    y = scaled.transform(query[:, nonconstant])
    pca = PCA(n_components=min(24, x.shape[0] - 1, x.shape[1]), svd_solver="randomized", random_state=14822)
    return (
        pd.DataFrame(pca.fit_transform(x), index=train_cells),
        pd.DataFrame(pca.transform(y), index=query_cells),
        len(genes),
    )


def _target_features(rows, rna, targets):
    matrix = np.empty((len(rows), 3), dtype=float)
    symbols = set(rna.columns)
    matched = {drug: sorted(symbols.intersection(genes)) for drug, genes in targets.items()}
    selected = sorted({gene for genes in matched.values() for gene in genes})
    rank_by_cell = rna.loc[rows.cell_id.unique()].rank(axis=1, pct=True)[selected]
    for index, (cell, drug) in enumerate(rows[["cell_id", "compound_id"]].itertuples(index=False, name=None)):
        found = matched.get(drug)
        if not found:
            matrix[index] = [0., 0., 0.]
            continue
        measured = rank_by_cell.loc[cell, found].to_numpy()
        matrix[index] = [1., np.mean(measured), np.max(measured)]
    return matrix


def _chemistry(frame):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256, includeChirality=True)
    features = {}
    for key, smiles in frame[["compound_id", "isomeric_smiles"]].drop_duplicates().itertuples(index=False, name=None):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid structure for {key}")
        fp = np.zeros(256, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), fp)
        descriptors = [Descriptors.MolWt(molecule) / 500, Descriptors.MolLogP(molecule) / 5,
                       Descriptors.TPSA(molecule) / 150, Descriptors.NumHDonors(molecule) / 5]
        if key in features:
            raise ValueError(f"Conflicting structures for {key}")
        features[key] = np.concatenate([fp, descriptors])
    return pd.DataFrame.from_dict(features, orient="index")


def _validate(train, query):
    if set(map(str.lower, query.columns)) & OUTCOME_COLUMNS:
        raise ValueError("External predictor received target outcomes")
    for key in KEY:
        if train[key].isna().any() or query[key].isna().any():
            raise ValueError(f"Missing external identity {key}")
    for key in ("row_id", "cell_id", "lineage_group"):
        overlap = set(train[key]) & set(query[key])
        if overlap:
            raise ValueError(f"Training and evaluation overlap in {key}")
    if query.row_id.duplicated().any() or query.duplicated(["cell_id", "compound_id"]).any():
        raise ValueError("External panel has duplicate observations")
    if not set(query.compound_id) <= set(train.compound_id):
        raise ValueError("The defined known-panel evaluation must have training support for every drug")
    if "lineage_role" in query and not (query.lineage_role == "adult_hcc").all():
        raise ValueError("External evaluation includes nonadult HCC identities")


def _source_rank_scores(fit, query, fit_rep, query_rep, chemistry, fit_rna, query_rna, targets):
    fit = fit.assign(model_value=endpoint_ranks(fit))
    y = fit.model_value.to_numpy()
    train_x = fit_rep.loc[fit.cell_id].to_numpy()
    test_x = query_rep.loc[query.cell_id].to_numpy()
    mean = fit.groupby("compound_id").model_value.mean()
    result = {"drug_mean": query.compound_id.map(mean).to_numpy()}
    if fit.compound_id.nunique() < 2 or fit.cell_id.nunique() < 2:
        raise ValueError("Source lacks crossed support for modeling")

    drug_id = OneHotEncoder(handle_unknown="error", sparse_output=False)
    drug_fit = drug_id.fit_transform(fit[["compound_id"]])
    drug_query = drug_id.transform(query[["compound_id"]])
    add_fit = np.column_stack([train_x, drug_fit])
    add_query = np.column_stack([test_x, drug_query])
    result["rna_additive"] = Ridge(alpha=10).fit(add_fit, y).predict(add_query)

    structure_fit = chemistry.loc[fit.compound_id].to_numpy()
    structure_query = chemistry.loc[query.compound_id].to_numpy()
    base_fit = np.column_stack([add_fit, structure_fit])
    base_query = np.column_stack([add_query, structure_query])
    # Regularized drug-by-RNA effects use the known-panel identity as well as the structural main effect.
    direct_fit = np.column_stack([base_fit, (train_x[:, :, None] * drug_fit[:, None, :]).reshape(len(fit), -1)])
    direct_query = np.column_stack([base_query, (test_x[:, :, None] * drug_query[:, None, :]).reshape(len(query), -1)])
    result["direct_structure_rna"] = Ridge(alpha=40).fit(direct_fit, y).predict(direct_query)
    if targets:
        target_fit = _target_features(fit, fit_rna, targets)
        target_query = _target_features(query, query_rna, targets)
        result["direct_with_targets"] = Ridge(alpha=40).fit(
            np.column_stack([direct_fit, target_fit]), y
        ).predict(np.column_stack([direct_query, target_query]))
    return pd.DataFrame(result, index=query.row_id)


def train_rankers(train, query, train_rna, query_rna, targets):
    _validate(train, query)
    chemistry = _chemistry(pd.concat([train, query]))
    source_outputs = []
    source_counts = []
    for (source, endpoint), group in train.groupby(["source_id", "endpoint_id"], sort=True):
        rows = group[group.compound_id.isin(query.compound_id)]
        eligible = query[query.compound_id.isin(rows.compound_id)]
        if eligible.compound_id.nunique() < 5:
            continue
        fit_cells = sorted(rows.cell_id.unique())
        query_cells = sorted(eligible.cell_id.unique())
        rep_fit, rep_query, genes = _prepare_expression(train_rna, query_rna, fit_cells, query_cells)
        forecasts = _source_rank_scores(rows, eligible, rep_fit, rep_query, chemistry, train_rna, query_rna, targets)
        # Each source contributes a fixed bounded native-endpoint rank model. A missing
        # source makes no contribution. Preserve all source-specific forecasts for audit.
        forecasts["source_id"] = source
        forecasts["endpoint_id"] = endpoint
        source_outputs.append(forecasts.reset_index())
        source_counts.append({"source_id": source, "endpoint_id": endpoint, "training_rows": len(rows),
                              "training_cells": len(fit_cells), "predicted_rows": len(eligible),
                              "compounds": int(eligible.compound_id.nunique()), "common_rna_genes": genes})
    if not source_outputs:
        raise ValueError("No sufficiently supported training source in the registered public axis")
    source_values = pd.concat(source_outputs, ignore_index=True)
    methods = [name for name in source_values if name not in {"row_id", "source_id", "endpoint_id"}]
    # Rank scores stay on a common sensitivity orientation, but have no LIMORE AA units.
    combined = source_values.groupby("row_id", as_index=False)[methods].mean()
    combined = query[KEY].merge(combined, on="row_id", how="left", validate="one_to_one")
    predictions = combined.melt(id_vars=KEY, var_name="method", value_name="predicted_value")
    return predictions.assign(track="X_C_LIMORE", fold=0), source_values, source_counts


def _rna(path):
    frame = read_table(path)
    if "cell_id" in frame.columns:
        frame = frame.set_index("cell_id")
    if frame.index.name != "cell_id" or frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("RNA table needs unique cell identifiers")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    args = parser.parse_args()
    paths = {f"{role}/{name}": args.bundle / role / f"{name}.parquet"
             for role in ("train", "test_inputs") for name in ("rows", "rna")}
    audit_path = args.bundle / "receipt.json"
    audit = json.loads(audit_path.read_text())
    for key, path in paths.items():
        if sha256(path) != audit["artifacts"][key + ".parquet"]["sha256"]:
            raise ValueError(f"External cohort input differs from its audit for {key}")
    train, query = (
        read_table(paths[f"{role}/rows"]).rename(columns={"record_id": "row_id", "cell_group": "lineage_group"})
        for role in ("train", "test_inputs")
    )
    train_rna, query_rna = (_rna(paths[f"{role}/rna"]) for role in ("train", "test_inputs"))
    targets = json.loads(args.targets.read_text()) if args.targets else {}
    if args.targets:
        paths["drug_targets"] = args.targets
    predictions, source_scores, counts = train_rankers(train, query, train_rna, query_rna, targets)
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"scope": "pan_cancer_to_public_limore_rank", "training_eligibility": counts,
                "cohort_audit_sha256": sha256(audit_path),
                "registered_observations_sha256": audit["artifacts"]["test_labels/outcomes.parquet"]["sha256"],
                "query_cells": int(query.cell_id.nunique()), "query_lineages": int(query.lineage_group.nunique()),
                "query_compounds": int(query.compound_id.nunique()), "query_rows": len(query),
                "input_sha256": {key: sha256(path) for key, path in paths.items()},
                "prediction_scale": "cross_source_sensitivity_rank", "requested_limore_model_count": 50}
    source_scores.to_parquet(args.out / "source_forecasts.parquet", index=False)
    metadata["source_forecasts_sha256"] = sha256(args.out / "source_forecasts.parquet")
    freeze_table(predictions, args.out / "predictions.parquet", provenance=metadata)
    publish_json(args.out / "metadata.json", metadata)


if __name__ == "__main__":
    main()
