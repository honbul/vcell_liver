"""Compare direct and state-supervised functional prediction on locked cell folds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .functional import chemistry_matrix
from .io import freeze_table, publish_json, read_table, sha256

METHODS = ("direct", "multi_output", "state_assisted")
SEEDS = (14822, 14823, 14824)
KEYS = ["row_id", "cell_id", "lineage_group", "compound_id"]


def load_auxiliary(directory, symbols_file, registry_file):
    receipt = json.loads((directory / "receipt.json").read_text())
    for name in ("rows.tsv", "genes.tsv", "delta.npy", "basal.tsv"):
        if sha256(directory / name) != receipt["files"][name]:
            raise ValueError(f"State auxiliary hash mismatch for {name}")
    rows = pd.read_csv(directory / "rows.tsv", sep="\t")
    basal = pd.read_csv(directory / "basal.tsv", sep="\t").set_index("cell_id")
    genes = pd.read_csv(directory / "genes.tsv", sep="\t").gene_id
    targets = np.load(directory / "delta.npy", mmap_mode="r", allow_pickle=False)
    if tuple(receipt["shape"]) != targets.shape or targets.shape != (len(rows), len(genes)):
        raise ValueError("State rows and target array disagree")
    if set(rows.time_h) != {24} or not rows.baseline_is_proxy.all():
        raise ValueError("State auxiliary must be 24h response with separately recorded basal proxy")
    registry = pd.read_csv(registry_file, sep="\t")
    accessions = registry[registry.study == "LIMORE"].set_index("cellosaurus_id").family_id.to_dict()
    hcc = rows.context_role == "adult_hcc"
    rows.loc[hcc, "lineage_group"] = rows.loc[hcc, "lineage_group"].str.removeprefix("RRID:").map(accessions)
    if rows.lineage_group.isna().any():
        raise ValueError("All adult auxiliary lineages need a verified LIMORE family mapping")
    with symbols_file.open(encoding="utf-8") as stream:
        columns = next(csv.reader(stream))
    mapping = {}
    for col in columns:
        matched = re.fullmatch(r"(.+) \(([0-9]+)\)", col)
        if matched:
            symbol, entrez = matched.groups()
            mapping[entrez] = symbol
    basal = basal.rename(columns=mapping)
    basal = basal.loc[:, ~basal.columns.duplicated(keep=False)]
    return rows.rename(columns={"smiles": "isomeric_smiles"}), basal, targets


def rank_features(data, genes):
    array = data.loc[:, genes].to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Joint branch requires measured shared RNA")
    return rankdata(array, axis=1) / len(genes)


def feature_blocks(train_rna, query_rna, state_rna):
    genes = sorted(set(train_rna) & set(query_rna) & set(state_rna))
    if len(genes) < 100:
        raise ValueError("Too few jointly measured basal genes")
    train, query, state = (rank_features(view, genes) for view in (train_rna, query_rna, state_rna))
    scale = StandardScaler().fit(train)
    pca = PCA(n_components=min(16, len(train) - 1), svd_solver="full").fit(scale.transform(train))
    features = [pca.transform(scale.transform(matrix)).astype(np.float32)
                for matrix in (train, query, state)]
    return features, genes


def fit_predict(inputs, observed, x_state, z_state, x_eval, *, method, seed):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dim, state_dim = inputs.shape[1], z_state.shape[1]

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
            self.state = nn.Linear(32, state_dim)
            self.functional_latent = nn.Linear(32, state_dim)
            self.functional = nn.Linear(32 + state_dim, 1)

        def predict_function(self, features):
            hidden = self.encoder(features)
            projection = self.state if method == "state_assisted" else self.functional_latent
            return self.functional(torch.cat([hidden, projection(hidden)], dim=-1)).squeeze(-1)

        def predict_state(self, features):
            return self.state(self.encoder(features))

    model = Network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    tensors = [torch.as_tensor(value, dtype=torch.float32)
               for value in (inputs, observed, x_state, z_state, x_eval)]
    x, y, s, z, q = tensors
    rng = np.random.default_rng(seed)
    for _ in range(300):
        indices = rng.integers(len(x), size=128)
        state_indices = rng.integers(len(s), size=128)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model.predict_function(x[indices]), y[indices])
        if method != "direct":
            loss = loss + 0.25 * nn.functional.mse_loss(model.predict_state(s[state_indices]), z[state_indices])
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        prediction = model.predict_function(q).numpy()
    return prediction


def _features(rows, cell_features, cells, chemical_scale):
    fingerprints, descriptors = chemistry_matrix(rows)
    return np.column_stack([cell_features[cells.get_indexer(rows.cell_id)],
                            fingerprints, chemical_scale.transform(descriptors)])


def joint_forecasts(function_rows, function_rna, split, state_rows, state_rna, state_targets):
    outcomes = function_rows.set_index("row_id", drop=False)
    forecasts = []
    support = []
    for fold, roles in split[split.track == "C"].groupby("fold"):
        fit = outcomes.loc[roles.loc[roles.role == "fit", "row_id"]].reset_index(drop=True)
        withheld = roles[roles.role == "withheld"]
        query = withheld[KEYS].merge(function_rows[["row_id", "isomeric_smiles"]], on="row_id", validate="one_to_one")
        query = query[query.compound_id.isin(fit.compound_id)].reset_index(drop=True)
        if set(fit.lineage_group) & set(query.lineage_group):
            raise ValueError("Function fit/test lineage overlap")
        use_state = ~state_rows.lineage_group.isin(withheld.lineage_group)
        use_state &= (state_rows.context_role != "adult_hcc") | state_rows.lineage_group.isin(fit.lineage_group)
        state_fit = state_rows[use_state].reset_index(drop=True)
        if not (state_fit.context_role == "adult_hcc").any():
            raise ValueError(f"No permitted matched adult state training in fold {fold}")
        state_observed = np.asarray(state_targets[use_state])
        if not np.isfinite(state_observed).all():
            raise ValueError("Non-finite state target")
        state_scale = StandardScaler().fit(state_observed)
        target_pca = PCA(n_components=12, svd_solver="randomized", random_state=14822)
        z = target_pca.fit_transform(state_scale.transform(state_observed))
        z = StandardScaler().fit_transform(z)
        groups = [pd.Index(frame.cell_id.unique()) for frame in (fit, query, state_fit)]
        reps, genes = feature_blocks(function_rna.loc[groups[0]], function_rna.loc[groups[1]],
                                     state_rna.loc[groups[2]])
        _, descriptors = chemistry_matrix(fit)
        chemical_scale = StandardScaler().fit(descriptors)
        x, q, s = (_features(rows, rep, cells, chemical_scale)
                   for rows, rep, cells in zip((fit, query, state_fit), reps, groups))
        means = fit.groupby("compound_id").endpoint_value.mean()
        anchor_fit = fit.compound_id.map(means).to_numpy()
        anchor_query = query.compound_id.map(means).to_numpy()
        residual = fit.endpoint_value.to_numpy() - anchor_fit
        scale = max(float(np.std(residual)), 1e-6)
        y = residual / scale
        for method in METHODS:
            estimates = [fit_predict(x, y, s, z, q, method=method, seed=seed) for seed in SEEDS]
            predicted = anchor_query + scale * np.mean(estimates, axis=0)
            forecasts.append(query[KEYS].assign(track="C_joint", fold=fold, method=method, predicted_value=predicted))
        support.append({"fold": int(fold), "fit_rows": len(fit), "forecast_rows": len(query),
                        "adult_state_fit_lineages": int(state_fit.loc[state_fit.context_role == "adult_hcc", "lineage_group"].nunique()),
                        "state_fit_lineages": int(state_fit.lineage_group.nunique()),
                        "state_fit_rows": len(state_fit), "basal_shared_genes": len(genes)})
    return pd.concat(forecasts, ignore_index=True), support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "splits", "state", "symbols", "registry", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from .train import prepared_data

    rows, rna = prepared_data(args.data)
    state, basal, targets = load_auxiliary(args.state, args.symbols, args.registry)
    split = read_table(args.splits)
    forecast, support = joint_forecasts(rows, rna, split, state, basal, targets)
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {
        "scope": "public auxiliary-state learning on the C functional folds",
        "support": support, "split_sha256": sha256(args.splits),
        "functional_rows_sha256": sha256(args.data / "rows.parquet"),
        "functional_basal_sha256": sha256(args.data / "rna.parquet"),
        "state_receipt_sha256": sha256(args.state / "receipt.json"),
        "basal_symbols_sha256": sha256(args.symbols), "model_registry_sha256": sha256(args.registry),
        "code_sha256": sha256(__file__),
    }
    freeze_table(forecast, args.out / "predictions.parquet", provenance=provenance)
    publish_json(args.out / "support.json", provenance)


if __name__ == "__main__":
    main()
