"""Fitted and deployable functional-endpoint models using basal RNA and structure."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class DomainError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256, includeChirality=True)
DESCRIPTORS = (
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds,
)
SEEDS = (14822, 14823, 14824)
SETTINGS = {
    "drug_mean": (0,),
    "rna_neighbor": (5,),
    "additive": (10,),
    "structure_ridge": (10, 100, 1000),
    "interaction_mlp": (16, 32, 64),
}
CONTEXT_METHODS = ("rna_neighbor", "interaction_mlp")
NONINTERACTION_METHODS = ("drug_mean", "additive", "structure_ridge")


@lru_cache(maxsize=4096)
def chemistry(smiles: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Preserve exact structure identity and derive parent-based descriptors."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise DomainError("structure_missing")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise DomainError("invalid_isomeric_smiles")
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    parent = rdMolStandardize.FragmentParent(molecule)
    if parent.GetNumAtoms() == 0:
        raise DomainError("structure_parent_empty")
    fingerprint = MORGAN.GetFingerprintAsNumPy(parent).astype(float)
    descriptors = np.array([function(parent) for function in DESCRIPTORS])
    return canonical, fingerprint, descriptors


def available_rna(rna: pd.DataFrame) -> pd.DataFrame:
    if not rna.index.is_unique or not rna.columns.is_unique:
        raise ValueError("Basal RNA requires unique cell and gene identifiers.")
    return rna.astype(float).replace([np.inf, -np.inf], np.nan)


def fit_rna(training: pd.DataFrame, lineage: pd.Series) -> dict:
    """Select high-variance genes and fit scaling and PCA inside the fit groups."""
    valid = training.notna().mean().ge(0.95)
    variances = training.loc[:, valid].var().sort_values(ascending=False, kind="stable")
    genes = sorted(variances[variances > 0].head(1000).index.tolist())
    if len(genes) < 2 or len(training) < 3:
        raise ValueError("Insufficient fit RNA for cell PCA.")
    medians = training[genes].median()
    scaled = StandardScaler().fit(training[genes].fillna(medians).to_numpy())
    standardized = scaled.transform(training[genes].fillna(medians).to_numpy())
    dimensions = min(16, len(training) - 1, len(genes))
    pca = PCA(n_components=dimensions, svd_solver="full").fit(standardized)
    pcs = pca.transform(standardized)
    pc_scale = StandardScaler().fit(pcs)
    coordinates = pc_scale.transform(pcs)
    distances = cdist(coordinates, coordinates)
    groups = lineage.loc[training.index].to_numpy()
    distances[groups[:, None] == groups[None, :]] = np.inf
    nearest = distances.min(axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    threshold = float(np.quantile(finite_nearest, 0.95)) if finite_nearest.size else None
    return {
        "genes": genes, "medians": medians.to_numpy(), "scale": scaled,
        "pca": pca, "pc_scale": pc_scale, "coordinates": coordinates,
        "fit_cells": training.index.tolist(), "distance_threshold": threshold,
        "min_gene_fraction": 0.90,
    }


def transform_rna(transform: dict, rna: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = rna.reindex(columns=transform["genes"]).to_numpy(dtype=float)
    observed = np.isfinite(values)
    filled = np.where(observed, values, transform["medians"])
    standardized = transform["scale"].transform(filled)
    state = transform["pc_scale"].transform(transform["pca"].transform(standardized))
    return state, observed.mean(axis=1)


def cell_coordinates(transform: dict, rows: pd.DataFrame, rna: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    index = pd.Index(rows["cell_id"].unique())
    state, fraction = transform_rna(transform, rna.reindex(index))
    lookup = index.get_indexer(rows["cell_id"])
    return state[lookup], fraction[lookup]


def chemistry_matrix(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fingerprints, descriptors = zip(*(chemistry(smiles)[1:] for smiles in rows["isomeric_smiles"]))
    return np.stack(fingerprints), np.stack(descriptors)


def structure_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    common = query @ reference.T
    union = query.sum(axis=1, keepdims=True) + reference.sum(axis=1) - common
    return np.divide(common, union, out=np.zeros_like(common), where=union > 0)


def compound_anchor(bundle: dict, compounds: pd.Series, fingerprint: np.ndarray) -> np.ndarray:
    means = bundle["drug_means"]
    values = compounds.map(means).to_numpy(dtype=float)
    novel = np.flatnonzero(~np.isfinite(values))
    if len(novel):
        similarities = structure_similarity(fingerprint[novel], bundle["fit_fingerprints"])
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :3]
        nearest = np.take_along_axis(similarities, order, axis=1)
        weights = nearest**2
        donor_means = bundle["fit_drug_mean_vector"][order]
        values[novel] = np.divide(
            (weights * donor_means).sum(axis=1), weights.sum(axis=1),
            out=np.full(len(novel), bundle["global_mean"]), where=weights.sum(axis=1) > 0,
        )
    return values


def feature_matrix(bundle: dict, fingerprint: np.ndarray, descriptors: np.ndarray,
                   state: np.ndarray) -> np.ndarray:
    chem = np.column_stack([fingerprint, bundle["descriptor_scale"].transform(descriptors)])
    return np.column_stack([chem, state])


def fit_base(train: pd.DataFrame, rna: pd.DataFrame) -> dict:
    """Share only fit-derived transforms and common drug anchor across arms."""
    cell_groups = train[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id")["lineage_group"]
    if not cell_groups.index.is_unique:
        raise ValueError("Each cell must have one lineage group.")
    transform = fit_rna(rna.loc[cell_groups.index], cell_groups)
    drug_rows = train[["compound_id", "isomeric_smiles"]].drop_duplicates().sort_values("compound_id")
    if drug_rows["compound_id"].duplicated().any():
        raise ValueError("Each assay compound identity must have one exact structure.")
    fingerprints, descriptors = chemistry_matrix(drug_rows)
    drug_means = train.groupby("compound_id")["endpoint_value"].mean()
    return {
        "rna": transform, "drug_means": drug_means.to_dict(),
        "fit_ids": drug_rows["compound_id"].tolist(),
        "structures": {
            drug: chemistry(smiles)[0]
            for drug, smiles in drug_rows.itertuples(index=False, name=None)
        },
        "fit_fingerprints": fingerprints,
        "fit_drug_mean_vector": drug_means.loc[drug_rows["compound_id"]].to_numpy(),
        "global_mean": float(train["endpoint_value"].mean()),
        "descriptor_scale": StandardScaler().fit(descriptors),
    }


def fit_arm(base: dict, train: pd.DataFrame, rna: pd.DataFrame, method: str, setting: int,
            seeds: tuple[int, ...] = SEEDS[:1]) -> dict:
    bundle = dict(base, method=method, setting=setting)
    if method == "drug_mean":
        return bundle
    state, _ = cell_coordinates(bundle["rna"], train, rna)
    observed = train["endpoint_value"].to_numpy()
    if method == "rna_neighbor":
        bundle["neighbor_data"] = {
            drug: (state[indices], observed[indices])
            for drug, indices in train.groupby("compound_id", sort=True).indices.items()
        }
        return bundle

    fingerprint, descriptors = chemistry_matrix(train)
    if method == "additive":
        features = state
    else:
        features = feature_matrix(bundle, fingerprint, descriptors, state)

    residual = observed - compound_anchor(bundle, train["compound_id"], fingerprint)
    if method == "interaction_mlp":
        scale = max(float(residual.std()), 1e-8)
        bundle["residual_scale"] = scale
        bundle["regressors"] = [
            MLPRegressor(
                hidden_layer_sizes=(setting,), activation="tanh", solver="adam",
                alpha=1.0, random_state=seed, batch_size=128, max_iter=100, tol=1e-5,
                n_iter_no_change=10, learning_rate_init=0.003,
            ).fit(features, residual / scale)
            for seed in seeds
        ]
    elif method in {"additive", "structure_ridge"}:
        bundle["regressors"] = [Ridge(alpha=setting).fit(features, residual)]
    else:
        raise ValueError(f"Unknown model arm {method!r}.")
    return bundle


def predict_arm(bundle: dict, rows: pd.DataFrame, rna: pd.DataFrame,
                *, drop_rna: bool = False) -> np.ndarray:
    fingerprint, descriptors = chemistry_matrix(rows)
    anchor = compound_anchor(bundle, rows["compound_id"], fingerprint)
    method = bundle["method"]
    if method == "drug_mean":
        return anchor
    state, fraction = cell_coordinates(bundle["rna"], rows, rna)
    if np.any(fraction < bundle["rna"]["min_gene_fraction"]):
        raise DomainError("insufficient_gene_coverage")
    if drop_rna:
        state = np.zeros_like(state)
    if method == "rna_neighbor":
        for drug, indices in rows.groupby("compound_id", sort=True).indices.items():
            donors = bundle["neighbor_data"].get(drug)
            if donors is None:
                continue
            coordinates, values = donors
            distance = cdist(state[indices], coordinates)
            order = np.argsort(distance, axis=1, kind="stable")[:, :bundle["setting"]]
            nearest = np.take_along_axis(distance, order, axis=1)
            weights = 1 / np.maximum(nearest, 1e-6)
            anchor[indices] = (weights * values[order]).sum(axis=1) / weights.sum(axis=1)
        return anchor
    features = state if method == "additive" else feature_matrix(bundle, fingerprint, descriptors, state)
    residual = np.mean([model.predict(features) for model in bundle["regressors"]], axis=0)
    return anchor + bundle.get("residual_scale", 1.0) * residual


class FunctionalModel:
    def __init__(self, bundle: dict, contract: dict):
        self.bundle = bundle
        self.contract = contract

    def _resolve(self, smiles: str | None, compound_id: str | None) -> tuple[str, str]:
        known = self.bundle["structures"]
        if compound_id is not None and compound_id not in known:
            raise DomainError("compound_outside_registered_panel")
        if smiles is None:
            if compound_id is None:
                raise DomainError("compound_identity_missing")
            return compound_id, known[compound_id]
        canonical = chemistry(smiles)[0]
        if compound_id is not None:
            if known[compound_id] != canonical:
                raise DomainError("compound_id_structure_mismatch")
            return compound_id, canonical
        matching = [drug for drug, structure in known.items() if structure == canonical]
        if len(matching) != 1:
            reason = "compound_outside_registered_panel" if not matching else "compound_alias_ambiguous"
            raise DomainError(reason)
        return matching[0], canonical

    def _rna(self, cell_rna_dict: dict) -> pd.DataFrame:
        if not isinstance(cell_rna_dict, dict) or not cell_rna_dict:
            raise DomainError("baseline_rna_missing")
        try:
            frame = pd.DataFrame([cell_rna_dict], index=["query"]).astype(float)
        except (TypeError, ValueError) as error:
            raise DomainError("invalid_baseline_expression") from error
        return frame

    def assess_domain(self, cell_rna_dict: dict, smiles: str | None = None,
                      compound_id: str | None = None) -> dict:
        drug, structure = self._resolve(smiles, compound_id)
        transform = self.bundle["rna"]
        rna = self._rna(cell_rna_dict)
        coords, fraction = transform_rna(transform, rna)
        if fraction[0] < transform["min_gene_fraction"]:
            raise DomainError("insufficient_gene_coverage")
        distance = float(cdist(coords, transform["coordinates"]).min())
        threshold = transform["distance_threshold"]
        fingerprint = chemistry(structure)[1].reshape(1, -1)
        similarity = float(structure_similarity(fingerprint, self.bundle["fit_fingerprints"]).max())
        return {
            "registered_compound_id": drug, "rna_gene_fraction": float(fraction[0]),
            "nearest_fit_cell_distance": distance, "rna_distance_threshold": threshold,
            "rna_ood": threshold is not None and distance > threshold,
            "nearest_structure_tanimoto": similarity,
        }

    def predict_many(self, cell_rna_dict: dict, candidates: list[dict]) -> np.ndarray:
        rna = self._rna(cell_rna_dict)
        _, fraction = transform_rna(self.bundle["rna"], rna)
        if fraction[0] < self.bundle["rna"]["min_gene_fraction"]:
            raise DomainError("insufficient_gene_coverage")
        records = []
        for candidate in candidates:
            drug, canonical = self._resolve(candidate.get("smiles"), candidate.get("compound_id"))
            records.append(("query", drug, canonical))
        rows = pd.DataFrame(records, columns=["cell_id", "compound_id", "isomeric_smiles"])
        return predict_arm(self.bundle, rows, rna)

    def predict(self, cell_rna_dict: dict, smiles: str | None = None,
                compound_id: str | None = None) -> float:
        return float(self.predict_many(cell_rna_dict, [{"smiles": smiles, "compound_id": compound_id}])[0])


def load_model(path: str | Path) -> FunctionalModel:
    directory = Path(path)
    contract = json.loads((directory / "contract.json").read_text())
    bundle = joblib.load(directory / "checkpoint.joblib")
    return FunctionalModel(bundle, contract)
