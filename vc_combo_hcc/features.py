"""Assay-independent compound and cell features for strictly held combinations."""
from dataclasses import dataclass
from functools import cached_property
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .context import verified
from .splits import record


def vectors(directory):
    file = directory / "vectors.npz"
    claim = json.loads((directory / "receipt.json").read_text())
    if record(file) != claim["output"]:
        raise ValueError(f"Molecular feature failed its receipt: {directory}")
    with np.load(file, allow_pickle=False) as arrays:
        ids, values = arrays["id"], arrays["vectors"].astype(np.float32)
    frame = pd.DataFrame(values, index=ids)
    if not frame.index.is_unique or not np.isfinite(values).all():
        raise ValueError("Invalid molecular representation")
    return frame


def transforms(values, train_rows):
    base = values[train_rows]
    mean, std = base.mean(axis=0), base.std(axis=0).clip(.03)
    return ((values - mean) / std).clip(-8, 8)


def pathway_lookup(path):
    with path.open() as stream:
        return {p[0]: p[2:] for line in stream if len(p := line.strip().split("\t")) > 2}


def memberships(genes, pathways):
    return np.stack([genes.isin(members).astype(np.float32) for members in pathways.values()], axis=1)


def make_target_features(chemicals, targets, curated, paths):
    targets_by_drug = targets.groupby("id").symbol.unique().to_dict()
    genes = pd.Index(sorted(targets.symbol.unique()))
    known = {drug: genes.get_indexer(target) for drug, target in targets_by_drug.items()}
    curation = pd.crosstab(curated.id, curated.pathway).clip(upper=1).reindex(chemicals, fill_value=0)
    mapping = memberships(genes, paths)
    embed = np.zeros((len(chemicals), len(paths) + 1), dtype=np.float32)
    for i, drug in enumerate(chemicals):
        ids = known.get(drug)
        if ids is not None:
            embed[i, :-1] = mapping[ids].mean(axis=0)
            embed[i, -1] = np.log1p(len(ids))
    return np.column_stack([embed, curation.to_numpy(dtype=np.float32)]), targets_by_drug


@dataclass
class FeatureView:
    left: np.ndarray
    right: np.ndarray
    cell: np.ndarray
    assay: np.ndarray
    dependency_left: np.ndarray
    dependency_right: np.ndarray
    mono: np.ndarray

    def classical(self, library):
        a, b = self.left, self.right
        left, right = library.ecfp_small[a], library.ecfp_small[b]
        projected = [left + right, np.abs(left - right), left * right]
        dependency = [self.dependency_left + self.dependency_right,
                      np.abs(self.dependency_left - self.dependency_right)]
        annotations = [library.targets[a] + library.targets[b], np.abs(library.targets[a] - library.targets[b])]
        morph = [library.rna[self.cell], library.crispr[self.cell]]
        return np.column_stack([*projected, *dependency, *annotations, *morph, self.assay]).astype(np.float32)


class Library:
    def __init__(self, context, embedding, hallmarks):
        ecfp = vectors(embedding / "ecfp4")
        molform = vectors(embedding / "molformer")
        self.ids = ecfp.index.intersection(molform.index).sort_values()
        self.ecfp = ecfp.loc[self.ids].to_numpy()
        self.molformer = molform.loc[self.ids].to_numpy()
        self.rna_raw = pd.read_parquet(verified(context, "rna.parquet"))
        self.effect = pd.read_parquet(verified(context, "chronos.parquet"))
        self.mono = pd.read_parquet(verified(context, "single_agent.parquet")).set_index(
            ["cell_id", "id"], verify_integrity=True
        ).inhibition
        t = pd.read_csv(verified(context, "target_registry.tsv"), sep="\t", dtype=str)
        p = pd.read_csv(verified(context, "curated_pathways.tsv"), sep="\t", dtype=str)
        hall = pathway_lookup(hallmarks)
        self.targets, self.target_map = make_target_features(self.ids, t, p, hall)
        self.hallmarks = hall

    def fold(self, train, optional_rna=None):
        return FoldFeatures(self, train, optional_rna)


class FoldFeatures:
    def __init__(self, base, train, optional_rna=None):
        self.base = base
        self.cell_ids = base.rna_raw.index
        raw = base.rna_raw
        if optional_rna is not None:
            # Missing screening genes carry the per-sample average rank.
            shared = raw.columns.intersection(optional_rna.columns)
            if len(shared) < 6000:
                raise ValueError("Too few organoid genes on the independent DepMap RNA axis")
            raw = pd.concat([raw, optional_rna.reindex(columns=raw.columns)], axis=0)
            if not raw.index.is_unique:
                raise ValueError("Organoid and cell-line identity overlap")
            self.cell_ids = raw.index
        self.ids = base.ids
        cells = self.cell_ids.get_indexer(train.cell_id.unique())
        drugs = self.ids.get_indexer(sorted(set(train.compound_a) | set(train.compound_b)))
        if cells.min() < 0 or drugs.min() < 0:
            raise ValueError("Rows outside verified features reached the fitter")
        matrix = raw.to_numpy(dtype=np.float32)
        columns = np.argsort(-matrix[cells].var(axis=0), kind="stable")[:384]
        ranked = rankdata(matrix, method="average", axis=1, nan_policy="omit")
        ranked = ranked / np.isfinite(matrix).sum(axis=1, keepdims=True)
        self.rna = transforms(np.nan_to_num(ranked[:, columns], nan=.5).astype(np.float32), cells)
        self.molecular = transforms(base.molformer, drugs)
        self.targets = base.targets
        # Compress sparse fingerprint bits for classical heads with a fixed
        # signed projection. Neural public comparators consume full bits.
        rng = np.random.default_rng(7712)
        projection = rng.choice(np.array([-1, 1], dtype=np.float32), size=(2048, 128)) / np.sqrt(128)
        graph = base.ecfp[:, :2048] @ projection.astype(np.float32)
        extra = transforms(base.ecfp[:, 2048:], drugs)
        self.ecfp_small = np.column_stack((graph, extra)).astype(np.float32)
        self.ecfp = np.column_stack((base.ecfp[:, :2048], extra)).astype(np.float32)
        self.effect = base.effect.reindex(self.cell_ids)
        genes = self.effect.columns
        self.effect_array = self.effect.to_numpy(dtype=np.float32)
        paths = memberships(genes, base.hallmarks)
        known = np.isfinite(self.effect_array)
        by_path = np.nan_to_num(self.effect_array) @ paths / np.maximum(known.astype(np.float32) @ paths, 1)
        self.crispr = np.column_stack((by_path, known.sum(axis=1) > 0)).astype(np.float32)
        self.target_columns = {
            drug: [i for i in genes.get_indexer(names) if i >= 0] for drug, names in base.target_map.items()
        }

    @cached_property
    def dependency_cache(self):
        return {}

    def dependencies(self, cells, compounds):
        count = len(cells)
        encoded = cells.astype(np.int64) * len(self.ids) + compounds
        unique, inv = np.unique(encoded, return_inverse=True)
        stats = np.zeros((len(unique), 4), dtype=np.float32)
        for j, key in enumerate(unique):
            if key not in self.dependency_cache:
                cell, drug = divmod(int(key), len(self.ids))
                indices = self.target_columns.get(self.ids[drug], [])
                observed = self.effect_array[cell, indices]
                observed = observed[np.isfinite(observed)]
                row = [float(np.min(observed)), float(np.mean(observed)), float(np.max(observed)),
                       float(np.log1p(len(observed)))] if len(observed) else [0., 0., 0., 0.]
                self.dependency_cache[key] = row
            stats[j] = self.dependency_cache[key]
        return stats[inv].reshape((count, 4))

    def view(self, rows):
        a, b, c = self.ids.get_indexer(rows.compound_a), self.ids.get_indexer(rows.compound_b), \
            self.cell_ids.get_indexer(rows.cell_id)
        if any(x.min() < 0 for x in (a, b, c)):
            raise ValueError("Unregistered held-out features")
        assay = np.column_stack([
            rows.source.str.startswith(prefix).to_numpy(dtype=np.float32)
            for prefix in ("Sanger:", "DrugCombDB:")
        ])
        mono = self.base.mono.reindex(
            pd.MultiIndex.from_arrays([pd.concat([rows.cell_id, rows.cell_id]),
                                       pd.concat([rows.compound_a, rows.compound_b])])
        ).to_numpy().reshape(2, -1).T
        return FeatureView(a, b, c, assay, self.dependencies(c, a), self.dependencies(c, b), mono)
