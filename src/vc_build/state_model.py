"""Small fixed models for an L1000 24 h vehicle-relative state response."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .state_data import exact_structure

ARMS = (
    "zero_change", "drug_common", "structure_nearest", "center_shrink", "rna_additive",
    "rna_interaction", "rna_fit_swap",
)
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512, includeChirality=True)
PAIR = ["cell_id", "compound_id"]


def conditions(rows, values):
    """Average wells in a plate, plates in each cell-drug condition, then give cells equal weight."""
    ix = pd.MultiIndex.from_frame(rows[[*PAIR, "rna_plate"]])
    frame = pd.DataFrame(values, index=ix)
    return frame.groupby(level=[0, 1, 2], sort=True).mean().groupby(level=[0, 1], sort=True).mean()


def fingerprints(smiles):
    return np.stack([GENERATOR.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]).astype(np.float32)


def row_interaction(cell, drug):
    return np.einsum("ni,nj->nij", cell, drug).reshape(len(cell), -1)


def max_similarity(query, fit):
    dot = query @ fit.T
    denom = query.sum(axis=1)[:, None] + fit.sum(axis=1) - dot
    return dot / np.maximum(denom, 1)


@dataclass
class StateModel:
    genes: np.ndarray
    programmes: PCA
    scale: np.ndarray
    basal_scale: StandardScaler
    basal_pca: PCA
    drug_pca: PCA
    known: pd.DataFrame
    fingerprints: np.ndarray
    additive: Ridge
    interaction: Ridge
    swap: Ridge

    @classmethod
    def fit(cls, rows, observed, basal, genes):
        values = conditions(rows, observed)
        mean = values.groupby(level=1).mean()
        reference = mean.loc[values.index.get_level_values("compound_id")].to_numpy()
        y = values.to_numpy()
        labels = PCA(n_components=8, svd_solver="randomized", random_state=1622).fit(y)
        scale = np.sqrt(labels.explained_variance_).clip(.001)

        pairs = values.index.to_frame(index=False)
        cells = pd.Index(pairs.cell_id.unique())
        if len(cells) < 5:
            raise ValueError("The HCC state context model requires at least five distinct fitting contexts")
        fit_x = basal.loc[cells].to_numpy()
        x_scale = StandardScaler().fit(fit_x)
        x_norm = x_scale.transform(fit_x)
        x_pca = PCA(n_components=4, svd_solver="full").fit(x_norm)
        coords = pd.DataFrame(x_pca.transform(x_norm), index=cells)
        x = coords.loc[pairs.cell_id].to_numpy()
        perm = np.roll(np.arange(len(cells)), 1)
        swap = coords.iloc[perm].set_axis(cells).loc[pairs.cell_id].to_numpy()

        compound = fingerprints(mean.index)
        drug_pca = PCA(n_components=16, svd_solver="randomized", random_state=1622).fit(compound)
        drug_coords = pd.DataFrame(drug_pca.transform(compound), index=mean.index)
        d = drug_coords.loc[pairs.compound_id].to_numpy()
        resid = ((y - reference) @ labels.components_.T) / scale
        # The source is imbalanced in condition count. Each fitting context contributes equal mass.
        weight = pairs.cell_id.map(1. / pairs.cell_id.value_counts()).to_numpy()
        weight *= len(weight) / weight.sum()
        additive = Ridge(alpha=100.).fit(x, resid, sample_weight=weight)
        interact = Ridge(alpha=100.).fit(np.column_stack((x, row_interaction(x, d))), resid, sample_weight=weight)
        swapped = Ridge(alpha=100.).fit(np.column_stack((swap, row_interaction(swap, d))), resid, sample_weight=weight)
        return cls(genes, labels, scale, x_scale, x_pca, drug_pca, mean, compound, additive, interact, swapped)

    def project(self, values):
        return (values @ self.programmes.components_.T) / self.scale

    def predict(self, smiles, basal):
        bp = np.asarray(basal, dtype=float)
        smiles = np.asarray(smiles, dtype=str)
        if bp.shape != (len(smiles), len(self.genes)) or not np.isfinite(bp).all():
            raise ValueError("Basal input must contain finite RNA for each registered landmark")
        index = self.known.index.get_indexer(smiles)
        if (index < 0).any():
            raise ValueError("Model supports only exact molecules represented in training")
        x = self.basal_pca.transform(self.basal_scale.transform(bp))
        fp = self.fingerprints[index]
        d = self.drug_pca.transform(fp)
        design = np.column_stack((x, row_interaction(x, d)))
        y = self.known.to_numpy()[index]
        nearest = max_similarity(fp, self.fingerprints)
        nearest[np.arange(len(index)), index] = -1
        simple = self.known.to_numpy()[nearest.argmax(axis=1)]

        def combine(r):
            return y + (r * self.scale) @ self.programmes.components_

        return dict(zip(ARMS, (
            np.zeros_like(y), y, simple, y * .5, combine(self.additive.predict(x)),
            combine(self.interaction.predict(design)), combine(self.swap.predict(design)),
        ), strict=True))

    def predict_request(self, user):
        if (user.get("assay_id") != "CMap2020_L1000_Level3" or user.get("dose_uM") != 10
                or user.get("time_h") != 24 or user.get("input_mode") != "baseline_rna"):
            raise ValueError("Expected baseline RNA and the registered 10 µM / 24 h CMap Level 3 assay")
        smiles = exact_structure(user.get("smiles", ""))[0]
        supplied = user.get("basal_expression")
        if not isinstance(supplied, dict) or not all(str(g) in supplied for g in self.genes):
            raise ValueError("Baseline expression must contain every registered Entrez landmark")
        if user.get("baseline_scale") != "log2(TPM+1)":
            raise ValueError("Basal expression units must be declared as log2(TPM+1)")
        vector = np.array([[float(supplied[str(g)]) for g in self.genes]])
        y = self.predict([smiles], vector)["rna_interaction"]
        return {
            "output_units": "fit-only standardized L1000 programme delta at 24 h versus vehicle",
            "programme_definition": "CMap-2020-VCB02-fit-only-PCA8",
            "programme_delta": self.project(y).ravel().tolist(),
            "experimental_status": "development-only state head; HCC promotion gate pending",
        }
