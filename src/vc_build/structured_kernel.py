"""Regularize cell responses across registered chemicals with separable kernels."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from .external import SOURCE, _chemistry


def dot_kernel(first, second):
    return first @ second.T / first.shape[1]


def radial(linear, row_energy, column_energy, width):
    distance = (row_energy[:, None] + column_energy[None, :] - 2 * linear).clamp_min(0)
    return torch.exp(-distance / width)


def ranked_expression(frame):
    ranks = rankdata(frame.to_numpy(), axis=1)
    return ((ranks - 0.5) / frame.shape[1] - 0.5).astype(np.float32)


@dataclass
class Standardization:
    center: np.ndarray
    spread: np.ndarray

    @classmethod
    def fit(cls, values, floor):
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), floor))

    def apply(self, values, device):
        return torch.as_tensor(np.clip((values - self.center) / self.spread, -8, 8), device=device)


@dataclass
class RankProjection:
    center: np.ndarray
    axes: np.ndarray

    def apply(self, ranks):
        return ranks - ((ranks - self.center) @ self.axes.T) @ self.axes


@dataclass
class CellKernelBasis:
    genes: pd.Index
    reference: tuple
    reference_energy: torch.Tensor
    reference_scale: float
    sequence: Standardization
    pathway: Standardization
    auxiliary: Standardization
    pathway_weights: sparse.csr_matrix
    auxiliary_models: tuple
    indices: np.ndarray
    projection: RankProjection | None = None

    def cpu(self):
        return replace(self, reference=tuple(x.cpu() for x in self.reference),
                       reference_energy=self.reference_energy.cpu())

    def vectors(self, query):
        device = self.reference[0].device
        raw = ranked_expression(query[self.genes])
        if self.projection is not None:
            raw = self.projection.apply(raw)
        q = self.sequence.apply(raw, device)
        s = self.pathway.apply(raw @ self.pathway_weights, device)
        proxy = auxiliary_features(self.auxiliary_models, auxiliary_inputs(q, s, self.indices))
        v = self.auxiliary.apply(proxy, device)
        return q, s, v

    def cross(self, query):
        x, p, u = self.reference
        q, s, v = self.vectors(query)
        linear = dot_kernel(q, x) / self.reference_scale
        energy = q.square().mean(dim=1) / self.reference_scale
        rna = 0.5 * (linear + radial(linear, energy, self.reference_energy, 2.0))
        return {"rna": rna, "full": 0.7 * rna + 0.2 * dot_kernel(s, p) + 0.1 * dot_kernel(v, u)}


def auxiliary_inputs(x, pathway, selected):
    ix = torch.as_tensor(selected, dtype=torch.long, device=x.device)
    return torch.cat((x[:, ix], pathway), dim=1).cpu().numpy()


def auxiliary_features(models, values):
    return np.concatenate([model.predict(values).reshape(len(values), -1) for model in models], axis=1)


def fit_cell_kernels(fit, signatures, auxiliary, *, device, projection_axes=None):
    """Fit the reusable expression basis and return the training Gram matrices."""
    genes = fit.columns
    raw = ranked_expression(fit)
    projection = None
    if projection_axes is not None:
        projection = RankProjection(raw.mean(axis=0), projection_axes)
        raw = projection.apply(raw)
    sequence = Standardization.fit(raw, 0.05)
    x = sequence.apply(raw, device)
    rna_linear = dot_kernel(x, x)
    scale = rna_linear.diagonal().mean().item()
    rna_linear = rna_linear / scale
    energy = rna_linear.diagonal().clone()
    rna = 0.5 * (rna_linear + radial(rna_linear, energy, energy, 2.0))

    annotated = signatures[signatures.gene.isin(genes)].copy()
    support = annotated.groupby("signature").gene.nunique()
    annotated = annotated[annotated.signature.isin(support[support.ge(10)].index)]
    pathways = pd.Index(sorted(annotated.signature.unique()))
    if not len(pathways):
        raise ValueError("No supported pathway/TF signatures")
    weights = sparse.csr_matrix(
        (annotated.weight.astype(np.float32), (genes.get_indexer(annotated.gene),
                                               pathways.get_indexer(annotated.signature))),
        shape=(len(genes), len(pathways)),
    )
    column_norm = np.sqrt(np.asarray(weights.power(2).sum(axis=0))).ravel()
    weights = weights @ sparse.diags(1.0 / column_norm)
    pathway_values = raw @ weights
    pathway = Standardization.fit(pathway_values, 0.05)
    p = pathway.apply(pathway_values, device)

    # Fit public Chronos and genomic annotations as auxiliary targets on fit cells only.
    selected = x.var(dim=0).argsort(descending=True)[:768].cpu().numpy()
    x_small = auxiliary_inputs(x, p, selected)
    models = []
    for table in auxiliary.values():
        measured = table.reindex(fit.index)
        by_support = {}
        for name, column in measured.items():
            mask = column.notna().to_numpy()
            if mask.sum() >= 40 and column.std() >= 1e-6:
                by_support.setdefault(mask.tobytes(), []).append(name)
        for mask, columns in by_support.items():
            valid = np.frombuffer(mask, dtype=bool)
            values = measured.loc[valid, columns].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Invalid public auxiliary target")
            models.append(Ridge(alpha=1000).fit(x_small[valid], values))
    if not models:
        raise ValueError("Insufficient fit-only support for biological auxiliaries")
    proxy = auxiliary_features(models, x_small)
    aux = Standardization.fit(proxy, 0.005)
    u = aux.apply(proxy, device)
    basis = CellKernelBasis(genes, (x, p, u), energy, scale, sequence, pathway, aux,
                            weights, tuple(models), selected, projection)
    enriched = 0.7 * rna + 0.2 * dot_kernel(p, p) + 0.1 * dot_kernel(u, u)
    return basis, {"rna": rna, "full": enriched}


def cell_kernels(fit, query, signatures, auxiliary, *, device):
    """Build blocked fit and cross Gram matrices with train-only scaling."""
    basis, training = fit_cell_kernels(fit, signatures, auxiliary, device=device)
    held = basis.cross(query)
    return {name: (training[name], held[name]) for name in training}, basis.reference[-1].shape[1]


def drug_kernels(rows, targets, *, device):
    fingerprints = _chemistry(rows, fp_size=2048).iloc[:, :2048].sort_index()
    bits = torch.as_tensor(fingerprints.to_numpy(dtype=np.float32), device=device)
    intersection = bits @ bits.T
    unions = bits.sum(dim=1)
    tanimoto = intersection / (unions[:, None] + unions[None, :] - intersection).clamp_min(1)
    genes = sorted(set().union(*(targets.get(name, ()) for name in fingerprints.index)))
    gene_index = {gene: i for i, gene in enumerate(genes)}
    matrix = torch.zeros((len(bits), len(genes)), device=device)
    for i, name in enumerate(fingerprints.index):
        matrix[i, [gene_index[gene] for gene in targets.get(name, ())]] = 1
    matrix /= matrix.norm(dim=1, keepdim=True).clamp_min(1)
    target_overlap = matrix @ matrix.T
    identity = torch.eye(len(bits), device=device)
    return fingerprints.index, {
        "structured": 0.65 * identity + 0.15 * tanimoto + 0.10 * target_overlap + 0.10,
        "no_drug_sharing": identity,
        "rna_chem_only": 0.75 * identity + 0.15 * tanimoto + 0.10,
    }


def training_grid(rows, cells, drugs, *, device, row_weight=None):
    """Weight assays equally within each drug while preserving its total effective support."""
    groups = rows.groupby([*SOURCE, "compound_id"]).residual
    sizes = groups.transform("size")
    largest = rows.assign(size=sizes).groupby("compound_id")["size"].transform("max")
    assays = rows[[*SOURCE, "compound_id"]].drop_duplicates().groupby("compound_id").size()
    weight = (largest / (sizes * rows.compound_id.map(assays))).to_numpy()
    if row_weight is not None:
        if row_weight.shape != weight.shape or not np.isfinite(row_weight).all() or (row_weight <= 0).any():
            raise ValueError("Nonpositive or invalid kernel observation weight")
        weight *= row_weight
    shape = (len(cells), len(drugs))
    index = cells.get_indexer(rows.cell_id), drugs.get_indexer(rows.compound_id)
    total, weighted = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    np.add.at(total, index, weight)
    np.add.at(weighted, index, weight * rows.residual.to_numpy())
    response = np.divide(weighted, total, out=weighted, where=total > 0)
    return tuple(torch.as_tensor(value, device=device) for value in (total, response))


def solve_residuals(cell, drug, weights, response, penalty, *, per_drug_prior=None, tolerance=5e-5, max_steps=220):
    """Solve weighted kernel ridge in the cell-by-drug grid without a dense pair Gram matrix."""
    root = weights.sqrt()
    rhs = root * response
    kernel_diagonal = cell.diagonal()[:, None] * drug.diagonal()[None, :]
    if per_drug_prior is not None:
        if per_drug_prior.ndim not in (2, 3) or per_drug_prior.shape[:2] != weights.shape:
            raise ValueError("The per-drug kernel features must share the observation grid")
        if per_drug_prior.ndim == 2:
            per_drug_prior = per_drug_prior.unsqueeze(-1)
        kernel_diagonal = kernel_diagonal + per_drug_prior.square().sum(dim=-1)
    diagonal = penalty + weights * kernel_diagonal
    coefficient = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual / diagonal
    energy = (direction * residual).sum()
    threshold = max(1e-7, tolerance * rhs.norm().item())
    for step in range(max_steps):
        weighted = root * direction
        kernel_product = cell @ weighted @ drug
        if per_drug_prior is not None:
            kernel_product = kernel_product + project_drug_features(per_drug_prior, per_drug_prior, weighted)
        product = root * kernel_product + penalty * direction
        update = energy / (direction * product).sum().clamp_min(1e-20)
        coefficient += update * direction
        residual -= update * product
        error = residual.norm().item()
        if error <= threshold:
            return root * coefficient, step + 1, error
        scaled = residual / diagonal
        next_energy = (scaled * residual).sum()
        direction = scaled + (next_energy / energy) * direction
        energy = next_energy
    raise RuntimeError(f"Kernel solver residual {error:.4g} exceeds {threshold:.4g}")


def project_drug_features(fit, query, coefficients):
    """Apply independent linear feature kernels for each drug."""
    effects = (fit * coefficients.unsqueeze(-1)).sum(dim=0)
    return (query * effects).sum(dim=-1)


@dataclass
class KernelGrid:
    cells: pd.Index
    drugs: pd.Index
    cell: dict
    drug: dict
    weights: torch.Tensor
    response: torch.Tensor
    auxiliary_features: int

    @classmethod
    def prepare(cls, source, fit, query, edges, targets, auxiliary, *, device):
        cell, count = cell_kernels(fit, query, edges, auxiliary, device=device)
        drugs, drug = drug_kernels(source, targets, device=device)
        weights, response = training_grid(source, fit.index, drugs, device=device)
        return cls(query.index, drugs, cell, drug, weights, response, count)

    def forecast(self, roster, arm, penalty):
        cell, cross = self.cell["rna" if arm == "rna_chem_only" else "full"]
        drug = self.drug[arm]
        coefficient, steps, error = solve_residuals(cell, drug, self.weights, self.response, penalty)
        predictions = (cross @ coefficient @ drug).detach().cpu().numpy()
        indices = self.cells.get_indexer(roster.cell_id), self.drugs.get_indexer(roster.compound_id)
        if any((index < 0).any() for index in indices):
            raise ValueError("Missing registered kernel cell or compound")
        return predictions[indices], {"iterations": steps, "solver_residual": error}


@dataclass
class KernelExpressionSchema:
    genes: pd.Index
    indices: np.ndarray
    raw_median: np.ndarray


@dataclass
class StructuredPredictor:
    features: KernelExpressionSchema
    basis: CellKernelBasis
    drugs: pd.Index
    anchor: pd.Series
    coefficients: torch.Tensor
    drug_kernel: torch.Tensor

    def predict(self, query, rna):
        cells = rna.index.get_indexer(query.cell_id)
        drugs = self.drugs.get_indexer(query.compound_id)
        if (cells < 0).any() or (drugs < 0).any():
            raise ValueError("Query requires a matched expression profile and compound")
        cross = self.basis.cross(rna)["full"]
        residual = (cross @ self.coefficients @ self.drug_kernel).numpy()
        return query.compound_id.map(self.anchor).to_numpy(), residual[cells, drugs]


@dataclass
class PanelPredictor:
    features: KernelExpressionSchema
    basis: CellKernelBasis
    drugs: pd.Index
    anchor: pd.Series
    projection: torch.Tensor | None
    cell_center: torch.Tensor | None = None

    def predict(self, query, rna, *, cross=None):
        rows = rna.index.get_indexer(query.cell_id)
        columns = self.drugs.get_indexer(query.compound_id)
        if (rows < 0).any() or (columns < 0).any():
            raise ValueError("Query requires a matched expression profile and registered compound")
        base = self.anchor.to_numpy()[columns]
        if self.projection is None:
            return base, np.zeros_like(base)
        if cross is None:
            cross = self.basis.cross(rna)["full"]
        if self.cell_center is not None:
            cross = cross - cross.mean(dim=1, keepdim=True) - self.cell_center + self.cell_center.mean()
        estimates = (cross @ self.projection).cpu().numpy()
        return base, estimates[rows, columns]
