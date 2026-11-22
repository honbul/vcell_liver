"""Regularize cell responses across registered chemicals with separable kernels."""

from dataclasses import dataclass

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


def cell_kernels(fit, query, signatures, auxiliary, *, device):
    """Build blocked fit and cross Gram matrices with train-only scaling."""
    genes = fit.columns
    fit_values, query_values = (
        ((rankdata(frame[genes].to_numpy(), axis=1) - 0.5) / len(genes) - 0.5).astype(np.float32)
        for frame in (fit, query)
    )

    def standardized(a, b, floor):
        center, spread = a.mean(axis=0), np.maximum(a.std(axis=0), floor)
        return tuple(torch.as_tensor(np.clip((value - center) / spread, -8, 8), device=device)
                     for value in (a, b))

    x, q = standardized(fit_values, query_values, 0.05)
    linear = dot_kernel(x, x)
    cross = dot_kernel(q, x)
    scale = linear.diagonal().mean().item()
    linear, cross = linear / scale, cross / scale
    query_energy = q.square().mean(dim=1) / scale
    nonlinear = radial(linear, linear.diagonal(), linear.diagonal(), 2.0)
    nonlinear_cross = radial(cross, query_energy, linear.diagonal(), 2.0)
    raw = (0.5 * (linear + nonlinear), 0.5 * (cross + nonlinear_cross))

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
    p, s = standardized(fit_values @ weights, query_values @ weights, 0.05)
    path = (dot_kernel(p, p), dot_kernel(s, p))

    # Fit public Chronos and genomic annotations as auxiliary targets on fit cells only.
    selected = x.var(dim=0).argsort(descending=True)[:768]
    x_small = torch.cat((x[:, selected], p), dim=1).cpu().numpy()
    q_small = torch.cat((q[:, selected], s), dim=1).cpu().numpy()
    proxy_fit, proxy_query = [], []
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
            estimator = Ridge(alpha=1000).fit(x_small[valid], values)
            proxy_fit.append(estimator.predict(x_small).reshape(len(x_small), len(columns)))
            proxy_query.append(estimator.predict(q_small).reshape(len(q_small), len(columns)))
    if not proxy_fit:
        raise ValueError("Insufficient fit-only support for biological auxiliaries")
    u, v = standardized(np.concatenate(proxy_fit, axis=1), np.concatenate(proxy_query, axis=1), 0.005)
    aux = (dot_kernel(u, u), dot_kernel(v, u))
    enriched = tuple(0.7 * r + 0.2 * a + 0.1 * b for r, a, b in zip(raw, path, aux))
    return {"full": enriched, "rna": raw}, u.shape[1]


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


def solve_residuals(cell, drug, weights, response, penalty, *, tolerance=5e-5, max_steps=220):
    """Solve weighted kernel ridge in the cell-by-drug grid without a dense pair Gram matrix."""
    root = weights.sqrt()
    rhs = root * response
    diagonal = penalty + weights * cell.diagonal()[:, None] * drug.diagonal()[None, :]
    coefficient = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual / diagonal
    energy = (direction * residual).sum()
    threshold = max(1e-7, tolerance * rhs.norm().item())
    for step in range(max_steps):
        product = root * (cell @ (root * direction) @ drug) + penalty * direction
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
