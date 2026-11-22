"""Fit and evaluate separable sample sensitivity and drug-specific interaction kernels."""

import numpy as np
import pandas as pd

from .assay_aware_cv import source_grids
from .external import KEY, _validate
from .v0plus_cv import panel_targets

PENALTY = 4.0
ARMS = ("marginal_only", "interaction_only", "separated", "separated_ridge_average")
PRIMARY = "separated"


def orthogonal_parts(y, w):
    """Weighted projection on observed rows, solving only a drug-sized Schur system."""
    if y.ndim != 2 or y.shape != w.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Invalid observed response grid")
    valid = w > 0
    if not valid.any() or not np.isfinite(y[valid]).all():
        raise ValueError("Empty or nonfinite observed response grid")
    mask = valid.any(axis=0)
    response = np.where(valid, y, 0.0).astype(np.float64)
    weights = w[:, mask].astype(np.float64)
    row_count = weights.sum(axis=1)
    col_count = weights.sum(axis=0)
    inverse = np.divide(1.0, row_count, out=np.zeros(len(row_count)), where=row_count > 0)
    wy = weights * response[:, mask]
    row_total = wy.sum(axis=1)
    column_total = wy.sum(axis=0)
    if valid.all() and np.ptp(w) == 0:
        # Complete equally weighted panels have a closed-form projection.
        sample = response.mean(axis=1) - response.mean()
        drug_effect = response.mean(axis=0)
        rank = len(col_count) - 1
    else:
        schur = np.diag(col_count) - weights.T @ (weights * inverse[:, None])
        rhs = column_total - weights.T @ (row_total * inverse)
        drug_effect, _, rank, _ = np.linalg.lstsq(schur, rhs, rcond=1e-11)
        sample = (row_total - weights @ drug_effect) * inverse
        if rank + 1 < len(col_count):
            # Each disconnected support block has its own additive gauge.
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components

            _, blocks = connected_components(csr_matrix(weights.T @ weights > 0), directed=False)
            for block in np.unique(blocks):
                cols = blocks == block
                rows = weights[:, cols].sum(axis=1) > 0
                offset = np.average(sample[rows], weights=row_count[rows])
                drug_effect[cols] += offset
                sample[rows] -= offset
        else:
            offset = np.average(sample, weights=row_count)
            drug_effect += offset
            sample[row_count > 0] -= offset
    drug = np.zeros(w.shape[1], dtype=np.float64)
    drug[mask] = drug_effect
    residual = np.where(valid, response - sample[:, None] - drug, 0.0)
    orthogonal = w * residual
    tolerance = 1e-8 * (1 + np.sum(np.abs(w * response)))
    if max(abs(orthogonal.sum(axis=0)).max(), abs(orthogonal.sum(axis=1)).max()) > tolerance:
        raise RuntimeError("Two-way decomposition failed orthogonality on the observed support")
    return sample, drug, residual


def centered_kernels(fit, cross):
    col = fit.mean(dim=0, keepdim=True)
    grand = col.mean()
    return (fit - fit.mean(dim=1, keepdim=True) - col + grand,
            cross - cross.mean(dim=1, keepdim=True) - col + grand)


def forecast(fit, query, rna, edges, targets, auxiliary, device):
    import torch

    from .structured_kernel import cell_kernels, drug_kernels, solve_residuals

    _validate(fit, query)
    _, anchor, train, _ = panel_targets(fit)
    train_rna = rna.loc[sorted(fit.cell_id.unique())]
    held_rna = rna.loc[sorted(query.cell_id.unique())]
    cells, features = cell_kernels(train_rna, held_rna, edges, auxiliary, device=device)
    compounds, drugs = drug_kernels(train, targets, device=device)
    w_by_assay, y_by_assay = source_grids(train, train_rna.index, compounds, device)
    two_way = [orthogonal_parts(y, w) for y, w in zip(y_by_assay.cpu().numpy(), w_by_assay.cpu().numpy())]
    compounds_kernel = drugs["structured"]
    compounds_centered, _ = centered_kernels(compounds_kernel, compounds_kernel)
    centered_cell, crossed = centered_kernels(*cells["full"])
    cell_targets = torch.as_tensor(np.stack([x[0] for x in two_way]), dtype=w_by_assay.dtype, device=device)
    interaction_targets = torch.as_tensor(np.stack([x[2] for x in two_way]), dtype=w_by_assay.dtype, device=device)
    assay_cell_weights = w_by_assay.sum(dim=2)
    cell_weights = assay_cell_weights.sum(dim=0)
    sample_target = (assay_cell_weights * cell_targets).sum(dim=0) / cell_weights.clamp_min(1e-8)
    cell_weights = cell_weights / cell_weights[cell_weights > 0].mean()
    pair_weights = w_by_assay.sum(dim=0)
    interaction = (w_by_assay * interaction_targets).sum(dim=0) / pair_weights.clamp_min(1e-8)
    single = torch.ones((1, 1), dtype=w_by_assay.dtype, device=device)
    broad, broad_steps, broad_err = solve_residuals(centered_cell, single, cell_weights[:, None],
                                                    sample_target[:, None], PENALTY)
    specific, steps, err = solve_residuals(centered_cell, compounds_centered, pair_weights, interaction, PENALTY)
    column = compounds.get_indexer(query.compound_id)
    row = held_rna.index.get_indexer(query.cell_id)
    if (column < 0).any() or (row < 0).any():
        raise ValueError("Component forecaster requires exact fit drug and query RNA identities")
    baseline = query.compound_id.map(anchor).to_numpy()
    offset = (crossed @ broad).flatten().cpu().numpy()[row]
    selectivity = (crossed @ specific @ compounds_centered).cpu().numpy()[row, column]
    values = (baseline + offset, baseline + selectivity, baseline + offset + selectivity)
    parts = [query[KEY].assign(method=name, predicted_value=value) for name, value in zip(ARMS[:-1], values)]
    return pd.concat(parts, ignore_index=True), {
        "fit_rna_genes": train_rna.shape[1], "auxiliary_features": features,
        "marginal_iterations": broad_steps, "marginal_solver_residual": broad_err,
        "interaction_iterations": steps, "interaction_solver_residual": err,
        "posterior_interaction_scale": float(np.std(selectivity)), "posterior_marginal_scale": float(np.std(offset)),
    }


def safe_corr(a, b, weights=None):
    if len(a) < 3:
        return None
    weight = np.full(len(a), 1 / len(a)) if weights is None else weights / weights.sum()
    a, b = [(x - weight @ x) * np.sqrt(weight) for x in (a, b)]
    magnitude = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / magnitude) if magnitude > 1e-20 else None


def component_scores(frame, *, cell_axis):
    """Measure component recovery on a common support, using registered observations once."""
    keys = [cell_axis, "compound_id"]
    joined = frame.groupby([*keys, "method"], as_index=False)[["endpoint_value", "predicted_value"]].mean()
    target = joined.drop_duplicates(keys).pivot(index=cell_axis, columns="compound_id", values="endpoint_value")
    weights = target.notna().to_numpy(dtype=np.float64)
    truth = target.to_numpy()
    m, d, inter = orthogonal_parts(truth, weights)
    support = weights > 0
    real = truth[support]
    denominator = np.square(real - real.mean()).sum()
    fit_error = np.square(inter[support]).sum()
    values = {}
    for method, block in joined.groupby("method"):
        matrix = block.pivot(index=cell_axis, columns="compound_id", values="predicted_value").reindex_like(target).to_numpy()
        pc, pd_, pi = orthogonal_parts(matrix, weights)
        ir, ip = inter[support], pi[support]
        variation = np.linalg.norm(ir)
        predicted = np.linalg.norm(ip)
        amplitude = None if variation < 1e-10 else float(predicted / variation)
        interaction_corr = safe_corr(ir, ip) if amplitude is not None and amplitude > 1e-8 else None
        values[method] = {
            "sample_marginal_pearson": safe_corr(m, pc, weights.sum(axis=1)),
            "drug_marginal_pearson": safe_corr(d, pd_, weights.sum(axis=0)),
            "interaction_pearson": interaction_corr, "interaction_amplitude": amplitude,
            "raw_scale_interaction_r2": None if not fit_error else float(1 - np.square(ir - ip).sum() / fit_error),
        }
    return {"support": int(weights.sum()), "observed_interaction_fraction": None if not denominator else float(fit_error / denominator),
            "scores": values}


def interaction_intervals(frame, candidate, comparator, *, repeats=2400, seed=111071):
    use = frame[frame.method.isin((candidate, comparator))]
    values = use.groupby(["lineage_group", "compound_id", "method"], as_index=False)[
        ["endpoint_value", "predicted_value"]].mean()
    observed = values.drop_duplicates(["lineage_group", "compound_id"]).pivot(
        index="lineage_group", columns="compound_id", values="endpoint_value",
    )
    reference = [values[values.method.eq(name)].pivot(index="lineage_group", columns="compound_id", values="predicted_value")
                 .reindex_like(observed) for name in (candidate, comparator)]
    matrices = [observed.to_numpy(), *(x.to_numpy() for x in reference)]
    if not all(np.isfinite(x).all() for x in matrices):
        raise ValueError("Joint drug/patient interaction resampling requires a full evaluated panel")
    rng = np.random.default_rng(seed)
    samples, drugs = observed.shape
    draws = []
    for start in range(0, repeats, 100):
        size = min(100, repeats - start)
        row = rng.integers(samples, size=(size, samples))[:, :, None]
        column = rng.integers(drugs, size=(size, drugs))[:, None, :]
        projected = []
        for matrix in matrices:
            drawn = matrix[row, column]
            projected.append(drawn - drawn.mean(axis=1, keepdims=True) - drawn.mean(axis=2, keepdims=True)
                             + drawn.mean(axis=(1, 2), keepdims=True))
        y, a, b = projected
        yn = np.linalg.norm(y, axis=(1, 2))
        correlations = []
        for p in (a, b):
            denominator = yn * np.linalg.norm(p, axis=(1, 2))
            correlations.append(np.divide((y*p).sum(axis=(1, 2)), denominator,
                                          out=np.full(size, np.nan), where=denominator > 1e-10))
        draws.extend(correlations[0] - correlations[1])
    return np.nanquantile(draws, [.025, .975]).tolist()
