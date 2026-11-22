"""Keep the cis target RNA and learn a drug-specific post-transcriptional prior."""

import numpy as np
import pandas as pd
import torch

from .external import KEY, _validate
from .pdo_transfer_cv import device_cuda
from .protein_proxy import donor_calibration, main, normalized_ranks
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import assess
from .structured_kernel import (
    drug_kernels, fit_cell_kernels, project_drug_features, solve_residuals, training_grid,
)
from .v0plus_cv import panel_targets
from .v0plus_score import axis_metrics, crossed_patient_drug_ci

RECEIPT = "7b8c6fd0ea7e3cee1604d8299f374f0b04deda5ea9f0063fae34a3dc23e52d06"
CIS = "target_cis"
ADDITIVE = "target_protein_additive"
PRIMARY = "target_protein_interaction"
SHAM = "target_protein_shuffled"
OBSERVED = "target_measured_cohort"
CALIBRATED = "target_residual_cohort"
ARMS = (CIS, ADDITIVE, PRIMARY, SHAM)
TECHNICAL = (CALIBRATED, OBSERVED)
CONTROLS = ("structured", CIS, SHAM)


def standardize(reference, query):
    center = reference.mean(dim=0)
    sigma = reference.std(dim=0, correction=0).clamp_min(.05)
    return tuple(((value - center) / sigma).clamp(-8, 8) for value in (reference, query))


def nuisance(cis):
    return torch.column_stack((torch.ones_like(cis), cis))


def partial_context(cis, x, y):
    """Estimate the ridge component after removing the unpenalized cis predictor."""
    control = nuisance(cis)
    design = torch.linalg.lstsq(control, x).solution
    cis_fit = torch.linalg.lstsq(control, y).solution
    residual = x - control @ design
    rhs = residual.T @ (y - control @ cis_fit)
    beta = torch.linalg.solve(residual.T @ residual + 1000 * torch.eye(x.shape[1], device=x.device), rhs)
    return design @ beta, beta


def conditional_profiles(basis, held, fit, learned, genes, *, replay):
    rna, pathways, _ = basis.reference
    query = basis.vectors(held)
    indices = basis.genes.get_indexer(genes)
    if not (indices >= 0).all():
        raise ValueError("Target list contains a missing RNA gene")
    candidates = rna.var(dim=0)
    candidates[indices] = -torch.inf
    chosen = candidates.argsort(descending=True, stable=True)[:256]
    x = torch.cat([rna[:, chosen], pathways], dim=1)
    q = torch.cat([query[0][:, chosen], query[1]], dim=1)
    cis, held_cis = rna[:, indices], query[0][:, indices]
    # Use both omics from the same physical fit model before combining independent lineages.
    fit_index = pd.Index(sorted(fit.cell_id.unique()))
    observed = learned.reindex(fit_index).dropna(how="all").reindex(columns=genes)
    lineage = fit.drop_duplicates("cell_id").set_index("cell_id").lineage_group
    protein = observed.groupby(lineage.reindex(observed.index)).mean()
    present = protein.notna().sum()
    eligible = present.ge(max(100, .60 * len(protein))) & protein.std().gt(1e-7)
    positions = fit_index.get_indexer(observed.index)
    device = x.device
    lineage_mean = (pd.DataFrame(torch.cat((x[positions], cis[positions]), dim=1).cpu().numpy(),
                                 index=observed.index)
                    .groupby(lineage.reindex(observed.index)).mean().loc[protein.index].to_numpy())
    merged = x.new_tensor(lineage_mean)
    z, z_cis = merged[:, :x.shape[1]], merged[:, x.shape[1]:]
    fit_innov, held_innov = torch.zeros_like(cis), torch.zeros_like(held_cis)
    support, coefs = {}, {}
    for j, name in enumerate(genes):
        if not eligible[name]:
            continue
        ranks = normalized_ranks(protein[name].to_numpy())
        valid = torch.as_tensor(np.isfinite(ranks), device=device)
        local_cis = z_cis[valid, j]
        if local_cis.std(correction=0) < .05:
            continue
        control, weight = partial_context(local_cis, z[valid], x.new_tensor(ranks[np.isfinite(ranks)]))
        train = x @ weight - nuisance(cis[:, j]) @ control
        held_values = q @ weight - nuisance(held_cis[:, j]) @ control
        a, b = standardize(train, held_values)
        fit_innov[:, j], held_innov[:, j] = a, b
        support[name] = int(present[name])
        coefs[j] = (control, weight, train.mean(), train.std(correction=0).clamp_min(.05))
    audit = {}
    if replay:
        vectors = basis.vectors(held.iloc[:1])
        first = torch.cat((vectors[0][:, chosen], vectors[1]), dim=1)
        target = vectors[0][:, indices]
        errors = []
        for j, (control, weight, center, scale) in coefs.items():
            value = first @ weight - nuisance(target[:, j]) @ control
            errors.append(float((((value-center)/scale).clamp(-8,8) - held_innov[0,j]).abs().max()))
        audit["single_query_max_error"] = max(errors, default=0.)
        if audit["single_query_max_error"] > 2e-5:
            raise ValueError("Single-RNA residual differs from batch inference")
    audit["fit_protein_lineages"] = len(protein)
    audit["observed_protein_target_lineages"] = support
    return (cis, held_cis), (fit_innov, held_innov), support, audit


def drug_maps(drugs, targets, genes, support):
    direct, correct, wrong = [], [], []
    index = dict(zip(genes, range(len(genes))))
    for drug in drugs:
        own = set(targets.get(drug, ()))
        direct.append(sorted(index[name] for name in own & index.keys()))
        eligible = sorted(own & support.keys())
        available = set(support) - own
        if not eligible or len(available) < len(eligible):
            correct.append([])
            wrong.append([])
            continue
        replacement = []
        for name in eligible:
            alternate = min(available, key=lambda gene: (abs(support[gene] - support[name]), gene))
            available.remove(alternate)
            replacement.append(index[alternate])
        correct.append([index[name] for name in eligible])
        wrong.append(replacement)
    return direct, correct, wrong


def block(fit, held, links, weight):
    if not any(links):
        return tuple(value.new_zeros((len(value), len(links), 1)) for value in (fit, held))
    a, b = standardize(fit, held)
    out = tuple(value.new_zeros((len(value), len(links), max(map(len, links)))) for value in (a, b))
    for j, columns in enumerate(links):
        if columns:
            scale = (weight / len(columns))**.5
            for tensor, value in zip(out, (a, b)):
                tensor[:, j, :len(columns)] = scale * value[:, columns]
    return out


def linked_blocks(cis_pair, innovation_pair, direct, actual, sham, technical):
    if not any(actual):
        raise ValueError("No target with measured-protein and sham support")
    main = block(*cis_pair, direct, .10)
    only_h = block(*innovation_pair, actual, .10)
    product_link = []
    start = 0
    for columns in actual:
        product_link.append(list(range(start, start + len(columns))))
        start += len(columns)

    def tails(link, query_innovation=None):
        product, residual = [], []
        for axis, (cis, one) in enumerate(zip(cis_pair, innovation_pair)):
            one = query_innovation if axis == 1 and query_innovation is not None else one
            joint = [
                cis[:, matching] * one[:, values] for matching, values in zip(actual, link) if matching
            ]
            product.append(torch.cat(joint, dim=1))
            residual.append(one)
        values = block(*residual, link, .05)
        combination = block(*product, product_link, .05)
        return tuple(torch.cat(parts, dim=-1) for parts in zip(values, combination))

    arms = {CIS: main, ADDITIVE: tuple(torch.cat(parts, dim=-1) for parts in zip(main, only_h))}
    variants = {PRIMARY: tails(actual), SHAM: tails(sham)}
    variants.update({name: tails(actual, held) for name, held in technical.items()})
    arms.update({name: tuple(torch.cat(parts, dim=-1) for parts in zip(main, value))
                 for name, value in variants.items()})
    return arms


def observed_residual(measured, query, held, genes, cis, proxy, supported):
    donor = query.drop_duplicates("cell_id").set_index("cell_id").donor_id.reindex(held.index)
    data = measured.reindex(held.index).reindex(columns=genes)
    patient = data.groupby(donor).mean()
    cols = patient.columns[patient.notna().mean().ge(.75) & patient.std().gt(1e-7)]
    cols = cols.intersection(supported)
    if not len(cols):
        raise ValueError("LICOB has no matched drug target proteome")
    data = data[cols].astype("float64")
    measured_ranks = donor_calibration(data, donor)
    rank_donors = measured_ranks.groupby(donor).mean()
    idx = genes.get_indexer(cols)
    own_rna = pd.DataFrame(cis[:, idx].cpu().numpy(), index=held.index, columns=cols).where(data.notna())
    rna_donors = own_rna.groupby(donor).mean()
    center_x = rna_donors.mean()
    center_y = rank_donors.mean()
    covariance = ((rank_donors - center_y) * (rna_donors - center_x)).mean()
    slope = covariance / ((rna_donors - center_x).pow(2).mean().clip(lower=1e-6))
    inferred = pd.DataFrame(proxy[:, idx].cpu().numpy(), index=held.index, columns=cols).where(data.notna())
    residual = measured_ranks - center_y - (own_rna - center_x) * slope
    correlation = residual.groupby(donor).mean().corrwith(inferred.groupby(donor).mean(), method="spearman").dropna()
    technical = {}
    for name, table in ((OBSERVED, residual), (CALIBRATED, inferred)):
        alter = proxy.clone()
        value = proxy.new_tensor(donor_calibration(table.astype("float64"), donor).to_numpy())
        alter[:, idx] = torch.where(value.isfinite(), value, proxy[:, idx])
        technical[name] = alter
    return technical, {"measured_target_count": len(cols), "measured_target_genes": list(cols),
                       "target_residual_correlations": correlation.to_dict()}


def forecast(fit, query, rna, edges, targets, auxiliary, learned, measured, *, replay):
    _validate(fit, query)
    device = device_cuda()
    x = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    _, anchor, rows, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(rows, targets, device=device)
    drug = kernels["structured"]
    basis, train = fit_cell_kernels(x, edges, auxiliary, device=device)
    genes = pd.Index(sorted(set(basis.genes) & set().union(*(targets.get(key, ()) for key in drugs))))
    cis, innovation, support, audit = conditional_profiles(basis, held, fit, learned, genes, replay=replay)
    direct, actual, wrong = drug_maps(drugs, targets, genes, support)
    technical = {}
    if measured is not None:
        used = {genes[index] for links in actual for index in links}
        technical, extra = observed_residual(measured, query, held, genes, cis[1], innovation[1], used)
        audit.update(extra)
    priors = linked_blocks(cis, innovation, direct, actual, wrong, technical)
    assay, y = training_grid(rows, x.index, drugs, device=device)
    cell, cross = train["full"], basis.cross(held)["full"]
    index = held.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((axis < 0).any() for axis in index):
        raise ValueError("Registered compound or cell is missing")
    output, shared = [], None
    for name, (a, b) in priors.items():
        if name in TECHNICAL:
            coefficients = shared
        else:
            coefficients, steps, error = solve_residuals(cell, drug, assay, y, 4., per_drug_prior=a)
            audit[name] = {"solver_iterations": steps, "solver_residual": error}
        if name == PRIMARY:
            shared = coefficients
        residual = cross @ coefficients @ drug + project_drug_features(a, b, coefficients)
        output.append(query[KEY].assign(method=name, predicted_value=query.compound_id.map(anchor) +
                                        residual.cpu().numpy()[index]))
    requested = set(query.compound_id)
    audit["direct_rna_supported_drugs"] = [name for name, links in zip(drugs, direct) if links and name in requested]
    audit["protein_supported_drugs"] = {key: [genes[j] for j in links] for key, links in zip(drugs, actual)
                                        if links and key in requested}
    audit["sham_genes_by_drug"] = {key: [genes[j] for j in links] for key, links in zip(drugs, wrong)
                                  if links and key in requested}
    return pd.concat(output, ignore_index=True), audit


def assess_target(frame, *, external):
    result = assess(frame, external=external, primary=PRIMARY, controls=CONTROLS, require_precision=True)
    if external:
        result["interaction_ci95"] = interaction_intervals(frame, PRIMARY, "structured")
        result["gates"]["structured"] &= result["interaction_ci95"][0] > 0
        result["components"] = component_scores(frame, cell_axis="lineage_group")
    if OBSERVED in frame.method.unique():
        result["measured_technical"] = {
            "across_cell_ci95": crossed_patient_drug_ci(frame, OBSERVED, CALIBRATED),
            "within_cell": axis_metrics(frame, [(OBSERVED, CALIBRATED)])[0]["paired_contrasts"],
        }
    return result


if __name__ == "__main__":
    main(forecast=forecast, arms=ARMS, technical=TECHNICAL, assessor=assess_target, protein_receipt=RECEIPT)
