"""Use compound-specific, lineage-disjoint RNA neighborhoods for local response estimates."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import fit_cell_kernels
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary

PRIMARY = "rna_support_blend"
FIXED = "rna_fixed_blend"
LOCAL = "rna_neighbor"
ARMS = (LOCAL, FIXED, PRIMARY)
BLOCK_WEIGHTS = (.7, .2, .1)
NEIGHBORS = 20
TEMPERATURE = .05


def lineage_vectors(fit, identities, groups):
    ix = torch.as_tensor(groups.get_indexer(identities), device=fit[0].device)
    result = []
    for view in fit:
        accumulated = view.new_zeros((len(groups), view.shape[1]))
        accumulated.index_add_(0, ix, normalize(view))
        result.append(normalize(accumulated))
    return result


def similarity(normalized, reference):
    return sum(w * (x @ y.T) for w, x, y in zip(BLOCK_WEIGHTS, normalized, reference))


def highest(matrix):
    indices = torch.argsort(matrix, dim=1, descending=True, stable=True)[:, :NEIGHBORS]
    return matrix.gather(1, indices), indices


def local_values(observed, query_similarity, values, mirror=None):
    """Fit an exclusion-aware support scale from the observed lineages of this drug."""
    own = observed.clone()
    own.fill_diagonal_(-torch.inf)
    close, _ = highest(own)
    low, typical = torch.quantile(close.mean(dim=1), own.new_tensor([.10, .50]))
    if typical - low <= 1e-6:
        return None

    def estimate(query):
        close, top = highest(query)
        weight = torch.softmax((close - close[:, :1]) / TEMPERATURE, dim=1)
        raw = (values[top] * weight).sum(dim=1)
        near = close.mean(dim=1)
        return raw, .5 * ((near - low) / (typical-low)).clamp(0, 1), near

    main = estimate(query_similarity)
    proof = None if mirror is None else estimate(mirror)
    return main, proof, (float(low), float(typical))


def held_forecasts(fit, query, rna, edges, targets, auxiliary, *, replay):
    _validate(fit, query)
    device = device_cuda()
    source = rna.loc[sorted(fit.cell_id.unique())]
    held = rna.loc[sorted(query.cell_id.unique())]
    fitted, _ = fit_cell_kernels(source, edges, auxiliary, device=device)
    reference_group = fit[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id")
    identities = reference_group.loc[source.index, "lineage_group"]
    groups = pd.Index(sorted(identities.unique()))
    ref = lineage_vectors(fitted.reference, identities, groups)
    train_similarity = similarity(ref, ref)
    query_similarity = similarity([normalize(view) for view in fitted.vectors(held)], ref)
    single = similarity([normalize(view) for view in fitted.vectors(held.iloc[:1])], ref) if replay else None
    cell_values, anchor, _, _ = panel_targets(fit)
    panel = cell_values.groupby(["lineage_group", "compound_id"]).residual.mean().unstack().reindex(groups)
    estimate = np.zeros((len(held), len(panel.columns)))
    alpha = np.zeros_like(estimate)
    close = np.zeros_like(estimate)
    below = np.zeros_like(estimate, dtype=bool)
    support = np.zeros(len(panel.columns), dtype=bool)
    error = np.zeros(3)
    trace = {}
    for j, (drug, outcomes) in enumerate(panel.items()):
        ids = np.flatnonzero(outcomes.notna().to_numpy())
        if len(ids) <= NEIGHBORS:
            continue
        indices = torch.as_tensor(ids, device=device)
        observed = train_similarity[indices[:, None], indices]
        targets_at = torch.as_tensor(outcomes.iloc[ids].to_numpy(), device=device, dtype=observed.dtype)
        local = local_values(observed, query_similarity[:, indices], targets_at,
                             mirror=None if single is None else single[:, indices])
        if local is None:
            continue
        (residual, share, distance), proof, cuts = local
        if proof is not None:
            error = np.maximum(error, [float((a[:1]-b).abs().max()) for a,b in zip((residual, share, distance), proof)])
        estimate[:, j], alpha[:, j], close[:, j] = [x.cpu().numpy() for x in (residual, share, distance)]
        below[:, j] = close[:, j] < cuts[0]
        support[j] = True
        trace[drug] = {"fit_groups": len(ids), "q10": cuts[0], "median": cuts[1]}
    if replay and error.max() > 2e-5:
        raise ValueError(f"Single RNA forecast changed with batch size: {error}")
    cols, rows = panel.columns.get_indexer(query.compound_id), held.index.get_indexer(query.cell_id)
    if (cols < 0).any() or (rows < 0).any():
        raise ValueError("Missing paired query compound or expression")
    weight, supported = alpha[rows, cols], support[cols]
    local_value = query.compound_id.map(anchor).to_numpy() + estimate[rows, cols]
    observations = query[KEY].assign(local=local_value, weight=weight, supported=supported)
    diagnostics = {
        "support_by_drug": trace, "unsupported_drugs": int((~support).sum()), "query_rows": len(query),
        "query_below_source_q10_count": int(below[rows, cols].sum()),
        "query_zero_weight_count": int((weight == 0).sum()), "query_full_weight_count": int((weight == .5).sum()),
        "average_neighbor_weight": float(weight.mean()), "single_profile_max_error": error.tolist() if replay else None,
        "query_supported_mean_similarity": float(close[rows, cols][supported].mean()) if supported.any() else None,
    }
    return observations, diagnostics


def predict(args):
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        prior, reference = external_reference(args, args.mode, query)
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        prior, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    edges, targets, source_priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, auxiliary_receipt = load_auxiliary(args.auxiliary)
    fresh, diagnostic = held_forecasts(fit, query, rna, edges, targets, auxiliary, replay=external)
    fresh = fresh.assign(track=args.mode, fold=0) if external else with_assay(fresh, check, args.fold)
    keys = [*KEY, "track", "fold"]
    base = prior[prior.method.eq("structured")][[*keys, "predicted_value"]]
    matched = base.merge(fresh, how="left", on=keys, validate="one_to_one")
    if not np.isfinite(matched[["local", "weight"]]).all(axis=None):
        raise ValueError("The local model differs from the registered row support")
    difference = (matched.local-matched.predicted_value) * matched.supported
    diagnostic["local_vs_structured_difference_rms"] = float(np.sqrt(np.square(difference).mean()))
    new = [matched[keys].assign(method=name, predicted_value=matched.predicted_value + scale * difference)
           for name, scale in ((LOCAL, 1.), (FIXED, .5), (PRIMARY, matched.weight))]
    forecasts = pd.concat([prior,*new], ignore_index=True)[FORECAST_SCHEMA]
    if external:
        shared_support(forecasts, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        panel = forecasts.pivot(index=keys, columns="method", values="predicted_value")
        if set(panel) != {*BASELINES,*ARMS} or not np.isfinite(panel).all(axis=None):
            raise ValueError("Source neighbor prediction support changed")
    freeze_table(forecasts, args.out, provenance={
        "mode": args.mode, "fold": args.fold, "core_sha256": RECEIPTS["core"], "inputs_sha256": registered,
        "reference_sha256": reference, "priors": source_priors, "auxiliary": auxiliary_receipt, "diagnostics": diagnostic,
    })


def assess_neighbors(frame, *, external):
    result = assess(frame, external=external, primary=PRIMARY, controls=("structured", FIXED), require_precision=True)
    del result["gates"][FIXED]
    if external:
        result["interaction_ci95"] = interaction_intervals(frame, PRIMARY, "structured")
        result["gates"]["structured"] &= result["interaction_ci95"][0] > 0
        result["components"] = component_scores(frame, cell_axis="lineage_group")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
              "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels")}
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *names, "out"):
            stage.add_argument(f"--{name}", type=Path, required=True)
        if action == "predict":
            stage.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            stage.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assess_neighbors)
    else:
        score_external(args, assessor=assess_neighbors, arms=ARMS)


if __name__ == "__main__":
    main()
