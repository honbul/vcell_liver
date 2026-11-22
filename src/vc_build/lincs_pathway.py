"""Use non-liver drug signatures and signed pathway context in a matched HCC benchmark."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table, sha256
from .limore_corrected import shared_support
from .lincs_signature_data import CELLS
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import (
    drug_kernels, fit_cell_kernels, project_shared_features, ranked_expression, solve_residuals, training_grid,
)
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .v0plus_score import axis_metrics
from .yang_hcc_external import require_hash

PRESENCE = "lincs_presence"
GEOMETRY = "lincs_geometry"
PRIMARY = "lincs_pathway_context"
SHUFFLED = "lincs_shuffled_context"
ARMS = (PRESENCE, GEOMETRY, PRIMARY, SHUFFLED)
CONTROLS = ("structured", PRESENCE, GEOMETRY, SHUFFLED)


def checked_receipt(path):
    code = json.loads((path/"receipt.json").read_text())
    if (code["source"] != "GSE92742" or code["core_sha256"] != RECEIPTS["core"]
            or not set(code["profile_cells"]) <= set(CELLS) or code["landmark_genes"] != 978):
        raise ValueError("Unexpected registered GSE92742 source")
    require_hash(path/"signatures.parquet", code["consensus_sha256"])
    return code


def standardize(fit, query):
    mean = fit.mean(dim=0)
    scale = fit.std(dim=0, correction=0).clamp_min(.05)
    return tuple(((value - mean) / scale).clamp(-4, 4) for value in (fit, query))


def same_landmarks(fit, held, signatures, edges, support, device):
    common = fit.columns.intersection(signatures.columns)
    block = edges[edges.gene.isin(common) & edges.signature.str.startswith(("PROGENY_", "HALLMARK_"))]
    counts = block.groupby("signature").gene.nunique()
    block = block[block.signature.isin(counts[counts.ge(10)].index)]
    weights = block.pivot_table(index="gene", columns="signature", values="weight", aggfunc="sum").fillna(0.)
    genes = weights.index
    if len(weights.columns) < 10 or len(genes) < 300:
        raise ValueError("Insufficient measured landmark pathway coverage")
    pathways = normalize(torch.as_tensor(weights.to_numpy(dtype=np.float32), device=device), dim=0)
    projections = [
        torch.as_tensor(ranked_expression(rna)[:, fit.columns.get_indexer(genes)], device=device) @ pathways
        for rna in (fit, held)
    ]
    x, q = standardize(*projections)
    y = torch.as_tensor(signatures[genes].fillna(0.).to_numpy(dtype=np.float32), device=device) @ pathways
    _, y = standardize(y[support], y)
    y = y * support[:, None]
    # The product's average norm is computed using only fit RNA and supported fit compounds.
    energy = (x.square().mean(dim=0) * y[support].square().mean(dim=0)).sum()
    if energy <= 1e-8:
        raise ValueError("Degenerate shared CTS × expression pathway features")
    y = y * (.10 / energy).sqrt()
    return x, q, y, {"shared_markers": len(genes), "pathways": list(weights.columns)}


def permutation(support):
    result = np.arange(len(support))
    eligible = np.flatnonzero(support)
    result[eligible] = np.random.default_rng(15149).permutation(eligible)
    return result


def forecast(fit, query, rna, edges, targets, auxiliary, signatures):
    _validate(fit, query)
    device = device_cuda()
    x = rna.loc[sorted(fit.cell_id.unique())]
    q = rna.loc[sorted(query.cell_id.unique())]
    _, anchor, rows, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(rows, targets, device=device)
    base_drug = kernels["structured"]
    basis, kernel = fit_cell_kernels(x, edges, auxiliary, device=device)
    cell, cross = kernel["full"], basis.cross(q)["full"]
    aligned = signatures.reindex(drugs)
    available = aligned.notna().all(axis=1).to_numpy()
    if not available.any() or int(available.sum()) < 8:
        raise ValueError("Insufficient registered Level5 compound support")
    support = torch.as_tensor(available, device=device)
    values = torch.as_tensor(aligned.fillna(0.).to_numpy(dtype=np.float32), device=device)
    unit = normalize(values - values.mean(dim=1, keepdim=True))
    molecular = unit @ unit.T
    cells_fit, cells_query, led, metadata = same_landmarks(x, q, aligned, edges, support, device)
    shuffled = permutation(available)
    index = torch.as_tensor(shuffled, device=device)
    mask = support.float()
    controls = {
        PRESENCE: (mask[:, None] * mask[None, :], None),
        GEOMETRY: (molecular, None),
        PRIMARY: (molecular, led),
        SHUFFLED: (molecular[index[:, None], index], led[index]),
    }
    w, response = training_grid(rows, x.index, drugs, device=device)
    selector = q.index.get_indexer(query.cell_id), drugs.get_indexer(query.compound_id)
    if any((part < 0).any() for part in selector):
        raise ValueError("Missing registered RNA or compound")
    result, diagnostics = [], {}
    for name, (extra, state) in controls.items():
        drug = base_drug + .20 * extra
        shared = None if state is None else (cells_fit, state)
        coefficients, steps, error = solve_residuals(cell, drug, w, response, 4., shared_prior=shared)
        prediction = cross @ coefficients @ drug
        if state is not None:
            prediction += project_shared_features(cells_fit, cells_query, state, coefficients)
        prediction = query.compound_id.map(anchor) + prediction.cpu().numpy()[selector]
        result.append(query[KEY].assign(method=name, predicted_value=prediction))
        diagnostics[name] = {"iterations": steps, "solver_residual": error}
    detail = {
        "fit_supported": int(available.sum()), "fit_parents": len(drugs), "permutation_changed": int(
            np.count_nonzero(shuffled != np.arange(len(shuffled)))),
        "query_supported": query[query.compound_id.isin(signatures.index)].compound_id.nunique(),
        "query_parents": query.compound_id.nunique(),
    }
    return pd.concat(result, ignore_index=True), diagnostics | metadata | detail


def predict(args):
    receipt = checked_receipt(args.signatures)
    signatures = pd.read_parquet(args.signatures / "signatures.parquet")
    if (len(signatures.columns) != 978 or not signatures.index.is_unique
            or set(signatures.index) != set(receipt["supported"]) or not np.isfinite(signatures.to_numpy()).all()):
        raise ValueError("Invalid registered consensus data")
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
    else:
        original, corrected, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, corrected, args.fold, args.mode)
        query = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    new, diagnostics = forecast(fit, query, rna, edges, targets, auxiliary, signatures)
    new = new.assign(track=args.mode, fold=0) if external else with_assay(new, check, args.fold)
    joint = pd.concat([previous, new], ignore_index=True)[FORECAST_SCHEMA]
    if external:
        shared_support(joint, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        panel = joint.pivot(index=[*KEY, "track", "fold"], columns="method", values="predicted_value")
        if set(panel) != {*BASELINES, *ARMS} or not np.isfinite(panel).all(axis=None):
            raise ValueError("Source forecasts have inconsistent reference support")
    freeze_table(joint, args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": aux_receipt,
        "lincs_sha256": sha256(args.signatures / "receipt.json"), "diagnostics": diagnostics,
    })


def assess_lincs(frame, *, external, supported):
    result = assess(frame, external=external, primary=PRIMARY, controls=CONTROLS, require_precision=True)
    if external:
        result["interaction_ci95"] = interaction_intervals(frame, PRIMARY, "structured")
        result["gates"]["structured"] &= result["interaction_ci95"][0] > 0
        result["components"] = component_scores(frame, cell_axis="lineage_group")
    part = frame[frame.compound_id.isin(supported)]
    if part.compound_id.nunique() >= 5:
        result["supported"] = {"drugs": part.compound_id.nunique(), "metrics": axis_metrics(part, ())[0]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {"predict": ("licob", "archive", "cv-archive", "priors", "hallmark", "targets", "auxiliary"),
              "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels")}
    for name, options in fields.items():
        sub = stages.add_parser(name)
        for flag in ("original", "core", "out", "signatures", *options):
            sub.add_argument("--" + flag, type=Path, required=True)
        if name == "predict":
            sub.add_argument("--mode", choices=(*ASSAYS, *EXTERNAL_TRACKS), required=True)
            sub.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
        return
    receipt = checked_receipt(args.signatures)
    callback = partial(assess_lincs, supported=set(receipt["supported"]))
    if args.stage == "source":
        score_source(args, assessor=callback)
    else:
        score_external(args, assessor=callback, arms=ARMS)


if __name__ == "__main__":
    main()
