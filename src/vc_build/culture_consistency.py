"""Use independent matched 2D/3D RNA to constrain prediction differences."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .expanded_panel_cv import PENALTY, RECEIPTS, registered_inputs, same_split
from .external import KEY, _validate
from .io import freeze_table, publish_json
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference,
)
from .rna_state_projection import score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import KernelGrid, dot_kernel, drug_kernels, fit_cell_kernels, radial, training_grid
from .v0plus_cv import molecular_inputs, panel_targets, validation_roster
from .v0plus_neural import load_auxiliary
from .yang_hcc_external import require_hash

PAIRED = "GSE298810_gene_fpkm_Huh7_HLF_2D_Inject_Embed.csv.gz"
PAIRED_SHA = "9e0c127b428e4cb5fad62bd111941817b7cf6d617a5cb92fd2aa8253bffbe686"
PAIR_MODELS = {"ACH-000393", "ACH-000480"}
CELL_LINES = ("Huh7", "HLF")
FORMATS = ("2D", "Inj", "Emb")
REPLICATES = (1, 2)
PRIMARY = "paired_consistency"
ARMS = ("common_axis", "replicate_consistency", "swapped_consistency", PRIMARY)
SOURCE_CONTROLS = ARMS[:-1]
RANK_TOLERANCE = .005


def read_bridge(path):
    require_hash(path, PAIRED_SHA)
    samples = [f"{line}_{form}_{rep}" for line in CELL_LINES for form in FORMATS for rep in REPLICATES]
    data = pd.read_csv(path, usecols=["gene_name", "gene_biotype", *samples])
    data = data[data.gene_biotype.eq("protein_coding") & data.gene_name.notna()]
    unique = data.loc[~data.gene_name.duplicated(keep=False)].set_index("gene_name")
    assays = unique[samples].astype(np.float32).T
    if not np.isfinite(assays.to_numpy()).all() or (assays < 0).any().any():
        raise ValueError("Invalid paired 2D/3D expression")
    return assays


def contrast_design(index):
    # Each contrast compares means of two biological replicates, except for the replicate controls.
    def paired_rows(swapped):
        for i, left in enumerate(CELL_LINES):
            right = CELL_LINES[1 - i] if swapped else left
            for form in FORMATS[1:]:
                yield {f"{left}_2D_1": -.5, f"{left}_2D_2": -.5,
                       f"{right}_{form}_1": .5, f"{right}_{form}_2": .5}

    noise = [{f"{line}_{form}_1": -1, f"{line}_{form}_2": 1}
             for line in CELL_LINES for form in FORMATS[:2]]
    pairs = {"replicate_consistency": noise, "swapped_consistency": list(paired_rows(True)),
             PRIMARY: list(paired_rows(False))}
    return {name: pd.DataFrame(rows).reindex(columns=index).fillna(0).to_numpy(np.float32)
            for name, rows in pairs.items()}


def kernel_between(basis, left, right):
    sequence, pathway, auxiliary = left
    other, other_pathway, other_auxiliary = right
    scale = basis.reference_scale
    dot = dot_kernel(sequence, other) / scale
    left_energy = sequence.square().mean(dim=1) / scale
    right_energy = other.square().mean(dim=1) / scale
    rna = .5 * (dot + radial(dot, left_energy, right_energy, 2.))
    return .7 * rna + .2 * dot_kernel(pathway, other_pathway) + .1 * dot_kernel(auxiliary, other_auxiliary)


def consistency_grams(train, held, fit_bridge, query_bridge, bridge, design):
    """Condition a PSD kernel on unit-variance bridge contrasts with unit noise."""
    contrast = torch.as_tensor(design, device=train.device, dtype=train.dtype)
    prior = contrast @ bridge @ contrast.T
    variance = prior.diagonal()
    if not torch.isfinite(variance).all() or (variance <= 1e-8).any():
        raise ValueError("A registered culture or repeat contrast is degenerate")
    contrast = contrast / variance.sqrt()[:, None]
    normalized = contrast @ bridge @ contrast.T
    system = (normalized + normalized.T) / 2 + torch.eye(len(contrast), device=train.device)
    root = torch.linalg.cholesky(system)
    fit_delta = fit_bridge @ contrast.T
    query_delta = query_bridge @ contrast.T
    corrected = torch.cholesky_solve(fit_delta.T.contiguous(), root)
    updated = train - fit_delta @ corrected
    audit = {"mean_unscaled_contrast_variance": variance.mean().item(),
             "fraction_mean_train_kernel_trace_removed": (1 - updated.trace() / train.trace()).item()}
    return (updated, held - query_delta @ corrected), audit


def candidate_forecasts(fit, query, rna, edges, targets, auxiliary, bridge):
    _validate(fit, query)
    device = device_cuda()
    genes = rna.columns.intersection(bridge.columns)
    if len(genes) < 10_000:
        raise ValueError("Too few shared protein-coding RNA genes")
    source = rna.loc[sorted(fit.cell_id.unique()), genes]
    held = rna.loc[sorted(query.cell_id.unique()), genes]
    bridge = bridge[genes]
    basis, first = fit_cell_kernels(source, edges, auxiliary, device=device)
    fit_gram = first["full"]
    held_gram = basis.cross(held)["full"]
    vectors = basis.vectors(bridge)
    bridge_gram = kernel_between(basis, vectors, vectors)
    source_bridge = basis.cross(bridge)["full"].T
    held_bridge = kernel_between(basis, basis.vectors(held), vectors)
    terms = {"common_axis": (fit_gram, held_gram)}
    diagnostics = {}
    for name, design in contrast_design(bridge.index).items():
        terms[name], diagnostics[name] = consistency_grams(
            fit_gram, held_gram, source_bridge, held_bridge, bridge_gram, design,
        )

    _, anchor, aligned, _ = panel_targets(fit)
    drugs, kernels = drug_kernels(aligned, targets, device=device)
    weights, response = training_grid(aligned, source.index, drugs, device=device)
    base = query.compound_id.map(anchor).to_numpy()
    outputs = [query[KEY].assign(method="drug_mean", predicted_value=base)]
    for name, grams in terms.items():
        grid = KernelGrid(held.index, drugs, {"full": grams}, kernels, weights, response, basis.reference[-1].shape[1])
        estimates, audit = grid.forecast(query, "structured", PENALTY)
        outputs.append(query[KEY].assign(method=name, predicted_value=base + estimates))
        diagnostics[name] = diagnostics.get(name, {}) | audit
    return pd.concat(outputs, ignore_index=True), {"genes": len(genes), "arms": diagnostics}


def drop_bridge_lines(frame):
    lineages = set(frame.loc[frame.cell_id.isin(PAIR_MODELS), "lineage_group"])
    return frame[~frame.lineage_group.isin(lineages)]


def predict(args):
    bridge = read_bridge(args.bridge)
    edges, targets, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    if args.mode in EXTERNAL_TRACKS:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
        candidate, audit = candidate_forecasts(fit, query, rna, edges, targets, auxiliary, bridge)
        joint = pd.concat([previous, candidate[candidate.method.ne("drug_mean")].assign(track=args.mode, fold=0)],
                          ignore_index=True)
        shared_support(joint, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        fit, check = drop_bridge_lines(fit), drop_bridge_lines(check)
        query = validation_roster(check)
        candidate, audit = candidate_forecasts(fit, query, rna, edges, targets, auxiliary, bridge)
        shared_support(candidate, query, ("drug_mean", *ARMS))
        joint = with_assay(candidate, check, args.fold)
        registered, reference = RECEIPTS["core"], None
    freeze_table(joint[FORECAST_SCHEMA], args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "priors": priors, "auxiliary": aux,
        "bridge_sha256": PAIRED_SHA, "diagnostics": audit,
    })


def assessor(frame, *, external):
    from .structured_cv import paired_intervals
    from .v0plus_score import axis_metrics, crossed_patient_drug_ci

    controls = (*SOURCE_CONTROLS, "drug_mean")
    if external:
        controls = (*controls, "structured", "drug_ridge", *PUBLIC)
    interval = crossed_patient_drug_ci if external else paired_intervals
    overall, _, _ = axis_metrics(frame, [(PRIMARY, name) for name in controls])
    across = {name: interval(frame, PRIMARY, name) for name in controls}
    within = {(row["baseline"], row["metric"]): row["ci95"] for row in overall["paired_contrasts"]}
    gates = {name: all(across[name][metric][0] > 0 for metric in ("spearman", "pearson"))
             and all(within[name, metric][0] > -RANK_TOLERANCE
                     for metric in ("spearman", "ndcg_at_5", "top5_overlap")) for name in controls}
    return {"metrics": overall, "across_cell_ci95": across, "gates": gates,
            "groups": frame.lineage_group.nunique(), "observations": frame.row_id.nunique()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fields = {
        "predict": ("licob", "archive", "priors", "hallmark", "targets", "auxiliary", "bridge"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, names in fields.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", *names, "out"):
            stage.add_argument("--" + name, required=True, type=Path)
        if action == "predict":
            stage.add_argument("--mode", required=True, choices=(*ASSAYS, *EXTERNAL_TRACKS))
            stage.add_argument("--fold", type=int, default=0, choices=range(3))
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=assessor)
    else:
        score_external(args, assessor=assessor, arms=ARMS)


if __name__ == "__main__":
    main()
