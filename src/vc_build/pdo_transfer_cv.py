"""Evaluate adaptation from cell lines to donor-disjoint non-HCC organoid studies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import score, score_across_cells
from .expanded_panel_cv import PENALTY, RIDGE, expand_aligned, expanded_targets, registered_inputs
from .external import KEY, SOURCE
from .io import freeze_table, publish_json, sha256, verify_forecast
from .v0plus_cv import fit_baseline, molecular_inputs, panel_targets, validation_roster
from .v0plus_features import ExpressionFeatures
from .v0plus_score import crossed_patient_drug_ci

PDO_HASH = "2db35388de8e6a7688c6d1d10842c0ef22be7b644ef65bacd1595048cf764f90"
STUDIES = ("IMPROVE dataset 1", "IMPROVE dataset 2", "IMPROVE dataset 3", "Pilot 1")
META = [*KEY, "source", "isomeric_smiles"]
TEACHER = ("2d_mean", "2d_ridge", "2d_structured")
PRIMARY = "pdo_residual_transfer"
ADAPTED = ("pdo_mean", "pdo_anchor_transfer", "pdo_only_residual", PRIMARY)
PDO_PENALTY = 2.
ANCHOR_PRIOR = 6.


def inputs(path, library=None):
    receipt_file = path / "receipt.json"
    if sha256(receipt_file) != PDO_HASH:
        raise ValueError("Unexpected non-HCC organoid receipt")
    receipt = json.loads(receipt_file.read_text())
    for filename, checksum in receipt["artifacts"].items():
        if sha256(path / filename) != checksum:
            raise ValueError(f"Organoid input failed verification: {filename}")
    metadata = pd.read_parquet(path / "observations.parquet", columns=META)
    rna = pd.read_parquet(path / "rna.parquet")
    if library is not None:
        metadata = metadata[metadata.compound_id.isin(library)]
    return metadata, rna


def cells_for_study(metadata, study):
    held = metadata[metadata.source.eq(study)]
    fit = metadata[~metadata.source.eq(study) & ~metadata.lineage_group.isin(held.lineage_group)]
    if fit.empty or held.empty or set(fit.cell_id) & set(held.cell_id):
        raise ValueError("Invalid donor or study split")
    return fit, held


def device_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The registered PDO transfer experiment requires G1 GPU")
    return torch.device("cuda")


def parameter_inputs(args):
    edges, target, features = molecular_inputs(args.priors, args.hallmark, args.targets)
    target = {key: list(value) for key, value in target.items()}
    return edges, target, features


def teacher_predict(args):
    from .structured_kernel import cell_kernels, drug_kernels, solve_residuals, training_grid
    from .v0plus_neural import load_auxiliary

    device = device_cuda()
    _, core, cells, extra = registered_inputs(args, include_extension=True)
    library = set(core.compound_id) | set(extra.compound_id)
    roster, rna = inputs(args.pdo, library)
    query = validation_roster(roster)
    axis = cells.columns.intersection(rna.columns).sort_values()
    cells, rna = cells[axis], rna[axis]
    if set(cells.index) & set(query.cell_id):
        raise ValueError("A development organoid overlaps the 2D lines")
    edges, targets, priors = parameter_inputs(args)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    _, base, core_rows, calibration = panel_targets(core)
    extra_rows = expand_aligned(extra, calibration)
    mean_extra = extra_rows.groupby([*SOURCE, "compound_id"]).model_value.mean().groupby("compound_id").mean()
    anchor = pd.concat([base, mean_extra], verify_integrity=True)
    combined = pd.concat([core_rows, extra_rows], ignore_index=True)
    targets, known = expanded_targets(extra, targets, axis)
    cell_terms, auxiliary_columns = cell_kernels(cells, rna, edges, auxiliary, device=device)
    cell, cross = cell_terms["full"]
    compounds, kernels = drug_kernels(combined, targets, device=device)
    drug = kernels["structured"]
    weight, residual = training_grid(combined, cells.index, compounds, device=device)
    coefficients, steps, norm = solve_residuals(cell, drug, weight, residual, PENALTY)
    dimensions = rna.index.get_indexer(query.cell_id), compounds.get_indexer(query.compound_id)
    if any((dim < 0).any() for dim in dimensions):
        raise ValueError("The registered common drug or cell library changed")
    base_values = query.compound_id.map(anchor).to_numpy()
    estimates = (cross @ coefficients @ drug).cpu().numpy()[dimensions]

    info = combined[combined.compound_id.isin(query.compound_id)]
    panel = info.groupby(["cell_id", "lineage_group", "compound_id"], as_index=False).residual.mean()
    reference_features = ExpressionFeatures.fit(cells, edges, targets, n_variable=512, rank=True)
    ridge = fit_baseline("drug_ridge", RIDGE[1:3], panel, anchor, cells, reference_features, targets)
    anchor_ridge, residual_ridge = ridge.predict(query, rna)
    if not np.array_equal(anchor_ridge, base_values):
        raise ValueError("The 2D anchors disagree")
    arm = dict(zip(TEACHER, (base_values, anchor_ridge + residual_ridge, base_values + estimates)))
    if not all(np.isfinite(values).all() for values in arm.values()):
        raise ValueError("Missing 2D teacher value on the registered common panel")
    forecast = pd.concat([query[KEY].assign(method=name, predicted_value=arm[name]) for name in TEACHER])
    freeze_table(forecast.assign(track="PDO_source_only", fold=0), args.out, provenance={
        "pdo_receipt": PDO_HASH, "core_receipt": sha256(args.core / "receipt.json"),
        "extension_receipt": sha256(args.extension / "receipt.json"),
        "fit_cells": len(cells), "fit_drugs": len(anchor), "exact_added_drug_targets": known,
        "rna_genes": len(axis), "auxiliary_columns": auxiliary_columns, "solver_steps": steps,
        "solver_residual": norm, "features": priors, "auxiliary_receipt": aux_receipt,
    })


def source_teacher(path, metadata):
    data, seal = verify_forecast(path)
    if seal["provenance"]["pdo_receipt"] != PDO_HASH or set(data.method) != set(TEACHER):
        raise ValueError("The frozen source-only PDO teacher changed")
    panel = data.pivot(index=["cell_id", "compound_id"], columns="method", values="predicted_value")
    library = panel.index.get_level_values("compound_id").unique()
    meta = metadata[metadata.compound_id.isin(library)]
    if set(panel.index) != set(meta.set_index(["cell_id", "compound_id"]).index):
        raise ValueError("Source teacher and registered panel have different exact support")
    return panel, meta, seal["sha256"]


def training_labels(path, fit, held, study):
    raw = pd.read_parquet(path / "observations.parquet", columns=[*META, "endpoint_value"],
                          filters=[("source", "!=", study), ("lineage_group", "not in", list(held.lineage_group.unique()))])
    fields = ["cell_id", "compound_id", "source"]
    data = fit.merge(raw[[*fields, "endpoint_value"]], on=fields, validate="one_to_one")
    if len(data) != len(fit) or set(data.lineage_group) & set(held.lineage_group):
        raise ValueError("PDO adaptation fit attempted to read a held study or model")
    return data


def adaptation_targets(train, teacher):
    data = train.join(teacher[["2d_mean", "2d_structured"]], on=["cell_id", "compound_id"],
                      validate="many_to_one")
    arms = ["source", "compound_id"]
    oriented = -data.endpoint_value
    reference = data["2d_mean"].groupby([data[k] for k in arms]).first()
    offset_by_drug = oriented.groupby([data[k] for k in arms]).mean() - reference
    offset = offset_by_drug.groupby("source").median()
    centered = oriented.groupby([data[k] for k in arms]).transform("mean")
    source_n = data.groupby(arms).lineage_group.nunique()
    local_shift = (offset_by_drug - offset) * source_n / (source_n + ANCHOR_PRIOR)
    shift = local_shift.groupby("compound_id").mean()
    data["teacher_residual"] = data["2d_structured"] - data["2d_mean"]
    mean_teacher = data.groupby("compound_id").teacher_residual.mean()
    study_mean_teacher = data.groupby(arms).teacher_residual.transform("mean")
    data["pdo_residual"] = oriented - centered
    data["transfer_residual"] = data.pdo_residual - (data.teacher_residual - study_mean_teacher)
    return data.assign(source_id=data.source, endpoint_id="native_pdo_AUC"), shift, mean_teacher


def domain_kernel_terms(rna, fit_cells, query_cells, edges, targets, device):
    import torch

    from .structured_kernel import dot_kernel, radial

    fit = rna.loc[fit_cells]
    expression = ExpressionFeatures.fit(fit, edges, targets, n_variable=512, rank=True)
    train, query = (torch.as_tensor(expression.transform(rna.loc[ids]).to_numpy(np.float32), device=device)
                    for ids in (fit_cells, query_cells))
    linear, cross = dot_kernel(train, train), dot_kernel(query, train)
    scale = linear.diagonal().mean().item()
    if scale <= 1e-6:
        raise ValueError("Insufficient variation among fit organoid RNA profiles")
    linear, cross = linear / scale, cross / scale
    base_energy = linear.diagonal()
    query_energy = query.square().mean(dim=1) / scale
    return {
        "linear": (linear, cross),
        "nonlinear": (.6 * linear + .4 * radial(linear, base_energy, base_energy, 2.),
                      .6 * cross + .4 * radial(cross, query_energy, base_energy, 2.)),
    }


def domain_kernels(rna, fit_cells, query_cells, edges, targets, device):
    return domain_kernel_terms(rna, fit_cells, query_cells, edges, targets, device)["nonlinear"]


def adapt(args):
    from .structured_kernel import drug_kernels, solve_residuals, training_grid

    device = device_cuda()
    metadata, rna = inputs(args.pdo)
    teacher, metadata, teacher_hash = source_teacher(args.teacher, metadata)
    fit, held = cells_for_study(metadata, args.study)
    train = training_labels(args.pdo, fit, held, args.study)
    edges, targets, _ = parameter_inputs(args)
    data, shift, mean_teacher = adaptation_targets(train, teacher)
    fit_cells, query_cells = (pd.Index(sorted(table.cell_id.unique())) for table in (data, held))
    cell, cross = domain_kernels(rna, fit_cells, query_cells, edges, targets, device)
    drug_ids, terms = drug_kernels(data, targets, device=device)
    known = held.compound_id.isin(drug_ids).to_numpy()
    drug = terms["structured"]
    test = teacher.reindex(pd.MultiIndex.from_frame(held[["cell_id", "compound_id"]])).reset_index(drop=True)
    anchor = test["2d_mean"].to_numpy() + held.compound_id.map(shift).fillna(0).to_numpy()
    centered_teacher = test["2d_structured"].to_numpy() - test["2d_mean"].to_numpy()
    centered_teacher -= held.compound_id.map(mean_teacher).fillna(0).to_numpy()
    variants, solver = {"pdo_mean": anchor, "pdo_anchor_transfer": anchor + centered_teacher}, {}
    for name, column in (("pdo_only_residual", "pdo_residual"), (PRIMARY, "transfer_residual")):
        weight, associated = training_grid(data.assign(residual=data[column]), fit_cells, drug_ids, device=device)
        coef, steps, norm = solve_residuals(cell, drug, weight, associated, PDO_PENALTY)
        y_fit = cell @ coef @ drug
        mean_by_drug = (weight * y_fit).sum(dim=0) / weight.sum(dim=0).clamp_min(1e-8)
        f = (cross @ coef @ drug - mean_by_drug).cpu().numpy()
        positions = query_cells.get_indexer(held.loc[known, "cell_id"]), drug_ids.get_indexer(
            held.loc[known, "compound_id"],
        )
        q = test["2d_structured"].to_numpy().copy()
        offset = centered_teacher[known] if name == PRIMARY else 0.
        q[known] = anchor[known] + offset + f[positions]
        variants[name] = q
        solver[name] = {"iterations": steps, "residual": norm}
    methods = dict(zip(TEACHER, (test[name].to_numpy() for name in TEACHER))) | variants
    if not all(np.isfinite(values).all() for values in methods.values()):
        raise ValueError("A registered PDO prediction is missing")
    outputs = [held[KEY].assign(method=name, predicted_value=v) for name, v in methods.items()]
    freeze_table(pd.concat(outputs).assign(track=args.study, fold=0), args.out, provenance={
        "pdo_receipt": PDO_HASH, "teacher_sha256": teacher_hash, "study": args.study,
        "fit_pdo": len(fit_cells), "held_pdo": len(query_cells), "fit_pdo_assays": sorted(data.source.unique()),
        "fit_pdo_rows": len(fit), "held_pdo_rows": len(held), "held_drugs": int(held.compound_id.nunique()),
        "direct_3d_drugs": int(held.loc[known, "compound_id"].nunique()), "direct_3d_rows": int(known.sum()),
        "solver": solver,
    })


def tableau(section):
    pairs = [(method, baseline) for method in ("pdo_anchor_transfer", "pdo_only_residual", PRIMARY)
             for baseline in ("2d_mean", "2d_structured", "2d_ridge")]
    ranking, _ = score(section, endpoint_direction="higher", rank_only=True, comparisons=pairs)
    enough = section.lineage_group.nunique() >= 5
    drugs, _ = score_across_cells(section, min_lineages=5 if enough else 4)
    associations = drugs.groupby("method")[["spearman", "pearson"]].mean()
    count = drugs.groupby("method").spearman.count()
    within = {row["method"]: row for row in ranking["scores"]}
    metrics = {}
    for method in (*TEACHER, *ADAPTED):
        values = associations.loc[method]
        metrics[method] = {"within_drug_spearman": None if pd.isna(values.spearman) else float(values.spearman),
                           "within_drug_pearson": None if pd.isna(values.pearson) else float(values.pearson),
                           "within_drug_eligible_drugs": int(count[method]), **within[method]}
    intervals = {baseline: crossed_patient_drug_ci(section, PRIMARY, baseline, repeats=2400)
                 for baseline in ("2d_ridge", "2d_structured")} if enough else {}
    return {"measured_rows": section.row_id.nunique(), "drugs": section.compound_id.nunique(),
            "pdo_models": section.cell_id.nunique(), "technical_four_donor_correlations": not enough,
            "metrics": metrics, "within_drug_contrasts": intervals, "within_lineage_contrasts": ranking["paired_contrasts"]}


def evaluate(args):
    metadata, _ = inputs(args.pdo)
    _, metadata, teacher_hash = source_teacher(args.teacher, metadata)
    truth = pd.read_parquet(args.pdo / "observations.parquet", columns=["row_id", "endpoint_value"])
    result, seals, support = {}, {}, {}
    for study in STUDIES:
        fit, held = cells_for_study(metadata, study)
        file = args.forecasts / (study.replace(" ", "_") + ".parquet")
        frame, receipt = verify_forecast(file)
        meta = receipt["provenance"]
        if (meta["pdo_receipt"], meta["teacher_sha256"], meta["study"]) != (PDO_HASH, teacher_hash, study):
            raise ValueError("Unexpected PDO forecast provenance")
        if set(frame.method) != set(TEACHER + ADAPTED):
            raise ValueError("A locked PDO model is missing")
        for _, part in frame.groupby("method"):
            if not part[KEY].set_index("row_id").sort_index().equals(held[KEY].set_index("row_id").sort_index()):
                raise ValueError("Forecasts must have identical registered PDO support")
        joined = frame.merge(truth, on="row_id", validate="many_to_one")
        joined.endpoint_value *= -1
        if len(joined) != len(frame):
            raise ValueError("Missing a registered PDO outcome")
        full = tableau(joined)
        direct = tableau(joined[joined.compound_id.isin(fit.compound_id)])
        result[study] = {"full_panel": full, "direct_3d_support": direct}
        seals[study] = receipt["sha256"]
        support[study] = meta
    primary = result["Pilot 1"]["full_panel"]
    across = primary["within_drug_contrasts"]["2d_structured"]
    within = [entry["ci95"][0] for entry in primary["within_lineage_contrasts"]
              if entry["candidate"] == PRIMARY and entry["baseline"] == "2d_structured"
              and entry["metric"] in ("spearman", "ndcg_at_5")]
    secondary = result["IMPROVE dataset 3"]["full_panel"]
    ndcg = [entry["ci95"][0] for entry in secondary["within_lineage_contrasts"]
            if entry["candidate"] == PRIMARY and entry["baseline"] == "2d_structured" and entry["metric"] == "ndcg_at_5"]
    gate = {"primary_both_within_drug_gains": len(across) == 2 and min(v[0] for v in across.values()) > 0,
            "primary_both_within_lineage_gains": len(within) == 2 and min(within) > 0,
            "secondary_top5_noninferior": len(ndcg) == 1 and ndcg[0] > -.01}
    publish_json(args.out, {"study_metrics": result, "study_support": support, "gate": gate,
                            "promoted": all(gate.values()), "teacher_sha256": teacher_hash, "forecast_sha256": seals})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    params = ("priors", "hallmark", "targets")
    for action, fields in {
        "teacher": ("original", "core", "extension", "auxiliary", *params),
        "adapt": ("teacher", *params),
        "score": ("teacher", "forecasts"),
    }.items():
        command = actions.add_parser(action)
        for field in ("pdo", "out", *fields):
            command.add_argument("--" + field, type=Path, required=True)
        if action == "adapt":
            command.add_argument("--study", choices=STUDIES, required=True)
    args = parser.parse_args()
    {"teacher": teacher_predict, "adapt": adapt, "score": evaluate}[args.action](args)


if __name__ == "__main__":
    main()
