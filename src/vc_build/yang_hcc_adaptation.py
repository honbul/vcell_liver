"""Grouped development of 3D HCC response models with separate score execution."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .evaluate import join_observations, score, score_across_cells
from .external import KEY
from .function_data import new_output
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import domain_kernel_terms
from .structured_cv import paired_intervals
from .v0plus_cv import molecular_inputs
from .yang_hcc_audit import COHORT
from .yang_hcc_external import (
    EXPOSED_DONORS, PRIMARY as SOURCE_ARM, SMALL_MOLECULE_TARGETS, aggregate_patients,
    outcomes, read_inputs, require_hash, summarize,
)

BENCHMARK_INPUTS = "a9574193e4489a9805ac197de6b9f7dfaa28e5de682e33fb219b2888640f5124"
BENCHMARK_FORECAST = "00c685ec949ed8c8abe4c135b390c56c2f24b9be8d193ed19074f00129094d79"
TRAINED = ("teacher_affine", "pdo_ridge", "pdo_nonlinear", "pdo_structured", "pdo_signed_transfer")
ANCHORED = ("drug_mean", "2d_mean", "2d_teacher")
PRIMARY = "pdo_signed_transfer"
PENALTIES = (2., 8., 32.)
REFERENCES = ("drug_mean", "pdo_ridge")
EXTRA_COMPARISONS = ("2d_teacher", "teacher_affine", "pdo_structured", "pdo_nonlinear")


def patient_splits(roster, seed):
    patients = pd.Index(sorted(roster.lineage_group.unique()))
    for fit, held in KFold(3, shuffle=True, random_state=seed).split(patients):
        yield (roster.lineage_group.isin(patients[fit]), roster.lineage_group.isin(patients[held]))


def original_inputs(inputs, frozen):
    read_inputs(inputs, BENCHMARK_INPUTS)
    archive, receipt = verify_forecast(frozen)
    if receipt["sha256"] != BENCHMARK_FORECAST:
        raise ValueError("Unexpected source-only teacher forecasts")
    query = pd.read_parquet(inputs / "query.parquet")
    query = query[~query.lineage_group.isin(EXPOSED_DONORS)]
    teacher = (archive[archive.method.isin({"drug_mean", SOURCE_ARM})]
               .pivot(index="row_id", columns="method", values="predicted_value")
               .rename(columns={"drug_mean": "2d_mean", SOURCE_ARM: "2d_teacher"}))
    roster = query.join(teacher, on="row_id", validate="one_to_one")
    if (len(roster) != 1085 or roster.lineage_group.nunique() != 74
            or not np.isfinite(roster[list(ANCHORED[1:])].to_numpy()).all()):
        raise ValueError("Unexpected donor or teacher support")
    return roster


def prepare(args):
    roster = original_inputs(args.inputs, args.teacher)
    response = outcomes(args.workbook, roster)
    if not np.isfinite(response.endpoint_value).all():
        raise ValueError("A previously complete Yang development outcome is missing")
    hashes, support = {}, {}
    with new_output(args.out) as tmp:
        for fold, (fit, held) in enumerate(patient_splits(roster, 1719)):
            for role, data in (("fit", response[fit]), ("held", roster[held])):
                file = tmp / f"{role}-{fold}.parquet"
                data.to_parquet(file, index=False)
                hashes[file.name] = sha256(file)
            support[str(fold)] = {"fit_donors": roster.loc[fit, "lineage_group"].nunique(),
                                  "held_donors": roster.loc[held, "lineage_group"].nunique(),
                                  "held_rows": int(held.sum())}
        publish_json(tmp / "receipt.json", {"teacher_sha256": BENCHMARK_FORECAST,
                                             "inputs_sha256": BENCHMARK_INPUTS,
                                             "seed": 1719, "support": support, "artifacts": hashes})


def load_fold(path, fold):
    record = json.loads((path / "receipt.json").read_text())
    if (record["teacher_sha256"], record["inputs_sha256"], record["seed"]) != (
            BENCHMARK_FORECAST, BENCHMARK_INPUTS, 1719):
        raise ValueError("Wrong Yang grouped development manifest")
    tables = []
    for role in ("fit", "held"):
        file = path / f"{role}-{fold}.parquet"
        require_hash(file, record["artifacts"][file.name])
        tables.append(pd.read_parquet(file))
    return tables


def predictor(fit, held, rna, edge, target, device, selected):
    import torch

    from .structured_kernel import drug_kernels, solve_residuals

    if set(fit.lineage_group) & set(held.lineage_group):
        raise ValueError("An organoid donor crossed the fold boundary")
    if "endpoint_value" in held:
        raise ValueError("Held organoid responses cannot enter prediction")
    fit_cells, held_cells = (pd.Index(sorted(frame.cell_id.unique())) for frame in (fit, held))
    drugs, kernels = drug_kernels(fit, target, device=device)
    terms = domain_kernel_terms(rna, fit_cells, held_cells, edge, target, device)
    mean = fit.groupby(["lineage_group", "compound_id"])[["endpoint_value", "2d_teacher"]].mean()
    averages = mean.groupby("compound_id").mean().reindex(drugs)
    stds = mean.groupby("compound_id").std(ddof=0).reindex(drugs)
    if (stds < 1e-6).any().any() or stds.isna().any().any():
        raise ValueError("The development fold cannot standardize a constant phenotype or teacher")
    donors = fit[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id", verify_integrity=True)
    counts = donors.lineage_group.value_counts()
    model_weight = 1 / donors.reindex(fit_cells).lineage_group.map(counts).to_numpy()
    grid = torch.as_tensor(np.broadcast_to(model_weight[:, None], (len(fit_cells), len(drugs))).copy(),
                           dtype=torch.float32, device=device)

    def standardize(frame, cells, name):
        matrix = frame.pivot(index="cell_id", columns="compound_id", values=name).reindex(index=cells, columns=drugs)
        z = ((matrix - averages[name]) / stds[name]).to_numpy(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Incomplete registered organoid–drug training or prediction grid")
        return torch.as_tensor(z, device=device)

    y = standardize(fit, fit_cells, "endpoint_value")
    teacher_fit = standardize(fit, fit_cells, "2d_teacher")
    teacher_test = standardize(held, held_cells, "2d_teacher")
    dr = drugs.get_indexer(held.compound_id)
    row = held_cells.get_indexer(held.cell_id)
    baseline = held.compound_id.map(averages.endpoint_value).to_numpy()
    scale = stds.endpoint_value.to_numpy()[dr]
    methods = {arm: held[arm].to_numpy() for arm in ANCHORED[1:]} | {"drug_mean": baseline}
    diagnostics = {}
    for arm, penalties in selected.items():
        for penalty in penalties:
            if arm == "teacher_affine":
                gain = (grid * teacher_fit * y).sum(dim=0) / ((grid * teacher_fit.square()).sum(dim=0) + penalty)
                prediction = teacher_test * gain
                details = {"teacher_slopes_standardized": dict(zip(drugs, gain.cpu().tolist()))}
            else:
                cell, cross = terms["linear" if arm == "pdo_ridge" else "nonlinear"]
                chemistry = kernels["no_drug_sharing" if arm in ("pdo_ridge", "pdo_nonlinear") else "structured"]
                signed = arm == PRIMARY
                alpha, steps, error = solve_residuals(cell, chemistry, grid, y, penalty,
                                                       per_drug_prior=teacher_fit if signed else None)
                prediction = cross @ alpha @ chemistry
                details = {"steps": steps, "solver_residual": error}
                if signed:
                    gain = (teacher_fit * alpha).sum(dim=0)
                    prediction = prediction + teacher_test * gain
                    details["teacher_slopes_standardized"] = dict(zip(drugs, gain.cpu().tolist()))
            key = f"{arm}/{penalty}"
            values = baseline + scale * prediction.cpu().numpy()[row, dr]
            if not np.isfinite(values).all():
                raise ValueError("A candidate forecast contains invalid values")
            methods[key], diagnostics[key] = values, details
    forecast = pd.concat([held[KEY].assign(method=name, predicted_value=p) for name, p in methods.items()])
    return forecast.assign(track=COHORT, fold=0), diagnostics


def aggregate(frame):
    parts = []
    for _, block in frame.groupby("fold", sort=True):
        donors = aggregate_patients(block, "mean")
        source_anchor = block[block.method.eq("2d_mean")].groupby("compound_id").predicted_value
        if not source_anchor.nunique().eq(1).all():
            raise ValueError("The archived 2D baseline must be constant within a drug")
        source = donors.method.eq("2d_mean")
        donors.loc[source, "predicted_value"] = donors.loc[source, "compound_id"].map(source_anchor.first())
        parts.append(donors)
    return pd.concat(parts, ignore_index=True)


def utility(frame):
    """The preregistered utility on patient averages at the inner split level."""
    donors = aggregate(frame)
    ranking, _ = score(donors, endpoint_direction="higher", rank_only=True)
    across, _ = score_across_cells(donors)
    by_drug = across.groupby("method")[["spearman", "pearson"]].mean()
    ranks = pd.DataFrame(ranking["scores"]).set_index("method")
    return .4 * by_drug.spearman + .15 * by_drug.pearson + .225 * (ranks.macro_spearman + ranks.macro_ndcg_at_5)


def predict(args):
    from .pdo_transfer_cv import device_cuda

    device = device_cuda()
    read_inputs(args.inputs, BENCHMARK_INPUTS)
    fit, held = load_fold(args.splits, args.fold)
    rna = pd.read_parquet(args.inputs / "rna.parquet").loc[sorted(set(fit.cell_id) | set(held.cell_id))]
    edges, targets, provenance = molecular_inputs(args.priors, args.hallmark, args.targets)
    panel = set(fit.compound_id)
    targets = {name: genes for name, genes in (targets | SMALL_MOLECULE_TARGETS).items() if name in panel}
    trials = dict.fromkeys(TRAINED, PENALTIES)
    inner = []
    for i, (train, validation) in enumerate(patient_splits(fit, 1720)):
        check = fit[validation]
        result, _ = predictor(fit[train], check.drop(columns="endpoint_value"), rna, edges, targets, device, trials)
        inner.append(join_observations(result.assign(fold=i), check))
    fitted = utility(pd.concat(inner, ignore_index=True))
    selected = {arm: max(PENALTIES, key=lambda x: (fitted[f"{arm}/{x}"], x)) for arm in TRAINED}
    if not np.isfinite(fitted.to_numpy()).all():
        raise ValueError("Missing registered inner-fold model utility")
    result, diagnostics = predictor(fit, held, rna, edges, targets, device,
                                    {name: [penalty] for name, penalty in selected.items()})
    result.method = result.method.str.split("/").str[0]
    freeze_table(result.assign(fold=args.fold), args.out, provenance={
        "fold": args.fold, "splits_sha256": sha256(args.splits / "receipt.json"),
        "teacher_sha256": BENCHMARK_FORECAST, "priors": provenance, "selected_penalties": selected,
        "inner_utility": fitted.to_dict(), "diagnostics": diagnostics,
        "fit_patients": fit.lineage_group.nunique(), "held_patients": held.lineage_group.nunique(),
    })


def sensitivity_contrasts(frame, pairs):
    return [{"candidate": a, "baseline": b, "ci95": paired_intervals(frame, a, b, repeats=2400)}
            for a, b in pairs]


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    roster = original_inputs(args.inputs, args.teacher)
    predictions, seals = [], {}
    manifest_hash = sha256(args.splits / "receipt.json")
    for fold, (_, held) in enumerate(patient_splits(roster, 1719)):
        file = args.forecasts / f"fold-{fold}.parquet"
        model, receipt = verify_forecast(file)
        info = receipt["provenance"]
        if (info["fold"], info["splits_sha256"], info["teacher_sha256"]) != (fold, manifest_hash, BENCHMARK_FORECAST):
            raise ValueError("Mismatched grouped validation identities")
        expected = roster[held].row_id
        if (set(model.method) != set(TRAINED+ANCHORED) or not model.fold.eq(fold).all()
                or any(len(group) != len(expected) or set(group.row_id) != set(expected)
                       for _, group in model.groupby("method"))):
            raise ValueError("Each method must predict all registered held cases")
        predictions.append(model)
        seals[str(fold)] = receipt
    observed = outcomes(args.workbook, roster)
    regions = join_observations(pd.concat(predictions, ignore_index=True), observed)
    donors = aggregate(regions)
    pairs = [(PRIMARY, b) for b in (*REFERENCES, *EXTRA_COMPARISONS)]
    summary = summarize(donors, ())
    summary["sensitivity_ci95"] = sensitivity_contrasts(donors, pairs)
    ranks, _ = score(donors, endpoint_direction="higher", rank_only=True, comparisons=pairs)
    summary["ranking_contrasts"] = ranks["paired_contrasts"]
    gate = {}
    for base in REFERENCES:
        across = next(row["ci95"] for row in summary["sensitivity_ci95"] if row["baseline"] == base)
        within = [r["ci95"][0] for r in summary["ranking_contrasts"] if r["baseline"] == base
                  and r["metric"] in ("spearman", "ndcg_at_5")]
        gate[base] = {"both_sensitivity_positive": len(across) == 2 and min(ci[0] for ci in across.values()) > 0,
                      "both_ranking_noninferior": len(within) == 2 and min(within) > -.01}
    publish_json(args.out, {"scope": "within Yang HCC donor-held-out development",
                            "summary": summary, "gates": gate, "promoted": all(all(g.values()) for g in gate.values()),
                            "freezes": seals, "compounds": observed.set_index("compound_id").response_label.to_dict()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for stage, fields in (
        ("prepare", ("inputs", "teacher", "workbook")),
        ("predict", ("inputs", "splits", "priors", "hallmark", "targets")),
        ("score", ("inputs", "teacher", "splits", "forecasts", "workbook")),
    ):
        command = sub.add_parser(stage)
        for field in (*fields, "out"):
            command.add_argument("--" + field, type=Path, required=True)
        if stage == "predict":
            command.add_argument("--fold", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "score": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
