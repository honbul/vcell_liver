"""Measure local assay calibration with patient-disjoint few-shot prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold

from .evaluate import join_observations, score, score_across_cells
from .external import KEY
from .function_data import new_output
from .hcc_3d_reciprocal import checked_inputs
from .hcc_3d_study_transfer import STUDIES, parameters, recipient_outcomes
from .io import freeze_table, publish_json, sha256, verify_forecast
from .pdo_transfer_cv import device_cuda, domain_kernel_terms
from .v0plus_score import compound_strata
from .yang_hcc_external import require_hash

INPUT_SHA = "d9778cf77d3732f81f8bbcbe4edc9c5a15aecf454c5047bc1f53278bfd629716"
SOURCE_SHA = "df76a2992c07532cfac09fd2ce6d8fb846f7a21a7e7c4f9c40d2d84e6d775bad"
BUDGETS = {"LICOB": (2, 4, 8, 12), "Wong": (2, 4)}
REPEATS = 5
SOURCE = ("2d_mean", "2d_ridge", "2d_teacher", "2d_cellhit", "2d_pdo_average")
PRIMARY = "local_residual_kernel"
METHODS = {*SOURCE, "local_mean", "local_shrink_mean", "local_calibrated_2d",
           "local_calibrated_cross3d", "local_ridge", PRIMARY}
REFERENCES = ("local_shrink_mean", "local_calibrated_2d")
COMPARISONS = tuple((PRIMARY, ref) for ref in (*REFERENCES, "2d_teacher", "2d_ridge", "2d_cellhit", "local_ridge")) + (
    ("local_shrink_mean", "2d_mean"), ("local_calibrated_2d", "2d_teacher"),
)


def fully_registered(args):
    meta, digest = checked_inputs(args.inputs)
    arch, seal = verify_forecast(args.prior)
    if (digest, seal["sha256"], seal["provenance"]["input_sha256"]) != (INPUT_SHA, SOURCE_SHA, INPUT_SHA):
        raise ValueError("Unexpected original 21-drug predictions")
    return meta, arch


def layouts(roster, study, repeat):
    patients = np.array(sorted(roster.lineage_group.unique()), dtype=object)
    seed = 31041 + repeat
    for side, (fit, held) in enumerate(KFold(2, shuffle=True, random_state=seed).split(patients)):
        order = np.random.default_rng(seed + 1000 + side).permutation(patients[fit]).tolist()
        yield 2 * repeat + side, order[:max(BUDGETS[study])], patients[held].tolist()


def fold_path(split, study, fold):
    return split / f"{study}-{fold}.parquet"


def prepare(args):
    _, _ = fully_registered(args)
    checksums, definitions = {}, {}
    with new_output(args.out) as folder:
        for study, book in zip(STUDIES, (args.licob_book, args.wong_book)):
            roster = pd.read_parquet(args.inputs / f"{study}.parquet")
            actual = recipient_outcomes(study, roster, book)
            if len(actual) != len(roster):
                raise ValueError("Unexpected missing outcome in the registered complete response grid")
            definitions[study] = {}
            for trial in range(REPEATS):
                for fold, fit, held in layouts(roster, study, trial):
                    path = fold_path(folder, study, fold)
                    actual[actual.lineage_group.isin(fit)].to_parquet(path, index=False)
                    checksums[path.name] = sha256(path)
                    definitions[study][str(fold)] = {"fit_order": fit, "held_donors": held}
        publish_json(folder / "receipt.json", {"source": SOURCE_SHA, "inputs": INPUT_SHA,
                                              "budgets": BUDGETS, "matches": definitions, "artifacts": checksums})


def split_manifest(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if (receipt["source"], receipt["inputs"], receipt["budgets"]) != (
            SOURCE_SHA, INPUT_SHA, {key: list(value) for key, value in BUDGETS.items()}):
        raise ValueError("Unexpected few-shot input declaration")
    return receipt


def local_models(fit, held, rna, donor_rna, edges, targets, device):
    import torch

    from .structured_kernel import drug_kernels, solve_residuals

    if set(fit.lineage_group) & set(held.lineage_group) or "endpoint_value" in held:
        raise ValueError("Calibration raw rows crossed the patient partition")
    fit_cells = pd.Index(sorted(fit.cell_id.unique()))
    held_cells = pd.Index(sorted(held.cell_id.unique()))
    feature_fit = fit_cells.append(donor_rna)
    compounds, drug = drug_kernels(fit, targets, device=device)
    kernels = domain_kernel_terms(rna, feature_fit, held_cells, edges, targets, device)
    donor_fit = fit.groupby(["lineage_group", "compound_id"])[
        ["endpoint_value", "2d_mean", "2d_teacher", "2d_anchor_pdo_ridge"]
    ].mean()
    by_drug = donor_fit.groupby("compound_id")
    mean = by_drug.mean().reindex(compounds)
    observed = mean.endpoint_value
    n = float(fit.lineage_group.nunique())
    slope = observed.std(ddof=0) / mean["2d_mean"].std(ddof=0)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("Drug anchors require nonzero spread on the observed training panel")
    base = observed.mean() + slope * (mean["2d_mean"] - mean["2d_mean"].mean())
    anchor = (n * observed + 4 * base) / (n + 4)
    variance = by_drug.endpoint_value.var(ddof=0).reindex(compounds)
    sd = np.sqrt((n * variance + 4 * variance.mean()) / (n + 4)).clip(lower=1e-6)
    model_groups = fit[["cell_id", "lineage_group"]].drop_duplicates().set_index("cell_id", verify_integrity=True)
    weights = 1. / model_groups.lineage_group.map(model_groups.lineage_group.value_counts()).reindex(fit_cells)
    weight_grid = torch.tensor(np.repeat(weights.to_numpy()[:, None], len(compounds), axis=1),
                               dtype=torch.float32, device=device)

    def shape(data, method):
        return (data[method] - data.compound_id.map(mean[method])).to_numpy() * slope

    result = {name: held[name].to_numpy() for name in SOURCE}
    prediction_base = held.compound_id.map(anchor).to_numpy()
    t = shape(held, "2d_teacher")
    result.update({
        "local_mean": held.compound_id.map(observed).to_numpy(), "local_shrink_mean": prediction_base,
        "local_calibrated_2d": prediction_base + t,
        "local_calibrated_cross3d": prediction_base + shape(held, "2d_anchor_pdo_ridge"),
    })
    index = held_cells.get_indexer(held.cell_id), compounds.get_indexer(held.compound_id)
    scale = held.compound_id.map(sd).to_numpy()
    for name, kind, chemistry in (("local_ridge", "linear", "no_drug_sharing"), (PRIMARY, "nonlinear", "structured")):
        center = fit.endpoint_value.to_numpy() - fit.compound_id.map(observed).to_numpy()
        if name == PRIMARY:
            center -= shape(fit, "2d_teacher")
        matrix = (fit.assign(value=center / fit.compound_id.map(sd).to_numpy())
                  .pivot(index="cell_id", columns="compound_id", values="value")
                  .reindex(index=fit_cells, columns=compounds))
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Missing few-shot RNA or drug residual")
        response = torch.as_tensor(matrix.to_numpy(np.float32), device=device)
        fit_term, cross = kernels[kind]
        d = drug[chemistry]
        m = len(fit_cells)
        coef, _, _ = solve_residuals(fit_term[:m, :m], d, weight_grid, response, 2.)
        delta = (cross[:, :m] @ coef @ d).cpu().numpy()[index] * scale
        shape_term = t if name == PRIMARY else 0.
        result[name] = prediction_base + shape_term + delta
    if not all(np.isfinite(p).all() for p in result.values()):
        raise ValueError("Nonfinite few-shot prediction")
    return pd.concat([held[KEY].assign(method=name, predicted_value=p) for name, p in result.items()])


def predict(args):
    _, past = fully_registered(args)
    manifest = split_manifest(args.splits)
    cohort = pd.read_parquet(args.inputs / f"{args.study}.parquet")
    other = next(name for name in STUDIES if name != args.study)
    donor_rna = pd.Index(sorted(pd.read_parquet(args.inputs / f"{other}.parquet", columns=["cell_id"]).cell_id.unique()))
    cohort_prediction = past[past.track.eq(args.study)].pivot(index="row_id", columns="method", values="predicted_value")
    roster = cohort.join(cohort_prediction, on="row_id", validate="one_to_one")
    rna = pd.read_parquet(args.inputs / "rna.parquet")
    device = device_cuda()
    edges, targets, priors = parameters(args)
    targets = {k: v for k, v in targets.items() if k in set(roster.compound_id)}
    predictions = []
    for fold, donors, test in layouts(roster, args.study, args.repeat):
        plan = manifest["matches"][args.study][str(fold)]
        if plan != {"fit_order": donors, "held_donors": test} or set(donors) & set(test):
            raise ValueError("The few-shot patient layout changed")
        path = fold_path(args.splits, args.study, fold)
        require_hash(path, manifest["artifacts"][path.name])
        labelled = pd.read_parquet(path).join(cohort_prediction, on="row_id", validate="one_to_one")
        if set(labelled.lineage_group) != set(donors):
            raise ValueError("Fit label file contains other donor IDs")
        held = roster[roster.lineage_group.isin(test)]
        for budget in BUDGETS[args.study]:
            local = labelled[labelled.lineage_group.isin(donors[:budget])]
            frame = local_models(local, held, rna, donor_rna, edges, targets, device)
            predictions.append(frame.assign(track=f"{args.study}/n{budget}", fold=fold))
    freeze_table(pd.concat(predictions, ignore_index=True), args.out, provenance={
        "source": SOURCE_SHA, "inputs": INPUT_SHA, "splits": sha256(args.splits / "receipt.json"),
        "study": args.study, "repeat": args.repeat, "priors": priors,
    })


def patient_averages(frame):
    fields = ["fold", "track", "lineage_group", "compound_id", "method"]
    groups = frame.groupby(fields, sort=True)
    values = groups[["endpoint_value", "predicted_value"]].mean()
    limits = groups.predicted_value.agg(["min", "max"])
    # Preserve bitwise constant controls through unequal regional PDO counts.
    values["predicted_value"] = values.predicted_value.where(limits["min"].ne(limits["max"]), limits["min"])
    values = values.reset_index()
    return values.assign(cell_id=values.lineage_group, row_id=values.lineage_group + "|" + values.compound_id)


def weighted_correlations(x, y, weight):
    """Weighted compound-wise correlations for every bootstrap draw and model."""
    w = weight / weight.sum(axis=1, keepdims=True)
    a, b = w @ x, np.einsum("bn,mnd->bmd", w, y, optimize=True)
    var_a = np.maximum(w @ (x*x) - a*a, 0.)
    var_b = np.maximum(np.einsum("bn,mnd->bmd", w, y*y, optimize=True) - b*b, 0.)
    covariance = np.einsum("bn,mnd->bmd", w, y*x, optimize=True) - a[:, None, :]*b
    denom = np.sqrt(var_a[:, None, :] * var_b)
    return np.divide(covariance, denom, out=np.zeros_like(denom), where=denom > 1e-10)


def clustered_sensitivity(donors, contrasts, *, draws=2400):
    methods = sorted({name for pair in contrasts for name in pair})
    patients = pd.Index(sorted(donors.lineage_group.unique()))
    drugs = pd.Index(sorted(donors.compound_id.unique()))
    rng = np.random.default_rng(4147)
    patient_weights = rng.exponential(size=(draws, len(patients)))
    drug_weights = rng.exponential(size=(draws, len(drugs)))
    drug_weights /= drug_weights.sum(axis=1, keepdims=True)
    totals = {key: np.zeros((draws, len(methods), len(drugs))) for key in ("pearson", "spearman")}
    for _, fold in donors.groupby("fold"):
        native = fold[fold.method.eq(methods[0])].pivot(index="lineage_group", columns="compound_id",
                                                       values="endpoint_value").reindex(columns=drugs)
        pred = np.stack([fold[fold.method.eq(name)].pivot(index="lineage_group", columns="compound_id",
                                                           values="predicted_value").loc[native.index, drugs]
                         .to_numpy() for name in methods])
        w = patient_weights[:, patients.get_indexer(native.index)]
        x = native.to_numpy()
        # Normalized fold-local centering reduces cancellation in near-constant columns.
        # Constants remain exactly zero after centering rather than acquiring rounding ranks.
        def center_and_scale(values, axis):
            centered = values - np.take(values, [0], axis=axis)
            scale = centered.std(axis=axis, keepdims=True)
            return np.divide(centered, scale, out=np.zeros_like(values), where=scale > 1e-12)

        totals["pearson"] += weighted_correlations(center_and_scale(x, 0), center_and_scale(pred, 1), w)
        totals["spearman"] += weighted_correlations(center_and_scale(rankdata(x, axis=0), 0),
                                                    center_and_scale(rankdata(pred, axis=1), 1), w)
    output = []
    for name, base in contrasts:
        first, second = methods.index(name), methods.index(base)
        result = {metric: np.quantile(np.sum((matrix[:, first] - matrix[:, second])*drug_weights, axis=1)
                                      / donors.fold.nunique(), [.025, .975]).tolist()
                  for metric, matrix in totals.items()}
        output.append({"candidate": name, "baseline": base, "ci95": result})
    return output


def score_budget(regions, annotations):
    donors = patient_averages(regions)
    ranking, _ = score(donors, endpoint_direction="higher", rank_only=True, comparisons=COMPARISONS)
    measures, _ = score_across_cells(donors)
    scores = measures.groupby("method", as_index=False)[["spearman", "pearson"]].mean().to_dict("records")
    sensitivity = clustered_sensitivity(donors, COMPARISONS)
    gates = {}
    for base in REFERENCES:
        d = next(p["ci95"] for p in sensitivity if p["candidate"] == PRIMARY and p["baseline"] == base)
        a = [p["ci95"][0] for p in ranking["paired_contrasts"] if p["candidate"] == PRIMARY
             and p["baseline"] == base and p["metric"] in {"spearman", "ndcg_at_5"}]
        gates[base] = len(a) == 2 and min([x[0] for x in d.values()] + a) > 0
    strata = {name: {"drugs": len(parents),
                     "scores": measures[measures.compound_id.isin(parents)].groupby("method", as_index=False)[
                         ["spearman", "pearson"]].mean().to_dict("records")}
              for name, parents in compound_strata(annotations).items() if parents}
    by_drug = measures.groupby(["method", "compound_id"], as_index=False)[["spearman", "pearson"]].mean()
    by_repeat = measures.assign(repeat=measures.fold // 2).groupby(["method", "repeat"], as_index=False)[
        ["spearman", "pearson"]].mean()
    return {"patients": donors.lineage_group.nunique(), "folds": donors.fold.nunique(),
            "sensitivity": scores, "ranking": ranking["scores"], "ranking_ci95": ranking["paired_contrasts"],
            "sensitivity_ci95": sensitivity, "gates": gates, "strata": strata,
            "by_drug": by_drug.to_dict("records"), "by_repeat": by_repeat.to_dict("records")}


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    inputs, _ = fully_registered(args)
    manifest_hash = sha256(args.splits / "receipt.json")
    annotations = pd.read_parquet(args.inputs / "Wong.parquet").drop_duplicates("compound_id")
    frozen, forecasts = {}, {}
    rosters = {study: pd.read_parquet(args.inputs / f"{study}.parquet") for study in STUDIES}
    for study, roster in rosters.items():
        blocks = []
        for trial in range(REPEATS):
            path = args.forecasts / f"{study}-{trial}.parquet"
            frame, seal = verify_forecast(path)
            status = seal["provenance"]
            if (status["source"], status["splits"], status["study"], status["repeat"]) != (
                    SOURCE_SHA, manifest_hash, study, trial):
                raise ValueError("Unaudited few-shot fold")
            folds = dict((fold, set(roster[roster.lineage_group.isin(test)].row_id))
                         for fold, _, test in layouts(roster, study, trial))
            expected_methods = {(f"{study}/n{n}", fold, name) for n in BUDGETS[study]
                                for fold in folds for name in METHODS}
            groups = frame.groupby(["track", "fold", "method"])
            if set(groups.groups) != expected_methods or any(len(rows) != len(folds[fold])
                    or set(rows.row_id) != folds[fold] for (_, fold, _), rows in groups):
                raise ValueError("A frozen comparison missed a registered held patient")
            blocks.append(frame)
            frozen[path.stem] = seal["sha256"]
        forecasts[study] = pd.concat(blocks, ignore_index=True)
    results = {}
    for study, book in zip(STUDIES, (args.licob_book, args.wong_book)):
        outcomes = recipient_outcomes(study, rosters[study], book)
        evaluation = join_observations(forecasts[study], outcomes)
        results[study] = {track.split("/")[-1]: score_budget(part, annotations)
                          for track, part in evaluation.groupby("track")}
    publish_json(args.out, {"role": "few-shot development on previously evaluated cohorts",
                            "results": results, "compounds": inputs["compound_labels"], "forecasts": frozen,
                            "both_primary_gates": all(all(b["n4"]["gates"].values()) for b in results.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    options = {
        "prepare": ("licob_book", "wong_book"),
        "predict": ("splits", "priors", "hallmark", "targets"),
        "score": ("licob_book", "wong_book", "splits", "forecasts"),
    }
    for mode, keys in options.items():
        child = commands.add_parser(mode)
        for key in ("inputs", "prior", *keys, "out"):
            child.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
        if mode == "predict":
            child.add_argument("--study", choices=STUDIES, required=True)
            child.add_argument("--repeat", type=int, choices=range(REPEATS), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "score": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
