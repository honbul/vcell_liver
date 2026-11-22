"""Patient-disjoint few-shot prediction and paired scoring on already registered 3D cohorts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.hcc_3d_study_transfer import DRUGS
from vc_build.io import freeze_table, publish_json, sha256, verify_forecast

from .contract import TRAIN_SHA, read_json, verified
from .ledger import save
from .metrics import AXES, RESIDUAL, association, interval, mean, metrics, pivot, within_cell
from .pdo_data import BUDGETS, REPEATS, STUDIES, patient_table

A = ("a_2d_structured", "a_2d_pairwise_rank")
B = "b_local_mean"
C = "c_eb_shrinkage"
D = ("d_eb_structured", "d_eb_pairwise_rank")
MODELS = (*A, B, C, *D)
DRAWS = 500


def inputs(args, *, calibration=False):
    prior, source_stamp = verify_forecast(args.prior)
    root = args.registration
    stamp = read_json(root / "registration.json")
    if (stamp["forecast_sha256"], stamp["contract_sha256"], source_stamp["provenance"]["train_data_sha256"]) != (
            source_stamp["sha256"], sha256(args.contract), TRAIN_SHA):
        raise ValueError("3D registration does not match the frozen 2D priors and scoring rules")
    if calibration:
        # This file contains only the declared K patients for each repeat.
        rel = f"{args.study}/k{args.k}-calibration-only.parquet"
        outcomes = pd.read_parquet(verified(root / rel, stamp["artifacts"][rel]))
    else:
        rel = f"{args.study}/test-outcomes.parquet"
        outcomes = pd.read_parquet(verified(root / rel, stamp["artifacts"][rel]))
    rel = f"{args.study}/k{args.k}-layout.json"
    layout = read_json(verified(root / rel, stamp["artifacts"][rel]))["layouts"]
    return pivot_prior(prior, args.study), outcomes, layout, source_stamp


def pivot_prior(prior, study):
    frame = prior[prior.track.eq(study)].pivot(index=["lineage_group", "cell_id", "compound_id"],
                                              columns="method", values="predicted_value").reset_index()
    return patient_table(frame, ["drug_mean", "structured", "pairwise_rank"])


def drug_prior(means, chem):
    d = chem - chem.mean()
    slope = max(0., float(np.dot(d, means - means.mean()) / max(1e-12, d @ d)))
    return means.mean() + slope * d, slope


def row_center(m):
    return m - m.mean(axis=1, keepdims=True)


def local_predict(y, fit_vol, held_vol, known_mask, source_mean):
    """Fit empirical Bayes drug shrinkage and one nonnegative coefficient per fixed RNA model."""
    k = len(y)
    b = y.mean(axis=0)
    c = b.copy()
    supported = y[:, known_mask]
    eta, slope = drug_prior(b[known_mask], source_mean)
    sigma2 = float(np.mean(np.var(supported, axis=0, ddof=1)))
    discrepancy = b[known_mask] - eta
    noise = sigma2 / k
    tau2 = max(float(np.mean(discrepancy**2)) - noise, 0.)
    reliability = tau2 / (tau2 + noise) if (tau2 + noise) > 0 else 1.0
    c[known_mask] = eta + reliability * discrepancy
    result = {B: np.tile(b, (held_vol.shape[1], 1)), C: np.tile(c, (held_vol.shape[1], 1))}
    parameters = {"k": k, "slope": slope, "pooled_noise_variance": sigma2,
                  "between_drug_variance": tau2, "local_mean_weight": reliability}
    for i, (zero, adjusted) in enumerate(zip(A, D)):
        avg = fit_vol[i].mean(axis=0)
        centered = row_center(fit_vol[i] - avg)
        held_centered = row_center(held_vol[i] - avg)
        weighted = float(np.sum(centered * (supported - c[known_mask])))
        squared = float(np.sum(centered**2))
        gamma = max(0., weighted / squared) if squared > 1e-12 else 0.0
        values = result[C].copy()
        values[:, known_mask] += gamma * held_centered
        result[adjusted] = values
        if known_mask.all():
            result[zero] = held_vol[i]
        parameters[f"gamma_{adjusted}"] = gamma
    return result, parameters


def predict(args):
    patient_scores, calibration, layouts, source_stamp = inputs(args, calibration=True)
    source = {name: pivot(patient_scores, name) for name in ("drug_mean", "structured", "pairwise_rank")}
    chemistry = source["drug_mean"]
    if not chemistry.nunique().eq(1).all():
        raise ValueError("The source compound anchor must be constant within drug")
    compounds = pd.Index(sorted(calibration.compound_id.unique()))
    known = compounds.isin(chemistry.columns)
    panels = ["exact68"] + (["full_local"] if not known.all() else [])
    result, fits = [], []
    for event in layouts:
        repeat, fit, held = event["repeat"], event["fit"], event["held"]
        if len(fit) != args.k or set(fit) & set(held) or set(fit) | set(held) != set(chemistry.index):
            raise ValueError("Each episode requires distinct calibration and held patient sets on identical support")
        labels = pivot(calibration[calibration.repeat.eq(repeat)], "endpoint_value")
        if set(labels.index) != set(fit):
            raise ValueError("Calibration-only rows contain the wrong patient's labels")
        fit_p = np.stack([source[name].loc[fit, compounds[known]].to_numpy() for name in ("structured", "pairwise_rank")])
        held_p = np.stack([source[name].loc[held, compounds[known]].to_numpy() for name in ("structured", "pairwise_rank")])
        anchor = chemistry.loc[fit[0], compounds[known]].to_numpy()
        for panel in panels:
            exact = panel == "exact68"
            drug_axis = compounds[known] if exact else compounds
            support = np.ones(len(drug_axis), dtype=bool) if exact else known
            scores, info = local_predict(labels.loc[fit, drug_axis].to_numpy(), fit_p, held_p, support, anchor)
            track = f"{args.study}/{panel}/K{args.k}"
            fits.append({"track": track, "repeat": repeat, **info})
            index = pd.MultiIndex.from_product([held, drug_axis], names=["lineage_group", "compound_id"])
            keys = index.to_frame(index=False)
            keys["row_id"] = keys.lineage_group + "|" + keys.compound_id
            keys["cell_id"] = keys.lineage_group
            for name, values in scores.items():
                result.append(keys.assign(track=track, fold=repeat, method=name, predicted_value=values.ravel()))
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "calibration_parameters.tsv", pd.DataFrame(fits))
    freeze_table(pd.concat(result, ignore_index=True), args.out / "forecasts.parquet", provenance={
        "patient_aggregation": "mean of ordinal-oriented AUC or AAC across PDO regions",
        "source_sha256": source_stamp["sha256"], "registration_sha256": sha256(args.registration / "registration.json"),
        "study": args.study, "k": args.k, "repeats": REPEATS,
        "parameters_sha256": sha256(args.out / "calibration_parameters.tsv"),
    })


def cross_bootstrap(y, forecasts, splits, k_precision, *, seed=57214):
    """Shared cohort-level draws preserve overlapping held donors among the 100 split episodes."""
    rng = np.random.default_rng(seed)
    patients = rng.integers(len(y), size=(DRAWS, len(y)))
    compounds = rng.integers(y.shape[1], size=(DRAWS, y.shape[1]))
    interest = (B, C, *D)
    results = {name: {key: np.zeros(DRAWS) for key in (*AXES, *RESIDUAL)} for name in interest}
    counts = {name: {key: np.zeros(DRAWS, dtype=int) for key in (*AXES, *RESIDUAL)} for name in interest}
    # Test membership is filtered separately for each episode, with the same original resample
    # across all episodes and methods. Conditions with zero valid observations remain undefined.
    for event, values in zip(splits, forecasts):
        reverse = np.full(len(y), -1, dtype=int)
        reverse[event] = np.arange(len(event))
        positions = reverse[patients]
        sizes = (positions >= 0).sum(axis=1)
        sorted_positions = np.sort(positions, axis=1)[:, ::-1]
        for size in np.unique(sizes[sizes >= 2]):
            use = np.flatnonzero(sizes == size)
            block = sorted_positions[use, :size, None]
            chosen = compounds[use, None, :]
            observed = y[event][block, chosen]
            for name in interest:
                # Outcomes that are constant after repeated donor resampling produce undefined correlations.
                measures = metrics(observed, values[name][block, chosen], precision_k=k_precision, min_lineages=2)
                for key, vector in measures.items():
                    valid = np.isfinite(vector)
                    results[name][key][use] += np.where(valid, vector, 0.)
                    counts[name][key][use] += valid
    return {name: {key: np.divide(value, counts[name][key], out=np.full(DRAWS, np.nan),
                                  where=counts[name][key] > 0) for key, value in data.items()}
            for name, data in results.items()}


def evaluate(args):
    _, labels, layouts, source = inputs(args)
    forecast, stamp = verify_forecast(args.forecast / "forecasts.parquet")
    meta = stamp["provenance"]
    if (meta["source_sha256"], meta["registration_sha256"], meta["study"], meta["k"], meta["repeats"]) != (
            source["sha256"], sha256(args.registration / "registration.json"), args.study, args.k, REPEATS):
        raise ValueError("Scoring receipt differs from forecasts")
    verified(args.forecast / "calibration_parameters.tsv", meta["parameters_sha256"])
    rows, intervals, episodes = [], [], []
    for track, data in forecast.groupby("track"):
        is_exact = "/exact68/" in track
        names = MODELS if is_exact else (B, C, *D)
        compounds = pd.Index(sorted(data.compound_id.unique()))
        y = pivot(labels, "endpoint_value").loc[:, compounds]
        k_precision = 3 if args.study == "Yang" else 5
        arrays, splits, raw = [], [], []
        for event in layouts:
            repeat, held = event["repeat"], event["held"]
            frame = data[data.fold.eq(repeat)]
            if (set(frame.lineage_group) != set(held) or set(frame.method) != set(names)
                or len(frame) != len(held) * len(compounds) * len(names)
                or frame.duplicated(["lineage_group", "compound_id", "method"]).any()):
                raise ValueError("Each calibration budget and held roster must match the frozen registration")
            per_model = {}
            for model, part in frame.groupby("method"):
                p = pivot(part, "predicted_value").loc[held, compounds].to_numpy()
                q = y.loc[held].to_numpy()
                if not np.isfinite(p).all() or set(held) & set(event["fit"]):
                    raise ValueError("Nonfinite or fit and test donors overlap")
                per_model[model] = p
                info = metrics(q, p, precision_k=k_precision, min_lineages=2)
                top1 = mean(within_cell(q, p, 5, 1)[1])
                raw.append({"model_id": model, "repeat": repeat, **info, "precision_at_1": top1})
            splits.append(y.index.get_indexer(held))
            arrays.append(per_model)
        scores = pd.DataFrame(raw)
        macro = scores.groupby("model_id").mean(numeric_only=True)
        for model, m in macro.drop(columns="repeat").iterrows():
            rows.append({"study": args.study, "track": track, "model_id": model, "k": args.k,
                         "k_labels": 0 if model in A else args.k, "patient_n": len(y),
                         "held_patients_each": len(y) - args.k, "drugs": len(compounds),
                         "precision_k": k_precision, "repeats": REPEATS, **m.to_dict()})
        for m in raw:
            episodes.append({"track": track, **m})
        boot = cross_bootstrap(y.to_numpy(), arrays, splits, k_precision)
        for name in D:
            for base in (B, C):
                for metric in (*AXES, *RESIDUAL):
                    values = boot[name][metric] - boot[base][metric]
                    lo, hi = interval(values)
                    intervals.append({"study": args.study, "track": track, "k": args.k, "model_id": name,
                                      "baseline": base, "metric": metric,
                                      "delta": macro.loc[name, metric] - macro.loc[base, metric],
                                      "ci95_low": lo, "ci95_high": hi, "bootstrap_nonmissing": np.isfinite(values).sum()})
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "metrics.tsv", pd.DataFrame(rows))
    save(args.out / "paired.tsv", pd.DataFrame(intervals))
    save(args.out / "episodes.tsv", pd.DataFrame(episodes))
    publish_json(args.out / "score_receipt.json", {
        "predictions_sha256": stamp["sha256"], "registration_sha256": meta["registration_sha256"],
        "study": args.study, "k": args.k, "repeats": REPEATS, "paired_bootstrap_draws": DRAWS,
        "files": {p.name: sha256(p) for p in args.out.iterdir()},
    })


def transfer(args):
    prior, receipt = verify_forecast(args.prior)
    registration = read_json(args.registration / "registration.json")
    if registration["forecast_sha256"] != receipt["sha256"]:
        raise ValueError("Transfer input altered")
    drugs = sorted(DRUGS)
    sorafenib = next(k for k, v in DRUGS.items() if v == "Sorafenib")
    j = drugs.index(sorafenib)
    arrays = {}
    for study in STUDIES:
        label_key = f"{study}/test-outcomes.parquet"
        patients = pd.read_parquet(verified(args.registration / label_key, registration["artifacts"][label_key]))
        y = pivot(patients, "endpoint_value").loc[:, drugs]
        source = pivot_prior(prior, study)
        p = np.stack([pivot(source, name).loc[y.index, drugs].to_numpy() for name in ("structured", "pairwise_rank")])
        anchor = pivot(source, "drug_mean").loc[y.index, drugs].iloc[0].to_numpy()
        arrays[study] = y, p, anchor
    args.out.mkdir(parents=True, exist_ok=False)
    frozen, transfers = [], []

    def freeze_panel(fit, test, y, drug_axis, results, panel):
        index = pd.MultiIndex.from_product([y.index, drug_axis], names=["lineage_group", "compound_id"])
        keys = index.to_frame(index=False)
        keys["row_id"] = keys.lineage_group + "|" + keys.compound_id
        keys["cell_id"] = keys.lineage_group
        for name, value in results.items():
            frozen.append(keys.assign(track=f"{fit}/{test}/{panel}", fold=0, method=name,
                                      predicted_value=value.ravel()))

    for fit, (y_cal, p_cal, anchor) in arrays.items():
        for test, (y, p, _) in arrays.items():
            if fit == test:
                continue
            results, _ = local_predict(y_cal.to_numpy(), p_cal, p, np.ones(len(drugs), dtype=bool), anchor)
            freeze_panel(fit, test, y, drugs, results, "common_3_exact_TKIs")
            common_sorafenib = {name: v[:, [j]] for name, v in results.items()}
            freeze_panel(fit, test, y, [sorafenib], common_sorafenib, "sorafenib_only_from_common_3_fit")

            # Fit the single-drug analysis from the fit study's sorafenib values alone.
            y_s = y_cal.to_numpy()[:, j]
            center = y_s.mean()
            mono = {name: common_sorafenib[name] for name in A}
            mono.update({name: np.full((len(y), 1), center) for name in (B, C)})
            for i, name in enumerate(D):
                avg = p_cal[i, :, j].mean()
                x = p_cal[i, :, j] - avg
                gamma = max(0., float(x @ (y_s - center) / max(1e-12, x @ x)))
                mono[name] = (center + gamma * (p[i, :, j] - avg))[:, None]
            freeze_panel(fit, test, y, [sorafenib], mono, "sorafenib_only_direct_single_drug")

    path = args.out / "transfer_forecasts.parquet"
    seal = freeze_table(pd.concat(frozen, ignore_index=True), path, provenance={
        "fit_inputs_sha256": registration["artifacts"], "source_sha256": receipt["sha256"],
    })
    prediction, _ = verify_forecast(path)
    for (track, name), part in prediction.groupby(["track", "method"]):
        fit, test, panel = track.split("/")
        labels, _, _ = arrays[test]
        p = pivot(part, "predicted_value")
        y = labels.loc[p.index, p.columns].to_numpy()
        p = p.to_numpy()
        if panel == "common_3_exact_TKIs":
            m = metrics(y, p, precision_k=1, min_lineages=2)
            values = {"ndcg_at_3": m["ndcg_at_5"], "precision_at_1": m["precision_at_k"],
                      **{k: m[k] for k in (*AXES[:2], "within_cell_spearman", *RESIDUAL)}}
        else:
            values = {"within_drug_spearman": association(y[:, 0], p[:, 0], axis=0, rank=True).item(),
                      "within_drug_pearson": association(y[:, 0], p[:, 0], axis=0).item()}
        transfers.append({"fit_study": fit, "test_study": test, "fit_patients": len(arrays[fit][0]),
                          "test_patients": len(y), "unit": "patient", "panel": panel, "drugs": p.shape[1],
                          "model_id": name, **values})
    save(args.out / "TRANSFER_MATCHED_PATIENT_PANELS.tsv", pd.DataFrame(transfers))
    publish_json(args.out / "receipt.json", {"prediction_sha256": seal["sha256"], "cross_study_directions": 6,
                 "patient_ids_and_units": "unique donor only", "table_sha256": sha256(
                     args.out / "TRANSFER_MATCHED_PATIENT_PANELS.tsv")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "score", "transfer"))
    parser.add_argument("--study", choices=STUDIES)
    parser.add_argument("--k", type=int, choices=BUDGETS)
    for name in ("contract", "prior", "registration", "out", "forecast"):
        parser.add_argument("--" + name, type=Path, required=name not in ("forecast",))
    args = parser.parse_args()
    {"predict": predict, "score": evaluate, "transfer": transfer}[args.stage](args)


if __name__ == "__main__":
    main()
