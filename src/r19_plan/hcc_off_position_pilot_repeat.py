"""Quantify independent-position repeated response and paired forecast error in the pilot."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256
from src.r15_lincs.design import fold_id
from src.r19_plan.hcc_off_position_pilot_native_score import native_predictions
from src.r19_plan.hcc_off_position_pilot_score import measurements, prepared, quantile, write_stage
from src.r19_plan.hcc_pilot_layout import LINES, ROLES
from src.r19_plan.hcc_pilot_score import DRAWS, vehicle_contrasts
from src.r19_plan.jhh5_off_position import CALIBRATED
from src.r19_plan.jhh7_off_position_native import REFERENCES


RAW, FITTED = "direct_repeat", "crossfit_repeat_gain"
FIT_MIN = 10


def directed_pairs(targets):
    pairs = []
    for idx in targets.groupby(["cell", "canonical_parent"], sort=True).indices.values():
        group = targets.iloc[idx]
        if not all(group[k].is_unique for k in ("plate_id", "culture_id", "independent_start_id", "well")):
            raise ValueError("Two observations are not separate physical and culture repeats")
        pairs.extend((src, target) for target in idx for src in idx if src != target)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def weights_by_target(pair):
    target = pair[:, 1]
    return 1./np.bincount(target)[target]


def normalized_pairs(targets, target, pair_weight):
    parent, labels = pd.factorize(targets.canonical_parent.iloc[target], sort=True)
    normalized = pair_weight/np.bincount(parent, weights=pair_weight)[parent]
    return parent, labels, normalized


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan, dtype=float), where=denominator>0)


def crossfit(targets, pair, source, truth, base):
    src, dest = pair.T
    slots = targets.iloc[dest].parent_slot
    folds = slots.map(lambda x: fold_id(x, "R19-pilot-repeat-193501", 3)).to_numpy()
    parent, labels, effective = normalized_pairs(targets, dest, base)
    dot = np.mean(source*truth, axis=(0,2))
    energy = np.mean(source**2, axis=(0,2))
    scale, fits = np.empty(len(pair)), []
    for k in sorted(set(folds)):
        fit = folds != k
        eval_parents, fit_parents = np.unique(parent[~fit]), np.unique(parent[fit])
        fit_energy = np.dot(effective[fit], energy[fit])
        held_ids, fit_ids = set(labels[eval_parents]), set(labels[fit_parents])
        if held_ids & fit_ids:
            raise ValueError("One chemical parent crossed repeat calibration folds")
        gain = float(np.clip(np.dot(effective[fit], dot[fit])/fit_energy, 0, 2)) if len(fit_ids)>=FIT_MIN and fit_energy>0 else np.nan
        scale[~fit] = gain
        fits.append(dict(fold=int(k), fit_parents=len(fit_ids), held_parents=len(held_ids), gain=gain))
    return (source*scale[None,:,None] if np.isfinite(scale).all() else None), fits


def metrics(y, source, estimates):
    errors = np.stack([np.mean(np.square(y-value), axis=2) for value in estimates], axis=2)
    rmse, sse = np.sqrt(errors).mean(axis=0), errors.mean(axis=0)
    energy = np.square(y).mean(axis=(0,2))
    dot = np.sum(y*source, axis=2)
    norm = np.linalg.norm(y, axis=2)*np.linalg.norm(source, axis=2)
    cosine = divide(dot, norm)
    valid = np.isfinite(cosine)
    agreement = np.where(valid, cosine, 0.).sum(axis=0)
    agreement_count = valid.sum(axis=0)
    repeat_noise = .5*np.square(y-source).mean(axis=(0,2))
    return rmse, sse, energy, agreement, agreement_count, repeat_noise


def distributions(targets, pair, pair_weight, stats, crossed, draws):
    """Use shared independent culture weights at both endpoints of each repeat pair."""
    src, dest = pair.T
    parent, labels, _ = normalized_pairs(targets, dest, pair_weight)
    culture, levels = pd.factorize(targets.culture_id, sort=True)
    one_hot = np.eye(len(labels))[parent]
    rmse, sse, energy, similarity, n_similar, noise = stats

    def calculate(p, cultures):
        blocks = cultures[:, culture[src]]*cultures[:, culture[dest]]
        within = pair_weight[None,:]*blocks
        denominator = within @ one_hot
        pooled = p[:, parent]*within
        macro = pooled/denominator[:, parent]
        macro /= macro.sum(axis=1, keepdims=True)
        r2 = 1-divide(pooled @ sse, (pooled @ energy)[:,None])
        cosine = divide(macro @ similarity, macro @ n_similar)
        return macro @ rmse, r2, np.column_stack((cosine, np.sqrt(macro @ noise)))

    point = calculate(np.ones((1,len(labels))), np.ones((1,len(levels))))
    rng = np.random.default_rng(193501)
    distribution = [[] for _ in point]
    for start in range(0,draws,128):
        count = min(128, draws-start)
        p = rng.exponential(size=(count,len(labels)))
        blocks = rng.exponential(size=(count,len(levels))) if crossed else np.ones((count,len(levels)))
        for bucket, sample in zip(distribution, calculate(p,blocks), strict=True):
            bucket.append(sample)
    return tuple(v[0] for v in point), tuple(np.concatenate(v) for v in distribution)


def score_tables(names, context, center, draws):
    model, pairs, repeat = [], [], []
    for metric, point, samples, choose in zip(("RMSE","pooled_R2"), center[:2], draws[:2], (np.argmin,np.argmax), strict=True):
        for i, name in enumerate(names):
            values = samples[:,i]
            lo, hi = quantile(values)
            model.append(dict(context, metric=metric, arm=name, point=point[i], low99=lo, high99=hi,
                              valid_draws=int(np.isfinite(values).sum())))
        references = [names.index(r) for r in REFERENCES]
        best_i = references[int(choose(point[references]))]
        op = np.min if metric == "RMSE" else np.max
        best = op(samples[:,references], axis=1)
        pairs_to_score = [(CALIBRATED, "B0"), (RAW,"B0"), (CALIBRATED,RAW), (CALIBRATED,"best_VC")]
        if FITTED in names:
            pairs_to_score.extend(((FITTED,"B0"), (FITTED,RAW), (CALIBRATED,FITTED)))
        for arm, comparator in pairs_to_score:
            candidate = names.index(arm)
            baseline = best_i if comparator=="best_VC" else names.index(comparator)
            baseline_draw = best if comparator=="best_VC" else samples[:,baseline]
            delta = samples[:,candidate]-baseline_draw
            lo, hi = quantile(delta)
            relative = (1-divide(samples[:,candidate], baseline_draw)) if metric=="RMSE" else np.array([])
            rlo, rhi = quantile(relative)
            pairs.append(dict(context, metric=metric, arm=arm, reference=comparator, reference_at_point=names[baseline],
                arm_value=point[candidate], reference_value=point[baseline], difference=point[candidate]-point[baseline],
                low99=lo, high99=hi, relative_gain=1-point[candidate]/point[baseline] if metric=="RMSE" and point[baseline]>0 else np.nan,
                relative_low99=rlo, relative_high99=rhi, valid_draws=int(np.isfinite(delta).sum())))
    for i, metric in enumerate(("repeat_cosine","repeat_disagreement_rms")):
        values=draws[2][:,i]
        lo, hi=quantile(values)
        repeat.append(dict(context, metric=metric, estimate=center[2][i], low99=lo, high99=hi,
                           valid_draws=int(np.isfinite(values).sum())))
    return model,pairs,repeat


def evaluate(targets, truth, forecasts, paired, draws=DRAWS):
    pairs = directed_pairs(targets)
    expected = np.unique(pairs[:,1])
    if not np.array_equal(expected,np.flatnonzero(paired)):
        raise ValueError("The enrolled repeat support differs from the independently locked physical assay")
    table_names = ("support","fitted_gains","points","paired","repeat_agreement")
    output = {name:[] for name in table_names}
    for cell in LINES:
        pair = pairs[targets.cell.iloc[pairs[:,1]].eq(cell).to_numpy()]
        if not len(pair):
            output["support"].append(dict(cell=cell, parents=0, wells=0, directed_pairs=0, undirected_pairs=0,
                different_batch_pairs=0, cultures=0, roles=len(ROLES), calibration_available=False))
            continue
        src, dest = pair.T
        source, y = truth[:,src], truth[:,dest]
        base = weights_by_target(pair)
        fitted, gains = crossfit(targets,pair,source,y,base)
        models = {"B0":np.zeros_like(y), **{arm:forecasts[arm][:,dest] for arm in (CALIBRATED,*REFERENCES)}, RAW:source}
        if fitted is not None:
            models[FITTED]=fitted
        stats = metrics(y, source, list(models.values()))
        tgt = targets.iloc[np.unique(dest)]
        batches = targets.batch_id.to_numpy()
        info = dict(cell=cell, parents=tgt.canonical_parent.nunique(), wells=len(tgt), directed_pairs=len(pair),
            undirected_pairs=len(pair)//2, different_batch_pairs=int((batches[src]!=batches[dest]).sum())//2,
            cultures=tgt.culture_id.nunique(), roles=len(ROLES), calibration_available=fitted is not None)
        output["support"].append(info)
        output["fitted_gains"].extend(dict(cell=cell, **record) for record in gains)
        for label, cross in (("parent",False),("parent_and_both_cultures",True)):
            center, sample = distributions(targets,pair,base,stats,cross,draws)
            contents = score_tables(list(models),dict(info,resampling=label),center,sample)
            for name, values in zip(table_names[2:],contents,strict=True):
                output[name].extend(values)
    return {name:pd.DataFrame(content) for name,content in output.items()}


def run(args):
    _, accepted, controls, targets, paired, _, genes, forecasts, proof = prepared(args)
    receipt = args.intake/"receipt.json"
    locked = read_json(receipt)
    if (sha256(receipt)!=args.intake_sha256 or {k:v for k,v in locked.items() if k!="files"}!=proof
            or any(sha256(args.intake/n)!=d for n,d in locked["files"].items())):
        raise ValueError("Repeat scoring requires the original locked assay and prediction metadata")
    forecasts.update(native_predictions(args,proof["forecast_sha256"],genes,targets))
    expression = measurements(args.measurements,proof["measurements_sha256"],accepted,genes)
    truth = vehicle_contrasts(expression,accepted,controls,targets)
    result = evaluate(targets,truth,forecasts,paired)
    write_stage(args.output,dict(proof,expression_read=True,actual_measurement_used=not args.synthetic,
        independent_confirmation=False,intake_sha256=args.intake_sha256,native_freeze_sha256=sha256(args.native_pin),
        repeat_protocol_sha256=sha256(args.repeat_protocol),repeat_scorer_sha256=sha256(__file__),
        draws=DRAWS), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("audit","forecast","forecast-pin","native","native-pin","native-protocol","wells","cultures","qc-record",
                  "assay-receipt","protocol","intake","measurements","repeat-protocol","output"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--intake-sha256",required=True)
    parser.add_argument("--synthetic",action="store_true")
    run(parser.parse_args())
