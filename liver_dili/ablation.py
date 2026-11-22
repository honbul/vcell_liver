"""Paired connectivity-bootstrap intervals for frozen DILI modality contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import digest, receipt


DRAWS = 2000


def binary_metrics(y, p, predictions, weights):
    positives = weights @ y
    negatives = weights.sum(axis=1) - positives
    order = np.argsort(-p, kind="stable")
    edges = np.r_[np.flatnonzero(p[order][1:] != p[order][:-1]), len(y) - 1]
    tp = np.cumsum(weights[:, order] * y[order], axis=1)[:, edges]
    fp = np.cumsum(weights[:, order] * (1 - y[order]), axis=1)[:, edges]
    tpr, fpr = tp / positives[:, None], fp / negatives[:, None]
    zeros = np.zeros((len(weights), 1))
    delta_tpr = np.diff(tpr, prepend=zeros, axis=1)
    delta_fpr = np.diff(fpr, prepend=zeros, axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tpr), where=tp + fp > 0)
    tpr_previous = np.c_[zeros, tpr[:, :-1]]
    sensitivity = (weights @ (predictions * y)) / positives
    specificity = (weights @ ((1 - predictions) * (1 - y))) / negatives
    return dict(auroc=((tpr + tpr_previous) * delta_fpr / 2.).sum(axis=1),
                auprc=(precision * delta_tpr).sum(axis=1),
                balanced_accuracy=(sensitivity + specificity) / 2.,
                brier=(weights @ (p - y)**2) / weights.sum(axis=1))


def contrast(focal, anchor, seed):
    keys = ["probability", "decision"]
    joint = focal.join(anchor[keys], how="inner", rsuffix="_reference", validate="one_to_one")
    if joint.label.nunique() < 2 or len(joint) < 20:
        return []
    group, parents = pd.factorize(joint.connectivity, sort=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(parents), size=(DRAWS, len(parents)))
    mass = np.vstack([np.bincount(draw, minlength=len(parents)) for draw in draws])[:, group]
    y = joint.label.to_numpy()
    valid = (mass @ y > 0) & (mass @ (1 - y) > 0)
    all_weights = np.vstack([np.ones(len(y), dtype=int), mass[valid]])
    a = binary_metrics(y, joint.probability.to_numpy(), joint.decision.to_numpy(dtype=int), all_weights)
    b = binary_metrics(y, joint.probability_reference.to_numpy(),
                       joint.decision_reference.to_numpy(dtype=int), all_weights)
    result = []
    for name in a:
        values = a[name] - b[name]
        lo, hi = np.quantile(values[1:], [.005, .995])
        result.append(dict(metric=name, difference=values[0], low99=lo, high99=hi,
                           rows=len(y), parents=len(parents), draws=int(valid.sum())))
    return result


def evaluate(args):
    scores = json.loads((args.scored / "receipt.json").read_text())
    if scores["output_sha256"] != digest(args.scored / "metrics.tsv"):
        raise ValueError("Independent aggregate result changed")
    split_receipt = args.splits / "receipt.json"
    if scores["split_receipt"] != digest(split_receipt):
        raise ValueError("Different FDA split")
    source = json.loads(split_receipt.read_text())["files"]
    observed = []
    for fold, h in scores["forecast_freezes"].items():
        root = args.forecasts / fold
        freeze = root / "freeze.json"
        pred = root / "forecast.parquet"
        labels = args.splits / fold / "score_labels.parquet"
        query = args.splits / fold / "query.parquet"
        if (digest(freeze) != h or json.loads(freeze.read_text())["forecast_sha256"] != digest(pred)
                or source[f"{fold}/score_labels.parquet"] != digest(labels)
                or source[f"{fold}/query.parquet"] != digest(query)):
            raise ValueError(f"Changed forecast or label inputs for {fold}")
        observed.append(pd.read_parquet(pred).join(pd.read_parquet(labels), validate="many_to_one")
                        .join(pd.read_parquet(query)[["connectivity"]], validate="many_to_one")
                        .assign(split=fold.rsplit("-", 1)[0]))
    data = pd.concat(observed)
    records = []
    comparisons = (
        ("structure", "ecfp_properties", False), ("predicted_pathways", "structure", False),
        ("predicted_pathways", "pathway_shuffle", False), ("cell_painting", "structure", False),
        ("cell_painting", "structure", True), ("cell_painting", "painting_mask", True),
        ("cell_painting", "painting_shuffle", True), ("measured_rna", "structure", True),
    )
    for split, strata in data.groupby("split"):
        for method in ("logistic", "lightgbm"):
            by_method = strata.loc[strata.method.eq(method)]
            for name, ref, matched in comparisons:
                arm = by_method.loc[by_method["mode"].eq(name)]
                if matched:
                    arm = arm.loc[arm.measured_available]
                control = by_method.loc[by_method["mode"].eq(ref)]
                values = contrast(arm, control, seed=20261122)
                records.extend(dict(split=split, mode=name, reference=ref, method=method,
                                    support="matched" if matched else "all", **v) for v in values)
        nn = strata.loc[strata["mode"].eq("structure") & strata.method.eq("multitask")]
        lgbm = strata.loc[strata["mode"].eq("structure") & strata.method.eq("lightgbm")]
        records.extend(dict(split=split, mode="structure_multitask", method="multitask", reference="structure_lightgbm",
                            support="all", **v) for v in contrast(nn, lgbm, seed=20261122))
    args.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(args.out / "paired.tsv", sep="\t", index=False)
    receipt(args.out / "receipt.json", dict(scored_receipt=digest(args.scored / "receipt.json"),
        code_sha256=digest(Path(__file__)), output_sha256=digest(args.out / "paired.tsv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("splits", "forecasts", "scored", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    evaluate(parser.parse_args())
