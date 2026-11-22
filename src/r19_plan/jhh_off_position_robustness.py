"""Attribute the fixed adult HCC gains to parents and test chemical and physical clustering."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_transfer import effect
from src.r19_plan.i_score import BOOTSTRAPS, checked_array, checked_rows, squared_rows
from src.r19_plan.jhh5_off_position import CALIBRATED, SCALAR, STATE
from src.r19_plan.jhh5_off_position_native import collect, references as jhh5
from src.r19_plan.jhh7_donor_transfer import read_plan
from src.r19_plan.jhh7_off_position_native import MODELS, REFERENCES, best_comparison, references as jhh7
from src.r19_plan.jhh_low_dose_transfer import load_plan
from src.r19_plan.t4_workflow import axes


COMPARED = (CALIBRATED, *REFERENCES, "B0")
SCOPE = {"JHH5": (305, 309, 6), "JHH7": (88, 88, 1)}
METRICS = ("relative_rmse_gain", "delta_r2")
REFERENCES_TO_SCORE = ("best_VC", "B0")


def verified_score(args):
    freeze = args.run / "freeze.json"
    receipt_file = args.run / "score/receipt.json"
    best = args.run / "score/best.tsv"
    record = read_json(receipt_file)
    pinned = read_json(args.native_pin)
    if (sha256(freeze) != sha256(args.native_pin) or sha256(receipt_file) != sha256(args.score_pin)
            or sha256(freeze) != record["freeze_sha256"]
            or sha256(best) != record["files"]["best.tsv"]
            or any(record[k] != value for k, value in pinned.items())):
        raise ValueError("The original public benchmark freeze or score changed")
    return pinned, pd.read_csv(best, sep="\t")


def jhh7_forecasts(args, old, ref, genes, frozen, native):
    queries, pieces = {}, {}
    for role in "ABC":
        query = read_plan(old, ref, "predict", role)["query"]
        source = args.previous / "forecasts" / role
        receipt = source / "receipt.json"
        array = source / "predictions.npz"
        if sha256(receipt) != frozen["roles"][role] or sha256(array) != read_json(receipt)["array_sha256"]:
            raise ValueError("Original JHH-7 forecast changed")
        shape = (len(query), len(genes))
        with np.load(array, allow_pickle=False) as data:
            values = {arm: checked_array(data[arm], shape, arm) for arm in (SCALAR, STATE, CALIBRATED)}
        for model in MODELS:
            folder = args.run / "forecasts" / role / model
            receipt = folder / "receipt.json"
            array = folder / "prediction.npy"
            if sha256(receipt) != native["roles"][role][model] or sha256(array) != read_json(receipt)["output_sha256"]:
                raise ValueError("An adapted JHH-7 forecast changed")
            values[model] = checked_array(np.load(array, allow_pickle=False), shape, model)
        queries[role] = query
        pieces[role] = values
    rows = checked_rows(pd.concat([q.assign(role=r) for r, q in queries.items()], ignore_index=True), None)
    return queries, rows, pieces


def frozen_vectors(args):
    pinned, best = verified_score(args)
    is_jhh5 = args.cell == "JHH5"
    original_args = argparse.Namespace(**(vars(args) | {"protocol": args.native_protocol}))
    old, ref, genes, cols, base, stamp = (jhh5 if is_jhh5 else jhh7)(original_args)
    if is_jhh5:
        queries, rows, pieces, _, manifest = collect(args, old, ref, genes, base, stamp)
        if manifest != pinned:
            raise ValueError("Reconstructed JHH-5 model lineup differs")
    else:
        if any(pinned[k] != value for k, value in stamp.items()):
            raise ValueError("JHH-7 forecast lineage changed")
        queries, rows, pieces = jhh7_forecasts(args, old, ref, genes, base, pinned)
    support = rows.parent.nunique(), rows.inst_id.nunique(), rows.rna_plate.nunique()
    if (support != SCOPE[args.cell] or not rows.groupby("inst_id").role.nunique().eq(3).all()
            or not rows.cell_id.eq(args.cell).all() or not rows.actual_dose_uM.eq("10").all()
            or not rows.b1_seen.eq("True").all()):
        raise ValueError("The original registered physical comparison support changed")
    if any(not np.array_equal(d[SCALAR], d["PRnet-well-adaptation"]) for d in pieces.values()):
        raise ValueError("The rank reference no longer equals the fixed common fitted scalar")
    # Open the existing score-only partitions after the complete model lineup has been checked.
    truth = []
    for key, query in queries.items():
        if is_jhh5:
            tables = load_plan(old, "score", key, ref["slices"][key]["score"])
            target, controls = tables["query.tsv"], tables["score_controls.tsv"]
            role = axes(key)[3]
        else:
            tables = read_plan(old, ref, "score", key)
            target, controls = tables["query"], tables["controls"]
            role = "ABC"[("ABC".index(key)+2)%3]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("Recorded target coordinates differ from the fixed prediction query")
        truth.append(effect(target, controls, ref, genes, cols, role))
    truth = np.concatenate(truth)
    arrays = {arm: np.concatenate([d[arm] for d in pieces.values()]) for arm in COMPARED[:-1]}
    arrays["B0"] = np.zeros_like(truth)
    replay, _ = best_comparison(rows, truth, arrays, "all")
    replay = pd.DataFrame(replay)
    fields = ["arm_value", "ref_value", "difference", "low99", "high99"]
    np.testing.assert_allclose(best[fields], replay[fields], atol=1e-12, rtol=0)
    return rows, truth, arrays


def frozen_comparison(args):
    rows, truth, arrays = frozen_vectors(args)
    return rows, truth.shape[1], squared_rows(truth), np.column_stack([squared_rows(truth, arrays[a]) for a in COMPARED])


def moments(rows, genes, energy, errors):
    labels = rows[["parent", "scaffold"]].drop_duplicates().sort_values("parent")
    if labels.parent.duplicated().any() or labels.scaffold.str.strip().eq("").any():
        raise ValueError("The original scaffold assignment needs one nonempty key per parent")
    plate_index, plates = pd.factorize(rows.rna_plate, sort=True)
    parent_index, parents = pd.factorize(rows.parent, sort=True)
    np.testing.assert_array_equal(parents, labels.parent)
    flat = parent_index*len(plates)+plate_index

    def reduce(values=None):
        return np.bincount(flat, weights=values, minlength=len(parents)*len(plates)).reshape(len(parents), len(plates))

    counts, exposure = reduce(), reduce(energy)
    rmse_sum = np.stack([reduce(np.sqrt(e/genes)) for e in errors.T])
    sse = np.stack([reduce(e) for e in errors.T])
    return labels, counts, exposure, rmse_sum, sse


def gains(rmse, r2):
    return np.stack((
        np.column_stack((1-rmse[:, 0]/rmse[:, 1:-1].min(axis=1), r2[:, 0]-r2[:, 1:-1].max(axis=1))),
        np.column_stack((1-rmse[:, 0]/rmse[:, -1], r2[:, 0]-r2[:, -1])),
    ), axis=1)


def weighted_metrics(weights, rmse, sse, energy):
    global_rmse = (weights @ rmse.T)/weights.sum(axis=1, keepdims=True)
    global_r2 = 1-(weights @ sse.T)/(weights @ energy)[:, None]
    return gains(global_rmse, global_r2)


def cluster_samples(rng, group, count, exposure, rmse_sum, sse, *, crossed):
    n_scaffolds = int(group.max()+1)
    samples = []
    for start in range(0, BOOTSTRAPS, 128):
        n = min(128, BOOTSTRAPS-start)
        if crossed:
            weights = rng.exponential(size=(n, n_scaffolds))[:, group]
            plate = rng.exponential(size=(n, count.shape[1]))
            norm = plate @ count.T
            rmse = np.column_stack([np.sum(weights*(plate @ a.T)/norm, axis=1)/weights.sum(axis=1)
                                   for a in rmse_sum])
            r2 = 1-np.column_stack([np.sum(weights*(plate @ a.T), axis=1) for a in sse])/(
                np.sum(weights*(plate @ exposure.T), axis=1)[:, None])
            samples.append(gains(rmse, r2))
        else:
            drawn = rng.integers(n_scaffolds, size=(n, n_scaffolds))
            weights = np.stack([np.bincount(d, minlength=n_scaffolds) for d in drawn])[:, group]
            samples.append(weighted_metrics(weights, rmse_sum.sum(axis=2)/count.sum(axis=1),
                sse.sum(axis=2), exposure.sum(axis=1)))
    combined = np.concatenate(samples)
    if not np.isfinite(combined).all() or len(combined) != BOOTSTRAPS:
        raise ValueError("Invalid clustered metric samples")
    return np.quantile(combined, [.005, .995], axis=0)


def results(rows, genes, energy, errors):
    labels, count, exposure, rmse_sum, sse = moments(rows, genes, energy, errors)
    rmse, squared, power = rmse_sum.sum(axis=2)/count.sum(axis=1), sse.sum(axis=2), exposure.sum(axis=1)
    group, scaffolds = pd.factorize(labels.scaffold, sort=True)
    all_weights = np.ones((1, len(labels)))
    point = weighted_metrics(all_weights, rmse, squared, power)[0]
    by_parent, distribution, removed = [], [], []
    for rank_reference, index, compare_index in (("scalar", COMPARED.index("PRnet-well-adaptation"), 0), ("B0", -1, 1)):
        for metric, per_parent in (("rmse_decrease", rmse[index]-rmse[0]), ("sse_decrease", squared[index]-squared[0])):
            positive = np.flatnonzero(per_parent > 0)
            ordered = positive[np.argsort(-per_parent[positive], kind="stable")]
            distribution.append(dict(rank_reference=rank_reference, metric=metric, parents=len(labels),
                improved_parents=len(positive), worse_parents=int(np.count_nonzero(per_parent<0)),
                median_decrease=float(np.median(per_parent))))
            by_parent.extend(dict(rank_reference=rank_reference, metric=metric, parent=p, scaffold=s, decrease=d)
                for p, s, d in zip(labels.parent, labels.scaffold, per_parent, strict=True))
            for n in (5, 10, 20):
                omit = ordered[:n]
                weights = all_weights.copy()
                weights[:, omit] = 0
                gain = weighted_metrics(weights, rmse, squared, power)[0, compare_index]
                removed.append(dict(rank_reference=rank_reference, metric=metric, omitted=len(omit),
                    retained=len(labels)-len(omit), reference=REFERENCES_TO_SCORE[compare_index],
                    share_positive=per_parent[omit].sum()/per_parent[positive].sum() if len(positive) else np.nan,
                    **dict(zip(METRICS, gain)), practical_points=bool(np.all(gain >= .005))))
    intervals = []
    designs = [("scaffold", False)]
    if count.shape[1] > 1:
        designs.append(("scaffold_x_plate", True))
    for name, crossed in designs:
        bounds = cluster_samples(np.random.default_rng(150104), group, count, exposure, rmse_sum, sse, crossed=crossed)
        for i, ref in enumerate(REFERENCES_TO_SCORE):
            for m, metric in enumerate(METRICS):
                lo, hi = bounds[:, i, m]
                intervals.append(dict(design=name, reference=ref, metric=metric, point=point[i,m],
                    low99=lo, high99=hi, parents=len(labels), scaffolds=len(scaffolds), plates=count.shape[1], draws=BOOTSTRAPS))
    return dict(distribution=distribution, parents=by_parent, top_removed=removed, clustered=intervals)


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    diagnostics = results(*frozen_comparison(args))
    with TemporaryDirectory(dir=args.out.parent, prefix=".robustness-") as work:
        stage = Path(work)
        for name, rows in diagnostics.items():
            pd.DataFrame(rows).to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(cell=args.cell, original_score_replayed=True,
            original_freeze_sha256=sha256(args.native_pin), original_score_sha256=sha256(args.score_pin),
            protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            files={p.name:sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=tuple(SCOPE), required=True)
    for field in ("run", "previous", "previous-protocol", "previous-pin", "access-pin", "native-pin", "native-protocol", "score-pin", "protocol", "out"):
        parser.add_argument("--"+field, required=True, type=Path)
    run(parser.parse_args())
