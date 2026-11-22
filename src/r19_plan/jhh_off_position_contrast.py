"""Score within-plate drug profiles and retrieval from the existing adult HCC freezes."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r19_plan.i_score import BOOTSTRAPS
from src.r19_plan.jhh5_donor_attribution import wrong_parent
from src.r19_plan.jhh5_off_position import CALIBRATED, SCALAR, STATE, WRONG
from src.r19_plan.jhh7_off_position_native import REFERENCES, best_comparison
from src.r19_plan.jhh_off_position_robustness import SCOPE, frozen_vectors
from src.r19_plan.t4_position import POOL, compare, contrast


RANKED = (CALIBRATED, SCALAR, *REFERENCES, WRONG)
PAIR_REFERENCES = (SCALAR, STATE, WRONG)


def wrong_predictions(rows, forecast):
    shuffled = np.empty_like(forecast)
    for (fold, _, plate, _), ids in rows.groupby(POOL, sort=False).indices.items():
        query = rows.iloc[ids]
        parent_rows = query.drop_duplicates("parent")
        source_idx = pd.Series(parent_rows.index, index=parent_rows.parent)
        permutation = wrong_parent(query, fold, plate)
        if permutation.eq(query.parent).any() or set(permutation) != set(query.parent):
            raise ValueError("The wrong-drug permutation is not a fixed-point-free bijection")
        parent_prediction = forecast[source_idx.loc[query.parent].to_numpy()]
        np.testing.assert_array_equal(parent_prediction, forecast[ids])
        shuffled[ids] = forecast[source_idx.loc[permutation].to_numpy()]
    return shuffled


def retrieval_scores(hits):
    arm_results, paired = [], []
    for metric, data, measure in (
        ("cosine", hits.loc[hits.k.eq(1)], "cosine"),
        ("top1", hits.loc[hits.k.eq(1)], "top"),
        ("top5", hits.loc[hits.k.eq(5) & hits.pool_size.ge(6)], "top"),
    ):
        points = data.groupby(["parent", "arm"])[[measure, "chance"]].mean()
        values = points[measure].unstack().loc[:, list(RANKED)]
        rng = np.random.default_rng(38105)
        draws = rng.integers(len(values), size=(BOOTSTRAPS, len(values)))
        sampled = values.to_numpy()[draws].mean(axis=1)
        candidate = values[CALIBRATED].mean()
        for arm, column in zip(RANKED, sampled.T, strict=True):
            lo, hi = np.quantile(column, [.005, .995])
            arm_results.append(dict(metric=metric, parents=len(values), arm=arm, point=values[arm].mean(), low99=lo, high99=hi))
        baselines = {arm: values[arm].to_numpy() for arm in PAIR_REFERENCES}
        if metric != "cosine":
            chance = points.xs(CALIBRATED, level="arm").chance.reindex(values.index).to_numpy()
            baselines["chance"] = chance
            lo, hi = np.quantile(chance[draws].mean(axis=1), [.005, .995])
            arm_results.append(dict(metric=metric, parents=len(values), arm="chance", point=chance.mean(), low99=lo, high99=hi))
        refs = [RANKED.index(arm) for arm in REFERENCES]
        best = sampled[:, refs].max(axis=1)
        ref_mean = values[list(REFERENCES)].mean(axis=0).max()
        contrasts = [(arm, base.mean(), sampled[:, 0]-base[draws].mean(axis=1)) for arm, base in baselines.items()]
        contrasts.append(("best_VC", ref_mean, sampled[:, 0]-best))
        for arm, reference, diff in contrasts:
            lo, hi = np.quantile(diff, [.005, .995])
            paired.append(dict(metric=metric, parents=len(values), arm=CALIBRATED, reference=arm,
                               arm_value=candidate, reference_value=reference, difference=candidate-reference, low99=lo, high99=hi))
    return arm_results, paired


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, truth, arrays = frozen_vectors(args)
    arrays[WRONG] = wrong_predictions(rows, arrays[CALIBRATED])
    target, centered, hits = contrast(rows, truth, arrays, arms=RANKED)
    centered["B0"] = np.zeros_like(target)
    best, frequency = best_comparison(rows, target, centered, "all")
    pairs = tuple((CALIBRATED, r) for r in dict.fromkeys((SCALAR, *REFERENCES, WRONG, "B0")))
    retrieval, ranks_paired = retrieval_scores(hits)
    support = rows.groupby(POOL, sort=True).agg(parents=("parent", "nunique"), wells=("inst_id", "nunique")).reset_index()
    if len(support) != 9*SCOPE[args.cell][2]:
        raise ValueError("A role, fold, or plate is missing from the fixed score pools")
    tables = dict(centered_best=best, centered=compare(rows, target, centered, pairs, "all"), strongest_frequency=frequency,
                  retrieval=retrieval, retrieval_paired=ranks_paired, support=support)
    with TemporaryDirectory(dir=args.out.parent, prefix=".jhh-centered-") as temp:
        stage = Path(temp)
        for name, values in tables.items():
            pd.DataFrame(values).to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(cell=args.cell, absolute_public_scores_replayed=True,
            protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            original_freeze_sha256=sha256(args.native_pin), original_score_sha256=sha256(args.score_pin),
            files={p.name:sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=tuple(SCOPE), required=True)
    for key in ("run", "previous", "previous-protocol", "previous-pin", "access-pin", "native-pin", "native-protocol", "score-pin", "protocol", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    run(parser.parse_args())
