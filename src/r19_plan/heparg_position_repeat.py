"""Score incremental chemistry and repeat effects over frozen HepaRG location anchors."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.heparg import DATASETS, PRIMARY, active_rows, cache_receipts, peers, read_source, response, verify
from src.goal_tracks.heparg_models import PERM_SEEDS, strata
from src.p123.p3_run import T3_FREEZE
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import checked_array, paired_effects, parent_summaries, squared_rows

LABELS = ("ge10uM", "rank7", "rank8")
LOCATION = "B_context_plus_position"
CHEMISTRY = "joint_dose_response"
REPEAT = "repeat_residual_x0.25"
ANCHORS = (LOCATION, "B_pos", "B_context", *(f"B_context_plus_perm{s}" for s in PERM_SEEDS))


def bind(args):
    original = args.t3 / "freeze.json"
    if sha256(original) != T3_FREEZE or read_json(args.t3_score)["frozen_sha256"] != T3_FREEZE:
        raise ValueError("Expected the original scored T3 freeze")
    freeze = read_json(original)
    if freeze["inputs"] != cache_receipts(args):
        raise ValueError("The original HepaRG caches changed")
    parts = {}
    for fold in range(3):
        folder = args.t3 / "predict" / f"{args.dataset}-{fold}"
        stamp = folder / "receipt.json"
        digest = sha256(stamp)
        if freeze["part_receipts"][f"{folder.name}/receipt.json"] != digest:
            raise ValueError("Original prediction receipt changed")
        files = read_json(stamp)["files"]
        verify(folder, {name: files[name] for name in ("keys.npz", *(f"{label}.npz" for label in LABELS))})
        parts[folder.name] = digest
    return dict(dataset=args.dataset, protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                source_sha256=T3_FREEZE, source_score_sha256=sha256(args.t3_score), parts=parts)


def repeat_predict(rows, observed, dataset):
    receiver, donors, weight = peers(rows, dataset)
    out = np.zeros_like(observed)
    np.add.at(out, receiver, weight[:, None] * observed[donors])
    return out, np.unique(receiver)


def effect_moments(rows, residual, prediction, fold):
    energy = squared_rows(residual)
    error = squared_rows(residual, prediction)
    parent = rows.rename(columns={"parent_key": "parent"})
    rmse, sse, denominator = parent_summaries(parent, residual.shape[1], energy, error, energy)
    return rows.parent_key.unique(), rmse, sse, denominator, fold, len(rows)


def mean_direction(rows, y, p):
    denom = np.linalg.norm(y, axis=1) * np.linalg.norm(p, axis=1)
    dot = np.einsum("ij,ij->i", y, p)
    cosine = np.divide(dot, denom, out=np.full(len(rows), np.nan), where=denom > 1e-12)
    return pd.Series(cosine).groupby(rows.parent_key.to_numpy()).mean()


def score_fold(source, dataset, folder, fold, effects, directions, support):
    query = active_rows(source, [fold])
    with np.load(folder / "keys.npz", allow_pickle=False) as keys:
        if not np.array_equal(keys["index"], query.index) or not query.outer_fold.eq(fold).all():
            raise ValueError("Original held row axis changed")
        panel = pd.Index(keys["genes"])
        if len(panel) != 2000 or not panel.is_unique:
            raise ValueError("Original fit-only 2,000-gene panel changed")
    observed = response(source, query, panel)
    for label in LABELS:
        keep = strata(query)[label]
        rows = query.loc[keep].reset_index(drop=True)
        y = observed[keep]
        with np.load(folder / f"{label}.npz", allow_pickle=False) as archive:
            if not np.array_equal(np.flatnonzero(keep), archive["row"]):
                raise ValueError("The scored stratum row axis changed")
            model = checked_array(archive[CHEMISTRY], y.shape, CHEMISTRY)
            for anchor in ANCHORS:
                value = checked_array(archive[anchor], y.shape, anchor)
                coverage = archive[f"support__{anchor}"]
                if coverage.dtype != bool or coverage.shape != (len(rows),):
                    raise ValueError("Original location support has the wrong axis")
                residual = y - value
                key = (label, anchor, "all", CHEMISTRY)
                effects.setdefault(key, []).append(effect_moments(rows, residual, model, fold))
                support.append(dict(stratum=label, anchor=anchor, fold=fold, parents=rows.parent_key.nunique(),
                                    wells=len(rows), anchor_supported_wells=int(coverage.sum()),
                                    parents_with_anchor_support=rows.loc[coverage, "parent_key"].nunique()))
                if label != "ge10uM" or anchor != LOCATION:
                    continue
                repeat, idx = repeat_predict(rows, residual, dataset)
                if not len(idx):
                    continue
                common = rows.iloc[idx]
                for name, pred in ((CHEMISTRY, model), (REPEAT, repeat)):
                    key = (label, anchor, "repeat_supported", name)
                    effects.setdefault(key, []).append(effect_moments(common, residual[idx], pred[idx], fold))
                    directions.setdefault(key, []).append(mean_direction(common, residual[idx], pred[idx]))


def summarize(dataset, key, blocks, directions):
    parents, rmse, sse, energy, fold, wells = zip(*blocks, strict=True)
    ids = np.concatenate(parents)
    if np.unique(ids).size != ids.size:
        raise ValueError("A chemical parent was scored in multiple folds")
    count = dict(zip(fold, map(len, parents), strict=True))
    rmse, sse, energy = map(np.concatenate, (rmse, sse, energy))
    points, interval, _ = paired_effects(rmse, sse, energy)
    relative_lo, relative_hi, _ = relative_interval(rmse)
    relative = 1.0 - points[0, 0] / points[0, 1]
    label, anchor, scope, arm = key
    primary = dataset == PRIMARY and label == "ge10uM" and anchor == LOCATION and (scope == "all" or arm == REPEAT)
    eligible = ids.size >= 100 and all(count.get(index, 0) >= 20 for index in range(3))
    cosine = pd.concat(directions, verify_integrity=True).mean() if directions else np.nan
    return dict(dataset=dataset, stratum=label, anchor=anchor, scope=scope, arm=arm, primary=primary,
                eligible=eligible, passes=bool(primary and eligible and relative_lo > 0 and interval[1, 0] > 0),
                parents=ids.size, wells=sum(wells), **{f"fold{i}_parents": count.get(i, 0) for i in range(3)},
                rmse=points[0, 0], reference_rmse=points[0, 1], relative_rmse_gain=relative,
                relative_rmse_low99=relative_lo, relative_rmse_high99=relative_hi,
                residual_R2=points[1, 0], residual_R2_low99=interval[1, 0], residual_R2_high99=interval[1, 1],
                residual_parent_cosine=cosine)


def run(args):
    proof = bind(args)
    target = args.work / args.dataset
    freeze = target / "freeze.json"
    if args.stage == "lock":
        target.mkdir(exist_ok=False)
        write_json(freeze, proof)
        return
    if read_json(freeze) != proof or sha256(freeze) != args.expected_sha256:
        raise ValueError("Independent scoring requires an unchanged locked comparison")
    source = read_source(args, args.dataset)
    effects, directions, support = {}, {}, []
    for fold in range(3):
        score_fold(source, args.dataset, args.t3 / "predict" / f"{args.dataset}-{fold}", fold,
                   effects, directions, support)
        print(f"Scored {args.dataset} fold {fold}", flush=True)
    path = target / "score"
    if path.exists():
        raise FileExistsError(path)
    rows = [summarize(args.dataset, key, blocks, directions.get(key)) for key, blocks in effects.items()]
    with TemporaryDirectory(dir=target) as temp:
        product = Path(temp)
        pd.DataFrame(rows).to_csv(product / "paired.tsv", sep="\t", index=False, na_rep="NA")
        pd.DataFrame(support).assign(dataset=args.dataset).to_csv(product / "support.tsv", sep="\t", index=False)
        write_json(product / "receipt.json", dict(proof, freeze_sha256=args.expected_sha256,
                   files={p.name: sha256(p) for p in product.iterdir()}))
        product.rename(path)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("lock", "score"))
    cli.add_argument("--dataset", choices=DATASETS, required=True)
    for field in ("work", "protocol", "t3", "t3-score", "cache", "index"):
        cli.add_argument(f"--{field}", type=Path, required=True)
    cli.add_argument("--expected-sha256")
    run(cli.parse_args())
