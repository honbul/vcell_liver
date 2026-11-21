"""Freeze T2 predictions and compare the candidate to the concurrent minimum VC benchmark."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY
from .t1_data import DOSES, MIN_CELLS, PROTOCOL_SHA256, slice_name, validate_protocol
from .t1_io import load_task, score_truth
from .t1_score import ScoreBook, comparisons, registry


REFERENCE = "proposed-v0"
BASELINES = ("B0", "B_abs", "B_pos", "B_pos-all", REFERENCE)


def dose_cases(dose):
    for cells in MIN_CELLS:
        for fold in range(3):
            for role in "AB":
                yield cells, fold, role


def verify_true_binding(t1_dir, stage, dose):
    freeze = t1_dir / "freeze.json"
    manifest = read_json(freeze)
    if (manifest["protocol_sha256"] != PROTOCOL_SHA256 or manifest["dose"] != dose
            or manifest["stage_sha256"] != sha256(stage / "receipt.json")):
        raise ValueError("T1 forecast parent receipt is from a different study slice")
    if str((t1_dir / "predictions").resolve()) != manifest["predictions"]:
        raise ValueError("T1 forecast storage changed")
    if set(BASELINES).difference(manifest["models"]):
        raise ValueError("T1 prediction baselines are incomplete")
    return freeze, manifest


def root_models(raw):
    result = {}
    for text in raw:
        name, sep, path = text.partition("=")
        if not sep or not name or not path or name in (*BASELINES, *result):
            raise ValueError("Provide each distinct known model as NAME=PATH")
        result[name] = Path(path).resolve()
    return result


def execution_hashes(stage):
    return dict(stage_sha256=sha256(stage / "receipt.json"),
                protocol_sha256=PROTOCOL_SHA256,
                scorer_sha256=sha256(__file__),
                shared_scorer_sha256=sha256(Path(__file__).with_name("t1_score.py")),
                loader_sha256=sha256(Path(__file__).with_name("t1_io.py")))


def freeze(args):
    validate_protocol(args.protocol)
    known = root_models(args.model)
    if not known:
        raise ValueError("No known comparator paths were registered")
    t1_freeze, t1_manifest = verify_true_binding(args.t1, args.stage, args.dose)
    frozen = {}
    for n, fold, role in dose_cases(args.dose):
        key = slice_name(args.dose, n, fold, role)
        paths = {name: root / key for name, root in known.items()}
        anchor = args.t1 / "predictions" / key
        frozen[key] = {name: sha256(path / "receipt.json") for name, path in paths.items()}
        frozen[key]["T1"] = sha256(anchor / "receipt.json")
        if t1_manifest["tasks"][key] != frozen[key]["T1"]:
            raise ValueError(f"T1 slice changed from initial freeze: {key}")
        _, rows, panel, _ = load_task(anchor, args.stage, args.protocol, args.dose, n, fold, role)
        for name, path in paths.items():
            receipt, ours, tokens, arrays = load_task(path, args.stage, args.protocol, args.dose, n, fold, role)
            if receipt["models"] != [name] or name not in arrays:
                raise ValueError(f"Unexpected submitted arm at {path}: {receipt['models']}")
            if not rows[KEY + [CONTROL, GROUP]].equals(ours[KEY + [CONTROL, GROUP]]) or not np.array_equal(panel, tokens):
                raise ValueError(f"Known VC baseline used a mismatched cohort or panel at {path}")
        print(f"frozen {key} with {len(known)} known comparators", flush=True)

    write_json(args.freeze, dict(
        execution_hashes(args.stage), eligible_known={k: str(v) for k, v in sorted(known.items())},
        t1_root=str(args.t1.resolve()), t1_freeze_sha256=sha256(t1_freeze), dose=args.dose, tasks=frozen,
        eligibility=("Parent-disjoint fresh fit or response-label-free frozen features, common physical wells, "
                     "role and panel. Scoring never chooses whether a comparison model is eligible."),
    ))


def compare_minimum(points, boot, known):
    result = []
    for stratum, subset in points.groupby("stratum", sort=False):
        measures = subset.set_index("arm")
        eligible = sorted(known)
        if set(eligible).difference(measures.index):
            raise ValueError(f"An eligible known arm lacks common rows in {stratum}")
        best = min(eligible, key=lambda arm: (measures.at[arm, "contrast_rmse"], arm))
        draw = np.minimum.reduce([boot[stratum, arm] for arm in eligible])
        lo, hi = np.quantile(boot[stratum, REFERENCE] - draw, (.005, .995))
        metadata = measures.loc[REFERENCE]
        formal = stratum == "liver" and metadata.dose_uM in (0.5, 5.0) and metadata.minimum_cells == 20
        rmse = float(measures.at[REFERENCE, "contrast_rmse"])
        baseline = float(measures.at[best, "contrast_rmse"])
        result.append(dict(dose_uM=metadata.dose_uM, minimum_cells=metadata.minimum_cells,
                           stratum=stratum, parents=metadata.parent_groups, rows_roles=metadata.rows_roles,
                           eligible_models=len(eligible), best_known=best,
                           proposed_contrast_rmse=rmse, best_known_contrast_rmse=baseline,
                           difference_rmse=rmse-baseline, ci99_low=lo, ci99_high=hi,
                           upper_below_zero=bool(hi < 0), formal_test=bool(formal),
                           tested_verdict=("pass" if hi < 0 else "fail") if formal else "descriptive"))
    return pd.DataFrame(result)


def score(args):
    validate_protocol(args.protocol)
    frozen = read_json(args.freeze)
    for key, expect in execution_hashes(args.stage).items():
        if frozen[key] != expect:
            raise ValueError(f"T2 freeze has changed execution dependency {key}")
    if args.dose != frozen["dose"]:
        raise ValueError("Dose differs from freeze")
    known = {name: Path(path) for name, path in frozen["eligible_known"].items()}
    t1 = Path(frozen["t1_root"])
    if sha256(t1 / "freeze.json") != frozen["t1_freeze_sha256"]:
        raise ValueError("T1 reference freeze changed after T2 freeze")
    groups = registry(args.stage)
    args.out.mkdir(parents=True, exist_ok=False)
    points_out, pair_out, minimum_out = [], [], []
    for n in MIN_CELLS:
        book = ScoreBook(args.dose, n, groups)
        checked_fold_roles = set()
        for fold in range(3):
            for role in "AB":
                key = slice_name(args.dose, n, fold, role)
                match = frozen["tasks"][key]
                anchor = t1 / "predictions" / key
                if sha256(anchor / "receipt.json") != match["T1"]:
                    raise ValueError("T1 receipt changed after T2 freeze")
                _, rows, panel, arrays = load_task(anchor, args.stage, args.protocol, args.dose, n, fold, role)
                forecasts = {arm: arrays[arm] for arm in BASELINES}
                for arm, root in known.items():
                    path = root / key
                    if sha256(path / "receipt.json") != match[arm]:
                        raise ValueError(f"T2 receipt changed after freeze: {arm} {key}")
                    _, other_rows, other_panel, result = load_task(
                        path, args.stage, args.protocol, args.dose, n, fold, role)
                    if not rows[KEY].equals(other_rows[KEY]) or not np.array_equal(panel, other_panel):
                        raise ValueError(f"T2 alignment changed after freeze: {arm} {key}")
                    forecasts[arm] = result[arm]
                observed = score_truth(args.stage, args.dose, n, fold, role, rows, panel,
                                       verify=(fold, role) not in checked_fold_roles)
                checked_fold_roles.add((fold, role))
                book.add(rows, observed, forecasts)
                print(f"scored {key}", flush=True)
        points, _, boot = book.summary()
        points["eligible_known_vc"] = points.arm.isin(known)
        pairs = comparisons(points, boot, ["B0", "B_pos-all", *sorted(known)])
        minimum = compare_minimum(points, boot, known)
        points_out.append(points)
        pair_out.append(pairs)
        minimum_out.append(minimum)
    pd.concat(points_out).to_csv(args.out / "benchmark.tsv", sep="\t", index=False)
    pd.concat(pair_out).to_csv(args.out / "paired.tsv", sep="\t", index=False)
    pd.concat(minimum_out).to_csv(args.out / "against_best.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(freeze_sha256=sha256(args.freeze),
               files={p.name: sha256(p) for p in sorted(args.out.iterdir())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for key in ("protocol", "stage", "freeze"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--dose", type=float, choices=DOSES, required=True)
    parser.add_argument("--t1", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    (freeze if args.action == "freeze" else score)(args)


if __name__ == "__main__":
    main()
