"""Seal R14 predictions, then score the unchanged SNU-423 contrast endpoint."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.stage import load_staged
from src.r13_optimization.objectives import center, row_products
from src.r13_optimization.score import draw_contrast

from .transfer import C3A, MIN, PROTOCOL, SNU, checked, eligible, files, own_control


PRIMARY = ("B0", "B_pos", "integrated", "matched_D0", "integrated_nopos")


def freeze(args):
    root = args.work / "predict" / args.split
    receipts = {}
    for fold in range(3):
        roles = []
        for j in range(2):
            name = f"task{2*fold+j}"
            folder = root / name
            path = folder / "receipt.json"
            record = read_json(path)
            if record["protocol_sha256"] != PROTOCOL:
                raise ValueError("R14 task uses another execution plan")
            for source, lock in record["implementation"].items():
                if sha256(Path(__file__).with_name(source)) != lock:
                    raise ValueError(f"R14 fit used another implementation: {source}")
            for item in record["files"]:
                checked(folder, record, item)
            roles.append((table(folder / "rows.tsv"), np.load(folder / "panel.npy")))
            receipts[name] = sha256(path)
        (a, raw), (b, other) = roles
        if not a[KEY + [GROUP]].equals(b[KEY + [GROUP]]) or a[CONTROL].eq(b[CONTROL]).any() or not np.array_equal(raw, other):
            raise ValueError("R14 opposing DMSO roles disagree")
    write_json(args.work / f"freeze-{args.split}.json",
               dict(protocol_sha256=PROTOCOL, split=args.split, task_receipts=receipts,
                    score_code_sha256=sha256(__file__)))


def read_freeze(args):
    path = args.work / f"freeze-{args.split}.json"
    frozen = read_json(path)
    if (frozen["protocol_sha256"], frozen["score_code_sha256"]) != (PROTOCOL, sha256(__file__)):
        raise ValueError("Scorer changed after freeze")
    root = args.work / "predict" / args.split
    for name, lock in frozen["task_receipts"].items():
        folder = root / name
        path = folder / "receipt.json"
        if sha256(path) != lock:
            raise ValueError("Prediction receipt changed after freeze")
        rec = read_json(path)
        for item in rec["files"]:
            checked(folder, rec, item)
    return root, path, frozen


def point_metrics(rows, observed, absolute, own, pred):
    metric = Contrast(rows, MIN)
    true, centered = center(metric, observed), center(metric, pred)
    rs = np.sqrt(row_products(centered - true, centered - true))
    norms = np.sqrt(row_products(centered, centered) * row_products(true, true))
    valid = norms > 1e-14
    dot = row_products(true, centered)
    cosine = np.divide(dot, norms, out=np.zeros_like(norms), where=valid)
    absolute_rmse = np.sqrt(row_products(pred + own - absolute, pred + own - absolute))
    return dict(contrast=float(metric.primary_weight @ rs),
                absolute=float(group_weights(rows) @ absolute_rmse),
                cosine=float(metric.primary_weight @ cosine),
                cosine_support=float(metric.primary_weight @ valid))


def difference_table(points, draws, mass, groups):
    available = set(points.index)
    comparisons = [("integrated", "B0"), ("integrated", "matched_D0"),
                   ("B_pos", "B0"), ("integrated_nopos", "B0"), ("integrated_nopos", "matched_D0")]
    results = []
    for arm, base in comparisons:
        if not {arm, base} <= available:
            continue
        lo, hi = np.quantile((draws[arm] - draws[base]) / mass, [0.005, 0.995])
        results.append(dict(arm=arm, baseline=base, difference=float(points.loc[arm,"contrast"] - points.loc[base,"contrast"]),
                            lower=float(lo), upper=float(hi), groups=groups, confidence=.99))
    return results


def score(args):
    root, _, frozen = read_freeze(args)
    stage = args.assets / ("staged" if args.split == "compound" else "staged-scaffold")
    rec = read_json(stage / "receipt.json")
    for name in rec["files"]:
        if "/score-" in name:
            checked(stage, rec, name)
    all_rows = {name: table(root / name / "rows.tsv") for name in frozen["task_receipts"]}
    universe = sorted(set().union(*(set(rows.loc[rows[LINE].eq(SNU), GROUP]) for rows in all_rows.values())))
    if args.split == "scaffold":
        component = table(args.assets / "scaffold/group_scaffolds.tsv").set_index(GROUP).scaffold_component
        labels = sorted(set(component.loc[universe]))
    else:
        component, labels = pd.Series(universe, index=universe), universe
    draws = np.random.default_rng(130113).exponential(size=(10_000, len(labels)))
    draws /= draws.mean(axis=1, keepdims=True)
    boot = defaultdict(lambda: np.zeros(len(draws)))
    mass = np.zeros(len(draws))
    records, by_similarity = [], []
    for name, allrow in sorted(all_rows.items()):
        dest = root / name
        receipt = read_json(dest / "receipt.json")
        if receipt["inputs"]["stage_sha256"] != sha256(stage / "receipt.json"):
            raise ValueError("Held-out stage snapshot mismatch")
        fold, role = int(name[4:]) // 2, "AB"[int(name[4:]) % 2]
        bg, counts = load_staged(stage / f"outer{fold}", "score", rec["genes"])
        panel = np.load(dest / "panel.npy")
        own = own_control(args.r11, role, allrow, panel)
        scored_wells = counts[align(allrow, bg)][:, panel].toarray()
        opposite = own_control(args.r11, "B" if role == "A" else "A", allrow, panel)
        target = scored_wells - opposite
        for label, mask in (("SNU-423", allrow[LINE].eq(SNU)), ("HepG2/C3A", allrow[LINE].eq(C3A)),
                            ("non-liver", ~allrow[LINE].isin((SNU, C3A)))):
            active = eligible(allrow.loc[mask].reset_index(drop=True))
            if active.empty:
                continue
            index = align(active, allrow)
            if label == "SNU-423":
                group_index = pd.Index(labels).get_indexer(component.loc[active[GROUP]])
                metric = Contrast(active, MIN)
                mass += draws @ np.bincount(group_index, weights=metric.primary_weight, minlength=len(labels))
            weight = float(group_weights(active).sum())
            for arm, file in receipt["predictions"].items():
                pred = np.load(dest / file, mmap_mode="r", allow_pickle=False)[index]
                measured = point_metrics(active, target[index], scored_wells[index], own[index], pred)
                metrics = {key: value / weight for key, value in measured.items()}
                records.append(dict(arm=arm, stratum=label, task=name, rows=len(active), group_mass=weight, **metrics))
                if label == "SNU-423" and arm in PRIMARY:
                    boot[arm] += draw_contrast(metric, pred, target[index], draws, group_index)
                if label == "SNU-423" and arm == "integrated":
                    feature = receipt["selected"]["integrated"] + "_similarity"
                    calc = np.sqrt(row_products(center(metric, pred - target[index]), center(metric, pred - target[index])))
                    for lo, hi in ((0, .25), (.25, .50), (.50, 1.01)):
                        take = active[feature].between(lo, hi, inclusive="left")
                        by_similarity.append(dict(task=name, low=lo, high=hi, groups=int(active.loc[take, GROUP].nunique()),
                                                  rows=int(take.sum()), rmse=float(np.mean(calc[take])) if take.any() else None))
        print(f"scored frozen {args.split} {name}", flush=True)
    raw = pd.DataFrame(records)
    groups = []
    for (stratum, arm), sub in raw.groupby(["stratum", "arm"]):
        metrics = {key: np.average(sub[key], weights=sub.group_mass) for key in ("contrast", "absolute", "cosine", "cosine_support")}
        groups.append(dict(stratum=stratum, arm=arm, **metrics))
    table_out = pd.DataFrame(groups)
    primary = table_out.loc[table_out.stratum.eq("SNU-423")].set_index("arm")
    pairs = difference_table(primary, boot, mass, len(labels))
    primary_gates = [x for x in pairs if x["arm"] == "integrated"]
    gate = len(primary_gates) == 2 and all(x["upper"] < 0 for x in primary_gates)
    out = args.work / f"score-{args.split}"
    out.mkdir(parents=True, exist_ok=False)
    raw.to_csv(out / "by_fold_role.tsv", sep="\t", index=False)
    table_out.to_csv(out / "arms.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired.tsv", sep="\t", index=False)
    pd.DataFrame(by_similarity).to_csv(out / "similarity.tsv", sep="\t", index=False)
    compare = primary.reset_index()[["arm", "contrast"]].rename(columns={"contrast": "contrast_rmse"})
    compare["rmse_reduction_percent"] = 100 * (1 - compare.contrast_rmse / primary.loc["B0", "contrast"])
    compare["groups"] = len(universe)
    compare["reference"] = "R14 SNU-423 study rows"
    paired = pd.DataFrame(pairs)
    if len(paired):
        against_zero = paired.loc[paired.baseline.eq("B0"), ["arm", "lower", "upper"]]
        compare = compare.merge(against_zero.rename(columns={"lower": "delta_ci99_low", "upper": "delta_ci99_high"}),
                                how="left", on="arm")
    compare = pd.concat([compare, pd.DataFrame([dict(arm="different_position_repeat_0.25",
                 groups=12, rmse_reduction_percent=3.47, reduction_ci95_low=-2.58, reduction_ci95_high=9.32,
                 reference="Prior step 3 combined endpoint; different rows and panel")])], ignore_index=True)
    compare.to_csv(out / "primary_comparison.tsv", sep="\t", index=False)
    pd.DataFrame([dict(task=name, arm=arm, same_position_rows=coverage["same_position_rows"],
                       sources=coverage["sources"], rows_by_source=coverage["rows_by_source"])
                  for name in sorted(all_rows)
                  for arm, coverage in read_json(root / name / "receipt.json")["inputs"]["coverage"].items()]
                 ).to_csv(out / "source_coverage.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(protocol_sha256=PROTOCOL, groups=len(universe), independent_units=len(labels),
                                         decision="improves" if gate else "criterion_not_met",
                                         independent_repeat_reference=dict(groups=12, percent=3.47, ci95=[-2.58, 9.32]),
                                         freeze_sha256=sha256(args.work / f"freeze-{args.split}.json"), files=files(out)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--split", choices=("compound", "scaffold"), required=True)
    for key in ("work", "assets", "r11", "protocol"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("R14 protocol differs from the committed scope")
    (freeze if args.action == "freeze" else score)(args)


if __name__ == "__main__":
    main()
