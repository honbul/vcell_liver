"""R14 exploratory lung holdout with compound-disjoint training and scoring."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r12_multiaxis.data import select_panel
from src.r12_multiaxis.stage import load_staged
from src.r13_optimization.score import draw_contrast

from . import transfer_models
from .transfer import (
    DS, MIN, PROTOCOL, SNU, WELL, Assay, baseline, checked, eligible, files, label, merge,
    output_task, own_control, scoreable_mask, subsets, validation_loss,
)
from .transfer_models import ALPHAS, ARMS, GAINS, NEIGHBORS, predict_grid
from .transfer_score import point_metrics


PRIMARY = ("B0", "integrated", "matched_D0")


def lung_lines(path):
    meta = pd.read_parquet(path, columns=["Organ", "Cell_ID_Cellosaur", "Cell_ID_DepMap"]).drop_duplicates()
    lines = meta.loc[meta.Organ.astype(str).str.strip().str.casefold().eq("lung")]
    cvcl = set(lines.Cell_ID_Cellosaur.dropna())
    depmap = set(lines.Cell_ID_DepMap.dropna())
    if "CVCL_0023" not in cvcl or SNU in cvcl:
        raise ValueError("Unexpected Tahoe organ metadata")
    return cvcl | depmap, meta


def prediction(args):
    stage = args.assets / "staged"
    rec = read_json(stage / "receipt.json")
    fold, role = args.fold, args.role
    unit = f"outer{fold}"
    for name in ("fit-data.npy", "fit-indices.npy", "fit-indptr.npy", "fit-rows.tsv", "score-rows.tsv"):
        checked(stage, rec, f"{unit}/{name}")
    split_path = args.assets / "partition" / "split.json"
    inner_map = read_json(split_path)["inner"][str(fold)]
    book_path = args.positions / "positions.tsv"
    if sha256(book_path) != read_json(args.positions / "receipt.json")["positions_sha256"]:
        raise ValueError("Tahoe position manifest differs from receipt")
    book = table(book_path).set_index("sample")
    raw, counts = load_staged(stage / unit, "fit", rec["genes"])
    held = label(table(stage / unit / "score-rows.tsv"), book)
    training = label(raw, book)
    if set(training.parent_key) & set(held.parent_key):
        raise ValueError("Held-out parent appears in Tahoe training")
    held_lines, meta = lung_lines(args.line_metadata)
    unknown = set(training[LINE]) | set(held[LINE])
    unknown -= set(meta.Cell_ID_Cellosaur.dropna())
    if unknown:
        raise ValueError(f"Unmapped Tahoe cell lines: {unknown}")
    later = eligible(held.loc[held[LINE].isin(held_lines)]).assign(**{CONTROL: lambda x: x[f"control_{role}"]})
    train_mask = ~training[LINE].isin(held_lines).to_numpy()
    excluded_lung = int(np.count_nonzero(~train_mask))
    training = training.loc[train_mask].reset_index(drop=True)
    counts = counts[train_mask]
    training["inner"] = training[GROUP].map({int(k): int(v) for k, v in inner_map.items()})
    if training.inner.isna().any() or later.empty:
        raise ValueError("Insufficient organ holdout cohort")
    genes = table(args.r11 / "metadata" / "genes.tsv").set_index("token_id").ensembl_id
    assays = {key: Assay(args.data, key) for key in DS}
    mix = assays["mixseq"]
    reserve = mix.rows.loc[mix.external]
    masks, excluded_assays = {}, {}
    for key, assay in assays.items():
        keep = assay.donor & ~assay.rows.parent_key.isin(held.parent_key)
        lung_row = assay.rows[LINE].isin(held_lines)
        excluded_assays[key] = int((keep & lung_row).sum())
        keep &= ~lung_row
        if key == "mixseq":
            keep &= ~assay.rows.parent_key.isin(reserve.parent_key) & ~assay.rows[WELL].isin(reserve[WELL])
            known = set(meta.Cell_ID_Cellosaur.dropna()) | set(meta.Cell_ID_DepMap.dropna())
            keep &= assay.rows[LINE].isin(known)
        if key.startswith("heparg"):
            if not assay.rows.loc[assay.donor, "outer_fold"].isin((0, 1, 2)).all():
                raise ValueError("HepaRG donor lacks a shared parent fold")
            keep &= assay.rows.outer_fold.ne(fold)
        masks[key] = keep
    trials = {arm: {} for arm in ARMS}
    selected, final, coverage = {}, {}, {}
    for inner in (0, 1, None):
        take = training.inner.ne(inner).to_numpy() if inner is not None else np.ones(len(training), bool)
        learn = training.loc[take].reset_index(drop=True)
        query = scoreable_mask(training.loc[~take]) if inner is not None else later
        excluded = training.loc[~take] if inner is not None else held
        panel = select_panel(counts[take & training[LINE].eq(SNU).to_numpy()], 2000)
        own = own_control(args.r11, role, learn, panel)
        target_control = own_control(args.r11, role, query, panel)
        y = counts[take][:, panel].toarray() - own
        parts = [(learn, y, own.mean(axis=1))]
        parts.extend(x.select(masks[key], pd.Index(genes.loc[panel]))
                     for key, x in assays.items() if masks[key].any())
        combined = merge(parts)
        if inner is not None:
            opposite = own_control(args.r11, "B" if role == "A" else "A", query, panel)
            actual = counts[align(query, training)][:, panel].toarray() - opposite
        else:
            final = {"B0": np.zeros_like(target_control), "B_abs": baseline(learn, y, query)}
        for arm in ARMS:
            (ref, response, context), _ = subsets(combined, query, excluded, arm, None)
            if ref[LINE].isin(held_lines).any():
                raise ValueError("Lung line entered fitting or tuning")
            grid, _ = predict_grid(ref, query, response, context, target_control.mean(axis=1))
            if inner is not None:
                for key, loss in validation_loss(grid, actual, query).items():
                    trials[arm][key] = trials[arm].get(key, 0) + loss
            else:
                coverage[arm] = ref.groupby("source").parent_key.nunique().to_dict()
                for family, settings in (("ridge", ALPHAS), ("neighbor", NEIGHBORS)):
                    setting, g = min(((s, j) for s in settings for j in range(len(GAINS))),
                                     key=lambda item: trials[arm][family, item[0]][item[1]])
                    name = f"{arm}_{family}"
                    selected[name] = dict(method=family, setting=setting, gain=float(GAINS[g]),
                                          inner_loss=float(trials[arm][family, setting][g] / 2))
                    final[name] = grid[family, setting] * GAINS[g]
            print(f"Lung fold {fold}, control {role}, inner {inner}, {arm}", flush=True)
    winner = min((arm for arm in selected if not arm.startswith("D0")), key=lambda ar: selected[ar]["inner_loss"])
    matched = "D0_" + selected[winner]["method"]
    final.update(integrated=final[winner], matched_D0=final[matched])
    selected.update(integrated=winner, matched_D0=matched)
    output_task(args.work / "predict" / f"task{2 * fold + int(role == 'B')}", later, panel, final, selected,
                dict(organ="Lung", excluded_tahoe_lung_rows=excluded_lung, excluded_assay_lung_rows=excluded_assays,
                     effective_mixseq_donor_rows=int(masks["mixseq"].sum()), assay_receipts={k: a.hash for k, a in assays.items()},
                     line_metadata_sha256=sha256(args.line_metadata), coverage=coverage, split_sha256=sha256(split_path),
                     stage_sha256=sha256(stage / "receipt.json"), script_sha256=sha256(__file__),
                     model_sha256=sha256(transfer_models.__file__)))


def freeze(args):
    lock = {}
    for fold in range(3):
        slots = []
        for n in (2 * fold, 2 * fold + 1):
            task = args.work / "predict" / f"task{n}"
            receipts = read_json(task / "receipt.json")
            if receipts["protocol_sha256"] != PROTOCOL or receipts["inputs"]["script_sha256"] != sha256(__file__):
                raise ValueError("Lung task differs from frozen protocol or source")
            if receipts["inputs"]["model_sha256"] != sha256(transfer_models.__file__):
                raise ValueError("Lung model changed before freeze")
            for name in receipts["files"]:
                checked(task, receipts, name)
            slots.append((table(task / "rows.tsv"), np.load(task / "panel.npy", allow_pickle=False)))
            lock[task.name] = sha256(task / "receipt.json")
        (a, ga), (b, gb) = slots
        if not a[KEY + [GROUP]].equals(b[KEY + [GROUP]]) or a[CONTROL].eq(b[CONTROL]).any() or not np.array_equal(ga, gb):
            raise ValueError("Lung opposing control roles have mismatched wells or panels")
    write_json(args.work / "freeze.json", dict(protocol_sha256=PROTOCOL, task_receipts=lock, scorer_sha256=sha256(__file__)))


def score(args):
    frozen_path = args.work / "freeze.json"
    locked = read_json(frozen_path)
    if (locked["protocol_sha256"], locked["scorer_sha256"]) != (PROTOCOL, sha256(__file__)):
        raise ValueError("Lung scorer or protocol differs from freeze")
    stage = args.assets / "staged"
    stage_rec = read_json(stage / "receipt.json")
    for name in stage_rec["files"]:
        if "/score-" in name:
            checked(stage, stage_rec, name)
    by_task = {}
    for task, digest in locked["task_receipts"].items():
        pred = args.work / "predict" / task
        if sha256(pred / "receipt.json") != digest:
            raise ValueError("Lung prediction changed after freeze")
        rec = read_json(pred / "receipt.json")
        for name in rec["files"]:
            checked(pred, rec, name)
        if rec["inputs"]["stage_sha256"] != sha256(stage / "receipt.json"):
            raise ValueError("Lung staged assay differs from prediction")
        by_task[task] = (pred, rec, table(pred / "rows.tsv"))
    groups = pd.Index(sorted(set().union(*(set(rows[GROUP]) for _, _, rows in by_task.values()))))
    draws = np.random.default_rng(130113).exponential(size=(10_000, len(groups)))
    boot = defaultdict(lambda: np.zeros(len(draws)))
    mass = np.zeros(len(draws))
    records = []
    for task, (pred, rec, rows) in by_task.items():
        fold, role = int(task[4:]) // 2, "AB"[int(task[4:]) % 2]
        meta, counts = load_staged(stage / f"outer{fold}", "score", stage_rec["genes"])
        panel = np.load(pred / "panel.npy", allow_pickle=False)
        own = own_control(args.r11, role, rows, panel)
        absolute = counts[align(rows, meta)][:, panel].toarray()
        observed = absolute - own_control(args.r11, "B" if role == "A" else "A", rows, panel)
        metric = Contrast(rows, MIN)
        indices = groups.get_indexer(rows[GROUP])
        mass += draws @ np.bincount(indices, weights=metric.primary_weight, minlength=len(groups))
        parts = [("all_lung", np.arange(len(rows))), *rows.groupby(LINE).indices.items()]
        for arm, filename in rec["predictions"].items():
            p = np.load(pred / filename, mmap_mode="r", allow_pickle=False)
            if arm in PRIMARY:
                boot[arm] += draw_contrast(metric, p, observed, draws, indices)
            for line, ix in parts:
                block = rows.iloc[ix].reset_index(drop=True)
                weight = group_weights(block).sum()
                computed = point_metrics(block, observed[ix], absolute[ix], own[ix], p[ix])
                records.append(dict(arm=arm, line=line, task=task, rows=len(ix), groups=weight,
                                    **{key: val / weight for key, val in computed.items()}))
    table_out = pd.DataFrame(records)
    points = []
    cols = ["contrast", "absolute", "cosine", "cosine_support"]
    for (line, arm), cohort in table_out.groupby(["line", "arm"]):
        points.append(dict(line=line, arm=arm, **{m: np.average(cohort[m], weights=cohort.groups) for m in cols}))
    comparisons = []
    for other in ("B0", "matched_D0"):
        point = {r["arm"]: r["contrast"] for r in points if r["line"] == "all_lung"}
        lo, hi = np.quantile((boot["integrated"] - boot[other]) / mass, [0.025, 0.975])
        comparisons.append(dict(arm="integrated", baseline=other, groups=len(groups),
                                difference=point["integrated"] - point[other], lower=lo, upper=hi, confidence=.95))
    out = args.work / "score"
    out.mkdir(parents=True, exist_ok=False)
    table_out.to_csv(out / "by_fold_role.tsv", sep="\t", index=False)
    pd.DataFrame(points).to_csv(out / "arms.tsv", sep="\t", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(protocol_sha256=PROTOCOL, organ="Lung", groups=len(groups),
                                         freeze_sha256=sha256(frozen_path), files=files(out), exploratory=True))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "freeze", "score"))
    for name in ("work", "assets", "r11", "positions", "data", "protocol", "line-metadata"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=range(3))
    parser.add_argument("--role", choices=("A", "B"))
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("R14 organ protocol differs from prospective commitment")
    if args.phase == "predict" and (args.fold is None or args.role is None):
        parser.error("prediction requires --fold and --role")
    {"predict": prediction, "freeze": freeze, "score": score}[args.phase](args)


if __name__ == "__main__":
    main()
