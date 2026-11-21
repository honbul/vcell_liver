"""Independent, compound-disjoint external R14 prediction and scoring."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r12_multiaxis.data import select_panel
from src.r12_multiaxis.stage import load_staged
from src.r13_optimization.objectives import center, row_products
from src.r13_optimization.score import draw_contrast

from .transfer import (
    DS, EXTERNAL, MIN, PROTOCOL, SNU, Assay, baseline, checked, eligible, files, label, merge,
    output_task, own_control, subsets, validation_loss,
)
from .transfer_models import ARMS, GAINS, predict_grid


def eligible_source(assay):
    row = assay.rows
    return (assay.external & row.parent_key.ne("") & row.smiles.ne("") &
            row.dose_uM.gt(0) & row.time_h.gt(0))


def tai(args, target):
    stage = args.assets / "staged"
    stage_rec = read_json(stage / "receipt.json")
    fold_path = stage / "outer0"
    records = [load_staged(fold_path, kind, stage_rec["genes"]) for kind in ("fit", "score")]
    book = table(args.positions / "positions.tsv").set_index("sample")
    from scipy.sparse import vstack
    rows = label(pd.concat([x[0] for x in records], ignore_index=True), book)
    matrix = vstack([x[1] for x in records], format="csr")
    assays = {k: Assay(args.data, k) for k in DS}
    mix = assays["mixseq"]
    hold_out = set(target.rows.loc[eligible_source(target), "parent_key"])
    reserved = set(mix.rows.loc[eligible_source(mix), "parent_key"]) | hold_out
    at = ~rows.parent_key.isin(reserved).to_numpy()
    rows, matrix = rows.loc[at].reset_index(drop=True), matrix[at]
    folds = read_json(args.assets / "partition/split.json")["outer"]
    rows["fold_id"] = rows[GROUP].map({int(k): int(f) for k, f in folds.items()})
    genes = table(args.r11 / "metadata/genes.tsv").set_index("token_id").ensembl_id
    supported = np.sort(genes[genes.isin(target.genes)].index.to_numpy())
    width = min(2000, len(supported))
    if width < 100:
        raise ValueError("Insufficient externally measured gene overlap")
    permitted = np.r_[[0, 1, 2], supported]
    return rows, matrix, assays, reserved, permitted, width, genes


def external_fit(args):
    target = Assay(args.data, args.name)
    eligible_rows = eligible_source(target)
    if not eligible_rows.any():
        raise ValueError(f"No eligible parent structure or treatment in {args.name}")
    rows, matrix, assays, reserved, allowed, width, genes = tai(args, target)
    scores = {arm: {} for arm in ARMS}
    predictions = {}
    coverage = {}
    selected = {}
    for k in (0, 1, 2, None):
        at = rows.fold_id.ne(k).to_numpy() if k is not None else np.ones(len(rows), dtype=bool)
        train = rows.loc[at].reset_index(drop=True)
        count = matrix[at]
        panel = allowed[select_panel(count[train[LINE].eq(SNU).to_numpy()][:, allowed], width)]
        names = pd.Index(genes.loc[panel])
        own = own_control(args.r11, "A", train, panel)
        delta = count[:, panel].toarray() - own
        parts = [(train, delta, own.mean(axis=1))]
        for key, assay in assays.items():
            mask = assay.donor & ~assay.rows.parent_key.isin(reserved)
            if key == "mixseq":
                mask &= ~assay.rows[KEY[0]].isin(assay.rows.loc[assay.external, KEY[0]])
            if mask.any():
                parts.append(assay.select(mask, names))
        if k is not None:
            holding = rows.loc[~at].reset_index(drop=True)
            mask = ~at & rows[LINE].eq(SNU).to_numpy()
            candidates = rows.loc[mask].reset_index(drop=True)
            candidates["origin"] = np.arange(len(candidates))
            query = eligible(candidates)
            c_query = own_control(args.r11, "A", query, panel).mean(axis=1)
            other = own_control(args.r11, "B", query, panel)
            actual = matrix[mask][query.origin.to_numpy()][:, panel].toarray() - other
        else:
            query, _, c_query = target.select(eligible_rows, names, responses=False)
            holding = query.assign(position=-1)
            query[GROUP] = query.parent_key
            query[ENTITY] = query.compound
            predictions["B0"] = np.zeros((len(query), len(panel)))
            predictions["B_abs"] = baseline(train.loc[train[LINE].eq(SNU)].reset_index(drop=True),
                                            delta[train[LINE].eq(SNU).to_numpy()], query)
        merged = merge(parts)
        for arm in ARMS:
            (ref, y, ctl), _ = subsets(merged, query, holding, arm, reserved)
            grid, _ = predict_grid(ref, query, y, ctl, c_query)
            if k is not None:
                for key, loss in validation_loss(grid, actual, query).items():
                    scores[arm][key] = scores[arm].get(key, 0) + loss
            else:
                coverage[arm] = ref.groupby("source").parent_key.nunique().to_dict()
                for family in ("ridge", "neighbor"):
                    setting, g = min(((key, i) for key in scores[arm] if key[0] == family for i in range(len(GAINS))),
                                     key=lambda x: scores[arm][x[0]][x[1]])
                    predictions[f"{arm}_{family}"] = grid[setting] * GAINS[g]
                    selected[f"{arm}_{family}"] = dict(setting=list(setting), gain=float(GAINS[g]),
                                                       inner_loss=float(scores[arm][setting][g] / 3))
        print(f"external candidate fit {args.name} holdout={k}", flush=True)
    chosen = min((arm for arm in selected if not arm.startswith("D0")), key=lambda a: selected[a]["inner_loss"])
    matched = "D0_" + chosen.split("_")[1]
    predictions.update(integrated=predictions[chosen], matched_D0=predictions[matched])
    selected["integrated"] = chosen
    selected["matched_D0"] = matched
    output_task(args.work / "external-predict" / args.name, query, names.to_numpy(dtype=str), predictions, selected,
                dict(target_receipt=target.hash, excluded_parent_groups=len(reserved), coverage=coverage,
                     held_out_treated_values_used_for_fit=False))


def external_freeze(args):
    saved = args.work / "external-predict" / args.name
    receipt = read_json(saved / "receipt.json")
    for key in receipt["files"]:
        checked(saved, receipt, key)
    prediction_hash = sha256(saved / "receipt.json")
    ref = {}
    if args.retry:
        original = args.work / f"freeze-external-{args.name}.json"
        if read_json(original)["prediction_receipt_sha256"] != prediction_hash:
            raise ValueError("Recovery requires the exact original frozen predictions")
        ref["original_freeze_sha256"] = sha256(original)
    write_json(args.work / f"freeze-external-{args.name}{args.retry}.json",
               dict(protocol_sha256=PROTOCOL, prediction_receipt_sha256=prediction_hash,
                    scorer_sha256=sha256(__file__), **ref))


def external_score(args):
    saved = args.work / "external-predict" / args.name
    frozen = args.work / f"freeze-external-{args.name}{args.retry}.json"
    freeze = read_json(frozen)
    rec = read_json(saved / "receipt.json")
    if (freeze["protocol_sha256"], freeze["prediction_receipt_sha256"], freeze["scorer_sha256"]) != (
            PROTOCOL, sha256(saved / "receipt.json"), sha256(__file__)):
        raise ValueError("External predictions or scorer changed after freeze")
    for key in rec["files"]:
        checked(saved, rec, key)
    target = Assay(args.data, args.name)
    if target.hash != rec["inputs"]["target_receipt"]:
        raise ValueError("The frozen external outcome snapshot differs")
    rows = table(saved / "rows.tsv")
    panel = pd.Index(np.load(saved / "panel.npy", allow_pickle=False))
    cohort, actual, _ = target.select(eligible_source(target), panel)
    if not cohort[KEY].equals(rows[KEY]):
        raise ValueError("External prediction rows differ from the frozen target")
    out = args.work / f"score-external-{args.name}{args.retry}"
    out.mkdir(parents=True, exist_ok=False)
    raw, intervals = [], []
    for role, frame in rows.groupby(LINE):
        ix = frame.index.to_numpy()
        part = frame.reset_index(drop=True)
        use = part.groupby(["plate_id", LINE, "dose_uM", "time_h"])[GROUP].transform("nunique").ge(MIN).to_numpy()
        pairs = pd.Index(sorted(part[GROUP].unique()))
        codes = pairs.get_indexer(part[GROUP])
        groups = np.random.default_rng(130113).exponential(size=(10_000, len(pairs)))
        sample = groups / groups.sum(axis=1)[:, None]
        weights = group_weights(part)
        absolute_draws, centered_draws = {}, {}
        if use.any():
            valid = part.loc[use].reset_index(drop=True)
            metric = Contrast(valid, MIN)
            bootstrap_mass = groups @ np.bincount(codes[use], weights=metric.primary_weight, minlength=len(pairs))
        for arm, name in rec["predictions"].items():
            y, p = actual[ix], np.load(saved / name, mmap_mode="r", allow_pickle=False)[ix]
            rmse = np.sqrt(row_products(y - p, y - p))
            contrast = None
            cosine = None
            absolute_draws[arm] = sample @ np.bincount(codes, weights=weights * rmse, minlength=len(pairs))
            if use.any():
                c_y, c_p = center(metric, y[use]), center(metric, p[use])
                contrast = metric.point(p[use], y[use])
                den = np.sqrt(row_products(c_y, c_y) * row_products(c_p, c_p))
                cosine = float(np.average(np.divide(row_products(c_y, c_p), den,
                                                     out=np.zeros_like(den), where=den > 1e-14),
                                          weights=metric.primary_weight))
                centered_draws[arm] = draw_contrast(metric, p[use], y[use], groups, codes[use]) / bootstrap_mass
            raw.append(dict(line=role, arm=arm, rows=len(part), groups=part[GROUP].nunique(),
                            contrast_rows=int(use.sum()), contrast_groups=int(part.loc[use, GROUP].nunique()),
                            delta_rmse=float(np.average(rmse, weights=weights)),
                            contrast_rmse=contrast, cosine=cosine))
        for name, entries in (("delta_rmse", absolute_draws), ("contrast_rmse", centered_draws)):
            for arm in ("integrated", "matched_D0"):
                for base in ("B0", "matched_D0"):
                    if arm == base or not {arm, base} <= entries.keys():
                        continue
                    q = np.quantile(entries[arm] - entries[base], [.025, .975])
                    intervals.append(dict(line=role, metric=name, arm=arm, baseline=base, lower95=q[0], upper95=q[1]))
    pd.DataFrame(raw).to_csv(out / "metrics.tsv", sep="\t", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(protocol_sha256=PROTOCOL, freeze_sha256=sha256(frozen), files=files(out)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    parser.add_argument("--name", required=True, choices=EXTERNAL)
    parser.add_argument("--retry", choices=("-retry",), default="")
    for key in ("work", "assets", "r11", "positions", "data", "protocol"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("R14 protocol differs from prospectively frozen version")
    {"predict": external_fit, "freeze": external_freeze, "score": external_score}[args.action](args)


if __name__ == "__main__":
    main()
