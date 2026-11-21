"""Score only pre-existing predictions on fixed Tahoe target sensitivity endpoints."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, POOL, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r6_followup_v1.metrics import grams


HISTORICAL_ARMS = {
    "R12": {"Integrated": "Integrated", "C2": "C2", "C3": "C3", "smooth_RMSE": "L1"},
    "R13": {"selector": "selector", "kernels": "kernels", "mechanism": "mechanism"},
}
STRATA = ("weak", "detectable", "strong", "unlabeled")


def verified(folder, receipt, name):
    file = folder / name
    if sha256(file) != receipt["files"][name]:
        raise ValueError(f"Artifact hash mismatch: {file}")
    return file


def history(args, task, rows):
    fold, role = task // 2, "AB"[task % 2]
    r12_key = f"outer{fold}/{role}/receipt.json"
    r12_freeze = read_json(args.r12 / "freeze.json")
    r13_freeze = read_json(args.r13 / "freeze.json")
    paths = {
        "R12": (args.r12 / "final-predictions" / f"outer{fold}" / role, r12_freeze["predictions_receipts"][r12_key]),
        "R13": (args.r13 / "frozen" / f"task{task}", r13_freeze["task_receipts"][f"task{task}"]),
    }
    available, rejected = {}, {}
    panels = []
    for study, (folder, digest) in paths.items():
        receipt_path = folder / "receipt.json"
        if sha256(receipt_path) != digest:
            raise ValueError(f"Historical receipt differs from freeze: {receipt_path}")
        rec = read_json(receipt_path)
        historical_rows = table(verified(folder, rec, "rows.tsv"))
        keys = KEY + [GROUP] + ([CONTROL] if CONTROL in rows else [])
        order = pd.MultiIndex.from_frame(historical_rows[keys]).get_indexer(pd.MultiIndex.from_frame(rows[keys]))
        if np.any(order < 0) or len(order) != len(historical_rows) or len(np.unique(order)) != len(order):
            raise ValueError(f"Task {task} changes the frozen row cohort for {study}")
        panel = np.load(verified(folder, rec, "panel.npy"), allow_pickle=False)
        panels.append(panel)
        index = read_json(verified(folder, rec, "prediction_index.json")) if study == "R12" else rec["predictions"]
        for label, key in HISTORICAL_ARMS[study].items():
            arm = f"{study}_{label}"
            names = index.get(key, ())
            if isinstance(names, dict):
                names = list(names.values())
            if not names or any(not (folder / n).is_file() for n in names):
                rejected[arm] = f"task{task}: frozen member unavailable"
                continue
            available[arm] = (folder, rec, names, order, panel)
    if not np.array_equal(*panels):
        raise ValueError("Historical R12 and R13 panels are not the same")
    return available, rejected, panels[0]


def load_members(source):
    folder, rec, names, order, panel = source
    predictions = [np.load(verified(folder, rec, name), allow_pickle=False)[order] for name in names]
    if any(p.shape != (len(order), len(panel)) or not np.isfinite(p).all() for p in predictions):
        raise ValueError(f"Invalid archived predictions in {folder}")
    return predictions


def weight_rows(rows, mask, weighted):
    weights = np.zeros(len(rows))
    part = rows.loc[mask]
    raw = group_weights(part)
    if weighted:
        raw *= part.n_cells.to_numpy()
        raw /= pd.Series(raw).groupby(part[GROUP].to_numpy()).transform("sum").to_numpy()
    weights[mask] = raw
    return weights


def scope_definitions(rows, panel, coding):
    every = np.ones(len(rows), dtype=bool)
    full = np.arange(len(panel))
    qc = np.arange(len(coding))
    overlap = np.flatnonzero(np.isin(coding, panel))
    yield Scope("original", every, full, False, False, False, True, True, True)
    yield Scope("position_corrected", every, full, False, True, False, False, True, True)
    for n in (50, 100, 200, 500):
        yield Scope(f"min_cells_{n}", rows.n_cells.ge(n).to_numpy(), full, False, False, False, False, True, False)
    yield Scope("weighted_cells", every, full, False, False, True, False, True, False)
    yield Scope("filtered_full", every, qc, True, False, False, False, False, False)
    yield Scope("filtered_overlap", every, overlap, True, False, False, False, True, False)
    enough = rows.n_cells.ge(200).to_numpy()
    yield Scope("combined_full", enough, qc, True, True, True, True, False, True)
    yield Scope("combined_overlap", enough, overlap, True, True, True, True, True, True)


@dataclass(frozen=True)
class Scope:
    name: str
    subset: np.ndarray
    columns: np.ndarray
    coding: bool
    corrected: bool
    weighted: bool
    stratified: bool
    historical: bool
    bootstrap: bool


@dataclass
class Totals:
    numerator: dict = field(default_factory=lambda: defaultdict(float))
    denominator: float = 0.0
    masks: set = field(default_factory=set)
    support: set = field(default_factory=set)
    per_task_pools: int = 0
    groups: set = field(default_factory=set)
    panel_sizes: set = field(default_factory=set)
    tasks: int = 0
    bootstrap_numerators: dict = field(default_factory=dict)
    bootstrap_denominator: np.ndarray | None = None

    def accumulate(self, rows, support, mask, weights, group_index, values, draws, genes, boot=None):
        active = rows.loc[mask]
        self.masks.update(map(tuple, active[KEY].to_numpy()))
        self.support.update(map(tuple, rows.loc[mask & support, KEY].to_numpy()))
        self.groups.update(active[GROUP])
        self.panel_sizes.add(genes)
        self.per_task_pools += len(active[POOL].drop_duplicates())
        self.tasks += 1
        self.denominator += float(weights.sum())
        for arm, value in values.items():
            self.numerator[arm] += float(value)
        if boot is not None:
            mass = draws @ np.bincount(group_index, weights=weights, minlength=draws.shape[1])
            if self.bootstrap_denominator is None:
                self.bootstrap_denominator = mass
            else:
                self.bootstrap_denominator += mass
            for arm, value in boot.items():
                if arm not in self.bootstrap_numerators:
                    self.bootstrap_numerators[arm] = value.copy()
                else:
                    self.bootstrap_numerators[arm] += value


def paired_numerators(metric, predictions, actual, cells, weights, group_index, draws, bootstrap):
    # On a fixed pool, the mean squared distance to a weighted mean comes from the error Gram matrix.
    points, replicates = {}, {}
    center = [base * cells[idx] for idx, base in zip(metric.pools, metric.centers)]
    center = [v / v.sum() for v in center]
    pool_draws = [draws[:, group_index[idx]] for idx in metric.pools] if bootstrap else ()
    probabilities = [v * w / (v @ w)[:, None] for v, w in zip(pool_draws, center)]
    for arm, members in predictions.items():
        point = np.zeros(weights.shape[1])
        replicate = np.zeros((len(draws), weights.shape[1])) if bootstrap else None
        for member in members:
            residual = member.astype(np.float64) - actual
            for k, (kernel, idx, probability) in enumerate(zip(grams(residual, metric.pools), metric.pools, center)):
                kernel_mean = probability @ kernel
                variance = np.diag(kernel) - 2 * kernel_mean + kernel_mean @ probability
                point += np.sqrt(np.maximum(variance, 0)) @ weights[idx]
                if bootstrap:
                    probability = probabilities[k]
                    kernel_mean = probability @ kernel
                    variance = np.diag(kernel) - 2 * kernel_mean + (kernel_mean * probability).sum(axis=1, keepdims=True)
                    replicate += (np.sqrt(np.maximum(variance, 0)) * pool_draws[k]) @ weights[idx]
        points[arm] = point / len(members)
        if bootstrap:
            replicates[arm] = replicate / len(members)
    return points, replicates


def task_scores(folder, rows, panel, frozen, groups, draws, totals, minimum):
    coding = np.load(folder / "panel_filtered.npy", allow_pickle=False)
    arrays = {}
    for suffix, genes in (("", panel), ("_filtered", coding)):
        observed, nuisance, support = (np.load(folder / f"{field}{suffix}.npy", allow_pickle=False)
                                       for field in ("observed", "nuisance", "supported"))
        if observed.shape != (len(rows), len(genes)) or nuisance.shape != observed.shape or support.shape != (len(rows),):
            raise ValueError(f"Common target dimensions do not match: {folder}")
        if not np.isfinite(observed).all() or not np.isfinite(nuisance).all() or np.any(nuisance[~support.astype(bool)] != 0):
            raise ValueError(f"Nonfinite target or nonzero nuisance outside supported rows: {folder}")
        arrays[bool(suffix)] = (observed, nuisance, support.astype(bool))
    archived = {arm: load_members(source) for arm, source in frozen.items()}
    for scope in scope_definitions(rows, panel, coding):
        if not len(scope.columns):
            continue
        indices = np.flatnonzero(scope.subset)
        qr = rows.iloc[indices].reset_index(drop=True)
        eligible = qr.groupby(POOL, sort=False)[GROUP].transform("nunique").ge(minimum).to_numpy()
        if not eligible.any():
            continue
        metric = Contrast(qr, minimum)
        strata = {"all": eligible}
        if scope.stratified:
            strata.update({tag: eligible & qr.strength.eq(tag).to_numpy() for tag in STRATA})
        strata = {key: mask for key, mask in strata.items() if mask.any()}
        weights = np.column_stack([weight_rows(qr, mask, scope.weighted) for mask in strata.values()])
        observed, nuisance, support = arrays[scope.coding]
        member_index = np.ix_(indices, scope.columns)
        target, position = observed[member_index], nuisance[member_index]
        if scope.corrected:
            target = target - position
        preds = {"B0": [np.zeros_like(target)], "B_pos": [position]}
        if scope.historical:
            histor_cols = pd.Index(panel).get_indexer((coding if scope.coding else panel)[scope.columns])
            if np.any(histor_cols < 0):
                raise ValueError("Selected genes are absent from historical predictions")
            prediction_index = np.ix_(indices, histor_cols)
            preds.update({arm: [p[prediction_index] for p in members] for arm, members in archived.items()})
        group_index = pd.Index(groups).get_indexer(qr[GROUP])
        cells = qr.n_cells.to_numpy() if scope.weighted else np.ones(len(qr))
        point, boot = paired_numerators(metric, preds, target, cells, weights, group_index, draws, scope.bootstrap)
        for i, (stratum, mask) in enumerate(strata.items()):
            totals[scope.name, stratum].accumulate(
                qr, support[indices], mask, weights[:, i], group_index,
                {arm: value[i] for arm, value in point.items()}, draws, len(scope.columns),
                {arm: value[:, i] for arm, value in boot.items()} if scope.bootstrap else None,
            )


def summarize(totals):
    points, intervals = [], []
    for (scope, strength), acc in sorted(totals.items()):
        common = dict(endpoint=scope, strength=strength, rows=len(acc.masks), supported_rows=len(acc.support),
                      compound_groups=len(acc.groups), role_fold_tasks=acc.tasks, pooled_slices=acc.per_task_pools,
                      genes_min=min(acc.panel_sizes), genes_max=max(acc.panel_sizes))
        b0 = acc.numerator["B0"] / acc.denominator
        for arm, numerator in acc.numerator.items():
            rmse = numerator / acc.denominator
            points.append(dict(**common, arm=arm, rmse=rmse, difference_vs_B0=rmse - b0,
                               relative_vs_B0_percent=100 * (rmse / b0 - 1) if b0 else None))
            if acc.bootstrap_denominator is None or arm == "B0":
                continue
            difference = (acc.bootstrap_numerators[arm] - acc.bootstrap_numerators["B0"]) / acc.bootstrap_denominator
            lo, hi = np.quantile(difference, [0.025, 0.975])
            intervals.append(dict(**common, arm=arm, difference_vs_B0=rmse - b0, lower=lo, upper=hi,
                                  confidence=0.95))
    return pd.DataFrame(points), pd.DataFrame(intervals)


def checked_targets(args):
    receipt_path = args.common / "receipt.json"
    receipt = read_json(receipt_path)
    freeze_path = args.common / "position_freeze.json"
    if sha256(freeze_path) != receipt["position_freeze_sha256"]:
        raise ValueError("Position freeze changed after preparing scoring targets")
    freeze = read_json(freeze_path)
    historical_sha256 = sha256(args.r13 / "freeze.json")
    if freeze["protocol_sha256"] != sha256(args.protocol) or freeze["historical_freeze_sha256"] != historical_sha256:
        raise ValueError("Target preparation used a different prescoring protocol or historical freeze")
    folders = [args.common / f"task{task}" for task in range(6)]
    for folder in folders:
        files = ["rows.tsv"] + [f"{field}{suffix}.npy" for field in ("panel", "observed", "nuisance", "supported")
                               for suffix in ("", "_filtered")]
        for file in files:
            verified(args.common, receipt, f"{folder.name}/{file}")
    return folders, sha256(receipt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("common", "r12", "r13", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--minimum-groups", type=int, default=3)
    args = parser.parse_args()
    totals = defaultdict(Totals)
    folders, target_receipt_sha256 = checked_targets(args)
    manifests = [table(folder / "rows.tsv") for folder in folders]
    if any(not set(rows.strength) <= set(STRATA) or not rows.n_cells.gt(0).all() for rows in manifests):
        raise ValueError("Unexpected common target stratum or cell-count metadata")
    groups = sorted(set().union(*(set(rows[GROUP]) for rows in manifests)))
    draws = np.random.default_rng(130113).exponential(size=(10_000, len(groups)))
    draws /= draws.mean(axis=1, keepdims=True)
    origins = [history(args, task, rows) for task, rows in enumerate(manifests)]
    common = set.intersection(*(set(available) for available, _, _ in origins))
    missing = defaultdict(list)
    for _, rejected, _ in origins:
        for arm, reason in rejected.items():
            missing[arm].append(reason)
    panels = []
    for task, (folder, rows, (arms, _, baseline_panel)) in enumerate(zip(folders, manifests, origins)):
        panel = np.load(folder / "panel.npy", allow_pickle=False)
        if not np.array_equal(panel, baseline_panel):
            raise ValueError(f"Common target has a different original panel: task{task}")
        coding = np.load(folder / "panel_filtered.npy", allow_pickle=False)
        panels.append(dict(task=task, fold=task // 2, role="AB"[task % 2], original=len(panel),
                           protein_coding=len(coding), overlap=np.isin(coding, panel).sum()))
        task_scores(folder, rows, panel, {arm: arms[arm] for arm in sorted(common)}, groups, draws, totals,
                    args.minimum_groups)
        print(f"rescored frozen role-fold task{task}", flush=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    point, paired = summarize(totals)
    point.to_csv(out / "points.tsv", sep="\t", index=False)
    paired.to_csv(out / "paired.tsv", sep="\t", index=False)
    pd.DataFrame(panels).to_csv(out / "panel_coverage.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(protocol_sha256=sha256(args.protocol), frozen_arms=sorted(common),
                                         target_receipt_sha256=target_receipt_sha256,
                                         historical_freeze_sha256={"R12": sha256(args.r12 / "freeze.json"),
                                                                   "R13": sha256(args.r13 / "freeze.json")},
                                         unavailable_arms=dict(missing), bootstrap_draws=10_000,
                                         bootstrap_endpoints=["original", "position_corrected", "combined_full", "combined_overlap"],
                                         files={p.name: sha256(p) for p in out.iterdir()}))


if __name__ == "__main__":
    main()
