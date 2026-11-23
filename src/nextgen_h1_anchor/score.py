"""One held-out scoring pass with paired independent-group bootstrap."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256 as digest
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.io import sha256
from vc_build.v0plus_score import HCC_DRUGS, compound_strata
from vc_build.evaluate import across_cell_metrics

from .inputs import PAN, json_write
from .metrics import NAMES, batch_stats, mean_valid, sample_summaries
from .models import KS
from .prepare import AUXILIARY, TRANSFER

DRAWS = 2000
EXPOSURE = {
    "GDSC2": "prior VC development", "CTRPv2": "prior VC development", "PRISM20Q2": "prior VC development",
    "Tier1-macro": "prior VC development", "LIMORE": "prior external and local evaluation",
    "LICOB": "prior external and local evaluation", TRANSFER: "prior target and source evaluation",
    AUXILIARY: "prior target and source evaluation",
}
PRIMARY = "H1-Anchor-Best"
DUAL = "H1-RNA+Anchor-Best"
SAMPLE_AXES = [0, 1, 2, 7, 8]


@dataclass
class Task:
    task: str
    y: np.ndarray
    p: dict
    group: np.ndarray
    tissue: np.ndarray
    meta: pd.DataFrame
    drawn_indices: np.ndarray | None = None

    @property
    def study(self):
        return self.task.rsplit("-", 1)[0]

    @property
    def absolute(self):
        return self.study not in {TRANSFER, AUXILIARY}


def read_tasks(root, anchor_dir, forecast_dir):
    initial = json.loads((root / "receipt.json").read_text())
    anchor = json.loads((anchor_dir / "anchors_receipt.json").read_text())
    forecasts = json.loads((forecast_dir / "forecast_receipt.json").read_text())
    if (forecasts["input_sha256"] != sha256(anchor_dir / "anchors_receipt.json")
            or forecasts["prediction_has_read_Q_truth"] or anchor["evaluation_query_values_exported"]
            or set(anchor["tasks"]) != set(initial["tasks"])):
        raise ValueError("Forecast does not verify against the sealed anchor pass")
    expected = {task for task, row in anchor["tasks"].items() if row["eligible"] >= 2}
    if expected != set(forecasts["files"]):
        raise ValueError("Frozen forecasts do not cover all eligible tasks")
    for task, key in forecasts["files"].items():
        if sha256(forecast_dir / f"{task}.npz") != key:
            raise ValueError(f"Frozen prediction changed: {task}")
    tasks = []
    for task in forecasts["files"]:
        study, fold = task.rsplit("-", 1)
        pred = dict(np.load(forecast_dir / f"{task}.npz", allow_pickle=False))
        idx, q, names = (pred.pop(key) for key in ("index", "q", "group"))
        outcome = root / "labels" / f"{task}.npz"
        if sha256(outcome) != initial["files"][f"labels/{task}.npz"]:
            raise ValueError("Held outcome hash changed")
        truth = dict(np.load(outcome, allow_pickle=False))
        if not np.array_equal(names, truth["groups"][idx]) or len(set(names)) != len(names):
            raise ValueError("Predictions and held score need identical independent group identities")
        y = truth["y"][idx][:, q]
        if any(x.shape != y.shape or not np.isfinite(x).all() for x in pred.values()):
            raise ValueError("Every forecast must cover the fixed Q exactly")
        f = root / study
        q_file = f / f"{fold}-query.npz"
        m_file = f / "chemistry.tsv"
        for path in (q_file, m_file):
            if sha256(path) != initial["files"][str(path.relative_to(root))]:
                raise ValueError("Q register artifact changed")
        query = np.load(q_file, allow_pickle=False)
        meta = (pd.read_csv(m_file, sep="\t", index_col=0).fillna("").loc[truth["panel"][q]])
        tasks.append(Task(task, y, pred, names, query["tissue"][idx], meta))
    return tasks, anchor


def bootstrap(tasks):
    """Draw the union of independent lineages once per fold across the public screens."""
    strata = {}
    for t in tasks:
        key = ("pan" if t.study in PAN else t.study, t.task.rsplit("-", 1)[1])
        strata.setdefault(key, []).append(t)
    for key, members in strata.items():
        union = np.unique(np.concatenate([t.group for t in members]))
        seed = int.from_bytes(digest(f"h1:bootstrap:{key}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        sampled = union[rng.integers(0, len(union), size=(DRAWS, len(union)))]
        for t in members:
            lookup = dict(zip(t.group, range(len(t.group))))
            index = np.array([[lookup[g] for g in row if g in lookup] + [-1] *
                              (len(union) - sum(g in lookup for g in row)) for row in sampled])
            t.drawn_indices = np.row_stack([np.arange(len(t.group)), index[:, :np.max((index >= 0).sum(axis=1))]]) \
                if index.shape[1] == len(t.group) and np.all(index >= 0) else np.row_stack((
                    np.pad(np.arange(len(t.group)), (0, max(index.shape[1] - len(t.group), 0)), constant_values=-1), index))
    return tasks


def fit_task(t):
    y = np.row_stack((t.y, np.full(t.y.shape[1], np.nan)))
    index = t.drawn_indices
    cache = {}
    stats = {}
    samples = {}
    for key, forecast in t.p.items():
        h = digest(np.ascontiguousarray(forecast).view(np.uint8)).digest()
        if h not in cache:
            p = np.row_stack((forecast, np.full(forecast.shape[1], np.nan)))
            single = sample_summaries(t.y, forecast)
            single = np.row_stack((single, np.full(single.shape[1], np.nan)))
            rows = [batch_stats(y[i], p[i], single[i], absolute=t.absolute)
                    for i in np.array_split(index, 1 + len(index) // 24)]
            cache[h] = np.concatenate(rows), single
        stats[key], samples[key] = cache[h]
    return t.task, stats, samples, (index >= 0).sum(axis=1)


def quantiles(values):
    valid = values[np.isfinite(values)]
    return tuple(np.quantile(valid, (.025, .975))) if len(valid) else (np.nan, np.nan)


def tier(study):
    return 1 if study in {*PAN, "Tier1-macro"} else {"LIMORE": 2, "LICOB": 3, TRANSFER: 4, AUXILIARY: "aux"}[study]


def score_rows(study, stacked, n_groups, q_range, n_entries):
    estimates = []
    contrasts = []
    for key, vals in stacked.items():
        method, k = key.split("/")
        for i, metric in enumerate(NAMES):
            point = vals[0, i]
            lo, hi = quantiles(vals[1:, i])
            estimates.append({"tier": tier(study), "study": study, "method": method, "k": int(k), "metric": metric,
                              "value": point, "lo95": lo, "hi95": hi, "independent_groups": n_groups[key],
                              "observed_entries": n_entries[key], "q_min": q_range[0], "q_max": q_range[1],
                              "exposure": EXPOSURE[study]})
    for k in KS:
        keys = [(f"{PRIMARY}/{k}", f"{arm}/{k}") for arm in ("Bsimple", "B0", "B1", "B2", "B3", "B4", "N1", "N2")]
        keys += [(f"B4/{k}", f"{arm}/{k}") for arm in ("B1", "B2")]
        keys += [(f"{DUAL}/{k}", f"{arm}/{k}") for arm in (PRIMARY, "N2")]
        if k:
            keys.append((f"{PRIMARY}/{k}", f"{PRIMARY}/0"))
        for a, b in keys:
            if a not in stacked or b not in stacked or n_groups[a] != n_groups[b]:
                continue
            delta = stacked[a] - stacked[b]
            for i, metric in enumerate(NAMES):
                lo, hi = quantiles(delta[1:, i])
                contrasts.append({"tier": tier(study), "study": study, "candidate": a, "comparator": b,
                                  "metric": metric, "delta": delta[0, i], "lo95": lo, "hi95": hi,
                                  "independent_groups": n_groups[a], "exposure": EXPOSURE[study], "draws": DRAWS})
    return estimates, contrasts


def combined(tasks, fit):
    keys = set.intersection(*(set(fit[t.task][0]) for t in tasks))
    columns = {}
    for key in sorted(keys):
        values = np.array([fit[t.task][0][key] for t in tasks])
        weights = np.array([fit[t.task][2] for t in tasks])
        # Within-drug and interaction quantities macro-average the separately computed folds.
        result = mean_valid(values, 0)
        for axis in SAMPLE_AXES:
            v = values[:, :, axis]
            if axis == 8:
                v = v ** 2
            weights_masked = weights * np.isfinite(v)
            denom = weights_masked.sum(axis=0)
            mean = np.divide(np.nansum(v * weights, axis=0), denom, out=np.full_like(denom, np.nan, dtype=float),
                             where=denom > 0)
            result[:, axis] = np.sqrt(mean) if axis == 8 else mean
        columns[key] = result
    return columns


def tissue_rows(tasks, fit):
    output = []
    for study in PAN:
        use = [t for t in tasks if t.study == study]
        for name in sorted(set(np.concatenate([t.tissue for t in use]))):
            panels = {t.task: np.where(t.tissue == name)[0] for t in use}
            n = sum(len(v) for v in panels.values())
            if not n:
                continue
            for k in KS:
                for arm in (PRIMARY, "Bsimple", "B1", "B4", "N1"):
                    keys = (f"{arm}/{k}", f"Bsimple/{k}")
                    block_sum = np.zeros((DRAWS + 1, 3))
                    baseline_sum = block_sum.copy()
                    counts = block_sum.copy()
                    for t in use:
                        subset = panels[t.task]
                        idx = np.where(np.isin(t.drawn_indices, subset), t.drawn_indices, -1)
                        a, b = (fit[t.task][1][key][idx][:, :, :3] for key in keys)
                        common = np.isfinite(a) & np.isfinite(b)
                        counts += common.sum(axis=1)
                        block_sum += np.sum(np.where(common, a, 0.), axis=1)
                        baseline_sum += np.sum(np.where(common, b, 0.), axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        values = block_sum / counts
                        paired = (block_sum - baseline_sum) / counts
                    for i, metric in enumerate(NAMES[:3]):
                        lo, hi = quantiles(paired[1:, i])
                        output.append({"study": study, "tissue": name, "method": arm, "k": k, "metric": metric,
                                       "n_groups": n, "value": values[0, i], "delta_vs_bsimple": paired[0, i],
                                       "paired_lo95": lo, "paired_hi95": hi, "exposure": EXPOSURE[study]})
    return output


def mechanism_rows(tasks):
    """Within-fold drug metrics with explicit support for each of the four HCC TKIs."""
    rows = []
    for t in tasks:
        if t.study not in {"LICOB", "LIMORE"}:
            continue
        moa = compound_strata(t.meta.reset_index())
        strata = {k: set(v) for k, v in moa.items()}
        for name in HCC_DRUGS:
            match = t.meta.source_compound_label.str.casefold().eq(name.casefold())
            strata[name] = set(t.meta.index[match])
        for label, drugs in sorted(strata.items()):
            indices = np.flatnonzero(t.meta.index.isin(drugs))
            if not len(indices):
                if label in HCC_DRUGS:
                    rows.append({"study": t.study, "fold": t.task, "stratum": label, "method": PRIMARY,
                                 "k": 5, "q_drugs": 0, "groups": len(t.group), "exposure": EXPOSURE[t.study]})
                continue
            for method in (PRIMARY, "Bsimple", "B1", "B4"):
                key = f"{method}/5"
                actual = t.y[:, indices]
                pred = t.p[key][:, indices]
                summaries = []
                for a, p in zip(actual.T, pred.T):
                    valid = np.isfinite(a)
                    frame = pd.DataFrame({"lineage_group": t.group[valid], "endpoint_value": a[valid],
                                          "predicted_value": p[valid]})
                    summaries.append(across_cell_metrics(frame))
                rows.append({"study": t.study, "fold": t.task, "stratum": label, "method": method,
                             "k": 5, "q_drugs": len(drugs), "groups": len(t.group),
                             "observations": int(np.isfinite(actual).sum()),
                             "pearson": mean_valid(np.array([m["pearson"] if m["pearson"] is not None else np.nan
                                                             for m in summaries]), 0),
                             "spearman": mean_valid(np.array([m["spearman"] if m["spearman"] is not None else np.nan
                                                              for m in summaries]), 0),
                             "mae": float(np.nanmean(np.abs(actual - pred))), "exposure": EXPOSURE[t.study]})
    return rows


def run(root, anchor_dir, forecast_dir, out):
    out.mkdir()
    claim = out / "score_claim.json"
    json_write(claim, {"score_passes_started": 1, "forecast_sha256": sha256(forecast_dir / "forecast_receipt.json")})
    tasks, anchor = read_tasks(root, anchor_dir, forecast_dir)
    tasks = bootstrap(tasks)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    with ProcessPoolExecutor(max_workers=n_workers) as workers:
        parts = workers.map(fit_task, tasks)
        fit = {}
        for name, values, samples, counts in parts:
            fit[name] = (values, samples, counts)
            print("scored", name, flush=True)
    estimates, contrasts = [], []
    grouped = {}
    for t in tasks:
        grouped.setdefault(t.study, []).append(t)
    all_metrics = {name: combined(block, fit) for name, block in grouped.items()}
    pan = set.intersection(*(set(all_metrics[study]) for study in PAN))
    all_metrics["Tier1-macro"] = {k: mean_valid(np.stack([all_metrics[s][k] for s in PAN]), 0) for k in pan}
    for study, parts in all_metrics.items():
        block = grouped.get(study, [t for t in tasks if t.study in PAN])
        n = len(set(np.concatenate([t.group for t in block])))
        q_dims = [t.y.shape[1] for t in block]
        entries = sum(int(np.isfinite(t.y).sum()) for t in block)
        a, b = score_rows(study, parts, {k: n for k in parts}, (min(q_dims), max(q_dims)),
                          {k: entries for k in parts})
        estimates.extend(a)
        contrasts.extend(b)
    written = {
        "H1_METRICS.tsv": pd.DataFrame(estimates),
        "H1_PAIRED_95CI.tsv": pd.DataFrame(contrasts),
        "H1_PANCANCER_BY_TISSUE.tsv": pd.DataFrame(tissue_rows(tasks, fit)),
        "H1_HCC_TKI_AND_MECHANISMS.tsv": pd.DataFrame(mechanism_rows(tasks)),
    }
    for name, frame in written.items():
        frame.to_csv(out / name, sep="\t", index=False, float_format="%.8g", na_rep="NA")
    json_write(out / "score_receipt.json", {
        "score_passes": 1, "paired_bootstrap_draws": DRAWS,
        "input_sha256": sha256(root / "receipt.json"), "score_claim_sha256": sha256(claim),
        "anchor_receipt_sha256": sha256(anchor_dir / "anchors_receipt.json"),
        "forecast_receipt_sha256": sha256(forecast_dir / "forecast_receipt.json"),
        "fit_group_overlap": 0, "masks": anchor["tasks"], "task_count": len(tasks),
        "outputs": {name: sha256(out / name) for name in written},
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "anchor_dir", "forecast_dir", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
