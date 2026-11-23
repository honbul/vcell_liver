"""Claim and execute the only v2 held-out scoring pass."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256 as digest
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h1_anchor.inputs import PAN, json_write
from nextgen_h1_anchor.metrics import NAMES, center2, mean_valid, ranks, sample_summaries
from nextgen_h1_anchor.score import (DRAWS, EXPOSURE, SAMPLE_AXES, Task, bootstrap, combined,
                                     quantiles, tier)
from vc_build.io import sha256
from vc_build.v0plus_score import correlations

from .data import BUDGETS, ROOT, STUDY, TASKS
from .freeze import ANCHOR, DUAL
from .predict import controls, models_predict, verify_freeze

TKIS = ("Sorafenib", "Lenvatinib", "Regorafenib", "Cabozantinib")
TRACKS = ("Tier1-macro", *PAN, "LIMORE", "LICOB", "LIMORE_to_LICOB")
TISSUE_ARMS = (DUAL, ANCHOR, "B1", "Bsimple", "v1-Anchor-Best", "v1-RNA+Anchor")


def validate(forecasts, prediction_gate, freezes, model_gate):
    prep = json.loads(prediction_gate.read_text())
    stamp = json.loads(model_gate.read_text())
    v1 = sha256(STUDY / "receipt.json")
    if (prep["outer_labels_opened"] or stamp["outer_labels_opened"] or prep["frozen_sha256"] != sha256(model_gate)
            or stamp["v1_study_sha256"] != v1 or set(stamp["packages"]) != set(TASKS)):
        raise ValueError("Held scoring must wait for the complete immutable v2 freeze")
    records = {}
    for task, expected in prep["prediction_receipts"].items():
        manifest = forecasts / f"{task}.json"
        row = json.loads(manifest.read_text())
        if sha256(manifest) != expected or sha256(forecasts / f"{task}.npz") != row["sha256"]:
            raise ValueError(f"Changed held forecast for {task}")
        if row["freeze_sha256"] != stamp["packages"][task] or sha256(freezes / task / "freeze.json") != stamp["packages"][task]:
            raise ValueError(f"Forecast model seal mismatch: {task}")
        records[task] = row
    if set(records) != set(TASKS):
        raise ValueError("Incomplete held forecast coverage")
    return records


def verify_prelabel(prelabel, frozen, predictions):
    receipt = json.loads(prelabel.read_text())
    if (receipt["holdout_outcomes_opened"] or set(receipt["tasks"]) != set(TASKS)
            or receipt["sealed_models"] != sha256(frozen) or receipt["sealed_forecasts"] != sha256(predictions)):
        raise ValueError("Prelabel validation does not cover the sealed models and predictions")
    return sha256(prelabel)


def open_once(task, frozen, model_gate, forecasts, aux, registered):
    """Restrict released extras to A12, then form observed query masks locally."""
    seal, package = verify_freeze(frozen, model_gate, task)
    file = STUDY / "labels" / f"{task}.npz"
    if sha256(file) != registered["files"][f"labels/{task}.npz"]:
        raise ValueError("Registered held label hash changed")
    with np.load(file, allow_pickle=False) as orig:
        y, groups = (orig[k] for k in ("y", "groups"))
        if not np.array_equal(orig["panel"], package["panel"]) or not np.array_equal(groups, package["groups"]):
            raise ValueError("Held outcome roster changed")
    study, fold = task.rsplit("-", 1)
    query = STUDY / study / f"{fold}-query.npz"
    chem = STUDY / study / "chemistry.tsv"
    for path in (query, chem):
        if sha256(path) != registered["files"][str(path.relative_to(STUDY))]:
            raise ValueError("Outcome-free query metadata changed")
    with np.load(query, allow_pickle=False) as identity:
        tissue = identity["tissue"]
    meta = pd.read_csv(chem, sep="\t", index_col=0).fillna("")
    with np.load(forecasts / f"{task}.npz", allow_pickle=False) as archive:
        index, q, ids = (archive[k] for k in ("index", "q", "group"))
        p = {k: archive[k] for k in archive.files if "/" in k}
    if not np.array_equal(q, package["q8"]) or not np.array_equal(ids, groups[index]):
        raise ValueError("Primary Q or donor mismatch")
    primary = Task(task, y[index][:, q], p, ids, tissue[index], meta.iloc[q])
    all_anchor = np.r_[np.asarray(seal["orders"]["anchor"][:8]), package["extension"]]
    q12 = np.setdiff1d(np.arange(len(package["panel"])), all_anchor)
    mask = np.isfinite(y[index][:, all_anchor]).all(1) & (np.isfinite(y[index][:, q12]).sum(1) >= 15)
    index12 = index[mask]
    records = {"primary_groups": len(index), "primary_q": len(q), "primary_pairs": int(np.isfinite(primary.y).sum()),
               "auxiliary_groups": len(index12), "auxiliary_q": len(q12)}
    if len(index12) < 2:
        return primary, None, records
    a = np.full(y.shape, np.nan)
    a[np.ix_(index12, all_anchor)] = y[np.ix_(index12, all_anchor)]
    result = models_predict(package, a, 12, q12, index12)
    result.update(controls(task, a, 12, q12, index12, package, {}, published_dual="v1-RNA+Anchor/5" in p))
    p12 = {f"{name}/12": b for name, b in result.items()}
    auxfile = aux / f"{task}.npz"
    np.savez_compressed(auxfile, anchors=a[index12][:, all_anchor], codes=all_anchor,
                        index=index12, q=q12, **p12)
    records |= {"auxiliary_pairs": int(np.isfinite(y[index12][:, q12]).sum()), "aux_file_sha256": sha256(auxfile)}
    return primary, Task(task, y[index12][:, q12], p12, groups[index12], tissue[index12], meta.iloc[q12]), records


def fit_metrics(task):
    """Compute all metrics on matched bootstrap draws, sharing the actual-side ranks."""
    index = task.drawn_indices
    count, width = task.y.shape
    y = np.vstack((task.y, np.full(width, np.nan)))
    hashes = {k: digest(np.ascontiguousarray(p).view(np.uint8)).digest() for k, p in task.p.items()}
    unique = {hashes[k]: p for k, p in task.p.items()}
    summary = {h: np.vstack((sample_summaries(task.y, p), np.full(5, np.nan))) for h, p in unique.items()}
    extended = {h: np.vstack((p, np.full(width, np.nan))) for h, p in unique.items()}
    results = {h: [] for h in unique}
    for ii in np.array_split(index, 1 + len(index) // 18):
        actual = y[ii]
        have = np.isfinite(actual)
        ranked = ranks(actual, 1)
        a = center2(actual).reshape(len(ii), -1, 1)
        ar = ranks(a, 1)
        for h, block in extended.items():
            pred = np.where(have, block[ii], np.nan)
            rho = mean_valid(correlations(ranked, ranks(pred, 1)), 1)
            pcc = mean_valid(correlations(actual, pred), 1)
            residual = center2(pred).reshape(len(ii), -1, 1)
            ir = correlations(ar, ranks(residual, 1))[:, 0]
            ip = correlations(a, residual)[:, 0]
            sample = mean_valid(summary[h][ii], 1)
            mae = sample[:, 3] if task.absolute else np.full(len(ii), np.nan)
            rmse = np.sqrt(sample[:, 4]) if task.absolute else np.full(len(ii), np.nan)
            results[h].append(np.column_stack((sample[:, :3], rho, pcc, ip, ir, mae, rmse)))
    merged = {h: np.concatenate(blocks) for h, blocks in results.items()}
    print("scored", task.task, "Q", width, flush=True)
    return task.task, {k: merged[h] for k, h in hashes.items()}, {k: summary[h] for k, h in hashes.items()}, (index >= 0).sum(1)


def study_axes(batch, axis):
    counts = defaultdict(list)
    for task in batch:
        counts[task.study].append(task)
    by_study = {s: combined(block, axis) for s, block in counts.items()}
    if all(n in by_study for n in PAN):
        shared = set.intersection(*(set(by_study[n]) for n in PAN))
        by_study["Tier1-macro"] = {k: mean_valid(np.stack([by_study[n][k] for n in PAN]), 0) for k in shared}
        counts["Tier1-macro"] = [t for t in batch if t.study in PAN]
    return by_study, counts


def contrasted(k, values):
    options = {v.split("/")[0] for v in values if v.endswith(f"/{k}")}
    refs = ("Bsimple", "B0", "B1", "B2", "B3", "B4", "v1-Bsimple", "v1-Anchor-Best", "v1-RNA+Anchor",
            "v1-RNA+Anchor-ungated", "N1", "N1-Dual", "N2")
    for model in (ANCHOR, DUAL):
        if model in options:
            for comp in refs:
                if comp in options:
                    yield f"{model}/{k}", f"{comp}/{k}"
            if 0 < k < 12:
                yield f"{model}/{k}", f"{model}/0"
    if ANCHOR in options and DUAL in options:
        yield f"{DUAL}/{k}", f"{ANCHOR}/{k}"
    for family in ("A", "B", "C", "D"):
        if f"Family-{family}" in options:
            yield f"Family-{family}/{k}", f"Bsimple/{k}"


def tables(batch, scoped, panel):
    estimates, pairs, bench = [], [], []
    axes, groups = study_axes(batch, scoped)
    for study, values in axes.items():
        roster = groups[study]
        base = dict(study=study, panel=panel, tier=tier(study), exposure=EXPOSURE[study],
                    independent_groups=len(np.unique(np.concatenate([t.group for t in roster]))),
                    min_q=min(t.y.shape[1] for t in roster), max_q=max(t.y.shape[1] for t in roster),
                    observed_pairs=sum(int(np.isfinite(t.y).sum()) for t in roster))
        for label, block in values.items():
            arm, k = label.rsplit("/", 1)
            for j, name in enumerate(NAMES):
                lo, hi = quantiles(block[1:, j])
                point = dict(**base, arm=arm, k=int(k), metric=name, value=block[0, j], lo95=lo, hi95=hi)
                estimates.append(point)
                if arm.startswith("Family-") and int(k) == 5:
                    bench.append(point)
        for k in (BUDGETS[:-1] if panel == "Q8" else (12,)):
            for model, comp in contrasted(k, values):
                delta = values[model] - values[comp]
                for j, name in enumerate(NAMES):
                    lo, hi = quantiles(delta[1:, j])
                    pairs.append(dict(**base, k=k, candidate=model, comparator=comp, metric=name,
                                      delta=delta[0, j], lo95=lo, hi95=hi, draws=DRAWS))
    return estimates, pairs, bench


def restrict_tissue(task, tissue):
    selected = np.flatnonzero(task.tissue == tissue)
    lookup = np.full(len(task.group) + 1, -1, dtype=int)
    lookup[selected] = np.arange(len(selected))
    draws = lookup[task.drawn_indices]
    width = int((draws >= 0).sum(axis=1).max())
    draws.sort(axis=1)
    forecasts = {f"{arm}/5": task.p[f"{arm}/5"][selected] for arm in TISSUE_ARMS}
    return Task(task.task, task.y[selected], forecasts, task.group[selected], task.tissue[selected],
                task.meta, draws[:, -width:])


def fit_tissues(task):
    result = []
    for label in sorted(set(task.tissue)):
        sub = restrict_tissue(task, label)
        _, metrics, _, counts = fit_metrics(sub)
        result.append((label, sub, (metrics, {}, counts)))
    return result


def tissue_table(scored_tissues):
    grouped = defaultdict(list)
    for block in scored_tissues:
        for name, sub, stats in block:
            grouped[sub.study, name].append((sub, stats))
    rows = []
    for (study, tissue), records in sorted(grouped.items()):
        subsets = [sub for sub, _ in records]
        values = combined(subsets, {sub.task: stats for sub, stats in records})
        n = sum(len(sub.group) for sub in subsets)
        for arm in TISSUE_ARMS:
            value = values[f"{arm}/5"]
            diff = value - values["Bsimple/5"]
            for j, metric in enumerate(NAMES):
                lower, upper = quantiles(value[1:, j])
                dl, du = quantiles(diff[1:, j])
                rows.append(dict(study=study, tissue=tissue, arm=arm, k=5, n_groups=n, metric=metric,
                                 value=value[0, j], lo95=lower, hi95=upper, delta_vs_bsimple=diff[0, j],
                                 delta_lo95=dl, delta_hi95=du, exposure=EXPOSURE[study]))
    return rows


def tki_table(tasks):
    results = []
    for study in ("LIMORE", "LICOB", "LIMORE_to_LICOB"):
        local = [t for t in tasks if t.study == study]
        methods = set.intersection(*(set(t.p) for t in local))
        arms = [f"{s}/5" for s in (ANCHOR, DUAL, "B0", "B1", "B2", "B3", "B4", "v1-Anchor-Best")
                if f"{s}/5" in methods]
        for drug in TKIS:
            site = []
            for t in local:
                matches = np.flatnonzero(t.meta.source_compound_label.str.casefold().eq(drug.casefold()))
                if len(matches):
                    site.append((t, matches[0]))
            if not site:
                results.append(dict(study=study, tki=drug, arm="", n=0, folds=0, exposure=EXPOSURE[study]))
                continue
            for arm in arms:
                records = []
                for t, j in site:
                    idx = t.drawn_indices
                    actual = np.r_[t.y[:, j], np.nan][idx, None]
                    pred = np.r_[t.p[arm][:, j], np.nan][idx, None]
                    pred = np.where(np.isfinite(actual), pred, np.nan)
                    rho = correlations(ranks(actual, 1), ranks(pred, 1))[:, 0]
                    pcc = correlations(actual, pred)[:, 0]
                    records.append(np.column_stack((rho, pcc)))
                summary = mean_valid(np.stack(records), 0)
                rlo, rhi = quantiles(summary[1:, 0])
                plo, phi = quantiles(summary[1:, 1])
                results.append(dict(study=study, tki=drug, arm=arm.rsplit("/", 1)[0],
                                    n=sum(int(np.isfinite(t.y[:, j]).sum()) for t, j in site), folds=len(site),
                                    spearman=summary[0, 0], spearman_lo95=rlo, spearman_hi95=rhi,
                                    pearson=summary[0, 1], pearson_lo95=plo, pearson_hi95=phi, exposure=EXPOSURE[study]))
    return results


def execute(freezes, frozen, forecasts, predictions, out, prelabel, study_lock=ROOT / "outer_score_claim.json"):
    verified = validate(forecasts, predictions, freezes, frozen)
    checked = verify_prelabel(prelabel, frozen, predictions)
    out.mkdir()
    # The exclusive claim is outside the output directory, so a different
    # output path cannot repeat the v2 scoring run.
    json_write(study_lock, {"score_passes_started": 1, "frozen_sha256": sha256(frozen),
                            "predictions_sha256": sha256(predictions), "prelabel_sha256": checked,
                            "cohort_sha256": sha256(STUDY / "receipt.json")})
    aux = out / "aux"
    aux.mkdir()
    register = json.loads((STUDY / "receipt.json").read_text())
    primary, auxiliary, support = [], [], {}
    for task in verified:
        main, secondary, record = open_once(task, freezes, frozen, forecasts, aux, register)
        primary.append(main)
        if secondary is not None:
            auxiliary.append(secondary)
        support[task] = record
    primary, auxiliary = bootstrap(primary), bootstrap(auxiliary)
    workers = min(32, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        points = {}
        for scope, batch in (("Q8", primary), ("Q12", auxiliary)):
            points[scope] = {name: (metrics, samples, counts)
                             for name, metrics, samples, counts in pool.map(fit_metrics, batch)}
        tissues = tissue_table(pool.map(fit_tissues, (t for t in primary if t.study in PAN)))
    m, p, b = tables(primary, points["Q8"], "Q8")
    m12, p12, _ = tables(auxiliary, points["Q12"], "Q12")
    outputs = {
        "H1V2_METRICS.tsv": m + m12, "H1V2_PAIRED_95CI.tsv": p + p12,
        "H1V2_FAMILY_BENCHMARK.tsv": b, "H1V2_PANCANCER_BY_TISSUE.tsv": tissues,
        "H1V2_HCC_TKI.tsv": tki_table(primary),
    }
    for name, payload in outputs.items():
        pd.DataFrame(payload).to_csv(out / name, index=False, sep="\t", float_format="%.9g", na_rep="NA")
    json_write(out / "score_receipt.json", {"score_passes": 1, "bootstrap_draws": DRAWS, "support": support,
                                            "claim_sha256": sha256(study_lock), "freeze_sha256": sha256(frozen),
                                            "predictions_sha256": sha256(predictions), "prelabel_sha256": checked,
                                            "previously_exposed": True,
                                            "outputs": {n: sha256(out / n) for n in outputs}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("freezes", "frozen", "forecasts", "predictions", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--prelabel", type=Path, required=True)
    execute(**vars(parser.parse_args()))
