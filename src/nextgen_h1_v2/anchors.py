"""Search nested fit-only anchor prefixes on a common withheld drug set."""
import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.linalg import solve

from nextgen_h1_anchor.inputs import json_write
from nextgen_h1_anchor.metrics import Retrieval
from nextgen_h1_anchor.models import select_anchors
from vc_build.io import sha256

from .data import ROOT, STUDY, TASKS, V1, drug_features, partitions, source, training_context, within_drug

POLICIES = ("chemistry", "logdet", "random-971", "random-972", "random-973",
            "Supervised-Greedy-NDCG", "Supervised-Greedy-Dual", "RNA-Residual-LogDet", "RNA-Residual-Greedy")
MODES = ("anchor", "dual")


def extend_reserve(ctx, current, count):
    remaining = np.setdiff1d(np.arange(len(ctx.mean)), current).tolist()
    selected = list(current)
    coverage = np.isfinite(ctx.train).mean(axis=0)
    for _ in range(count):
        # Prevent auxiliary support collapse at K=12.
        largest = max(coverage[remaining])
        allowed = [x for x in remaining if coverage[x] >= max(.80, largest - .035)]
        if not allowed:
            allowed = remaining
        choice = max(allowed, key=lambda j: (coverage[j] - .035 * ctx.similarity[j, selected].max(), -j))
        selected.append(choice)
        remaining.remove(choice)
    return np.array(selected)


def conditional(ctx, obs, order, dual):
    base = ctx.bvalid if dual else np.broadcast_to(ctx.mean, (len(obs), len(ctx.mean)))
    cov = ctx.cov_rna if dual else ctx.cov
    if not len(order):
        return base
    centered = (obs - base[:, order]) / ctx.sd[order]
    weights = solve(cov[np.ix_(order, order)] + np.eye(len(order)) * .3, cov[order], assume_a="pos")
    return base + (np.nan_to_num(centered) @ weights) * ctx.sd


@dataclass
class SearchSplit:
    context: object
    good: np.ndarray
    actual: np.ndarray
    measure: object


def search_splits(parts, reserve, query):
    for part in parts:
        ok = np.isfinite(part.valid[:, reserve]).all(axis=1) & (np.isfinite(part.valid[:, query]).sum(1) >= 15)
        if ok.sum() >= 2:
            y = part.valid[ok][:, query]
            yield SearchSplit(part, ok, y, Retrieval(y))


def greedy(parts, reserve, q, *, dual, joint, size=8):
    pieces = list(search_splits(parts, reserve, q))
    if len(pieces) < 2:
        raise ValueError("At least two nested subfolds must support the common anchor packet")

    def utility(order):
        scores = []
        for block in pieces:
            p = conditional(block.context, block.context.valid[:, order], order, dual)
            y, pred = block.actual, p[block.good][:, q]
            ndcg = np.nanmean(block.measure(pred)[:, 0])
            scores.append((ndcg, within_drug(y, pred) if joint else 0.))
        return np.mean(scores, axis=0)

    def winner(options, reference):
        values = np.array([utility(option) for option in options]) - utility(reference)
        if joint:
            values = ((values - values.mean(0)) / np.maximum(values.std(0), 1e-9)).mean(1)
        else:
            values = values[:, 0]
        return options[int(np.argmax(values))]

    order = []
    for k in range(1, size + 1):
        possibilities = [order + [int(j)] for j in reserve if j not in order]
        order = winner(possibilities, order)
        if k == 5:
            candidates = [order]
            for i in range(5):
                candidates.extend(order[:i] + [int(a)] + order[i + 1:] for a in reserve if a not in order)
            order = winner(candidates, order)
    return np.array(order)


def restricted_order(ctx, reserve, policy):
    similarity = ctx.similarity[np.ix_(reserve, reserve)]
    values = ctx.train - ctx.bfit if policy == "RNA-Residual-LogDet" else ctx.train
    method = "logdet" if policy == "RNA-Residual-LogDet" else policy
    return reserve[select_anchors(values[:, reserve], similarity, method)]


def nested_contexts(study, dev, fit_mask, drug):
    local = {k: v[fit_mask] if k in {"x", "y", "groups", "tissue"} else v for k, v in dev.items()}
    return [training_context(study, local, fit, val, drug, pca=False)
            for fit, val in partitions(local["groups"], "h1v2-anchor-selector")]


def order_policy(ctx, sub, reserve, policy):
    if policy.startswith("Supervised") or policy == "RNA-Residual-Greedy":
        q = np.setdiff1d(np.arange(len(ctx.mean)), reserve)
        return greedy(sub, reserve, q, dual=policy == "RNA-Residual-Greedy", joint=policy == "Supervised-Greedy-Dual")
    return restricted_order(ctx, reserve, policy)


def row_stats(ctx, order, k, q, dual):
    good = np.isfinite(ctx.valid[:, order]).all(axis=1) & (np.isfinite(ctx.valid[:, q]).sum(1) >= 15)
    actual = ctx.valid[good][:, q]
    if good.sum() < 2:
        return None
    pred = conditional(ctx, ctx.valid[:, order[:k]], order[:k], dual)[good][:, q]
    base = conditional(ctx, ctx.valid[:, :0], order[:0], dual)[good][:, q]
    primary = Retrieval(actual)
    ndcg, precision = np.nanmean(primary(pred), axis=0)
    rho = within_drug(actual, pred)
    left = actual - base
    right = actual - pred
    return {"ndcg_at_5": ndcg, "precision_at_5": precision, "within_drug_spearman": rho,
            "delta_ndcg": ndcg - np.nanmean(primary(base)[:, 0]),
            "delta_rho": rho - within_drug(actual, base),
            "residual_r2": 1 - np.nansum(right ** 2) / max(np.nansum(left ** 2), 1e-12),
            "n": int(good.sum()), "q": len(q)}


def chemistry_diversity(ctx, panel, meta):
    pairs = np.triu_indices(len(panel), k=1)
    similarity = ctx.tanimoto[np.ix_(panel, panel)][pairs]
    mechanisms = meta.iloc[panel].drug_target_source_labels.str.lower().str.strip()
    return {"tanimoto_diversity": float(1 - similarity.mean()),
            "distinct_mechanism_annotations": int(mechanisms[mechanisms.ne("")].nunique())}


def run(task, out, root=STUDY):
    study, dev, meta = source(task, root=root)
    drug = drug_features(meta)
    out.mkdir(parents=True)
    records = []
    states = []
    for j, (fit, val) in enumerate(partitions(dev["groups"], "h1v2-inner")):
        ctx = training_context(study, dev, fit, val, drug)
        sub = nested_contexts(study, dev, fit, drug)
        reserve = select_anchors(ctx.train, ctx.similarity, "chemistry")
        query = np.setdiff1d(np.arange(len(ctx.mean)), reserve)
        flags = [p for p in POLICIES if not ctx.levels or not p.startswith("RNA-")]
        orders = {p: order_policy(ctx, sub, reserve, p) for p in flags}
        all_orders = {name: extend_reserve(ctx, value, 4) for name, value in orders.items()}
        for name, order in orders.items():
            for modality in ("anchor",) if ctx.levels else MODES:
                stats = row_stats(ctx, order, 5, query, modality == "dual")
                records.append(dict(task=task, fold=j, policy=name, modality=modality, scope="fixed8",
                                    anchor5="|".join(dev["panel"][order[:5]]), **stats,
                                    **chemistry_diversity(ctx, order[:5], meta)))
        if study == "LICOB":
            wide = extend_reserve(ctx, reserve, 8)
            q = np.setdiff1d(np.arange(len(ctx.mean)), wide)
            for policy, dual, both in (("Supervised-Greedy-NDCG", False, False),
                                       ("Supervised-Greedy-Dual", False, True), ("RNA-Residual-Greedy", True, False)):
                picked = greedy(sub, wide, q, dual=dual, joint=both)
                stats = row_stats(ctx, wide, 5, q, dual) if np.array_equal(picked[:5], wide[:5]) else None
                if stats is None:
                    common = np.r_[picked, np.setdiff1d(wide, picked)]
                    stats = row_stats(ctx, common, 5, q, dual)
                if stats:
                    records.append(dict(task=task, fold=j, policy=policy, modality="dual" if dual else "anchor",
                                        scope="inner-reserve16", anchor5="|".join(dev["panel"][picked[:5]]), **stats,
                                        **chemistry_diversity(ctx, picked[:5], meta)))
        states.append((ctx, sub, all_orders, np.flatnonzero(fit), np.flatnonzero(val)))
        print(task, "round1", j, flush=True)
    frame = pd.DataFrame(records)
    winners = {}
    for mode, subset in frame[(frame.scope == "fixed8") & ~frame.policy.str.startswith("random")].groupby("modality"):
        stats = subset.groupby("policy")[["ndcg_at_5", "within_drug_spearman"]].mean()
        stats["utility"] = stats.ndcg_at_5 + (.12 if study == "LICOB" else .06) * stats.within_drug_spearman
        winners[mode] = stats.sort_values("utility", ascending=False).index[0]
    frame.to_csv(out / "anchors.tsv", index=False, sep="\t")
    with (out / "study.pkl").open("xb") as stream:
        pickle.dump((states, winners, list(meta.index), list(meta.source_compound_label),
                     list(meta.drug_target_source_labels)), stream, protocol=5)
    json_write(out / "round1.json", {"task": task, "winners": winners, "inner_folds": len(states),
                                     "rows": len(frame), "source_receipt_sha256": sha256(root / "receipt.json"),
                                     "table_sha256": sha256(out / "anchors.tsv"),
                                     "cache_sha256": sha256(out / "study.pkl"), "query_labels_opened": False})


def preflight(out):
    stamp = V1 / "score-v1/score_receipt.json"
    score = json.loads(stamp.read_text())
    if score["score_passes"] != 1 or score["input_sha256"] != sha256(STUDY / "receipt.json"):
        raise ValueError("Original study does not match the signed v1 benchmark")
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "torch", "lightgbm",
                                                         "pandas", "rdkit")}
    json_write(out, {"data_sha256": score["input_sha256"], "v1_score_sha256": sha256(stamp),
                     "groups": {t: score["masks"][t]["eligible"] for t in TASKS},
                     "versions": versions, "installed_packages": [], "outer_labels_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    sub = subs.add_parser("preflight")
    sub.add_argument("out", type=Path)
    sub = subs.add_parser("run")
    sub.add_argument("out", type=Path)
    sub.add_argument("task", nargs="?")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.out)
    else:
        from os import environ
        task = args.task if args.task else TASKS[int(environ["SLURM_ARRAY_TASK_ID"])]
        run(task, args.out / task)
