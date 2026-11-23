"""Three nested inner rounds followed by a sealed, outcome-blind outer fit."""
import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vc_build.io import sha256
from vc_build.v0plus_score import compound_strata

from .inputs import PAN, fold_order, json_write
from .metrics import Retrieval
from .models import (ALPHAS, BASELINES, KS, POLICIES, RIDGES, Fit, assay_prior, chemistry,
                     config, mean_scale, rna_oof, select_anchors)
from .prepare import AUXILIARY, TRANSFER


def config_key(candidate):
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def scaled_assay(y, transfer):
    if not transfer:
        return y, 0., 1.
    low, high = np.nanmin(y), np.nanmax(y)
    return (y - low) / max(high - low, 1e-9), float(low), float(high - low)


def baseline_grid(rna):
    return {
        "B0": [config("mean")],
        "B1": [config("rna")] if rna else [],
        "B2": [config("scalar", base=b) for b in (("mean", "rna") if rna else ("mean",))],
        "B3": [config("affine", base=b) for b in (("mean", "rna") if rna else ("mean",))],
        "B4": [config("knn", neighbors=k) for k in (3, 5, 10)],
    }


def linear_grid(study):
    ranks = (2, 4, 6) if study == "LICOB" else (2, 4, 8, 16)
    estimators = [(m, None) for m in ("sample", "lw", "oas")] + [("svd", rank) for rank in ranks]
    return [config("cov", estimator=m, rank=rank, ridge=penalty, scale=scale)
            for (m, rank), penalty, scale in product(estimators, RIDGES, (False, True))]


def round2_grid(best, outlier, rna, has_pan):
    result = [best]
    for prior in ("pan", "chem"):
        if prior == "pan" and not has_pan:
            continue
        for alpha, ridge in product(ALPHAS, (best["ridge"], .3, 1.)):
            result.append(config("cov", estimator="sample", prior=prior, alpha=alpha,
                                 ridge=ridge, scale=True))
    for transform in ("huber", "winsor", "rank"):
        for clip, ridge in product((1., 2., 3.) if outlier else (1.5, 3.), (.1, .3, 1.)):
            result.append(config("cov", estimator="sample", robust=transform, clip=clip, ridge=ridge, scale=True))
    result.extend(config("rbf", width=width, ridge=ridge)
                  for width, ridge in product((.25, 1., 4.), (.1, 1., 10.)))
    result.extend(config("cosine", ridge=ridge) for ridge in (.1, 1., 10.))
    result.extend(config("als", rank=rank, ridge=ridge) for rank, ridge in product((2, 4, 8), (.1, .3, 1., 3.)))
    if rna:
        result.extend(config("rna_cov", estimator=m, ridge=ridge, scale=scale)
                      for m, ridge, scale in product(("sample", "lw", "oas"), (.1, .3, 1., 3.), (False, True)))
        result.extend(config("rna_mix", ridge=ridge, width=width, weight=weight)
                      for ridge, width, weight in product((.1, 1., 10.), (.5, 2.), (0., .1, .25, .5)))
    return list({config_key(c): c for c in result}.values())


@dataclass
class Split:
    train: np.ndarray
    held: np.ndarray
    fit: Fit
    similarity: np.ndarray
    groups: np.ndarray
    tissue: np.ndarray
    offset: float
    span: float

    def with_policy(self, policy):
        a = select_anchors(self.train, self.similarity, policy)
        self.fit.anchors = a
        q = np.setdiff1d(np.arange(self.held.shape[1]), a)
        ok = np.isfinite(self.held[:, a]).all(axis=1) & (np.isfinite(self.held[:, q]).sum(axis=1) >= 15)
        if ok.sum() < 2:
            raise ValueError(f"{policy} has insufficient complete held anchor packets")
        return q, ok


def get_inputs(root, task):
    study, fold = task.rsplit("-", 1)
    files = json.loads((root / "receipt.json").read_text())["files"]

    def verified(relative):
        path = root / relative
        if sha256(path) != files[relative]:
            raise ValueError(f"Input packet changed: {relative}")
        return path

    train = dict(np.load(verified(f"{study}/{fold}-develop.npz"), allow_pickle=False))
    query = dict(np.load(verified(f"{study}/{fold}-query.npz"), allow_pickle=False))
    meta = pd.read_csv(verified(f"{study}/chemistry.tsv"), sep="\t", index_col=0).fillna("")
    if not np.array_equal(train["panel"], query["panel"]) or list(meta.index) != list(train["panel"]):
        raise ValueError("Fit and query drug identities disagree")
    if set(train["groups"]) & set(query["groups"]):
        raise ValueError("Held evaluation group found in development packet")
    return study, train, query, meta


def make_fit(block, train, held_x, study, chem, similarity):
    public = study in PAN
    transfer = study in {TRANSFER, AUXILIARY}
    y, offset, span = scaled_assay(block["y"][train], transfer)
    mean, sd = mean_scale(y)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Fit has an unsupported drug")
    groups = block["groups"][train]
    arrays = []
    for peer in block["prior_names"]:
        matrix = block[f"prior_{peer}"]
        if public:
            mask = np.isin(block[f"ids_{peer}"], groups)
            matrix = matrix[mask]
        arrays.append(matrix)
    fit = Fit(y, mean, sd, np.empty(8, dtype=int),
              prior_chem=similarity + np.diag(1. - similarity.diagonal()),
              prior_pan=assay_prior(arrays, len(mean)))
    if not transfer:
        fit.rna_fit, fit.rna_oof, fit.rna_cross, fit.rna_gram = rna_oof(y, block["x"][train], held_x, chem, groups)
    return fit, offset, span


def inner_splits(block, study, chem, similarity):
    order = fold_order(block["groups"], "h1-inner")
    splits = []
    for i in range(3):
        val = np.isin(block["groups"], order[i::3])
        fit, offset, span = make_fit(block, ~val, block["x"][val], study, chem, similarity)
        actual = (block["y"][val] - offset) / span
        splits.append(Split(fit.y, actual, fit, similarity, block["groups"][val], block["tissue"][val], offset, span))
    return splits


def evaluate(splits, policy, grid, log, stage, *, k=5):
    keys = [config_key(settings) for settings in grid]
    result = {tag: [] for tag in keys}
    for fold, split in enumerate(splits):
        q, good = split.with_policy(policy)
        actual = split.held[good][:, q]
        measure = Retrieval(actual)
        a = split.fit.anchors
        actual_anchors = split.held[good][:, a[:k]]
        query_pos = np.flatnonzero(good)
        for tag, settings in zip(keys, grid):
            pred = split.fit.predict(actual_anchors, k, settings, rna_index=query_pos)[:, q]
            ndcg, prec = np.nanmean(measure(pred), axis=0)
            result[tag].append((ndcg, prec))
            log.append({"round": stage, "policy": policy, "inner_fold": fold, "k": k, "recipe": tag,
                        "ndcg_at_5": ndcg, "precision_at_5": prec, "n": int(good.sum()),
                        "q_drugs": len(q), "a8_index": "|".join(map(str, a))})
    return {k: np.array(v) for k, v in result.items()}


def best_of(candidates, scores):
    return max(candidates, key=lambda c: (scores[config_key(c)][:, 0].mean(),
                                          scores[config_key(c)][:, 1].mean(), config_key(c)))


def improvement(candidate, baseline):
    gains = candidate - baseline
    return bool(gains[:, 0].mean() >= .002 and (gains[:, 0] > 0).sum() >= 2 and gains[:, 1].mean() >= -.02)


def errors(splits, policy, settings, meta):
    groups = compound_strata(meta.reset_index())
    positions = {n: np.flatnonzero(meta.index.isin(drugs)) for n, drugs in groups.items()}
    rows = []
    outliers = []
    for fold, split in enumerate(splits):
        q, good = split.with_policy(policy)
        a = split.fit.anchors[:5]
        actual = split.held[good]
        observation = actual[:, a]
        pred = split.fit.predict(observation, 5, settings, rna_index=np.flatnonzero(good))
        correction = pred[:, q] - split.fit.mean[q]
        anchor_z = np.abs((observation - split.fit.mean[a]) / split.fit.sd[a])
        outliers.extend(anchor_z.max(axis=1))
        groups_mae = {}
        for group, indices in positions.items():
            sub = np.intersect1d(q, indices)
            if len(sub) and np.isfinite(actual[:, sub]).any():
                groups_mae[group] = float(np.nanmean(np.abs(pred[:, sub] - actual[:, sub])))
        tissue_mae = {}
        tissues = split.tissue[good]
        for tissue in sorted(set(tissues)):
            mask = tissues == tissue
            tissue_mae[tissue] = {"n": int(mask.sum()),
                                  "mae": float(np.nanmean(np.abs(pred[mask][:, q] - actual[mask][:, q])))}
        peak = np.argmax(anchor_z, axis=1)
        perturb = observation.copy()
        perturb[np.arange(len(peak)), peak] += split.fit.sd[a][peak] * 2
        perturbed = split.fit.predict(perturb, 5, settings, rna_index=np.flatnonzero(good))[:, q]
        rows.append({"inner_fold": fold, "drug_differential_correction_variance": float(np.var(correction, axis=1).mean()),
                     "sample_constant_correction_variance": float(np.var(correction.mean(axis=1))),
                     "max_anchor_abs_z_p90": float(np.quantile(anchor_z.max(axis=1), .9)),
                     "two_sd_anchor_noise_mean_abs_query_change": float(np.abs(perturbed - pred[:, q]).mean()),
                     "mechanisms": groups_mae, "tissues": tissue_mae})
    return rows, bool(np.quantile(outliers, .90) > 3.)


def refine(c):
    if c["family"] == "knn":
        return [c]
    ridge = c["ridge"]
    return [c | {"ridge": r} for r in (ridge / 2, ridge, ridge * 2)]


def choose(splits, study, meta):
    log = []
    has_rna = study not in {TRANSFER, AUXILIARY}
    bs = baseline_grid(has_rna)
    bgrid = [c for candidates in bs.values() for c in candidates]
    linear = linear_grid(study)
    by_policy = {}
    for policy in POLICIES:
        scores = evaluate(splits, policy, bgrid + linear, log, 1)
        b = best_of(bgrid, scores)
        candidate = best_of(linear, scores)
        gain = scores[config_key(candidate)] - scores[config_key(b)]
        by_policy[policy] = {"gain": float(gain[:, 0].mean()), "best_baseline": b,
                             "best_linear": candidate, "baseline_ndcg": float(scores[config_key(b)][:, 0].mean()),
                             "linear_ndcg": float(scores[config_key(candidate)][:, 0].mean())}
    random_gain = float(np.mean([by_policy[p]["gain"] for p in POLICIES if p.startswith("random-")]))
    # Randomness is a three-seed control. A selected operational policy is fully deterministic.
    policy = max(POLICIES[:3], key=lambda p: by_policy[p]["gain"])
    r1 = by_policy[policy]["best_linear"]
    diagnostics, outlier = errors(splits, policy, r1, meta)
    grid2 = round2_grid(r1, outlier, has_rna, all(s.fit.prior_pan is not None for s in splits))
    scores2 = evaluate(splits, policy, bgrid + grid2, log, 2)
    anchors = [c for c in grid2 if not c["family"].startswith("rna")]
    rnas = [c for c in grid2 if c["family"].startswith("rna")]
    families = {}
    for candidate in anchors:
        key = (candidate["family"], candidate.get("prior"), candidate.get("robust"))
        families.setdefault(key, []).append(candidate)
    candidates = [best_of(group, scores2) for group in families.values()]
    ranked = sorted(candidates, key=lambda c: scores2[config_key(c)][:, 0].mean(), reverse=True)
    b4 = best_of(bs["B4"], scores2)
    qualifying = [c for c in ranked if improvement(scores2[config_key(c)], scores2[config_key(b4)])]
    dual = best_of(rnas, scores2) if rnas else None
    promising_dual = dual is not None and improvement(
        scores2[config_key(dual)], scores2[config_key(qualifying[0] if qualifying else b4)])
    grid3 = list({config_key(c): c for c in bgrid + [r1] +
                  [c for promising in qualifying[:2] for c in refine(promising)] +
                  (refine(dual) if promising_dual else [dual] if dual is not None else [])}.values())
    scores3 = evaluate(splits, policy, grid3, log, 3)
    baselines = {tag: best_of(options, scores3) for tag, options in bs.items() if options}
    baseline_name = max(baselines, key=lambda n: (scores3[config_key(baselines[n])][:, 0].mean(), -int(n[-1])))
    knn = baselines["B4"]
    primary = best_of([c for c in grid3 if c["family"] in {"knn", "cov", "rbf", "cosine", "als"}], scores3)
    if not improvement(scores3[config_key(primary)], scores3[config_key(knn)]):
        primary = knn
    ancillary = None if dual is None else best_of([c for c in grid3 if c["family"].startswith("rna_")], scores3)
    promote = ancillary is not None and improvement(scores3[config_key(ancillary)], scores3[config_key(primary)])
    for k in (0, 3, 8):
        evaluate(splits, policy, list(baselines.values()) + [primary] + ([ancillary] if promote else []),
                 log, 3, k=k)
    return {"policy": policy, "baseline": baseline_name, "baselines": baselines,
            "primary": primary, "rna_control": ancillary, "rna_secondary": ancillary if promote else None,
            "rna_inner_delta": float((scores3[config_key(ancillary)] - scores3[config_key(primary)])[:, 0].mean())
            if ancillary else None,
            "round1_by_policy": by_policy, "random_3seed_gain": random_gain,
            "round1_diagnostics": diagnostics, "round2_outliers": outlier,
            "round2_family_winners": [{"recipe": c, "ndcg": float(scores2[config_key(c)][:, 0].mean())}
                                      for c in ranked],
            "round3_retained": qualifying[:2],
            "inner_primary_ndcg": float(scores3[config_key(primary)][:, 0].mean()),
            "inner_selected_baseline_ndcg": float(scores3[config_key(baselines[baseline_name])][:, 0].mean())}, log


def freeze(root, run, task):
    study, block, query, meta = get_inputs(root, task)
    directory = run / task
    directory.mkdir(parents=True)
    chem, similarity = chemistry(meta)
    splits = inner_splits(block, study, chem, similarity)
    selection, history = choose(splits, study, meta)
    pd.DataFrame(history).to_csv(directory / "inner.tsv", sep="\t", index=False)
    fit, offset, span = make_fit(block, np.ones(len(block["y"]), dtype=bool), query["x"], study, chem, similarity)
    fit.anchors = select_anchors(fit.y, similarity, selection["policy"])
    arrays = {k: v for k, v in vars(fit).items() if isinstance(v, np.ndarray)}
    np.savez_compressed(directory / "model.npz", **arrays)
    selection.update(task=task, assay_shift=offset, assay_span=span, fit_groups=len(fit.y), outer_groups=len(query["groups"]),
                     anchors=block["panel"][fit.anchors].tolist(),
                     query_panel=np.delete(block["panel"], fit.anchors).tolist(),
                     code_sha256={path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                     data_receipt_sha256=sha256(root / "receipt.json"),
                     artifacts={name: sha256(directory / name) for name in ("model.npz", "inner.tsv")})
    json_write(directory / "freeze.json", selection)
    print(task, selection["policy"], selection["primary"], selection["baseline"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("run", type=Path)
    cli.add_argument("task", nargs="?")
    args = cli.parse_args()
    if args.task is None:
        from os import environ
        args.task = json.loads((args.root / "receipt.json").read_text())["tasks"][int(environ["SLURM_ARRAY_TASK_ID"])]
    freeze(args.root, args.run, args.task)
