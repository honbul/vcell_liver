"""Run only dose-gated, structure-disjoint HepaRG prediction and isolated scoring."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.kernels import _neighbor_weights, _tanimoto

from .heparg_eval import PRIMARY, PROTOCOL, SEED, center, load_dataset, panel, table, target
from .heparg_prepare import parent
from .transfer_models import fingerprint


ARM_NAMES = ("kernel", "mechanism", "dose_shared")
ALPHA = (0.1, 1.0, 10.0)
NEIGHBORS = (3, 8)
ANNOTATIONS = Path("/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet")


def annotated(rows):
    catalogue = pd.read_parquet(ANNOTATIONS, columns=["canonical_smiles", "targets", "moa-fine"]).fillna("")
    by_parent = defaultdict(set)
    relevant = set(rows.parent_key)
    for smiles, targets, moa in catalogue.itertuples(index=False, name=None):
        parsed = parent(smiles)
        if not parsed or parsed[0] not in relevant:
            continue
        by_parent[parsed[0]].update(f"target:{word.strip().casefold()}" for word in targets.split(",") if word.strip())
        if moa.casefold().strip() not in ("", "unknown", "unclear"):
            by_parent[parsed[0]].add(f"moa:{moa.strip().casefold()}")
    return by_parent


def profiles(rows, delta, idx, columns):
    row, y = target(rows, delta[np.ix_(idx, columns)], idx)
    conditions = row.groupby(["parent_key", "dose_rank"], sort=True).indices
    part = row.iloc[[members[0] for members in conditions.values()]].reset_index(drop=True)
    y = np.stack([y[members].mean(axis=0) for members in conditions.values()])
    return part, y


def group_average(rows, y):
    conditions = rows.groupby("parent_key", sort=True).indices
    meta = rows.iloc[[members[0] for members in conditions.values()]].reset_index(drop=True)
    return meta, np.stack([y[members].mean(axis=0) for members in conditions.values()])


def models(fit, actual, query, rank, annotations):
    take = fit.dose_rank.eq(rank).to_numpy()
    result, current = {}, query.reset_index(drop=True)
    qnames, qfp = fingerprint(current)
    for arm, use in (("kernel", take), ("dose_shared", np.ones(len(fit), dtype=bool)), ("mechanism", take)):
        if not use.any():
            for size in (NEIGHBORS if arm == "mechanism" else ALPHA):
                result[arm, size] = np.zeros((len(current), actual.shape[1]), dtype=np.float32)
            continue
        f, y = group_average(fit.loc[use].reset_index(drop=True), actual[use])
        fnames, ffp = fingerprint(f)
        if set(qnames) & set(fnames):
            raise ValueError("Held-out compound entered HepaRG donor fit")
        qi = pd.Index(qnames).get_indexer(current.parent_key)
        fi = pd.Index(fnames).get_indexer(f.parent_key)
        if arm == "mechanism":
            sim = np.zeros((len(qnames), len(fnames)), dtype=float)
            for i, key in enumerate(qnames):
                a = annotations[key]
                if a:
                    sim[i] = [len(a & annotations[p]) / len(a | annotations[p]) for p in fnames]
            for k in NEIGHBORS:
                weight = _neighbor_weights(sim[qi][:, fi], k, 2)
                total = weight.sum(axis=1, keepdims=True)
                weight /= np.where(total > 0, total, 1)
                result[arm, k] = weight @ y
            continue
        gram, cross = [sum(w * _tanimoto(a, b) for w, a, b in zip((0.7, 0.3), current_fp, ffp))
                       for current_fp in (ffp, qfp)]
        gram = gram[np.ix_(fi, fi)]
        cross = cross[np.ix_(qi, fi)]
        mean = gram.mean(axis=0)
        overall = mean.mean()
        gram = gram - mean[None, :] - mean[:, None] + overall
        cross = cross - mean[None, :] - cross.mean(axis=1, keepdims=True) + overall
        yc = y - y.mean(axis=0)
        for alpha in ALPHA:
            diagonal = gram + alpha * np.eye(len(gram))
            result[arm, alpha] = cross @ cho_solve(cho_factor(diagonal), yc)
    return result


def active_query(rows, idx, rank):
    chosen = rows.iloc[idx].loc[lambda fr: fr.dose_rank.eq(rank)].copy()
    active = chosen.groupby("plate").parent_key.transform("nunique").ge(3)
    return chosen.loc[active].reset_index(drop=True), chosen.loc[active].index.to_numpy()


def group_score(rows, error):
    return float(pd.DataFrame({"g": rows.parent_key, "r": np.sqrt(np.mean(error ** 2, axis=1))}).groupby("g").r.mean().mean())


def gate_path(root):
    return root / "HepaRG/repeat_296254/results.json"


def pred_dir(args):
    return args.root / "HepaRG" / f"predictions{args.run_tag}"


def frozen_path(args):
    return args.root / "HepaRG" / f"freeze{args.run_tag}.json"


def freeze_inputs(root, gate, out, files, outcomes):
    write_json(out / "receipt.json", dict(protocol_sha256=PROTOCOL, gate_sha256=sha256(gate),
               data_receipt_sha256=sha256(root / "data/heparg/receipt.json"),
               prediction_files={name: sha256(out / name) for name in files}, dose_ranks=outcomes))


def predict(args):
    gate = gate_path(args.root)
    repeat = read_json(gate)
    if repeat["protocol_sha256"] != PROTOCOL:
        raise ValueError("Repeat gate predates the active protocol")
    passing = repeat["passed_dose_ranks"]
    args.out.mkdir(parents=True, exist_ok=False)
    if not passing:
        freeze_inputs(args.root, gate, args.out, [], passing)
        return
    folder, raw, genes, delta, controls = load_dataset(args.root, PRIMARY)
    if sha256(folder / "receipt.json") != repeat["cache_receipt_sha256"]:
        raise ValueError("Repeat gate used a different assay cache")
    rows = raw.loc[raw.parent_key.ne("")].reset_index(drop=True)
    src = raw.loc[raw.parent_key.ne("")].index.to_numpy()
    if len(rows) != len(raw):
        # Only well IDs, structures and source-aligned index addresses enter the split.
        delta, controls = delta[src], controls[src]
    rows["repeat_group"] = rows.parent_key
    folds = rows.outer_fold.to_numpy()
    annotation = annotated(rows)
    files = []
    for outer in range(3):
        train = np.flatnonzero(folds != outer)
        held = np.flatnonzero(folds == outer)
        col = panel(delta, controls, train)
        fit, y = profiles(rows, delta, train, col)
        internals = []
        for validation in range(3):
            if validation == outer:
                continue
            eligible = np.flatnonzero((folds != outer) & (folds != validation))
            ic = panel(delta, controls, eligible)
            ir, iy = profiles(rows, delta, eligible, ic)
            internals.append((np.flatnonzero(folds == validation), ic, ir, iy))
        for rank in passing:
            query, hold_index = active_query(rows, held, rank)
            if query.empty:
                continue
            scores = defaultdict(list)
            for val, cols, ir, iy in internals:
                vr, vi = active_query(rows, val, rank)
                if vr.empty:
                    continue
                obs = delta[np.ix_(vi, cols)]
                predictions = models(ir, iy, vr, rank, annotation)
                for spec, prediction in predictions.items():
                    scores[spec].append(group_score(vr, center(vr, obs - prediction)))
            predictions = models(fit, y, query, rank, annotation)
            output = {}
            selections = {}
            for arm in ARM_NAMES:
                candidates = [spec for spec, performance in scores.items() if spec[0] == arm and performance]
                if candidates:
                    best = min(candidates, key=lambda spec: (np.mean(scores[spec]), spec[1]))
                    output[arm] = center(query, predictions[best]).astype(np.float32)
                    selections[arm] = best[1]
            annotated_rows = np.array([bool(annotation[g]) for g in query.parent_key])
            name = f"pred-fold{outer}-rank{rank}"
            np.savez_compressed(args.out / f"{name}.npz", indices=src[hold_index], columns=col,
                                annotation=annotated_rows, **output)
            write_json(args.out / f"{name}.json", dict(fold=outer, dose_rank=rank,
                       genes=genes.ensembl_id.iloc[col].tolist(), selections=selections,
                       rows=len(query), groups=int(query.parent_key.nunique())))
            files.extend((f"{name}.npz", f"{name}.json"))
    freeze_inputs(args.root, gate, args.out, files, passing)


def freeze(args):
    folder = pred_dir(args)
    receipt = read_json(folder / "receipt.json")
    if receipt["protocol_sha256"] != PROTOCOL or sha256(gate_path(args.root)) != receipt["gate_sha256"]:
        raise ValueError("Frozen predictor or HepaRG repeat gate changed")
    for name, digest in receipt["prediction_files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Unverified prediction: {name}")
    write_json(args.out, dict(protocol_sha256=PROTOCOL, prediction_receipt_sha256=sha256(folder / "receipt.json"),
               **{k: v for k, v in receipt.items() if k != "protocol_sha256"}))


def report(rows):
    scores = rows.groupby(["parent_key"])[["b0", "rmse"]].mean().to_numpy()
    points = scores.mean(axis=0)
    n = len(scores)
    repl = scores[np.random.default_rng(SEED).integers(n, size=(10000, n))].mean(axis=1)
    effect = float(100 * (1 - points[1] / points[0]))
    ci = np.quantile(100 * (1 - repl[:, 1] / repl[:, 0]), [0.025, 0.975]).tolist()
    return dict(groups=n, rows=len(rows), b0_rmse=float(points[0]), rmse=float(points[1]), pct=effect, ci95=ci)


def score(args):
    frozen = read_json(frozen_path(args))
    if frozen["protocol_sha256"] != PROTOCOL or sha256(gate_path(args.root)) != frozen["gate_sha256"]:
        raise ValueError("Scoring requires the prespecified gate and sealed predictions")
    folder = pred_dir(args)
    if sha256(folder / "receipt.json") != frozen["prediction_receipt_sha256"]:
        raise ValueError("Prediction receipt differs from freeze")
    if not frozen["dose_ranks"]:
        write_json(args.out, dict(status="gate_not_met", protocol_sha256=PROTOCOL,
                   freeze_sha256=sha256(frozen_path(args)), results=[]))
        return
    source, rows, _, delta, _ = load_dataset(args.root, PRIMARY)
    if sha256(source / "receipt.json") != frozen["data_receipt_sha256"]:
        raise ValueError("Score cache differs from frozen prediction cohort")
    all_metrics = defaultdict(list)
    for name, digest in frozen["prediction_files"].items():
        f = folder / name
        if sha256(f) != digest:
            raise ValueError(f"Prediction changed after freeze: {name}")
        if not name.endswith(".npz"):
            continue
        m = read_json(f.with_suffix(".json"))
        pred = np.load(f, allow_pickle=False)
        query = rows.iloc[pred["indices"]].reset_index(drop=True)
        if not query.outer_fold.eq(m["fold"]).all() or not query.dose_rank.eq(m["dose_rank"]).all():
            raise ValueError("Fold, row or dose provenance mismatch")
        obs = center(query, delta[np.ix_(pred["indices"], pred["columns"])])
        b0 = np.sqrt(np.mean(obs**2, axis=1))
        for arm in ARM_NAMES:
            if arm not in pred:
                continue
            score = query[["parent_key"]].assign(b0=b0, rmse=np.sqrt(np.mean((obs - pred[arm])**2, axis=1)))
            all_metrics[m["dose_rank"], arm, "all"].append(score)
            if arm == "mechanism" and pred["annotation"].any():
                all_metrics[m["dose_rank"], arm, "annotated"].append(score.loc[pred["annotation"]])
    result = [dict(dose_rank=key[0], arm=key[1], scope=key[2],
                   **report(pd.concat(slices, ignore_index=True)))
              for key, slices in sorted(all_metrics.items())]
    write_json(args.out, dict(status="completed", protocol_sha256=PROTOCOL,
               freeze_sha256=sha256(frozen_path(args)), results=result))
