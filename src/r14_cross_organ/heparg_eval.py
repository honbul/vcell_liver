"""Measure HepaRG repeat ceilings, then enforce the conditional prediction gate."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json

from .heparg_prepare import PRIMARY


SEED = 130113
GAINS = np.arange(9) / 8
POOLS = ["plate", "dose_rank"]
PROTOCOL = "09774371b5965afbee825e37f955756e90b84700985605a01d1f6c6230c7f63c"


def table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def load_dataset(root, dataset):
    folder = root / "data" / ("heparg" if dataset == PRIMARY else "heparg_284321")
    rec = read_json(folder / "receipt.json")
    if rec["protocol_sha256"] != PROTOCOL:
        raise ValueError("Execution protocol changed")
    manifest = root / "data/heparg_compound_folds.tsv"
    if sha256(manifest) != rec["manifest_sha256"]:
        raise ValueError("The shared HepaRG chemistry manifest changed")
    for name, digest in rec["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"HepaRG data changed: {name}")
    rows, genes = [table(folder / name) for name in ("rows.tsv", "genes.tsv")]
    arrays = [np.load(folder / name, mmap_mode="r", allow_pickle=False) for name in ("delta.npy", "control.npy")]
    if any(x.shape != (len(rows), len(genes)) for x in arrays):
        raise ValueError("Expression axes disagree with the receipt")
    return folder, rows, genes, *arrays


def panel(delta, control, idx):
    variance = np.zeros(delta.shape[1], dtype=np.float64)
    for start in range(0, delta.shape[1], 1000):
        sl = slice(start, start + 1000)
        variance[sl] = (delta[idx, sl] + control[idx, sl]).var(axis=0, dtype=np.float64)
    order = np.argsort(-variance, kind="stable")
    return order[variance[order] > 0][:2000]


def target(rows, values, idx):
    rows = rows.iloc[idx].reset_index(drop=True)
    active = rows.groupby(POOLS).repeat_group.transform("nunique").ge(3).to_numpy()
    out = rows.loc[active].reset_index(drop=True)
    return out, center(out, values[active].copy())


def center(rows, values):
    counts = rows.groupby(POOLS + ["repeat_group"]).repeat_group.transform("size").to_numpy()
    for inds in rows.groupby(POOLS, sort=False).indices.values():
        values[inds] -= np.average(values[inds], axis=0, weights=1 / counts[inds])
    return values


def positions(rows, seeds=(41, 42, 43)):
    coords = {"position": rows.position.to_numpy(copy=True)}
    all_coords = rows.position.to_numpy()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = coords[f"position_perm_{seed}"] = all_coords.copy()
        for idx in rows.groupby("plate", sort=True).indices.values():
            perm[idx] = rng.permutation(all_coords[idx])
    return coords


def position_predict(fit, y, query, fit_coord, query_coord):
    lookup = defaultdict(list)
    fit_plate = fit.plate.to_numpy()
    for j, loc in enumerate(fit_coord):
        lookup[loc].append(j)
    guess, count = np.zeros((len(query), y.shape[1]), dtype=y.dtype), np.zeros(len(query), dtype=int)
    for i, (plate, coordinate) in enumerate(zip(query.plate, query_coord)):
        uses = [j for j in lookup[coordinate] if fit_plate[j] != plate]
        if not uses:
            continue
        parts = defaultdict(list)
        for j in uses:
            parts[fit_plate[j]].append(j)
        guess[i] = np.array([y[ix].mean(axis=0) for ix in parts.values()]).mean(axis=0)
        count[i] = len(parts)
    return guess, count


def raw_pairs(rows, values, require_culture):
    lookup = defaultdict(list)
    for i, keys in enumerate(zip(rows.identity, rows.chemical, rows.dose_uM)):
        lookup[keys].append(i)
    plate, loc, culture = (rows[f].to_numpy() for f in ("plate", "position", "culture"))
    held, pred, pair_n = [], [], []
    for indices in lookup.values():
        for i in indices:
            donors = defaultdict(list)
            for j in indices:
                if plate[i] == plate[j] or loc[i] == loc[j]:
                    continue
                if require_culture and (not culture[i] or not culture[j] or culture[i] == culture[j]):
                    continue
                donors[plate[j]].append(j)
            if donors:
                held.append(i)
                pred.append(np.mean([values[ix].mean(axis=0) for ix in donors.values()], axis=0))
                pair_n.append(sum(map(len, donors.values())))
    keep = np.array(held, dtype=int)
    return rows.iloc[keep].reset_index(drop=True), values[keep], np.array(pred).reshape(-1, values.shape[1]), pair_n


def features(rows, actual, predicted):
    y2 = np.mean(actual ** 2, axis=1)
    yp = np.mean(actual * predicted, axis=1)
    p2 = np.mean(predicted ** 2, axis=1)
    denom = np.sqrt(y2 * p2)
    result = rows[["repeat_group", "dose_rank", "dose_uM"]].copy()
    result["cosine"] = np.divide(yp, denom, out=np.zeros_like(yp), where=denom > 1e-16)
    rss = y2[:, None] - 2 * yp[:, None] * GAINS + p2[:, None] * GAINS ** 2
    for idx, col in enumerate(np.sqrt(np.maximum(rss, 0)).T):
        result[f"z{idx}"] = col
    return result


def summarize(frame):
    if frame.empty:
        return {"groups": 0, "rows": 0}
    selected = ["cosine", "z0", "z2", "z8", "selected"]
    groups = frame.groupby("repeat_group", sort=True)[selected].mean().to_numpy()
    n = len(groups)
    point = groups.mean(axis=0)
    boot = np.empty((10000, len(selected)))
    rng = np.random.default_rng(SEED)
    for i in range(0, len(boot), 256):
        part = boot[i:i + 256]
        part[:] = groups[rng.integers(n, size=(len(part), n))].mean(axis=1)
    reductions = 100 * (1 - point[2:] / point[1])
    ci = np.quantile(100 * (1 - boot[:, 2:] / boot[:, [1]]), [0.025, 0.975], axis=0).T
    return {
        "groups": n, "rows": len(frame), "dose_min_uM": float(frame.dose_uM.min()),
        "dose_max_uM": float(frame.dose_uM.max()), "cosine": float(point[0]),
        "cosine_ci95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
        "b0_rmse": float(point[1]),
        **{name: dict(rmse=float(point[i + 2]), reduction_pct=float(reductions[i]), ci95=ci[i].tolist())
           for i, name in enumerate(("quarter", "full", "selected"))},
    }


def best_by_dose(frame):
    cols = [f"z{i}" for i in range(len(GAINS))]
    bank = frame.groupby(["dose_rank", "repeat_group"])[cols].mean().groupby("dose_rank").mean()
    return {int(rank): int(np.argmin(row.to_numpy())) for rank, row in bank.iterrows()}


def freeze_gains(fold_frame, values):
    gains = best_by_dose(pd.concat(values, ignore_index=True))
    fold_frame["selected"] = [fold_frame[f"z{gains.get(rank, 0)}"].iloc[i]
                              for i, rank in enumerate(fold_frame.dose_rank)]
    return gains


def repeat_block(rows, delta, controls, tr, te, coords, require_culture):
    use = panel(delta, controls, tr)
    query, obs = target(rows, delta[np.ix_(te, use)], te)
    train, fit = target(rows, delta[np.ix_(tr, use)], tr)
    lookup = pd.Index(rows.library)
    take = lookup.get_indexer(query.library)
    kept = lookup.get_indexer(train.library)
    matched, observed, direct, pair_n = raw_pairs(query, obs, require_culture)
    result = {"repeat": features(matched, observed, direct)}
    result["repeat"]["directed_supported_well_pairs"] = pair_n
    for method, coordinate in coords.items():
        guess, n = position_predict(train, fit, query, coordinate[kept], coordinate[take])
        result[method] = features(query, obs, center(query, guess))
        result[method]["supported"] = n > 0
    return result, len(use)


def vehicle_positions(folder):
    rows = table(folder / "vehicles.tsv")
    full = np.load(folder / "vehicles.npy", mmap_mode="r", allow_pickle=False)
    coords = positions(rows)
    by_method = defaultdict(list)
    for plate, test in rows.groupby("plate", sort=True).indices.items():
        if len(test) < 2:
            continue
        fitting = np.flatnonzero(rows.plate.ne(plate))
        left = panel(full, np.broadcast_to(np.float32(0), full.shape), fitting)
        x, y = full[np.ix_(fitting, left)], full[np.ix_(test, left)]
        f, q = rows.iloc[fitting].reset_index(drop=True), rows.iloc[test].reset_index(drop=True)
        for idx in f.groupby("plate", sort=False).indices.values():
            x[idx] -= x[idx].mean(axis=0)
            if len(idx) > 1:
                x[idx] *= len(idx) / (len(idx) - 1)
        y = (y - y.mean(axis=0)) * len(y) / (len(y) - 1)
        q = q.assign(repeat_group=plate, dose_rank=0, dose_uM=0)
        for method, location in coords.items():
            pp, support = position_predict(f, x, q, location[fitting], location[test])
            shape = features(q, y, pp)
            shape["selected"] = shape.z2
            shape["supported"] = support > 0
            by_method[method].append(shape)
    result = {}
    for method, parts in by_method.items():
        frame = pd.concat(parts, ignore_index=True)
        result[method] = {**summarize(frame), "supported_rows": int(frame.supported.sum())}
    return result


def repeat(args):
    folder, rows, _, delta, controls = load_dataset(args.root, args.dataset)
    cc = positions(rows)
    folds = rows.repeat_fold.to_numpy()
    if set(folds) != {0, 1, 2}:
        raise ValueError("Incorrect structure split")
    require_culture = args.dataset == PRIMARY
    scores, settings = defaultdict(list), []
    for outer in range(3):
        fit, query = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        frame, panel_size = repeat_block(rows, delta, controls, fit, query, cc, require_culture)
        inner = [
            repeat_block(rows, delta, controls, np.flatnonzero((folds != outer) & (folds != valid)),
                         np.flatnonzero(folds == valid), cc, require_culture)[0]
            for valid in range(3) if valid != outer
        ]
        selected = {}
        for method, part in frame.items():
            selected[method] = {str(rank): float(GAINS[gain])
                                for rank, gain in freeze_gains(part, [it[method] for it in inner]).items()}
            scores[method].append(part)
        settings.append(dict(fold=outer, panel_genes=panel_size, by_method=selected))
    by_method = {}
    for method, parts in scores.items():
        reports = {}
        frame = pd.concat(parts, ignore_index=True)
        for rank, part in frame.groupby("dose_rank"):
            s = summarize(part)
            if method == "repeat":
                s["directed_supported_well_pairs"] = int(part.directed_supported_well_pairs.sum())
                s["passes"] = bool(s["groups"] >= 10 and s["quarter"]["ci95"][0] > 1)
            else:
                s["supported_rows"] = int(part.supported.sum())
            reports[str(rank)] = s
        by_method[method] = reports
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "results.json"
    write_json(path, dict(dataset=args.dataset, protocol_sha256=PROTOCOL,
                         cache_receipt_sha256=sha256(folder / "receipt.json"),
                         expected_culture_independence=require_culture,
                         coordinated_position_source="GEO original library A01-P24 coordinate",
                         positions_by_vehicle=vehicle_positions(folder),
                         methods=by_method, selected_gains=settings,
                         passed_dose_ranks=[int(r) for r, x in by_method["repeat"].items()
                                            if require_culture and x["passes"]]))


def cross(args):
    left = load_dataset(args.root, PRIMARY)
    right = load_dataset(args.root, "GSE284321")
    a, b = [pd.Index(x[2].ensembl_id) for x in (left, right)]
    common = a.intersection(b)
    keys = ["parent_key", "dose_uM"]
    results = []
    for (_, rows, _, delta, _), genes in zip((left, right), (a, b)):
        linked = rows.loc[rows.parent_key.ne("")].groupby(keys).indices
        keeps = np.flatnonzero(rows.parent_key.ne(""))
        codes = genes.get_indexer(common)
        results.append({k: delta[np.ix_(keeps[v], codes)].mean(axis=0) for k, v in linked.items()})
    pairs = sorted(set(results[0]) & set(results[1]))
    a, b = [np.stack([r[k] for k in pairs]) for r in results]
    n = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    direction = np.divide(np.einsum("ij,ij->i", a, b), n, out=np.zeros_like(n), where=n > 1e-10)
    pairs = pd.DataFrame(pairs, columns=keys)
    pairs["cosine"] = direction
    pair_mu = pairs.groupby("parent_key").cosine.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = pair_mu[rng.integers(len(pair_mu), size=(10000, len(pair_mu)))].mean(axis=1)
    write_json(args.out, dict(protocol_sha256=PROTOCOL, matched_conditions=len(pairs), groups=len(pair_mu),
                              common_genes=len(common), mean_cosine=float(pair_mu.mean()),
                              ci95=np.quantile(boot, [0.025, 0.975]).tolist()))


def delegate(args):
    from . import heparg_model

    {"predict": heparg_model.predict, "freeze": heparg_model.freeze, "score": heparg_model.score}[args.stage](args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, entry in (("repeat", repeat), ("cross", cross), ("predict", delegate),
                        ("freeze", delegate), ("score", delegate)):
        sub = stages.add_parser(name)
        for arg in ("root", "protocol", "out"):
            sub.add_argument(f"--{arg}", type=Path, required=True)
        sub.add_argument("--dataset", default=PRIMARY)
        sub.add_argument("--run-tag", default="")
        sub.set_defaults(entry=entry)
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("Prespecified HepaRG protocol changed")
    args.entry(args)


if __name__ == "__main__":
    main()
