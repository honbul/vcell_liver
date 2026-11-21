"""Run the registered HepaRG design, prediction, freeze and scoring stages."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r14_cross_organ.heparg_eval import SEED, center, load_dataset, table
from src.r14_cross_organ.heparg_model import ANNOTATIONS, annotated
from src.r14_cross_organ.heparg_prepare import PRIMARY, SECONDARY

from .heparg_design import enrich, metadata_out_of_plate, plate_pairs, released_design
from .heparg_models import (
    MODELS, PERM_SEEDS, RESIDUAL, Kernel, candidates, location_candidates, parent_groups, permute,
    pick_settings, score_inner, strata,
)


DATASETS = (PRIMARY, SECONDARY)
PRIMARY_STRATA = ("rank7", "rank8", "ge10uM")
PROTOCOL_SHA256 = "129af982566410cf860bd494cda4ea526105569a716c171221d332f67fc13dd2"
PROTOCOL_COMMIT = "18163c80b83de37fa3eee5b5e1a9258ad12c57b1"


def receipt(folder, names):
    return {str(p): sha256(folder / p) for p in sorted(names)}


def verify(folder, digests):
    if receipt(folder, digests) != digests:
        raise ValueError(f"Sealed files changed in {folder}")


def read_source(args, dataset):
    folder, rows, genes, delta, control = load_dataset(args.cache, dataset)
    manifest = args.index / f"{dataset}_bulk_sample_index.tsv"
    if sha256(manifest) != read_json(folder / "receipt.json")["metadata_sha256"]:
        raise ValueError(f"Released layout for {dataset} differs from the cache")
    design = released_design(args.index, dataset)
    rows = enrich(rows, design)
    return folder, rows, pd.Index(genes.ensembl_id), delta, control


def design_stage(args):
    out = args.work / "design"
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    for dataset in DATASETS:
        folder = args.cache / "data" / ("heparg" if dataset == PRIMARY else "heparg_284321")
        manifest = args.index / f"{dataset}_bulk_sample_index.tsv"
        if sha256(manifest) != read_json(folder / "receipt.json")["metadata_sha256"]:
            raise ValueError(f"Released layout for {dataset} differs from the cache")
        design = released_design(args.index, dataset)
        structure = table(args.cache / "data/heparg_compound_folds.tsv")
        structure = structure.loc[structure.dataset.eq(dataset), ["identity", "chemical", "parent_key", "smiles"]]
        design = design.merge(structure, on=["identity", "chemical"], how="left", validate="many_to_one").fillna({"parent_key": "", "smiles": ""})
        design = enrich(design, design)
        pairs = plate_pairs(design)
        pairs.to_csv(out / f"{dataset}-plate-pairs.tsv", sep="\t", index=False)
        summary.append(metadata_out_of_plate(design).assign(dataset=dataset))
        design[["library", "plate", "block", "plategroup", "culture", "position", "identity", "chemical", "dose_uM",
                "max_uM", "design_rank", "dose_decade", "chemical_family", "generic_scaffold"]].to_csv(
            out / f"{dataset}-design.tsv", sep="\t", index=False,
        )
    pd.concat(summary, ignore_index=True).to_csv(out / "design.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", {"protocol_sha256": PROTOCOL_SHA256, "files": receipt(out, [p.name for p in out.iterdir()])})


def selected_panel(src, i, common):
    _, _, genes, delta, control = src
    columns = genes.get_indexer(common)
    variance = np.empty(len(common), dtype=np.float64)
    for start in range(0, len(common), 384):
        sl = slice(start, start + 384)
        part = np.ix_(i, columns[sl])
        variance[sl] = (delta[part] + control[part]).var(axis=0, dtype=np.float64)
    selected = np.argsort(-variance, kind="stable")
    return common[selected[variance[selected] > 0][:2000]]


def active_rows(src, folds):
    rows = src[1]
    eligible = rows.loc[rows.outer_fold.isin(folds)].copy()
    n = eligible.groupby(["plate", "dose_rank"]).parent_key.transform("nunique")
    return eligible.loc[n >= 3]


def response(src, rows, selected):
    cols = src[2].get_indexer(selected)
    if (cols < 0).any():
        raise ValueError("Selected panel missing in an assay")
    return center(rows, src[3][np.ix_(rows.index, cols)])


def stage_parts(sources, common, dataset, fit_folds, held_fold):
    source = sources[dataset]
    fits = {name: active_rows(cache, fit_folds) for name, cache in sources.items()}
    query = active_rows(source, [held_fold])
    if query.empty or any(f.empty for f in fits.values()):
        raise ValueError("An analysis role has no supported fold pools")
    i, j = fits[dataset].index.to_numpy(), query.index.to_numpy()
    panel = selected_panel(source, i, common)
    train = pd.concat(fits.values(), ignore_index=True)
    y = np.concatenate([response(sources[name], rows, panel) for name, rows in fits.items()])
    local = train.dataset.eq(dataset).to_numpy()
    hold_parents = frozenset(query.parent_key)
    if hold_parents & frozenset(train.parent_key):
        raise ValueError("A held parent entered this predictive role")
    return panel, train, y, local, i, j, query.reset_index(drop=True)


def peers(query, dataset):
    out_row, donor, weight = [], [], []
    plate, position, culture = [query[name].to_numpy() for name in ("plate", "position", "culture")]
    for members in query.groupby(["parent_key", "identity", "chemical", "dose_uM"], sort=True).indices.values():
        for i in members:
            use = (plate[members] != plate[i]) & (position[members] != position[i])
            if dataset == PRIMARY:
                use &= (culture[i] != "") & (culture[members] != "") & (culture[members] != culture[i])
            accepted = members[use]
            if not len(accepted):
                continue
            _, index, counts = np.unique(plate[accepted], return_inverse=True, return_counts=True)
            out_row.extend([int(i)] * len(accepted))
            donor.extend(accepted)
            weight.extend((0.25 / (len(counts) * counts[index])).tolist())
    return np.asarray(out_row, dtype=int), np.asarray(donor, dtype=int), np.asarray(weight, dtype=np.float32)


def record_predictions(query, selected, location, predictor, out):
    flags = strata(query)
    context = None
    keep = {label: {} for label, flag in flags.items() if flag.any()}
    for arm, pred, support in location:
        if arm == "B_context":
            context = pred
        for label, bank in keep.items():
            config = selected.get(label, {}).get(arm)
            if config is None:
                continue
            flag = flags[label]
            estimate = config["gain"] * pred[flag]
            if arm in RESIDUAL:
                estimate += selected[label]["B_context"]["gain"] * context[flag]
            bank[arm] = estimate
            bank[f"support__{arm}"] = support[flag]
    for arm, spec, pred in predictor:
        for label in PRIMARY_STRATA:
            config = selected.get(label, {}).get(arm)
            if config is not None and config["config"] == list(spec) and label in keep:
                keep[label][arm] = config["gain"] * pred[flags[label]]
    for label, bank in keep.items():
        needed = set(selected.get(label, {}))
        if not needed.issubset(bank):
            raise ValueError(f"Selected prediction missing for {label}: {needed - bank.keys()}")
        np.savez_compressed(out / f"{label}.npz", row=np.flatnonzero(flags[label]), **bank)


def predict_stage(args):
    out = args.work / "predict" / f"{args.dataset}-{args.fold}"
    out.mkdir(parents=True, exist_ok=False)
    design_receipt = args.work / "design/receipt.json"
    design = read_json(design_receipt)
    if design["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("The metadata audit predates this protocol")
    verify(design_receipt.parent, design["files"])
    sources = {name: read_source(args, name) for name in DATASETS}
    source = sources[args.dataset]
    common = sources[PRIMARY][2].intersection(sources[SECONDARY][2])
    training_folds = [f for f in range(3) if f != args.fold]
    all_known = pd.concat([src[1].loc[lambda d: d.parent_key.ne("")] for src in sources.values()], ignore_index=True)
    if all_known.groupby("parent_key").outer_fold.nunique().max() != 1:
        raise ValueError("A parent straddles the prespecified two-study partition")
    catalogue = annotated(all_known)
    kernel = Kernel(all_known)
    coords = {"position": source[1].position.to_numpy()}
    for seed in PERM_SEEDS:
        coords[f"perm{seed}"] = permute(source[1], seed)
        coords[f"matched_perm{seed}"] = permute(source[1], seed, True)
    validation = []
    for i, left in enumerate(training_folds):
        right = training_folds[1 - i]
        panel, train, y, local, fit_index, query_index, query = stage_parts(sources, common, args.dataset, [left], right)
        obs = response(source, source[1].iloc[query_index], panel)
        loc = location_candidates(train.loc[local].reset_index(drop=True), y[local], query, coords, fit_index, query_index)
        models = candidates(train, y, query, catalogue, kernel)
        validation.append(score_inner(obs, PRIMARY_STRATA, parent_groups(query, strata(query)), loc, models))
    settings = pick_settings(validation)
    panel, train, y, local, fit_index, query_index, query = stage_parts(sources, common, args.dataset, training_folds, args.fold)
    loc = location_candidates(train.loc[local].reset_index(drop=True), y[local], query, coords, fit_index, query_index)
    models = candidates(train, y, query, catalogue, kernel)
    record_predictions(query, settings, loc, models, out)
    receiver, donor, weight = peers(query, args.dataset)
    np.savez_compressed(out / "keys.npz", index=query_index, genes=panel.to_numpy(dtype=str),
                        receiver=receiver, donor=donor, weight=weight, annotation=np.array([bool(catalogue[p]) for p in query.parent_key]))
    write_json(out / "settings.json", settings)
    write_json(out / "receipt.json", {
        "protocol_sha256": PROTOCOL_SHA256, "design_receipt_sha256": sha256(design_receipt),
        "dataset": args.dataset, "fold": args.fold, "query_rows": len(query), "query_parents": int(query.parent_key.nunique()),
        "common_genes": len(common), "panel_genes": len(panel), "inputs": cache_receipts(args),
        "annotation_sha256": sha256(ANNOTATIONS),
        "files": receipt(out, [p.name for p in out.iterdir()]),
    })


def cache_receipts(args):
    return {name: sha256(args.cache / "data" / folder / "receipt.json")
            for name, folder in ((PRIMARY, "heparg"), (SECONDARY, "heparg_284321"))}


def verify_parts(args):
    root = args.work / "predict"
    inputs = cache_receipts(args)
    design = sha256(args.work / "design/receipt.json")
    annotations_hash = sha256(ANNOTATIONS)
    parts = [Path(f"{dataset}-{fold}") / "receipt.json" for dataset in DATASETS for fold in range(3)]
    folders = {}
    for part in parts:
        body = read_json(root / part)
        if (body["protocol_sha256"] != PROTOCOL_SHA256 or body["inputs"] != inputs
                or body["design_receipt_sha256"] != design or body["annotation_sha256"] != annotations_hash
                or part.parent.name != f"{body['dataset']}-{body['fold']}"):
            raise ValueError(f"Prediction provenance mismatch in {part}")
        verify(root / part.parent, body["files"])
        folders[part.parent.name] = body
    return folders, receipt(root, parts)


def freeze_stage(args):
    folders, receipts = verify_parts(args)
    source = Path(__file__).resolve().parent
    write_json(args.work / "freeze.json", {
        "protocol_sha256": PROTOCOL_SHA256, "protocol_commit": PROTOCOL_COMMIT, "parts": folders, "part_receipts": receipts,
        "code_commit": args.code_commit, "code": receipt(source, ("heparg.py", "heparg_design.py", "heparg_models.py")),
        "inputs": cache_receipts(args),
    })


def score_metrics(query, observed, predictions, repeat, support, annotation, positions):
    n = observed.shape[1]
    y2 = np.einsum("ij,ij->i", observed, observed) / n
    result = []
    baseline = np.sqrt(y2)
    for arm, value in {"B0": np.zeros_like(observed), **predictions}.items():
        p2 = np.einsum("ij,ij->i", value, value) / n
        yp = np.einsum("ij,ij->i", value, observed) / n
        denominator = np.sqrt(y2 * p2)
        r = query[["parent_key"]].copy()
        r["arm"] = arm
        r["row_rmse"] = np.sqrt(np.maximum(y2 - 2 * yp + p2, 0))
        r["cosine"] = np.divide(yp, denominator, out=np.zeros_like(yp), where=denominator > 1e-12)
        r["b0"] = baseline
        r["support"] = positions.get(arm, np.ones(len(query), dtype=bool))
        r["ridge"] = np.sqrt(np.mean((observed - predictions["dose_shared_ridge"]) ** 2, axis=1)) if "dose_shared_ridge" in predictions else np.nan
        r["context"] = np.sqrt(np.mean((observed - predictions["B_context"]) ** 2, axis=1)) if "B_context" in predictions else np.nan
        if arm == "dose_shared_ridge":
            r["ridge"] = r["row_rmse"]
        if arm == "B_context":
            r["context"] = r["row_rmse"]
        r["repeat"] = np.where(support, np.sqrt(np.mean((observed - repeat) ** 2, axis=1)), np.nan)
        result.append(r.assign(scope="all"))
        if arm in (*MODELS, "B0") and support.any():
            result.append(r.loc[support].assign(scope="repeat_supported"))
        if arm == "target_moa_neighbors" and annotation.any():
            result.append(r.loc[annotation].assign(scope="has_external_annotation"))
    return pd.concat(result, ignore_index=True)


def ratios(p, a, denominator):
    return np.divide(p, denominator, out=np.full_like(p, np.nan), where=(denominator > 0)) * a


def summarize(data):
    fields = ["b0", "row_rmse", "cosine", "ridge", "context", "repeat"]
    parent = data.groupby("parent_key", sort=True)[fields].mean().to_numpy(dtype=np.float64)
    n = len(parent)
    point = parent.mean(axis=0)
    boot = np.empty((10000, len(fields)))
    rng = np.random.default_rng(SEED)
    for i in range(0, len(boot), 128):
        batch = boot[i:i + 128]
        batch[:] = parent[rng.integers(n, size=(len(batch), n))].mean(axis=1)
    out = {"parents": n, "rows": len(data), "position_support": int(data.support.sum())}
    out.update({f"rmse_{name}": float(point[k]) for k, name in enumerate(("B0", "model", "cosine", "single_study_ridge", "context", "repeat")) if k != 2})
    out["cosine_mean"] = float(point[2])
    for index, label in ((0, "b0"), (3, "ridge"), (4, "context"), (5, "repeat")):
        p = float(ratios(np.asarray(point[index] - point[1]), 100, point[index]))
        draws = ratios(boot[:, index] - boot[:, 1], 100, boot[:, index])
        bounds = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975]) if np.isfinite(draws).any() else (np.nan, np.nan)
        out.update({f"reduction_vs_{label}_pct": p, f"{label}_lower95": bounds[0], f"{label}_upper95": bounds[1]})
    delta = point[0] - point[5]
    out["repeat_gap_pct"] = float(ratios(np.asarray(delta), 100, point[0]))
    if np.isfinite(delta):
        denom = boot[:, 0] - boot[:, 5]
        gap = np.quantile(100 * denom / boot[:, 0], [.025, .975])
        reaches = ratios(boot[:, 0] - boot[:, 1], 100, denom)
        out["repeat_gap_lower95"], out["repeat_gap_upper95"] = gap
        out["reach_pct"] = float(ratios(np.asarray(point[0] - point[1]), 100, delta))
        out["reach_denom_crosses_zero"] = bool(gap[0] <= 0 <= gap[1])
        if gap[0] > 0:
            out["reach_lower95"], out["reach_upper95"] = np.quantile(reaches[np.isfinite(reaches)], [.025, .975])
    return out


def score_stage(args):
    frozen_path = args.work / "freeze.json"
    frozen = read_json(frozen_path)
    _, receipts = verify_parts(args)
    if (frozen["protocol_sha256"] != PROTOCOL_SHA256 or frozen["part_receipts"] != receipts
            or frozen["code_commit"] != args.code_commit or frozen["inputs"] != cache_receipts(args)):
        raise ValueError("The score stage requires sealed out-of-parent predictions")
    verify(Path(__file__).resolve().parent, frozen["code"])
    out = args.work / "scored"
    out.mkdir(parents=True, exist_ok=False)
    points = []
    for dataset in DATASETS:
        cache = read_source(args, dataset)
        rows = cache[1]
        frames = defaultdict(list)
        selections = []
        for fold in range(3):
            folder = args.work / "predict" / f"{dataset}-{fold}"
            body = frozen["parts"][f"{dataset}-{fold}"]
            with np.load(folder / "keys.npz", allow_pickle=False) as keys:
                all_idx = keys["index"]
                query = rows.iloc[all_idx].reset_index(drop=True)
                expected = active_rows(cache, [fold]).index
                if not np.array_equal(expected, all_idx) or not query.outer_fold.eq(fold).all():
                    raise ValueError("Frozen row identity changed")
                selected = pd.Index(keys["genes"])
                if selected.nunique() != body["panel_genes"]:
                    raise ValueError("Frozen panel identity changed")
                observed = response(cache, query.set_axis(all_idx), selected)
                repeat = np.zeros_like(observed)
                receiver, donor, weight = keys["receiver"], keys["donor"], keys["weight"]
                independently_derived = peers(query, dataset)
                if not all(np.array_equal(a, b) for a, b in zip(independently_derived, (receiver, donor, weight))):
                    raise ValueError("Repeat linkage changed after sealing")
                np.add.at(repeat, receiver, weight[:, None] * observed[donor])
                support = np.isin(np.arange(len(query)), receiver)
                annotation = keys["annotation"]
            by_label = strata(query)
            settings = read_json(folder / "settings.json")
            for label, cfg in settings.items():
                selections.extend({"dataset": dataset, "outer_fold": fold, "stratum": label, "arm": arm,
                                   "gain": val["gain"], "hyperparameters": str(val["config"])}
                                  for arm, val in cfg.items())
            for label, flag in by_label.items():
                if not flag.any():
                    continue
                with np.load(folder / f"{label}.npz", allow_pickle=False) as archive:
                    keep = archive["row"]
                    if not np.array_equal(keep, np.flatnonzero(flag)):
                        raise ValueError("Score rows differ from preregistered cohort")
                    preds = {arm: archive[arm] for arm in settings.get(label, {})}
                    conf = {arm: archive[f"support__{arm}"] for arm in preds if f"support__{arm}" in archive}
                    frame = score_metrics(query.iloc[keep], observed[keep], preds, repeat[keep], support[keep], annotation[keep], conf)
                frames[label].append(frame.assign(outer_fold=fold))
        pd.DataFrame(selections).to_csv(out / f"{dataset}-settings.tsv", index=False, sep="\t")
        for label, parts in frames.items():
            source = pd.concat(parts, ignore_index=True)
            source.to_csv(out / f"{dataset}-{label}-paired.tsv.gz", index=False, sep="\t", compression="gzip")
            for (arm, scope), treated in source.groupby(["arm", "scope"]):
                r = summarize(treated)
                points.append({"dataset": dataset, "stratum": label, "arm": arm, "scope": scope,
                               "passes_b0": bool(arm in MODELS and scope == "all" and r["b0_lower95"] > 0),
                               "passes_single_study_ridge": bool(arm in MODELS and arm != "dose_shared_ridge"
                                                                 and scope == "all" and r["ridge_lower95"] > 0), **r})
    pd.DataFrame(points).to_csv(out / "metrics.tsv", index=False, sep="\t")
    write_json(out / "receipt.json", {
        "frozen_sha256": sha256(frozen_path), "protocol_sha256": PROTOCOL_SHA256,
        "GSE296254_repeat_requires_distinct_culture": True, "GSE284321_culture_independence_known": False,
        "repeat_reference_scale": 0.25, "bootstrap_replicates": 10000,
        "files": receipt(out, [p.name for p in out.iterdir()]),
    })


def main():
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "predict", "freeze", "score"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--fold", type=int, choices=range(3))
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("The preregistration hash changed")
    if args.stage == "predict" and (args.dataset is None or args.fold is None):
        parser.error("predict needs --dataset and --fold")
    {"design": design_stage, "predict": predict_stage, "freeze": freeze_stage, "score": score_stage}[args.stage](args)


if __name__ == "__main__":
    main()
