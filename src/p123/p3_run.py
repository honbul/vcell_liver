"""Run the preregistered HepaRG cross-assay predictions, freeze and separate score."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.goal_tracks.heparg import (
    DATASETS, PRIMARY, SECONDARY, active_rows, cache_receipts, peers, read_source, receipt,
    response, stage_parts, verify,
)
from src.p123.p3_models import Kernel, MODEL_NAMES, SignatureBank, candidates, dose_templates, pick, pooled_scores, strata
from src.p123.p3_score import evaluate_stratum
from src.r11_chemical_probe.common import read_json, sha256, write_json

PROTOCOL_COMMIT = "f2fc9ae571b68635b8395997bf1443074b4b40eb"
PROTOCOL_SHA256 = "d0680c422b41c63f895b8c1efc0fdedbcbc7965fdc1e35a40e6eb7bde6da8693"
T3_FREEZE = "e4ec8e273ced848ca040103afeb9118157c2acd2e3c7afe151f35e32e3b64299"
T3_DESIGN = "7b1daee38e02cbbd5809bfc754c9818b3d00799ae0db97544174a00923f94b43"


def codes():
    path = Path(__file__).resolve().parent
    return receipt(path, ("p3_models.py", "p3_run.py", "p3_score.py"))


def source_receipts(args):
    prior = args.lincs / "lincs_heparg.npz"
    receipt_file = args.lincs / "receipt.json"
    prior_receipt = read_json(receipt_file)
    lin_hash = sha256(prior)
    if (prior_receipt.get("protocol_sha256") != PROTOCOL_SHA256
            or prior_receipt.get("files", {}).get(prior.name) != lin_hash):
        raise ValueError("The LINCS prediction inputs differ from the source receipt")
    for path, digest in ((args.t3 / "freeze.json", T3_FREEZE),
                         (args.t3 / "design-balanced/design/receipt.json", T3_DESIGN)):
        if sha256(path) != digest:
            raise ValueError("T3 design or frozen response model differs from the official source")
    return {"heparg": cache_receipts(args), "t3_freeze_sha256": T3_FREEZE, "t3_design_sha256": T3_DESIGN,
            "lincs_sha256": lin_hash, "lincs_receipt_sha256": sha256(receipt_file),
            "landmark_map_sha256": sha256(args.genes), "tahoe_gene_annotation_only": True}


def source_data(args):
    sources = {name: read_source(args, name) for name in DATASETS}
    common = sources[PRIMARY][2].intersection(sources[SECONDARY][2])
    bank_receipt = read_json(args.lincs / "receipt.json")
    symbol_conflicts = [int(row["entrez"]) for row in bank_receipt["landmark_symbol_conflicts"]]
    prior = SignatureBank(args.lincs / "lincs_heparg.npz", args.genes, common, symbol_conflicts)
    return sources, common, prior


def historical(args, dataset, fold):
    base = args.t3 / f"predict/{dataset}-{fold}"
    body = read_json(args.t3 / "freeze.json")
    attested = body["part_receipts"][f"{dataset}-{fold}/receipt.json"]
    if sha256(base / "receipt.json") != attested:
        raise ValueError("T3 prediction role receipt changed")
    part = read_json(base / "receipt.json")
    verify(base, {name: part["files"][name] for name in ("keys.npz", "settings.json", "rank7.npz", "rank8.npz", "ge10uM.npz")})
    return base, part


def predict(args):
    inputs = source_receipts(args)
    sources, common, prior = source_data(args)
    reference, old = historical(args, args.dataset, args.fold)
    old_settings = read_json(reference / "settings.json")
    settings = {s: old_settings[s]["joint_dose_response"]["config"][0] for s in ("rank7", "rank8", "ge10uM")}
    all_known = pd.concat([src[1][src[1].parent_key.ne("")] for src in sources.values()], ignore_index=True)
    if all_known.groupby("parent_key").outer_fold.nunique().max() != 1:
        raise ValueError("A parent spans multiple study partitions")
    geometry = Kernel(all_known)
    training_folds = [i for i in range(3) if i != args.fold]
    source = sources[args.dataset]
    validation = {name: [] for name in settings}
    for fit_fold, val_fold in (training_folds, training_folds[::-1]):
        panel, train, y, _, _, held_idx, query = stage_parts(sources, common, args.dataset, [fit_fold], val_fold)
        y_h = response(source, source[1].iloc[held_idx], panel)
        templates = dose_templates(train, y)
        for label, keep in strata(query).items():
            preds = candidates(query, templates, geometry, prior, settings[label])
            validation[label].append(pooled_scores(keep, y_h, preds))
        print(f"Finished validation fold {val_fold}", flush=True)
    picked, validation_rows = {}, []
    for label, scores in validation.items():
        picked[label], search = pick(scores)
        validation_rows.extend({"stratum": label, "hill_half_dose": settings[label], **row} for row in search)

    panel, train, y, _, _, query_idx, query = stage_parts(sources, common, args.dataset, training_folds, args.fold)
    with np.load(reference / "keys.npz", allow_pickle=False) as keys:
        if not (np.array_equal(query_idx, keys["index"]) and np.array_equal(panel, keys["genes"])):
            raise ValueError("The T3 and P3 gene or evaluation axes disagree")
    templates = dose_templates(train, y)
    target = args.work / "predict" / f"{args.dataset}-{args.fold}"
    target.mkdir(parents=True, exist_ok=False)
    for label, keep in strata(query).items():
        finalized = {}
        for arm, spec, values in candidates(query, templates, geometry, prior, settings[label]):
            control = picked[label][arm]
            if list(spec) == control["settings"]:
                prediction = values[keep] * control["gain"]
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite frozen HepaRG forecast")
                finalized[arm] = prediction
        if set(finalized) != set(MODEL_NAMES):
            raise ValueError("A required forecast was not generated")
        np.savez_compressed(target / f"{label}.npz", row=np.flatnonzero(keep), **finalized)
    mapped = prior.mapped
    np.savez_compressed(target / "keys.npz", held_idx=query_idx, genes=panel.to_numpy(dtype=str),
                        parents=query.parent_key.to_numpy(dtype=str))
    pd.DataFrame(validation_rows).assign(settings=lambda z: z.settings.map(json.dumps)).to_csv(
        target / "validation.tsv", sep="\t", index=False)
    write_json(target / "selection.json", {"hill_half_dose": settings, "p3": picked})
    write_json(target / "receipt.json", {
        "dataset": args.dataset, "fold": args.fold, "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256, "code_commit": args.code_commit,
        "code": codes(), "input_receipts": inputs,
        "t3_fold_receipt_sha256": sha256(reference / "receipt.json"),
        "t3_fold_expected_rows": old["query_rows"], "t3_panel_genes": len(panel),
        "common_temposeq_genes": len(common),
        "mapped_lincs_landmark_count": int(mapped.both_temposeq_assays.sum()),
        "fit_panel_lincs_overlap": int(mapped.loc[mapped.both_temposeq_assays, "ensembl_id"].isin(panel).sum()),
        "files": receipt(target, [p.name for p in target.iterdir()]),
    })


def pred_receipts(args):
    provenance = source_receipts(args)
    code = codes()
    output = {}
    for dataset, fold in product(DATASETS, range(3)):
        key = f"{dataset}-{fold}/receipt.json"
        path = args.work / "predict" / key
        body = read_json(path)
        old, _ = historical(args, dataset, fold)
        values = (PROTOCOL_SHA256, PROTOCOL_COMMIT, args.code_commit, dataset, fold, provenance,
                  code, sha256(old / "receipt.json"))
        fields = ("protocol_sha256", "protocol_commit", "code_commit", "dataset", "fold",
                  "input_receipts", "code", "t3_fold_receipt_sha256")
        if tuple(body[k] for k in fields) != values:
            raise ValueError(f"Unexpected P3 provenance for {key}")
        verify(path.parent, body["files"])
        output[key] = sha256(path)
    return {"protocol_sha256": PROTOCOL_SHA256, "protocol_commit": PROTOCOL_COMMIT,
            "code_commit": args.code_commit, "code": code, "input_receipts": provenance, "part_receipts": output}


def freeze(args):
    output = args.work / "freeze.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, pred_receipts(args))


def load_for_scoring(args, source, dataset, fold):
    prev, _ = historical(args, dataset, fold)
    folder = args.work / "predict" / f"{dataset}-{fold}"
    with np.load(folder / "keys.npz", allow_pickle=False) as cur, np.load(prev / "keys.npz", allow_pickle=False) as before:
        idx, panel, parents = cur["held_idx"], pd.Index(cur["genes"]), cur["parents"]
        if not (np.array_equal(idx, before["index"]) and np.array_equal(panel, before["genes"])):
            raise ValueError("The prior and extension score axes disagree")
        query = active_rows(source, [fold])
        if not (np.array_equal(query.index, idx) and np.array_equal(query.parent_key, parents)):
            raise ValueError("The evaluation cohort disagrees with the freeze")
        obs = response(source, query, panel)
        query = query.reset_index(drop=True).assign(outer_fold=fold)
        receiver, donor, weight = peers(query, dataset)
        if not all(np.array_equal(x, before[name]) for x, name in zip((receiver, donor, weight), ("receiver", "donor", "weight"))):
            raise ValueError("T3 and P3 repeat pairs disagree")
        repeats = np.zeros_like(obs)
        np.add.at(repeats, receiver, weight[:, None] * obs[donor])
        query["repeat_supported"] = np.isin(np.arange(len(query)), receiver)
    print(f"Opened official held HepaRG fold {dataset} {fold} after verifying freeze", flush=True)
    return query, obs, repeats, folder, prev


def score(args):
    expected = read_json(args.work / "freeze.json")
    if expected != pred_receipts(args):
        raise ValueError("Scoring requires unchanged P3 and T3 predictions and inputs")
    sources, _, prior = source_data(args)
    directory = args.work / "scored"
    directory.mkdir(parents=True, exist_ok=False)
    metrics, concentrations, parent_data, selections = [], [], [], []
    map_data = prior.mapped.copy()
    for dataset in DATASETS:
        frames = [load_for_scoring(args, sources[dataset], dataset, fold) for fold in range(3)]
        for fold in range(3):
            p3 = frames[fold][3]
            with np.load(p3 / "keys.npz", allow_pickle=False) as keys:
                map_data[f"{dataset}_fold{fold}_panel"] = map_data.ensembl_id.isin(keys["genes"])
            selection = read_json(p3 / "selection.json")
            for label, selected in selection["p3"].items():
                for arm, params in selected.items():
                    selections.append({"dataset": dataset, "fold": fold, "stratum": label, "arm": arm,
                                       "hill_half_dose": selection["hill_half_dose"][label], **params})
        for label in ("rank7", "rank8", "ge10uM"):
            by_fold = []
            for query, truth, repeats, p3, t3 in frames:
                keep = strata(query)[label]
                with np.load(t3 / f"{label}.npz", allow_pickle=False) as prev, np.load(p3 / f"{label}.npz", allow_pickle=False) as cur:
                    if not (np.array_equal(prev["row"], np.flatnonzero(keep))
                            and np.array_equal(prev["row"], cur["row"])):
                        raise ValueError("Frozen prediction rows do not match the registered stratum")
                    models = {name: prev[name] for name in prev.files if name != "row" and not name.startswith("support__")}
                    models.update({name: cur[name] for name in MODEL_NAMES})
                    by_fold.append((query.loc[keep].reset_index(drop=True), truth[keep], models, repeats[keep]))
            for item, bank in zip((metrics, concentrations, parent_data), evaluate_stratum(dataset, label, by_fold, prior)):
                item.append(bank)
            print(f"Scored {dataset} {label}", flush=True)
    for name, table in (("metrics.tsv", pd.concat(metrics, ignore_index=True)),
                        ("concentration.tsv", pd.concat(concentrations, ignore_index=True)),
                        ("contributions.tsv", pd.concat(parent_data, ignore_index=True)),
                        ("landmark_map.tsv", map_data),
                        ("selection.tsv", pd.DataFrame(selections).assign(settings=lambda s: s.settings.map(json.dumps)))):
        table.to_csv(directory / name, sep="\t", index=False)
    write_json(directory / "receipt.json", {
        "freeze_sha256": sha256(args.work / "freeze.json"), "protocol_sha256": PROTOCOL_SHA256,
        "bootstrap_draws": 10000, "input_receipts": expected["input_receipts"],
        "files": receipt(directory, [p.name for p in directory.iterdir()]),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "freeze", "score"))
    for field in ("work", "protocol", "t3", "cache", "index", "genes", "lincs"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--fold", type=int, choices=range(3))
    args = parser.parse_args()
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("The P3 prespecification differs from the shared commit")
    if args.stage == "predict" and (args.dataset is None or args.fold is None):
        parser.error("predict requires a dataset and fold")
    {"predict": predict, "freeze": freeze, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
