"""Fit and freeze donor-blocked single-cell resistance forecasts offline."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
import torch

from .cohorts import file_record
from .singlecell_inputs import COHORTS, files_receipt
from .source_panels import DRUG_NAMES, frozen
from .transfer_models import Result, neural, ordinary, rank_skill


STEMNESS = ("EPCAM", "PROM1", "CD44", "SOX9", "BMI1", "KRT19", "ALDH1A1", "SOX2", "NANOG")
GROUPS = (*COHORTS, "pseudo", "tahoe")


@dataclass
class Profiles:
    index: pd.Index
    genes: pd.Index
    values: object


def available_profiles(path, row_metadata, genes_file):
    axis = pd.Index(pd.read_csv(genes_file, sep="\t").symbol)
    cells = sparse.load_npz(path)
    if cells.shape != (len(row_metadata), len(axis)) or not axis.is_unique or not row_metadata.id.is_unique:
        raise ValueError(f"Unaligned registered single-cell profile {path}")
    return Profiles(pd.Index(row_metadata.id), axis, cells)


def load_embedding(file):
    with np.load(file, allow_pickle=False) as stream:
        matrix = pd.DataFrame(stream["vectors"], index=stream["id"])
        if not matrix.index.is_unique or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f"Ambiguous embeddings in {file}")
        return matrix


def high_variance(bulk, ids, genes):
    values = bulk.loc[sorted(set(ids)), genes].to_numpy(dtype=np.float32)
    variance = values.var(axis=0)
    return np.lexsort((genes, -variance))[:min(512, len(genes))]


def extract(profile, target_axis, choose, signatures, size=256):
    pos = profile.genes.get_indexer(target_axis)
    if (pos < 0).any():
        raise ValueError("Profiles must have the same registered expression feature universe")
    features, biological = [], []
    signature_indices = [np.flatnonzero(target_axis.isin(genes)) for genes in signatures.values()]
    if min(map(len, signature_indices)) == 0:
        raise ValueError("No expression coverage for a registered biomarker")
    for start in range(0, len(profile.index), size):
        rows = profile.values[start:start + size, pos]
        if sparse.issparse(rows):
            rows = rows.toarray()
        ranks = (rankdata(rows, axis=1).astype(np.float32) - 0.5) / len(pos)
        features.append(ranks[:, choose])
        biological.append(np.column_stack([ranks[:, index].mean(axis=1) for index in signature_indices]))
    matrix = np.concatenate(features)
    return (pd.DataFrame(matrix, index=profile.index),
            pd.DataFrame(np.concatenate(biological), index=profile.index, columns=list(signatures)))


def load_sets(path):
    collection = {}
    with path.open() as stream:
        for line in stream:
            name, _, *members = line.rstrip("\n").split("\t")
            collection[name] = members
    return {"ABCB1": ("ABCB1",), "stemness": STEMNESS,
            "EMT": collection["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
            "hypoxia": collection["HALLMARK_HYPOXIA"]}


def source_targets(fitting, validation):
    for frames in (fitting, validation):
        if frames.duplicated("id").any():
            raise ValueError("Repeated source assay")
    candidates = fitting.groupby(["source_id", "drug"]).resistance_native
    distributions = {group: np.sort(values.to_numpy()) for group, values in candidates}

    def scale(rows):
        weights = pd.Series(index=rows.index, dtype=np.float32)
        for group, part in rows.groupby(["source_id", "drug"]):
            sample = distributions.get(group)
            if sample is None or len(sample) < 8:
                raise ValueError(f"Source endpoint lacks a fit-only distribution for {group}")
            left = np.searchsorted(sample, part.resistance_native, side="left")
            right = np.searchsorted(sample, part.resistance_native, side="right")
            weights.loc[part.index] = (left + right) / (2 * len(sample))
        return weights.to_numpy(np.float32)

    return scale(fitting), scale(validation)


def assemble_query(fold, datasets, panel, organs):
    parts = []
    for name, meta in datasets.items():
        if name in COHORTS:
            cells = meta[meta.fold.eq(fold)].copy()
            cells["role"] = "patient"
        else:
            cells = meta[meta.fold.eq(fold)].copy()
            cells["donor"] = cells.cell_group
            if not set(cells.cell_group).issubset(panel.cell_group):
                raise ValueError("A pseudo or real cell originates outside its held-out assay fold")
        cells["profile_group"] = name
        parts.append(cells)
    if fold < 4:
        organ = organs[organs.fold.eq(fold)].copy()
        organ["role"] = "organoid"
        organ["profile_group"] = "organoids"
        parts.append(organ)
    base = pd.concat(parts, ignore_index=True)
    genes = pd.DataFrame({"drug": DRUG_NAMES})
    interface = base.merge(genes, how="cross")
    interface = interface[~interface.role.eq("organoid") | interface.drug.eq("Sorafenib")]
    return interface.reset_index(drop=True)


def sample_adaptation(metas, fold, seed):
    rng = np.random.default_rng(seed + fold)
    development = pd.concat([metas[name].assign(group=name) for name in ("GSE149614", "GSE151530")])
    development = development[development.fold.ne(fold)]
    return [rng.choice(rows.id.to_numpy(), size=min(320, len(rows)), replace=False)
            for _, rows in development.groupby("donor", sort=True)]


def run_fold(a, fold, raw_bulk, organs_rna, source, anchor_receipt, signatures, metas, profiles, embeddings):
    folder = f"fold-{fold}"
    train, validation, panel = [pd.read_parquet(frozen(a.sources, f"{folder}/{role}.parquet", source))
                                for role in ("fit", "validation", "query")]
    if "resistance_native" in panel or not set(panel.cell_group).isdisjoint(set(train.cell_group) | set(validation.cell_group)):
        raise ValueError("External child, parent or outcome crossed into fitting")
    if set(train.cell_group) & set(validation.cell_group):
        raise ValueError("Fit and early-stopping cell lineage overlap")
    organs = pd.read_parquet(frozen(a.sources, "organoids/query.parquet", source))
    organs["fold"] = organs.donor.map({donor: i for i, donor in enumerate(sorted(organs.donor.unique()))})
    anchor_query_fold = fold % 4
    query = assemble_query(fold, metas, panel, organs)
    development = sample_adaptation(metas, fold, a.seed)
    adaptation_ids = np.concatenate(development)
    if set(adaptation_ids) & set(query.id):
        raise ValueError("A held-out patient or culture occurs in domain adaptation")

    # The predictor opens labels only from each fitting organoid role.
    # The sealed fold-specific fit table contains three donors and no query.
    anchor_path = f"anchors-{anchor_query_fold}.parquet"
    anchors = pd.read_parquet(frozen(a.anchors, anchor_path, anchor_receipt))
    if not set(anchors.donor).isdisjoint(set(organs.loc[organs.fold.eq(anchor_query_fold), "donor"])):
        raise ValueError("Organoid test phenotype crossed the donor holdout")
    anchors["drug"] = "Sorafenib"
    fit_ids = np.concatenate((train.cell_id.to_numpy(), anchors.id.to_numpy()))
    val_ids = validation.cell_id.to_numpy()
    types = {key: i for i, key in enumerate(DRUG_NAMES)}
    train_drug = np.array([types[v] for v in [*train.drug, *anchors.drug]], dtype=np.int64)
    val_drug = validation.drug.map(types).to_numpy(np.int64)
    query_drug = query.drug.map(types).to_numpy(np.int64)
    bulk_y, vy = source_targets(train, validation)
    y = np.concatenate((bulk_y, anchors.outcome.to_numpy(np.float32)))
    counts = train.groupby(["cell_group", "drug"]).id.transform("size").to_numpy()
    weights = np.concatenate((1 / counts, np.full(len(anchors), 2.0))).astype(np.float32)
    weights /= weights.mean()

    common = profiles["pseudo"].genes
    common = common.intersection(organs_rna.columns, sort=False)
    chosen = high_variance(raw_bulk, train.cell_id, common)
    feature_blocks, bio_blocks = [], []
    for name, profile in profiles.items():
        use = set(fit_ids) | set(val_ids) if name == "source_bulk" else set(query.id) | set(adaptation_ids)
        if name == "organoids":
            use |= set(anchors.id)
        positions = np.flatnonzero(profile.index.isin(use))
        if len(positions):
            frame, bio = extract(Profiles(profile.index[positions], profile.genes, profile.values[positions]),
                                 common, chosen, signatures)
            feature_blocks.append(frame)
            bio_blocks.append(bio)
    rank = pd.concat(feature_blocks)
    markers = pd.concat(bio_blocks)
    query_columns = ["id", "donor", "role", "fold", "profile_group", "drug"]
    query_columns += [key for key in ("cell_id", "cell_group", "depth", "sample") if key in query]
    prediction = query[query_columns].copy()
    model_receipts = {}

    def pick(matrix):
        return [matrix.loc[ids].to_numpy(dtype=np.float32)
                for ids in (fit_ids, val_ids, query.id.to_numpy(), adaptation_ids)]

    def add(name, result):
        if len(result.predictions) != len(query) or not np.isfinite(result.predictions).all():
            raise ValueError(f"Wrong forecast result for {name}")
        prediction[name] = result.predictions
        model_receipts[name] = {"source_validation_spearman": result.validation_spearman,
                                "sorafenib_high_resistance_threshold": result.threshold, **result.config}
        print(f"fold {fold}: {name} {result.validation_spearman:.5f} source validation", flush=True)

    for marker in markers.columns:
        train_v, valid_v, out, _ = pick(markers[[marker]])
        label = float(np.quantile(train_v[train_drug == 0], 0.75))
        add(marker, Result(out.flatten(), label, rank_skill(vy, valid_v.flatten(), val_drug)[0],
                           {"threshold_rule": "Source training 75th percentile"}))
    averages = pd.Series(y).groupby(train_drug).mean().to_numpy()
    add("mean", Result(averages[query_drug], float(averages[0]), 0.0,
                       {"threshold_rule": "Training average"}))
    rx, rv, rq, ra = pick(rank)
    for family in ("ridge", "elasticnet", "lightgbm"):
        count = len(train)
        add(f"{family}_bulk", ordinary(family, rx[:count], train_drug[:count], y[:count], weights[:count],
                                       rv, val_drug, vy, rq, query_drug, a.seed))
        add(family, ordinary(family, rx, train_drug, y, weights, rv, val_drug, vy, rq, query_drug, a.seed))
    fx, fv, fq, fa = pick(embeddings)
    for name, variant, blocks in (
        ("rank_mlp", "source", (rx, rv, rq, ra)),
        ("SCAD_family", "adversarial", (rx, rv, rq, ra)),
        ("scDEAL_family", "mmd", (rx, rv, rq, ra)),
        ("scgpt_mlp_no_adaptation", "source", (fx, fv, fq, fa)),
        ("scgpt_adversarial", "adversarial", (fx, fv, fq, fa)),
        ("scgpt_mmd", "mmd", (fx, fv, fq, fa)),
    ):
        nx, nv, nq, na = blocks
        result = neural(variant, nx, train_drug, y, weights, nv, val_drug, vy, nq,
                        query_drug, na, a.device, a.seed)
        add(name, result)
    for marker in markers.columns:
        prediction["marker_" + marker] = markers.loc[query.id, marker].to_numpy()
    return prediction, {"models": model_receipts, "source_assays": train.source_id.value_counts().to_dict(),
                        "domain_adaptation_cells": len(adaptation_ids),
                        "patient_test_donors": int(query.loc[query.role.eq("patient"), "donor"].nunique()),
                        "organoid_fitting_donors": len(set(anchors.donor))}


def predict(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    receipts = {label: json.loads((getattr(a, label) / "receipt.json").read_text())
                for label in ("sources", "singlecell", "scgpt", "anchors")}
    source, inputs, emb, anchor_receipt = (receipts[key] for key in ("sources", "singlecell", "scgpt", "anchors"))
    if emb["queries"] != file_record(a.singlecell / "receipt.json") or emb["source_rna"] != file_record(a.sources / "receipt.json"):
        raise ValueError("Feature store does not match split and query registration")
    if anchor_receipt["sources"] != emb["source_rna"]:
        raise ValueError("Organoid fitting roles are registered against different source phenotypes")
    raw_bulk = pd.read_parquet(frozen(a.sources, "cell_lines_rna.parquet", source))
    organs = pd.read_parquet(frozen(a.sources, "organoids/rna.parquet", source))
    raw_profiles = {
        name: Profiles(frame.index, frame.columns, frame.to_numpy(np.float32))
        for name, frame in (("source_bulk", raw_bulk), ("organoids", organs))
    }
    meta = {}
    for cohort in GROUPS:
        table = pd.read_parquet(frozen(a.singlecell, f"{cohort}/rows.parquet", inputs))
        raw_profiles[cohort] = available_profiles(
            frozen(a.singlecell, f"{cohort}/counts.npz", inputs), table,
            frozen(a.singlecell, f"{cohort}/genes.tsv", inputs),
        )
        meta[cohort] = table
    embeddings = pd.concat([load_embedding(frozen(a.scgpt, name + ".npz", emb)) for name in raw_profiles])
    if not embeddings.index.is_unique:
        raise ValueError("Source and single-cell embedding identities overlap")
    signatures = load_sets(a.hallmark)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        records = {}
        for fold in range(5):
            result, info = run_fold(
                a, fold, raw_bulk, organs, source, anchor_receipt, signatures, meta, raw_profiles, embeddings,
            )
            result.to_parquet(staged / f"fold-{fold}.parquet", index=False)
            records[str(fold)] = info
        (staged / "receipt.json").write_text(json.dumps({
            "inputs": {**{k: file_record(getattr(a, k) / "receipt.json") for k in receipts},
                       "hallmark": file_record(a.hallmark)},
            "folds": records, "device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
            "select_on": "Mean within-source held-lineage drug ranks and RMSE as tie break",
            "files": files_receipt(staged),
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("sources", "singlecell", "scgpt", "anchors", "output", "hallmark"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=51122)
    predict(p.parse_args())


if __name__ == "__main__":
    main()
