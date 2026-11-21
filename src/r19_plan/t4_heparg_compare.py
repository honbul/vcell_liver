"""Transfer exact-condition frozen T4 directions to admitted HepaRG wells."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_stage import parse_tsv, pinned_bytes
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.t4_access import CONDITION, SCORE, condition_filter, condition_index
from src.r19_plan.t4_attribution import ORIGINAL_SHA256
from src.r19_plan.t4_heparg_support import DATASETS
from src.r19_plan.t4_workflow import axes, deltas, plans, reference
from src.r19_plan.x_inputs import check_unique, control_axis, matrix_axes

ARMS = ("identity6", "operator6", "direct24")
COUNTS = {"GSE284321": (5, 21), "GSE296254": (3, 9)}


def contracts(args):
    hepa = read_json(args.bridge)
    files = read_json(args.p3)["files"]
    if (not hepa["metadata_only"] or hepa["expression_read"] or hepa["target_audit_sha256"] != sha256(args.p3)
            or sha256(args.original) != ORIGINAL_SHA256):
        raise ValueError("Unexpected HepaRG metadata or original T4 forecast lineage")
    target = read_pinned(args.allowlists, "target_allowlist.tsv", hepa["files"]["target_allowlist.tsv"])
    # This study uses HEPG2 only. The support audit has checked the other cells independently.
    target = target.loc[target.source_cell.eq("HEPG2")].copy()
    if (set(target.dataset) != set(COUNTS) or target.duplicated(["dataset", "library"]).any()
            or any((g.parent.nunique(), len(g)) != COUNTS[key] for key, g in target.groupby("dataset"))):
        raise ValueError("The exact-condition HepaRG support changed")
    axis = parse_tsv(pinned_bytes(args.genes, files["landmark_map.tsv"]))
    axis = axis.loc[axis.both_temposeq_assays.eq("True"), ["entrez_id", "ensembl_id"]].copy()
    axis["entrez_id"] = pd.to_numeric(axis.entrez_id, errors="raise")
    if len(axis) != 827 or not axis.entrez_id.is_unique or not axis.ensembl_id.is_unique:
        raise ValueError("Unexpected common LINCS–TempO-Seq landmark axis")
    t4 = reference(args)
    source = read_json(args.original)
    if source["access_sha256"] != sha256(args.pin) or set(source["slices"]) != set(t4["slices"]):
        raise ValueError("Original T4 access and forecast scope disagree")
    stamp = dict(protocol_sha256=sha256(args.protocol), bridge_sha256=sha256(args.bridge),
                 access_sha256=sha256(args.pin), original_sha256=sha256(args.original),
                 genes_sha256=sha256(args.genes), code_sha256=sha256(__file__))
    return target, axis.reset_index(drop=True), t4, source, stamp


def source_query(args, ref, key):
    return parse_tsv(pinned_bytes(args.access/"predict-plan"/key/"query.tsv",
                                  ref["slices"][key]["predict"]["query.tsv"]))


def source_queries(args, ref, conditions):
    result = {}
    for key in sorted(ref["slices"]):
        query = source_query(args, ref, key)
        keep = condition_index(query).isin(condition_index(conditions))
        if keep.any():
            result[key] = (query, keep)
    return result


def predict(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    target, common, ref, source, stamp = contracts(args)
    landmarks, _ = matrix_axes(args.access, ref)
    coordinates = pd.Index(landmarks.pr_gene_id).get_indexer(common.entrez_id)
    if (coordinates < 0).any():
        raise ValueError("P3 landmark IDs differ from the frozen T4 axis")
    inventory, predictions = [], {arm: [] for arm in ARMS}
    for key, (query, selected) in source_queries(args, ref, target).items():
        path = args.original_forecasts / key
        received = read_json(path/"receipt.json")
        if (sha256(path/"receipt.json") != source["slices"][key]
                or sha256(path/"forecasts.npz") != received["files"]["forecasts.npz"]
                or received["query_sha256"] != ref["slices"][key]["predict"]["query.tsv"]):
            raise ValueError("An original T4 forecast or its row order changed")
        cases = query.loc[selected, CONDITION].reset_index(drop=True)
        groups = cases.groupby(CONDITION, sort=True).indices
        with np.load(path/"forecasts.npz", allow_pickle=False) as archive:
            for arm in ARMS:
                vectors = checked_array(archive[arm], (len(query), len(landmarks)), arm)[selected][:, coordinates]
                packed = []
                for positions in groups.values():
                    if not np.all(vectors[positions] == vectors[positions[0]]):
                        raise ValueError("T4 forecasts vary within an admitted physical condition")
                    packed.append(vectors[positions[0]])
                predictions[arm].extend(packed)
        inventory.extend(dict(zip(CONDITION, condition, strict=True), key=key,
                              role=axes(key)[1], fold=axes(key)[0]) for condition in groups)
    inventory = pd.DataFrame(inventory)
    verify_conditions(inventory, target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".hepa-predict-") as temp:
        stage = Path(temp)
        inventory.to_csv(stage/"conditions.tsv", sep="\t", index=False)
        np.savez_compressed(stage/"forecasts.npz", **{arm: np.stack(a) for arm, a in predictions.items()})
        write_json(stage/"receipt.json", dict(stamp, files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out/"receipt.json")


def verify_conditions(inventory, target):
    if (inventory.empty or inventory.duplicated([*CONDITION, "role"]).any()
            or set(condition_index(inventory)) != set(condition_index(target))
            or not inventory.groupby(CONDITION).role.nunique().eq(3).all()
            or not inventory.groupby(CONDITION).fold.nunique().eq(1).all()):
        raise ValueError("The frozen transfer lacks a matched condition or independent role")


def read_predictions(args, stamp, target):
    path = args.forecasts
    receipt = read_json(path/"receipt.json")
    if (any(receipt[k] != h for k, h in stamp.items())
            or set(receipt["files"]) != {"conditions.tsv", "forecasts.npz"}
            or sha256(path/"forecasts.npz") != receipt["files"]["forecasts.npz"]):
        raise ValueError("The new HepaRG forecast does not match the fixed protocol and sources")
    rows = read_pinned(path, "conditions.tsv", receipt["files"]["conditions.tsv"])
    verify_conditions(rows, target)
    with np.load(path/"forecasts.npz", allow_pickle=False) as file:
        if set(file.files) != set(ARMS):
            raise ValueError("A technical comparison arm was added or omitted")
        values = {arm: checked_array(file[arm], (len(rows), 827), arm) for arm in ARMS}
    return rows, values


def cosine(truth, value):
    product = np.linalg.norm(truth, axis=1) * np.linalg.norm(value)
    return np.divide(truth @ value, product, out=np.full(len(truth), np.nan), where=product > 1e-12)


def hepa_assets(args, dataset, common):
    root = args.heparg/DATASETS[dataset]
    proof = read_json(args.bridge)
    if sha256(root/"receipt.json") != proof["targets"][dataset]:
        raise ValueError("HepaRG score array cache lacks the original QC receipt")
    receipt = read_json(root/"receipt.json")
    rows, controls, genes = [read_pinned(root, p+".tsv", receipt["files"][p+".tsv"])
                             for p in ("rows", "vehicles", "genes")]
    columns = pd.Index(genes.ensembl_id).get_indexer(common.ensembl_id)
    if (columns < 0).any() or not rows.library.is_unique or not controls.library.is_unique:
        raise ValueError("HepaRG physical or gene axis changed")
    active = controls.chemical.str.strip().str.lower().isin({"dmso", "dimethyl sulfoxide"})
    candidate = controls.loc[active & controls.time.eq("24h")].copy()
    candidate["crc"] = candidate.library.map(lambda value: crc32(value.encode()))
    candidate = candidate.sort_values(["plate", "crc", "library"])
    candidate["role"] = candidate.groupby("plate").cumcount().mod(3).map(dict(enumerate("ABC")))
    arrays = {}
    for name, frame in (("delta", rows), ("control", rows), ("vehicles", controls)):
        path = root/(name+".npy")
        if sha256(path) != receipt["files"][path.name]:
            raise ValueError("A HepaRG score array changed since R14")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        if arrays[name].shape != (len(frame), len(genes)):
            raise ValueError("HepaRG metadata and score arrays have different axes")
    return rows.set_index("library", drop=False), candidate, arrays, columns


def hep_truth(asset, target, role):
    rows, controls, values, columns = asset
    selected = rows.loc[target.library]
    if not selected.plate.to_numpy().tolist() == target.plate.to_numpy().tolist():
        raise ValueError("HepaRG matched row coordinates changed")
    idx = rows.index.get_indexer(selected.library)
    role_controls = controls.loc[controls.role.eq(role) & controls.plate.isin(selected.plate)]
    if set(role_controls.plate) != set(selected.plate):
        raise ValueError("One matched HepaRG score plate lacks the independent vehicle role")
    mean = {plate: values["vehicles"][np.ix_(group.index, columns)].mean(axis=0)
            for plate, group in role_controls.groupby("plate")}
    wanted = np.ix_(idx, columns)
    return values["delta"][wanted] + values["control"][wanted] - np.stack([mean[p] for p in selected.plate])


def source_truths(args, rows, ref, common):
    genes, indices = matrix_axes(args.access, ref)
    selected = pd.Index(genes.pr_gene_id).get_indexer(common.entrez_id)
    if (selected < 0).any():
        raise ValueError("Missing matched gene ID in the LINCS score matrix")
    scores = {}
    for key, group in rows.groupby("key"):
        query = source_query(args, ref, key)
        score = plans(args, ref, key, "score", SCORE)
        truth, controls = condition_filter(score["query"], group), score["controls"]
        if not truth.drop(columns="level3_row_index").equals(condition_filter(query, group)):
            raise ValueError("T4 technical score changed its independent 24 h coordinate")
        check_unique(pd.concat((truth, controls), ignore_index=True))
        control_axis(controls, truth, axes(key)[3])
        raw = extract_matrix(ref["matrix"], pd.concat((truth, controls), ignore_index=True),
                             indices[selected], common.entrez_id.to_numpy())
        delta = deltas(truth, raw[:len(truth)], controls, raw[len(truth):])
        for condition, indexes in truth.groupby(CONDITION).indices.items():
            scores[(key, *condition)] = delta[indexes]
    return scores


def score_pairs(args, ref, common, target, rows, values):
    source = source_truths(args, rows, ref, common)
    records = []
    for dataset, selected in target.groupby("dataset", sort=True):
        asset = hepa_assets(args, dataset, common)
        for idx, row in enumerate(rows.itertuples(index=False)):
            matched = selected.loc[selected.parent.eq(row.parent) & selected.actual_dose_uM.eq(row.actual_dose_uM)]
            if matched.empty:
                continue
            left = source[row.key, row.parent, row.actual_dose_uM].astype("float64")
            right = hep_truth(asset, matched, axes(row.key)[3]).astype("float64")
            for assay, truth in (("L1000_24h", left), ("HepaRG_24h", right)):
                for arm in ARMS:
                    for c in cosine(truth, values[arm][idx].astype("float64")):
                        records.append(dict(dataset=dataset, parent=row.parent, dose=row.actual_dose_uM,
                            score=assay, arm=arm, cosine=c))
    details = pd.DataFrame(records)
    parent = details.groupby(["dataset", "parent", "score", "arm"], as_index=False).agg(
        cosine=("cosine", "mean"), finite_rows=("cosine", "count"), role_and_well_rows=("cosine", "size"))
    summary = parent.groupby(["dataset", "score", "arm"], as_index=False).agg(
        parents=("parent", "nunique"), parents_with_cosine=("cosine", "count"),
        mean_cosine=("cosine", "mean"), min_parent_cosine=("cosine", "min"), max_parent_cosine=("cosine", "max"))
    return parent, summary


def evaluate(args):
    scoring = args.action == "score"
    if scoring and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("HepaRG scoring requires a fresh output and the independent forecast digest")
    target, common, ref, _, upstream = contracts(args)
    rows, values = read_predictions(args, upstream, target)
    stamp = dict(upstream, predict_sha256=sha256(args.forecasts/"receipt.json"), conditions=len(rows)//3)
    if not scoring:
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if read_json(args.freeze) != stamp:
        raise ValueError("HepaRG prediction arrays changed after the separate freeze")
    parent, result = score_pairs(args, ref, common, target, rows, values)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".hepa-score-") as temp:
        stage = Path(temp)
        result.to_csv(stage/"cosine.tsv", sep="\t", index=False, na_rep="NA")
        parent.to_csv(stage/"by_parent.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(stamp, freeze_sha256=args.expected_sha256, genes=len(common),
            files={p.name: sha256(p) for p in stage.iterdir()}))
        stage.rename(args.out)
    return sha256(args.out/"receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "freeze", "score"))
    for name in ("bridge", "allowlists", "p3", "access", "pin", "original", "original-forecasts", "protocol",
                 "genes", "heparg", "out", "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print((predict if args.action == "predict" else evaluate)(args))
