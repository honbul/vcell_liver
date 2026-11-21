"""Audit pre-existing public phenotype views on the frozen Tahoe SNU-423 axis."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.p4.common import WellProjection
from src.r11_chemical_probe.common import read_json, sha256
from src.r11_chemical_probe.dataset import GROUP, LINE, table
from src.r15_lincs.data import table as lincs_table
from src.r19_plan.hcc_scaffold import LINE_ID
from src.r19_plan.hcc_scaffold_refit import cut_training, require_hash
from src.r19_plan.i_audit import checked_sources, structure, write_receipt
from src.r19_plan.i_kernel import FitProfile
from src.r19_plan.i_phenotypes import load_crosswalk, morphology, prism
from src.r19_plan.r18_preflight import checked_tables


def parent_axis(rows):
    pairs = pd.concat([frame[[GROUP, "smiles"]] for frame in rows], ignore_index=True).drop_duplicates()
    pairs["parent"] = pairs.smiles.map(structure)
    parents = pairs.groupby(GROUP).parent.agg(lambda value: value.iloc[0] if value.nunique() == 1 else "")
    collision = parents.ne("") & parents.duplicated(keep=False)
    return parents.mask(collision, ""), int(collision.sum())


def fold_axes(args, receipt, blocks):
    data, checked = {}, {}
    for fold in range(3):
        fit, held = [table(require_hash(args.visible / name, receipt["files"][name]))
                     for name in (f"f{fold}-fit.tsv", f"f{fold}-held.tsv")]
        if any(rows.quarantined.any() or rows.plate_id.eq("plate13").any() for rows in (fit, held)):
            raise ValueError("A protected Tahoe row would enter the support audit")
        query = WellProjection(held).rows.loc[lambda r: r[LINE].eq(LINE_ID)]
        keep, _ = cut_training(fit, query, blocks, fold)
        train = WellProjection(fit.loc[keep["scaffold"]]).rows.loc[lambda r: r[LINE].eq(LINE_ID)]
        if set(train[GROUP]) & set(query[GROUP]) or set(train[GROUP].map(blocks)) & set(query[GROUP].map(blocks)):
            raise ValueError("Scaffold-blocked fit and evaluation metadata intersect")
        data[fold] = (sorted(set(train[GROUP])), sorted(set(query[GROUP])))
        checked[fold] = (fit, held)
    if set().union(*(set(q) for _, q in data.values())) != set(blocks.index) or sum(len(q) for _, q in data.values()) != 203:
        raise ValueError("The 203-parent evaluation axis changed")
    return data, checked


def profiles(path, parents):
    with np.load(path / "features.npz", allow_pickle=False) as file:
        source = pd.DataFrame(file["features"], index=pd.Index(file["parent"], name="parent"))
    return source.reindex(parents.to_numpy()).set_axis(parents.index).to_numpy()


def transform_support(array, fit_index, query_index):
    training, query = array[fit_index], array[query_index]
    transform = FitProfile.learn(training, rank=64)
    _, fit_mask = transform.transform(training)
    _, query_mask = transform.transform(query)
    return fit_mask, query_mask, 0 if transform.basis is None else int(transform.basis.n_components_)


def cover(parents, folds, arrays, reserved):
    rows = []
    for fold, (fit, query) in folds.items():
        fi, qi = [parents.index.get_indexer(axis) for axis in (fit, query)]
        conditions = {view: transform_support(array, fi, qi) for view, array in arrays.items()}
        a, b = conditions.values()
        conditions.update(both=(a[0] & b[0], a[1] & b[1], None), either=(a[0] | b[0], a[1] | b[1], None),
                          I2_only=(a[0] & ~b[0], a[1] & ~b[1], None), I4_only=(b[0] & ~a[0], b[1] & ~a[1], None),
                          unavailable=(~(a[0] | b[0]), ~(a[1] | b[1]), None))
        strict = ~parents.loc[query].isin(reserved).to_numpy()
        for view, (f_mask, q_mask, pca) in conditions.items():
            rows.append(dict(view=view, fold=fold, fit_total=len(fit), fit_support=int(f_mask.sum()),
                query_total=len(query), query_support=int(q_mask.sum()), pca_rank=pca,
                query_non_r18_parent=int(strict.sum()), query_non_r18_support=int((strict & q_mask).sum())))
    table = pd.DataFrame(rows)
    summary = (table.groupby("view", sort=False)
               .agg(query_total=("query_total", "sum"), query_support=("query_support", "sum"),
                    query_non_r18_parent=("query_non_r18_parent", "sum"), query_non_r18_support=("query_non_r18_support", "sum"),
                    min_fit_fold=("fit_support", "min"), min_query_fold=("query_support", "min")).reset_index())
    summary["gate"] = (summary.query_support.ge(100) & summary.min_fit_fold.ge(20) & summary.min_query_fold.ge(20)
                       & summary.view.isin(("I2", "I4", "both")))
    return table, summary


def main(args):
    proof = args.reports / "data"
    receipt = read_json(require_hash(args.visible / "receipt.json", sha256(proof / "hcc_scaffold_refit/visible/receipt.json")))
    blocks = table(require_hash(args.reports.parent / "r12_multiaxis/group_scaffolds.tsv", receipt["scaffold_sha256"]))
    blocks = blocks.set_index(GROUP, verify_integrity=True).scaffold_component
    folds, original = fold_axes(args, receipt, blocks)
    parents, ambiguous = parent_axis(frame for fold in original.values() for frame in fold)
    if not parents.index.equals(blocks.sort_index().index):
        raise ValueError("Expression-independent parent and scaffold axes disagree")
    checked_tables(args.reserved, proof / "r18-support-receipt.json")
    reserved = set().union(*(set(lincs_table(args.reserved / f"r18_{name}.tsv").parent) - {""}
                            for name in ("reserved_manifest", "B_donor_exceptions")))
    require_hash(args.data / "source_manifest.json", sha256(proof / "i_views/public_source_manifest.json"))
    checked_sources(args.data)
    population = pd.DataFrame({"parent": parents[parents.ne("")].to_numpy(), "pert_id": ""})
    paths, treatments, crosswalk = load_crosswalk(args.data, population)
    args.out.mkdir(exist_ok=False)
    morphology(paths, crosswalk, population, args.out / "I2")
    prism(args.data, treatments, crosswalk, population, args.out / "I4")
    by_fold, summary = cover(parents, folds, {view: profiles(args.out / view, parents) for view in ("I2", "I4")}, reserved)
    by_fold.to_csv(args.out / "fold_support.tsv", sep="\t", index=False)
    summary.to_csv(args.out / "summary.tsv", sep="\t", index=False)
    parents.to_csv(args.out / "parent_axis.tsv", sep="\t", header=True)
    write_receipt(args.out, dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        source_manifest_sha256=sha256(args.data / "source_manifest.json"), visible_sha256=sha256(args.visible / "receipt.json"),
        reservation_sha256=sha256(args.reserved / "receipt.json"), expression_read=False, raw_images_read=False,
        group_count=len(parents), resolved=int(parents.ne("").sum()), canonical_collision_groups=ambiguous,
        view_receipts={view: sha256(args.out / view / "receipt.json") for view in ("I2", "I4")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("visible", "data", "protocol", "reports", "reserved", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
