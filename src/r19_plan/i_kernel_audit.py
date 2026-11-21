"""Check numerical PSD and effective I3 support on fold-fitted independent views."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

from src.r11_chemical_probe.common import sha256
from src.r15_lincs.data import table
from src.r15_lincs.design import chemical_features, fold_id
from src.r15_lincs.estimators import _tanimoto
from src.r19_plan.i_audit import CAP_BYTES, write_receipt
from src.r19_plan.i_kernel import FitProfile, masked_gram
from src.r19_plan.r18_preflight import checked_tables

ASSAY_KEY = ["target_chembl_id", "target_organism", "assay_type", "standard_type", "standard_units"]
RANK = 64


def quantitative_rows(path):
    fields = ["parent", *ASSAY_KEY, "pchembl_value", "confidence_score"]
    rows = pd.read_csv(path, sep="\t", usecols=fields, keep_default_na=False)
    usable = rows[ASSAY_KEY].ne("").all(axis=1) & pd.to_numeric(rows.confidence_score, errors="coerce").ge(7)
    rows = rows.loc[usable, fields[:-1]]
    median = rows.groupby(["parent", *ASSAY_KEY], sort=True).pchembl_value.median().reset_index()
    median["feature_key"] = list(median[ASSAY_KEY].itertuples(index=False, name=None))
    return median[["parent", "feature_key", "pchembl_value"]]


def assay_matrix(records, parents, fitting):
    train = records.loc[records.parent.isin(parents[fitting])]
    support = train.groupby("feature_key").parent.nunique()
    keep = support.loc[support.ge(2)].index
    selected = records.loc[records.feature_key.isin(keep) & records.parent.isin(parents)]
    matrix = np.full((len(parents), len(keep)), np.nan, dtype=np.float32)
    row = parents.get_indexer(selected.parent)
    col = keep.get_indexer(selected.feature_key)
    matrix[row, col] = selected.pchembl_value
    return matrix


def phenotype_matrix(path, parents):
    with np.load(path, allow_pickle=False) as package:
        seen = pd.Index(package["parent"])
        original = package["features"]
    if not seen.is_unique or len(seen) != len(original):
        raise ValueError("An independent profile axis changed")
    lookup = seen.get_indexer(parents)
    result = np.full((len(parents), original.shape[1]), np.nan, dtype=np.float32)
    result[lookup >= 0] = original[lookup[lookup >= 0]]
    return result


def kernel_checks(structure, mapped, names, query):
    views = [mapped[name] for name in names]
    gram = masked_gram(structure, views)
    lowest = float(eigvalsh(gram, subset_by_index=[0, 0], check_finite=False)[0])
    counts = np.column_stack([m for _, m in views]).sum(axis=1)
    fallback = counts == 0
    fallback_error = float(np.max(np.abs(gram[np.ix_(fallback, fallback)]
                                           - structure[np.ix_(fallback, fallback)]), initial=0.0))
    if lowest < -1e-8 or fallback_error > 1e-12:
        raise ValueError("Combined kernel violated PSD or structure-only fallback")
    return dict(kernel="+".join(names), minimum_eigenvalue=lowest, fallback_error=fallback_error,
                query_parents=int(query.sum()), query_two_views=int((query & (counts >= 2)).sum()),
                query_all_views=int((query & (counts == len(names))).sum()),
                query_structure_only=int((query & fallback).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--views", nargs="+", choices=("I1", "I2", "I4"))
    args = parser.parse_args()
    root = args.root
    coverage_dir = args.coverage or root / "coverage"
    coverage = checked_tables(coverage_dir, coverage_dir / "receipt.json")
    cells = table(coverage_dir / "modality_coverage.tsv")
    available = cells.loc[cells.scope.eq("R17_DEV_assay_excluded") & cells.coverage_pass.eq("True"), "view"]
    eligible = [name for name in ("I1", "I2", "I4") if name in set(available)]
    if args.views:
        if len(set(args.views)) != len(args.views) or not set(args.views) <= set(eligible):
            raise ValueError("Selected I3 view lacks verified support")
        eligible = [name for name in eligible if name in args.views]
    if len(eligible) < 2:
        raise ValueError("The registered I3 coverage gate requires at least two eligible independent views")
    for name in eligible:
        if sha256(root / name / "receipt.json") != coverage["feature_view_receipts"][name]:
            raise ValueError(f"Feature view changed after coverage report: {name}")
        checked_tables(root / name, root / name / "receipt.json")
    available_rows = table(coverage_dir / "parent_availability.tsv").sort_values("parent")
    quantitative = quantitative_rows(root / "I1/bioassays.tsv.gz") if "I1" in eligible else None
    result, transforms = [], []
    for scope, rows in [("assay_excluded", available_rows), ("exclude_shared_R18_parent", available_rows.loc[
            available_rows.r18_identity_overlap.eq("False")])]:
        parents = pd.Index(rows.parent)
        folded = rows.outer_fold.astype(int).to_numpy()
        structure_vectors = chemical_features(parents)
        structure = _tanimoto(structure_vectors, structure_vectors).astype(np.float64)
        profiles = {name: phenotype_matrix(root / name / "features.npz", parents)
                    for name in eligible if name in ("I2", "I4")}
        primary = ["I1", "I2"] if {"I1", "I2"} <= set(eligible) else eligible[:2]
        blocks = [primary] if primary == eligible else [primary, eligible]
        for outer in range(3):
            selected = np.array([fold_id(p, f"R15-inner-{outer}", 5) == 0 for p in parents])
            for phase, fitting, query in [
                    ("inner", (folded != outer) & ~selected, (folded != outer) & selected),
                    ("outer", folded != outer, folded == outer)]:
                if (fitting & query).any():
                    raise ValueError("A held-out parent entered the feature preprocessing fit")
                mapped = {}
                for name in eligible:
                    matrix = assay_matrix(quantitative, parents, fitting) if name == "I1" else profiles[name]
                    fit = FitProfile.learn(matrix[fitting], rank=RANK)
                    vectors, mask = fit.transform(matrix)
                    mapped[name] = vectors, mask
                    transforms.append(dict(scope=scope, outer=outer, phase=phase, view=name,
                        fit_parents=int(fitting.sum()), fit_measured=int((fitting & mask).sum()),
                        query_parents=int(query.sum()), query_measured=int((query & mask).sum()),
                        features_retained=int(fit.columns.sum()), encoded_dim=vectors.shape[1]))
                for names in blocks:
                    result.append(dict(scope=scope, outer=outer, phase=phase,
                                       **kernel_checks(structure, mapped, names, query)))

    if sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) >= CAP_BYTES:
        raise ValueError("I-track storage exceeds the approved cap")
    out = args.out
    out.mkdir(exist_ok=False)
    pd.DataFrame(result).to_csv(out / "kernel_support.tsv", sep="\t", index=False)
    pd.DataFrame(transforms).to_csv(out / "transform_support.tsv", sep="\t", index=False)
    write_receipt(out, dict(coverage_sha256=sha256(coverage_dir / "receipt.json"),
        eligible_views=eligible, preprocessing="training-only standardized PCA up to 64 components",
        quantitative="target/resolved-assay grouped pChEMBL, confidence >=7, vocabulary support >=2 fit parents",
        structure="existing Morgan Tanimoto", PSD="row-scaled sum of PSD component kernels",
        RNA_scored=False, RNA_read=False, predictor_fit=False))


if __name__ == "__main__":
    main()
