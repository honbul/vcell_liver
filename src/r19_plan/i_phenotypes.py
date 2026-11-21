"""Audit independent Cell Painting and PRISM morphology/viability feature support."""

import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import sha256
from src.r15_lincs.data import STUDIES, hepatic_lines, table
from src.r17_lincs.inventory import ROOT
from src.r19_plan.i_audit import (base_brd, build_crosswalk, checked_sources, read_catalog, structure,
                                  write_receipt)
from src.r19_plan.r18_preflight import checked_tables

MORPH_CONTEXT = ["Metadata_broad_sample", "Metadata_cell_id", "Metadata_time_point", "Metadata_dose_recode"]
COMPOUND, CELL, TIME, DOSE = MORPH_CONTEXT
FEATURE_PREFIX = ("Cells_", "Cytoplasm_", "Nuclei_")
CPG_CONTEXT = {"cell": "A549", "time": "48H", "dose_recode": "6"}
LIVER_TEXT = r"liver|hepati|hepatoc|hepatob|intrahepatic|hepg2|huh.?7|snu.?423"


def load_crosswalk(data, population):
    legacy = pd.read_csv(data / "cpg/broad_id_map.csv", dtype=str, keep_default_na=False)
    paths = sorted((data / "cpg").glob("*_consensus_median_dmso.csv.gz"))
    if len(paths) != 2:
        raise ValueError("Expected both pinned Cell Painting consensus batches")
    treatment = pd.read_csv(data / "prism/primary-screen-replicate-collapsed-treatment-info.csv",
                            dtype=str, keep_default_na=False)
    samples = set(treatment.broad_id)
    for path in paths:
        samples.update(pd.read_csv(path, usecols=[COMPOUND], dtype=str,
                                   keep_default_na=False)[COMPOUND])
    catalog = read_catalog(ROOT / "LINCS2020/compoundinfo_beta.txt",
                           {base_brd(s) for s in samples} | {base_brd(s) for s in population.pert_id})
    return paths, treatment, build_crosswalk(samples, legacy, catalog, population)


def summarize_matrix(out, profile, fields, columns):
    # Parent-local medians retain missingness and learn no parameters across parents.
    array = profile[columns].to_numpy(dtype=np.float32)
    array[~np.isfinite(array)] = np.nan
    support = []
    profiles = []
    for parent, ids in profile.groupby("parent", sort=True).indices.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            summary = np.nanmedian(array[ids], axis=0)
        count = int(np.isfinite(summary).sum())
        if count:
            profiles.append(summary)
            support.append(dict(parent=parent, source_rows=len(ids), observed_features=count))
    if not support:
        raise ValueError("No usable parent profile was mapped to the frozen DEV population")
    registry = pd.DataFrame(support)
    registry.to_csv(out / "parents.tsv", sep="\t", index=False)
    profile[fields].to_csv(out / "source_rows.tsv", sep="\t", index=False)
    np.savez_compressed(out / "features.npz", parent=registry.parent.to_numpy(dtype=str),
                        features=np.stack(profiles), columns=np.asarray(columns, dtype=str))
    return registry


def morphology(paths, crosswalk, population, out):
    nodes = [set(pd.read_csv(path, nrows=0).columns) for path in paths]
    columns = sorted(name for name in set.intersection(*nodes) if name.startswith(FEATURE_PREFIX))
    if not columns:
        raise ValueError("Cell Painting batches have no common CellProfiler columns")
    aliases = crosswalk.set_index("source_id").parent
    wanted = set(population.parent)
    audited, accepted, source_rows = [], [], 0
    for path in paths:
        choices = {COMPOUND: str, CELL: str, TIME: str, DOSE: str, **{name: np.float32 for name in columns}}
        for rows in pd.read_csv(path, usecols=[*MORPH_CONTEXT, *columns], dtype=choices, chunksize=2000):
            source_rows += len(rows)
            rows["parent"] = rows[COMPOUND].map(aliases).fillna("")
            rows["batch"] = path.name.split("_consensus_", 1)[0]
            rows = rows.loc[rows.parent.isin(wanted)].copy()
            if len(rows):
                observed = np.isfinite(rows[columns].to_numpy()).any(axis=1)
                rows = rows.loc[observed]
                audited.append(rows[["parent", "batch", CELL, TIME, DOSE]])
            keep = (rows[CELL].eq(CPG_CONTEXT["cell"]) & rows[TIME].eq(CPG_CONTEXT["time"])
                    & rows[DOSE].eq(CPG_CONTEXT["dose_recode"]))
            accepted.append(rows.loc[keep])
    profiles = pd.concat(accepted, ignore_index=True)
    out.mkdir(exist_ok=False)
    registry = summarize_matrix(out, profiles, ["parent", "batch", *MORPH_CONTEXT], columns)
    pd.concat(audited).groupby(["batch", CELL, TIME, DOSE]).agg(
        parents=("parent", "nunique"), records=("parent", "size")).reset_index().to_csv(
            out / "context_coverage.tsv", sep="\t", index=False)
    write_receipt(out, dict(view="I2", license="CC0-1.0", total_published_consensus_rows=source_rows,
        parent_profiles=len(registry), profile_context=CPG_CONTEXT, raw_images_read=False,
        cellprofiler_features=len(columns), normalization="publisher plate-local DMSO RobustMAD, median consensus",
        global_feature_selection=False, global_sphering_or_harmony=False, standardization_or_pca_fitted=False))


def cell_filter(path):
    cell = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("row_name", verify_integrity=True)
    annotations = ["ccle_name", "primary_tissue", "secondary_tissue", "tertiary_tissue", "originally_assigned_ccle_name"]
    tissue_hit = cell[annotations].apply(lambda column: column.str.contains(
        LIVER_TEXT, regex=True, case=False, na=False)).any(axis=1)
    known = set.union(*(hepatic_lines(ROOT, study) for study in STUDIES))
    known = {re.sub(r"[^A-Z0-9]", "", name.upper()) for name in known}
    name = cell.ccle_name.str.split("_", n=1).str[0].str.replace(r"[^A-Z0-9]", "", regex=True)
    old_name = cell.originally_assigned_ccle_name.str.split("_", n=1).str[0].str.replace(r"[^A-Z0-9]", "", regex=True)
    liver_related = tissue_hit | name.isin(known) | old_name.isin(known)
    assigned = cell.depmap_id.str.fullmatch(r"ACH-[0-9]+") & cell.primary_tissue.str.strip().str.lower().ne("na")
    assigned &= cell.primary_tissue.str.strip().ne("")
    cell["decision"] = np.select([liver_related, ~assigned],
                                 ["excluded_liver_lineage", "excluded_unresolved_tissue"], default="nonhepatic")
    return cell


def prism(data, treatment, crosswalk, population, out):
    source = data / "prism"
    cells = cell_filter(source / "primary-screen-cell-line-info.csv")
    raw = pd.read_csv(source / "primary-screen-replicate-collapsed-logfold-change.csv", index_col=0)
    if not raw.index.is_unique or not raw.columns.is_unique or not raw.index.isin(cells.index).all():
        raise ValueError("PRISM row identities are not uniquely represented in official cell metadata")
    cells["in_published_matrix"] = cells.index.isin(raw.index)
    keep = cells.index[cells.decision.eq("nonhepatic") & cells.in_published_matrix]
    raw = raw.loc[keep].T
    if not cells.loc[raw.columns, "decision"].eq("nonhepatic").all():
        raise ValueError("The selected PRISM feature axis contains a nonallowed tissue")
    treat = treatment.set_index("column_name", verify_integrity=True).reindex(raw.index)
    keys = crosswalk.set_index("source_id").parent
    verified = treat.smiles.map(structure)
    id_parent = treat.broad_id.map(keys).fillna("")
    invalid = verified.ne("") & id_parent.ne("") & verified.ne(id_parent)
    parent = verified.where(verified.ne(""), id_parent).where(~invalid, "")
    treat = treat.assign(parent=parent, numeric_dose=pd.to_numeric(treat.dose, errors="coerce"))
    keep = treat.parent.isin(population.parent) & treat.numeric_dose.gt(0) & ~invalid
    treat = treat.loc[keep].copy()
    # The primary HTS reference is 2.5 uM; nearest source dose is fixed without RNA outcomes.
    distance = (treat.numeric_dose / 2.5).map(np.log2).abs()
    treat = treat.loc[distance.eq(distance.groupby(treat.parent).transform("min"))]
    columns = raw.columns.to_list()
    profiles = treat[["parent", "broad_id", "numeric_dose", "screen_id"]].join(raw).reset_index(names="column_name")
    out.mkdir(exist_ok=False)
    registry = summarize_matrix(out, profiles, ["parent", "column_name", "broad_id", "numeric_dose", "screen_id"], columns)
    cells.to_csv(out / "cell_line_audit.tsv", sep="\t")
    excluded = cells.in_published_matrix & cells.decision.eq("excluded_liver_lineage")
    write_receipt(out, dict(view="I4", doi="10.6084/m9.figshare.9393293.v4", license="CC-BY-4.0",
        parent_profiles=len(registry), original_cell_records=int(cells.in_published_matrix.sum()),
        nonhepatic_feature_cell_count=len(columns), excluded_liver_records=int(excluded.sum()),
        excluded_liver_cell_ids=cells.loc[excluded, "ccle_name"].to_list(),
        conflicting_structure_records=int(invalid.sum()), direct_liver_features=False,
        source_processing="Published 19Q4 primary replicate-collapsed DMSO log2-fold change; ComBat within treatment",
        upstream_lineage_refit=False, standardization_or_pca_fitted=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("I2", "I4"), required=True)
    for field in ("data", "population", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    checked_sources(args.data)
    checked_tables(args.population, args.population / "receipt.json")
    population = table(args.population / "rows.tsv")
    paths, treatment, crosswalk = load_crosswalk(args.data, population)
    if args.view == "I2":
        morphology(paths, crosswalk, population, args.out)
    else:
        prism(args.data, treatment, crosswalk, population, args.out)


if __name__ == "__main__":
    main()
