"""Register CC BY Oslo CRLM organoid RNA and dose measurements for 3D development."""

import argparse
from contextlib import closing
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.integrate import trapezoid

from .function_data import alias_token, new_output
from .io import publish_json, sha256
from .licob_cold import checked as checked_licob
from .licob_curves import DOSE_REGISTRATION
from .licob_dose_sources import COLD, registration
from .yang_hcc_external import require_hash

ACCESSION = "10.17632/hr94h42xdc.3"
OSLO_REGISTRATION = "4d6441a1126551ba5addd7cc4d5cecac7d6c25cfcda1be28100bbb5f292eeb58"
CHECKSUMS = {
    "Data S1.xlsx": "d125c5b3a1144aab2b5406cf8f26fa3c5998426ff452b4a742e651f0b253fcc7",
    "Data S2.xlsx": "3a4dbd7c444311752bb0894a21704d92d0c181d21757ccb9e4f9091d24493393",
    "Data S4.xlsx": "3847aa93b2a84c7d5d0b04c26494f39f35963fc41e96eae97d8a180fbc33d81c",
    "Data S5.xlsx": "05be116a1cc837a70b97ed24af1d7a01efdb57faa5e9d9b711882ebaec5ebae2",
    "mmc8.xlsx": "105c3edf7ff5cb69d8a6ef006026e5cf8dc43bc7cc78dd18a9bb738a7d26117e",
    "mmc11.xlsx": "b0c15ad0eb39c840c38e7357b1743b6c1189adcdd52ca2f4d231c9d27ba02f91",
}
PLATE = ["sample_id", "run_id", "library_id", "plate"]
WELL = [*PLATE, "drow", "dcol"]
CURVE = ["sample_id", "run_id", "assay_no", "library_id", "compound_fimm", "compound_name"]


def sheet(path, *, header=0, use=None):
    with closing(load_workbook(path, read_only=True, data_only=True)) as book:
        rows = book.active.iter_rows(min_row=header + 1, values_only=True)
        titles = next(rows)
        indexes = [titles.index(name) for name in use] if use is not None else list(range(len(titles)))
        return pd.DataFrame.from_records(([row[i] for i in indexes] for row in rows),
                                         columns=[titles[i] for i in indexes])


def patients(directory):
    clinical = sheet(directory / "mmc8.xlsx", header=3, use=("sample_id", "platform"))
    pattern = r"(?P<patient>Pt\d+)(?:\.\d+)?_t\d+(?:\.\d+)?"
    clinical = clinical[clinical.sample_id.astype(str).str.fullmatch(pattern)].copy()
    clinical["patient"] = clinical.sample_id.str.extract(pattern).patient
    genetic = sheet(directory / "Data S1.xlsx", use=("sample_id", "patient", "sample_type"))
    genetic = genetic[genetic.sample_type.eq("PDO")][["sample_id", "patient"]].drop_duplicates()
    matched = clinical.merge(genetic, on=["sample_id", "patient"], validate="one_to_one")
    overlap = set(clinical.sample_id) & set(genetic.sample_id)
    if set(matched.sample_id) != overlap:
        raise ValueError("Publication disagrees about some patient-to-organoid identities")
    if clinical.sample_id.duplicated().any():
        raise ValueError("Sample identities must be unique in the author's catalogue")
    return clinical.set_index("sample_id"), len(overlap)


def counts(directory, metadata):
    table = sheet(directory / "Data S2.xlsx")
    columns = pd.Index(table.columns[3:], name="sample_id")
    if not columns.is_unique or not set(columns) <= set(metadata.index):
        raise ValueError("Unexpected expression sample roster")
    gene = table.hgnc_symbol
    chosen = gene.notna() & gene.astype(str).ne("") & ~gene.duplicated(keep=False)
    matrix = table.loc[chosen, columns].to_numpy(dtype=float).T
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix != np.floor(matrix)).any():
        raise ValueError("Expected nonnegative integer RNA counts")
    return pd.DataFrame(matrix.astype(np.int32), index=columns, columns=gene[chosen].to_numpy()), {
        "released_gene_rows": len(table), "unique_nonempty_symbols": int(chosen.sum()),
        "rna_samples": len(columns), "rna_patients": int(metadata.loc[columns].patient.nunique()),
    }


def plate_check(rows):
    types = ["control_positive", "control_negative"]
    control = rows[rows.compound_type.isin(types)].drop_duplicates(WELL)
    summary = (control.groupby([*PLATE, "compound_type"]).signal.agg(["median", "size"])
               .unstack("compound_type"))
    keep = (summary["size"][types].ge(4).all(axis=1)
            & summary["median"].control_negative.gt(summary["median"].control_positive))
    plate = summary.loc[keep, "median"].rename(columns={"control_positive": "positive", "control_negative": "negative"})
    return plate, len(summary)


def measured(rows, metadata):
    data = rows.copy()
    for column in ("signal", "viability", "concentration"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    controls, plates = plate_check(data[np.isfinite(data.signal)])
    selected = (data.compound_type.eq("single") & data.concentration_unit.eq("nM")
                & data.concentration.gt(0) & np.isfinite(data.viability)
                & data.compound_fimm.astype(str).str.fullmatch(r"FIMM\d+-\d+")
                & ~data.compound_name.isin(["TAS-102", "Disulfiram+Cu"]))
    singles = data[selected].copy()
    if singles.duplicated(WELL).any():
        raise ValueError("Conflicting single-agent well coordinates")
    if not set(singles.sample_id) <= set(metadata.index):
        raise ValueError("A screened PDO lacks a published patient/lesion registration")
    paired = singles.join(controls, on=PLATE).dropna(subset=["positive", "negative", "signal"])
    normalized = (paired.signal - paired.positive) / (paired.negative - paired.positive)
    differences = (paired.viability - normalized).abs().to_numpy()
    singles = singles.assign(dose_uM=singles.concentration / 1000.)
    grid = (singles.groupby([*CURVE, "dose_uM"], as_index=False).viability.agg(
        viability="median", physical_wells="size"))
    npoints = grid.groupby(CURVE).dose_uM.transform("nunique")
    grid = grid[npoints.ge(5)].copy()
    grid["lineage_group"] = grid.sample_id.map(metadata.patient)
    stats = {"rows": len(data), "single_released_wells": int(data.compound_type.eq("single").sum()),
             "nM_explicit_product_eligible_wells": len(singles), "plate_groups_with_controls": plates,
             "reconstructible_plate_groups": len(controls), "reconstructible_single_wells": len(paired),
             "published_viability_minus_local_control_normalization_abs_quantiles":
                 np.quantile(differences, [.5, .9, .99]).tolist() if len(differences) else None}
    return grid, stats


def endpoint(doses, viability):
    log_dose = np.log10(doses)
    values = np.clip(viability, 0, 1)
    return float(trapezoid(-values, x=log_dose) / np.ptp(log_dose)), float(-values[-1])


def repeats(grid):
    all_pairs = []
    for (sample, drug), group in grid.groupby(["sample_id", "compound_fimm"], sort=True):
        sets = group.groupby(["library_id", "run_id", "assay_no"], sort=True)
        blocks = [(key, part.set_index("dose_uM").viability.sort_index()) for key, part in sets]
        chosen = set()
        for (left, a), (right, b) in combinations(blocks, 2):
            kind = "different_library" if left[0] != right[0] else "different_run" if left[1] != right[1] else None
            if kind is None or kind in chosen:
                continue
            shared = a.index.intersection(b.index).sort_values()
            if len(shared) < 5:
                continue
            chosen.add(kind)
            ap, ae = endpoint(shared.to_numpy(), a.loc[shared].to_numpy())
            bp, be = endpoint(shared.to_numpy(), b.loc[shared].to_numpy())
            all_pairs.append((kind, sample, drug, group.lineage_group.iloc[0], len(shared), ap, ae, bp, be))
    paired = pd.DataFrame(all_pairs, columns=["kind", "sample", "drug", "patient", "points",
                                             "auc_a", "emax_a", "auc_b", "emax_b"])
    results = {}
    for kind, rows in paired.groupby("kind"):
        donors = rows.groupby(["drug", "patient"], as_index=False)[["auc_a", "emax_a", "auc_b", "emax_b"]].mean()
        results[kind] = {"curves_paired": len(rows), "samples": rows["sample"].nunique(),
                         "donors": rows.patient.nunique(), "drug_products": rows.drug.nunique(),
                         "paired_donor_products": len(donors), "median_exact_dose_points": float(rows.points.median())}
        for feature in ("auc", "emax"):
            rank = []
            for _, specimen in donors.groupby("drug"):
                a, b = specimen[f"{feature}_a"], specimen[f"{feature}_b"]
                if len(specimen) >= 5 and a.nunique() > 1 and b.nunique() > 1:
                    rank.append((a.corr(b, method="spearman"), a.corr(b)))
            results[kind][feature] = {"eligible_drug_products": len(rank),
                "within_drug_spearman": float(np.mean(rank, axis=0)[0]) if rank else None,
                "within_drug_pearson": float(np.mean(rank, axis=0)[1]) if rank else None}
    return results


def support(grid):
    unique = grid[CURVE + ["lineage_group"]].drop_duplicates()
    return {"curves": len(unique), "wells": int(grid.physical_wells.sum()), "dose_points": len(grid),
            "samples": unique.sample_id.nunique(), "donors": unique.lineage_group.nunique(),
            "exact_source_products": unique.compound_fimm.nunique(), "published_drug_names": unique.compound_name.nunique()}


def register(args):
    for name, digest in CHECKSUMS.items():
        require_hash(args.source / name, digest)
    catalogue, independently_matched = patients(args.source)
    expression, expression_support = counts(args.source, catalogue)
    grid, plate = measured(sheet(args.source / "Data S4.xlsx"), catalogue)
    rna_grid = grid[grid.sample_id.isin(expression.index)].copy()
    if checked_licob(args.licob, ("query", "rna")) != COLD:
        raise ValueError("Unexpected LICOB comparison registration")
    query = pd.read_parquet(args.licob / "query.parquet")
    licob_genes = pd.read_parquet(args.licob / "rna.parquet").columns
    overlap = rna_grid[["compound_name", "compound_fimm"]].drop_duplicates()
    licob = query[["compound_id", "source_compound_label"]].drop_duplicates()
    # Nominal name overlap is explicitly not exact structure or lot confirmation.
    matched = overlap.assign(name=overlap.compound_name.map(alias_token)).merge(
        licob.assign(name=licob.source_compound_label.map(alias_token)), on="name").drop(columns="name")
    with new_output(args.out) as out:
        expression.to_parquet(out / "counts.parquet")
        rna_grid.to_parquet(out / "dose_points.parquet", index=False)
        catalogue.loc[expression.index, ["patient", "platform"]].to_parquet(out / "rna_identity.parquet")
        matched.sort_values(["compound_name", "compound_fimm"]).to_csv(out / "name_overlap.tsv", index=False, sep="\t")
        products = grid.groupby(["compound_fimm", "compound_name"]).sample_id.nunique()
        if products.index.get_level_values("compound_fimm").duplicated().any():
            raise ValueError("An exact FIMM code maps to multiple drug names")
        publish_json(out / "receipt.json", {"doi": ACCESSION, "license": "CC BY 4.0 (Mendeley data)",
            "input_hashes": CHECKSUMS, "publication_clinical_samples": len(catalogue),
            "identity_agreement_with_mutation_metadata": independently_matched,
            "plate_qc": plate, "RNA": expression_support | {"licob_common_symbols": len(licob_genes.intersection(expression.columns))},
            "full_dose_support": support(grid), "RNA_dose_support": support(rna_grid),
            "RNA_repeat_agreement": repeats(rna_grid),
            "licob_name_overlap": {"name_candidate_parents": matched.compound_id.nunique(),
                                    "source_products": matched.compound_fimm.nunique()},
            "outputs": {name: sha256(out / name) for name in ("counts.parquet", "dose_points.parquet",
                                                             "rna_identity.parquet", "name_overlap.tsv")}})


def bridge(args):
    receipt = args.registered / "receipt.json"
    require_hash(receipt, OSLO_REGISTRATION)
    meta = json.loads(receipt.read_text())
    for name in ("dose_points.parquet", "name_overlap.tsv"):
        require_hash(args.registered / name, meta["outputs"][name])
    if registration(args.dose) != DOSE_REGISTRATION:
        raise ValueError("Unexpected LICOB concentration registration")
    # Read dose and identity fields only. The bridge audit does not read viability.
    key = [*CURVE, "lineage_group"]
    grid = pd.read_parquet(args.registered / "dose_points.parquet", columns=[*key, "dose_uM"])
    curves = grid.groupby(key).dose_uM.agg(["min", "max"]).reset_index()
    doses = pd.read_parquet(args.dose / "dose.parquet", columns=["compound_id", "minimum", "maximum"])
    matched = pd.read_csv(args.registered / "name_overlap.tsv", sep="\t")
    linked = curves.merge(matched, on=["compound_name", "compound_fimm"], validate="many_to_one")
    linked = linked.merge(doses, on="compound_id", validate="many_to_one")
    linked = linked.assign(full_window=linked["min"].le(linked.minimum) & linked["max"].ge(linked.maximum),
                           highest=linked["min"].le(linked.maximum) & linked["max"].ge(linked.maximum))
    grouped = linked.groupby(["compound_id", "source_compound_label", "compound_fimm"], sort=True)
    support = []
    for (compound, name, code), group in grouped:
        support.append({"compound_id": compound, "name": name, "oslo_product": code, "donors": group.lineage_group.nunique(),
                        "oslo_curves": len(group), "full_window": int(group.full_window.sum()),
                        "licob_highest_dose_in_support": int(group.highest.sum())})
    lib = curves.groupby("library_id").agg(donors=("lineage_group", "nunique"),
            samples=("sample_id", "nunique"), curves=("sample_id", "size"), products=("compound_fimm", "nunique"))
    donors_per_product = curves.groupby(["compound_fimm", "library_id"]).lineage_group.nunique().unstack(fill_value=0)
    if set(donors_per_product.columns) != {"lib1", "lib2"}:
        raise ValueError("The registered comparison needs both published screening libraries")
    across_libraries = donors_per_product[donors_per_product.ge(10).all(axis=1)].index
    in_both = curves.groupby("sample_id").library_id.nunique().gt(1)
    groups_both = curves.groupby("lineage_group").library_id.nunique().gt(1)
    publish_json(args.out, {"oslo_registration_sha256": OSLO_REGISTRATION,
        "licob_dose_registration_sha256": DOSE_REGISTRATION, "library_support": lib.to_dict("index"),
        "samples_in_both_libraries": int(in_both.sum()), "donors_in_both_libraries": int(groups_both.sum()),
        "exact_products_with_ten_donors_each_library": sorted(across_libraries),
        "nominal_name_links": support, "full_window_nominal_drugs": linked.loc[linked.full_window, "compound_id"].nunique(),
        "upper_dose_nominal_drugs": linked.loc[linked.highest, "compound_id"].nunique()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, function, fields in (("register", register, ("source", "licob")),
                                   ("bridge", bridge, ("registered", "dose"))):
        stage = stages.add_parser(name)
        for field in fields:
            stage.add_argument(f"--{field}", type=Path, required=True)
        stage.set_defaults(execute=function)
    args = parser.parse_args()
    args.execute(args)


if __name__ == "__main__":
    main()
