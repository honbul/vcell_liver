"""Audit exact response coverage outside the original known-compound panel."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from .cross_source_data import (
    MIXTURE, MODEL_COLS, ctrp_rows, prism_parent, prism_rows, read_raw, response_rows, roster_maps, roster_table,
    unambiguous_parent,
)
from .function_data import new_output
from .io import publish_json, sha256
from .v0plus_cv import training_inputs


SOURCE_COLUMNS = {
    "CTRPv2": ("ctrp_compounds", "master_cpd_id", "cpd_name", "cpd_smiles", "gene_symbol_of_protein_target", "\t"),
    "PRISM20Q2": ("prism_curves", "broad_id", "name", "smiles", "target", ","),
}


def compound_panel(sources):
    records = []
    counts = {}
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        for source, (name, native, label, compound, target, delim) in SOURCE_COLUMNS.items():
            table = read_raw(sources[name]["path"], [native, label, compound, target], sep=delim).drop_duplicates()
            resolve = prism_parent if source == "PRISM20Q2" else unambiguous_parent
            lookup = {smiles: resolve(smiles) for smiles in table[compound].unique()}
            candidate = table[compound].map(lookup).mask(table[label].str.contains(MIXTURE)).dropna()
            keys = table[[native, label, target]].join(candidate.rename("parent"), how="inner")
            definitions = pd.DataFrame(keys.parent.tolist(), index=keys.index, columns=[
                "isomeric_smiles", "compound_id", "connectivity_group", "scaffold_group",
            ])
            merged = keys.drop(columns="parent").join(definitions)
            known = merged.groupby(native).compound_id.nunique()
            filtered_ids = set(table.loc[~table.index.isin(candidate.index), native])
            valid = set(known[known.eq(1)].index) - filtered_ids
            selected = merged[merged[native].isin(valid)].rename(columns={
                native: "native_compound_id", label: "source_compound_label", target: "drug_target_source_labels",
            }).assign(source_id=source)
            records.append(selected)
            counts[source] = {"catalogue_native_ids": int(table[native].nunique()),
                              "qualified_single_agent_native_ids": len(valid)}
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    return pd.concat(records, ignore_index=True), counts


def write_extension(path, fit, extensions, chemistry, provenance):
    names = ["cell_id", "lineage_group", "source_cell_label", "tissue_origin", "lineage_role"]
    cells = fit[names].drop_duplicates().set_index("cell_id", verify_integrity=True)
    rows = pd.concat(extensions, ignore_index=True).join(cells, on="cell_id", validate="many_to_one")
    join = ["source_id", "native_compound_id", "compound_id"]
    native = chemistry.sort_values(["source_compound_label", "isomeric_smiles"]).drop_duplicates(join)
    rows = rows.merge(native, on=join, validate="many_to_one")
    if len(rows) != sum(map(len, extensions)) or set(rows.compound_id) & set(fit.compound_id):
        raise ValueError("A new source row lost its exact compound identity or overlaps the original panel")
    rows["row_id"] = rows.source_id + "|" + rows.cell_id + "|" + rows.compound_id + "|" + rows.endpoint_id
    if not rows.row_id.is_unique or rows.lineage_group.isna().any():
        raise ValueError("Additional response rows require a unique identity in the original cell roster")
    # Use one canonical descriptor representation per parent in joint models.
    structures = native.sort_values("isomeric_smiles").drop_duplicates("compound_id").set_index("compound_id")
    rows["isomeric_smiles"] = rows.compound_id.map(structures.isomeric_smiles)
    with new_output(path) as tmp:
        destination = tmp / "additional_rows.parquet"
        rows.sort_values("row_id", ignore_index=True).to_parquet(destination, index=False)
        native_path = tmp / "chemistry.parquet"
        chemistry[chemistry.compound_id.isin(rows.compound_id)].to_parquet(native_path, index=False)
        publish_json(tmp / "receipt.json", provenance | {
            "additional_rows": len(rows), "additional_compounds": int(rows.compound_id.nunique()),
            "cell_models": int(rows.cell_id.nunique()), "lineages": int(rows.lineage_group.nunique()),
            "augmented_training_rows": len(fit) + len(rows),
            "artifacts": {p.name: sha256(p) for p in (destination, native_path)},
        })


def correct_core(output, bundle, fit, screened, provenance):
    affected = fit[fit.native_drug_label.str.contains(MIXTURE)]
    if affected.empty:
        raise ValueError("Expected the withdrawn combination measurements in the original core")
    affected_keys = affected[["source_id", "compound_id"]].drop_duplicates()
    registered = pd.MultiIndex.from_frame(affected_keys)
    matches = fit.merge(affected_keys, on=["source_id", "compound_id"], validate="many_to_one")
    if len(matches) != len(affected):
        raise ValueError("All records of an affected source and parent must be corrected consistently")
    columns = ["source_protocol", "source_compound_label", "isomeric_smiles", "connectivity_group",
               "scaffold_group", "drug_target_source_labels"]
    keys = ["source_id", "compound_id"]
    metadata = affected[[*keys, *columns]].drop_duplicates().set_index(keys, verify_integrity=True)
    cells = fit[["cell_id", "lineage_group", "source_cell_label", "tissue_origin", "lineage_role"]]
    cells = cells.drop_duplicates().set_index("cell_id", verify_integrity=True)
    combined = pd.concat(screened, ignore_index=True)
    replacement = combined[combined.set_index(keys).index.isin(registered)].drop(columns="source_protocol")
    replacement = replacement.join(metadata, on=keys, validate="many_to_one")
    replacement = replacement.join(cells, on="cell_id", validate="many_to_one")
    replacement["row_id"] = (replacement.source_id + "|" + replacement.cell_id + "|" +
                             replacement.compound_id + "|" + replacement.endpoint_id)
    remaining = fit[~fit.row_id.isin(affected.row_id)]
    clean = pd.concat([remaining, replacement[fit.columns]], ignore_index=True)
    if (set(replacement.set_index(keys).index) != set(registered) or clean.row_id.duplicated().any()
            or clean.native_drug_label.str.contains(MIXTURE).any()):
        raise ValueError("Incomplete correction to verified single-agent responses")
    changes = []
    for (source, drug), rows in affected.groupby(keys, sort=True):
        pure = replacement[replacement.source_id.eq(source) & replacement.compound_id.eq(drug)]
        changes.append({
            "source": source, "compound_id": drug, "compound_label": rows.source_compound_label.iloc[0],
            "removed_native_ids": sorted(rows.native_compound_id.unique()), "withdrawn_rows": len(rows),
            "withdrawn_adult_hcc_rows": int(rows.lineage_role.eq("adult_hcc").sum()),
            "single_agent_native_ids": sorted(pure.native_compound_id.unique()), "replacement_rows": len(pure),
            "replacement_adult_hcc_rows": int(pure.lineage_role.eq("adult_hcc").sum()),
        })
    with new_output(output) as tmp:
        train = tmp / "train"
        train.mkdir()
        os.link(bundle / "train/rna.parquet", train / "rna.parquet")
        sorted_rows = clean.rename(columns={"row_id": "record_id", "lineage_group": "cell_group"})
        sorted_rows.sort_values("record_id", ignore_index=True).to_parquet(train / "rows.parquet", index=False)
        publish_json(tmp / "receipt.json", {
            "dataset": "vc_build_v1_training_single_agent_correction",
            "original_bundle_sha256": provenance["original_bundle_sha256"],
            "source_sha256": provenance["source_sha256"], "affected_source_compounds": changes,
            "training_rows": len(clean), "training_cells": int(clean.cell_id.nunique()),
            "training_lineages": int(clean.lineage_group.nunique()), "training_drugs": int(clean.compound_id.nunique()),
            "unchanged_rows": len(remaining), "artifacts": {
                str(file.relative_to(tmp)): {"sha256": sha256(file), "bytes": file.stat().st_size}
                for file in sorted(train.iterdir())
            },
        })


def audit(args):
    fit, _, original = training_inputs(args.bundle)
    source_names = ("depmap_models", "ctrp_cells", "ctrp_compounds", "ctrp_experiments", "ctrp_curves",
                    "prism_cells", "prism_curves", "gdsc2")
    inputs = original["inputs"]
    for name in source_names:
        spec = inputs[name]
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Source changed: {name}")
    models = read_raw(inputs["depmap_models"]["path"], MODEL_COLS)
    ctrp = roster_table(read_raw(inputs["ctrp_cells"]["path"], ["ccl_name", "master_ccl_id"], sep="\t"),
                        "CTRPv2", "ccl_name", "master_ccl_id")
    prism = roster_table(read_raw(inputs["prism_cells"]["path"], ["ccle_name", "depmap_id"]),
                         "PRISM20Q2", "ccle_name", "depmap_id")
    roster = roster_maps(pd.concat([ctrp, prism], ignore_index=True), models)
    roster = roster[roster.cell_id.isin(fit.cell_id)]
    eligible, catalog_counts = compound_panel(inputs)
    hcc = set(fit.loc[fit.lineage_role.eq("adult_hcc"), "cell_id"])
    known = set(fit.compound_id)
    report, support, hcc_support, extensions, screened = {}, {}, {}, [], []
    for method, source, assay, time in (
        (ctrp_rows, "CTRPv2", "CellTiterGlo_72h", 72), (prism_rows, "PRISM20Q2", "PRISM_barcoded_viability", 120),
    ):
        allowed = eligible[eligible.source_id.eq(source)]
        rows = method(inputs, roster[roster.source_id.eq(source)], allowed,
                      native_ids=set(allowed.native_compound_id))
        collapsed, _ = response_rows(rows, source, assay, time, "original public screen",
                                     ["dose_min_uM", "dose_max_uM"])
        screened.append(collapsed)
        prior = fit[fit.source_id.eq(source)]
        actual = collapsed.set_index(["cell_id", "compound_id"]).endpoint_value
        same = prior.set_index(["cell_id", "compound_id"]).endpoint_value
        reproduced = actual.reindex(same.index)
        # Counts outside the core are independent of any changed protocol selection on the original panel.
        # The prepared core stays authoritative for its compounds.
        present = reproduced.notna()
        difference = (reproduced[present] - same[present]).abs()
        mismatched = difference.gt(1e-8)
        extra = collapsed[~collapsed.compound_id.isin(known)]
        extensions.append(extra)
        hcc_extra = extra[extra.cell_id.isin(hcc)]
        support[source] = set(extra.compound_id)
        hcc_support[source] = set(hcc_extra.compound_id)
        report[source] = catalog_counts[source] | {
            "prior_rows": len(prior), "prior_drugs": int(prior.compound_id.nunique()),
            "prior_rows_missing_after_recensus": int((~present).sum()),
            "prior_rows_difference_over_1e_8": int(mismatched.sum()),
            "prior_drugs_with_differing_values": int(
                difference[mismatched].index.get_level_values("compound_id").nunique()),
            "prior_reproduced_max_absolute_difference": float(difference.max()) if len(difference) else None,
            "prior_difference_p95_absolute": float(np.percentile(difference, 95)) if len(difference) else None,
            "qualified_rows": len(collapsed), "qualified_drugs": int(collapsed.compound_id.nunique()),
            "additional_rows_outside_core": len(extra), "additional_drugs_outside_core": len(support[source]),
            "additional_hcc_rows": len(hcc_extra), "additional_hcc_drugs": len(hcc_support[source]),
            "additional_drugs_at_least_8_hcc": int(hcc_extra.groupby("compound_id").cell_id.nunique().ge(8).sum()),
            "cells_with_additional_hcc_rows": int(hcc_extra.cell_id.nunique()),
        }
    meta = read_raw(inputs["gdsc2"]["path"], ["DRUG_ID"])
    receipt = {
        "original_bundle_sha256": sha256(args.bundle / "receipt.json"),
        "source_sha256": {name: inputs[name]["sha256"] for name in source_names},
        "retained_core_cells": int(fit.cell_id.nunique()), "retained_adult_hcc": len(hcc),
        "new_parents_with_both_sources": len(set.intersection(*support.values())),
        "new_parents_with_hcc_in_both_sources": len(set.intersection(*hcc_support.values())),
        "sources": report, "gdsc2_screened_native_ids": int(meta.DRUG_ID.nunique()),
        "gdsc2_original_exact_parent_compounds": int(fit.loc[fit.source_id.eq("GDSC2"), "compound_id"].nunique()),
    }
    if args.extension is not None:
        write_extension(args.extension, fit, extensions, eligible, receipt)
    if args.corrected_core is not None:
        correct_core(args.corrected_core, args.bundle, fit, screened, receipt)
    publish_json(args.out, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--corrected-core", type=Path)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
