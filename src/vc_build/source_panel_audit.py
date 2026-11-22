"""Audit exact response coverage outside the original known-compound panel."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from .cross_source_data import (
    MODEL_COLS, ctrp_rows, prism_rows, read_raw, response_rows, roster_maps, roster_table,
)
from .function_data import canonical_chemistry
from .io import publish_json, sha256
from .v0plus_cv import training_inputs


def compound_panel(sources):
    ctrp = read_raw(sources["ctrp_compounds"]["path"], ["cpd_smiles"], sep="\t").cpd_smiles
    prism = read_raw(sources["prism_curves"]["path"], ["smiles"]).smiles
    parents = set()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        for smiles in set(ctrp) | set(prism):
            if smiles:
                try:
                    parents.add(canonical_chemistry(smiles)[1])
                except (ValueError, RuntimeError):
                    continue
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    return pd.DataFrame({"compound_id": sorted(parents)})


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
    eligible = compound_panel(inputs)
    hcc = set(fit.loc[fit.lineage_role.eq("adult_hcc"), "cell_id"])
    known = set(fit.compound_id)
    report, support, hcc_support = {}, {}, {}
    for method, source, assay, time in (
        (ctrp_rows, "CTRPv2", "CellTiterGlo_72h", 72), (prism_rows, "PRISM20Q2", "PRISM_barcoded_viability", 120),
    ):
        rows = method(inputs, roster[roster.source_id.eq(source)], eligible)
        collapsed, _ = response_rows(rows, source, assay, time, "original public screen",
                                     ["dose_min_uM", "dose_max_uM"])
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
        hcc_extra = extra[extra.cell_id.isin(hcc)]
        support[source] = set(extra.compound_id)
        hcc_support[source] = set(hcc_extra.compound_id)
        report[source] = {
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
    publish_json(args.out, {
        "original_bundle_sha256": sha256(args.bundle / "receipt.json"),
        "source_sha256": {name: inputs[name]["sha256"] for name in source_names},
        "retained_core_cells": int(fit.cell_id.nunique()), "retained_adult_hcc": len(hcc),
        "new_parents_with_both_sources": len(set.intersection(*support.values())),
        "new_parents_with_hcc_in_both_sources": len(set.intersection(*hcc_support.values())),
        "sources": report, "gdsc2_screened_native_ids": int(meta.DRUG_ID.nunique()),
        "gdsc2_original_exact_parent_compounds": int(fit.loc[fit.source_id.eq("GDSC2"), "compound_id"].nunique()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
