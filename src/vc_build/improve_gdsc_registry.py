"""Cross-check the released IMPROVE drug structure catalogue against GDSC 8.5."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from .cross_source_data import (
    MIXTURE, MODEL_COLS, gdsc_rows, read_raw, response_rows, roster_maps, roster_table,
    unambiguous_parent,
)
from .expanded_panel_cv import RECEIPTS, registered_inputs
from .function_data import alias_token, new_output
from .io import publish_json, sha256

DRUG_INFO_SHA256 = "134234a2d2bae7f02a970308a0a728bafbc44d83459e1604fff933ba6c83b138"


def matched_structures(table, metadata, ids):
    result, audits = [], {"publisher_native_ids": 0, "name_conflicts": [], "structure_conflicts": []}
    names = {
        key: {t for name in values for t in [alias_token(name)] if t}
        for key, values in metadata.assign(names=lambda frame: frame.DRUG_NAME + "," + frame.SYNONYMS)
        .groupby("DRUG_ID").names.agg(",".join).str.split(",").items()
    }
    candidates = table.assign(native_compound_id=table.DrugID.str.extract(r"^GDSC[.](\d+)$")[0])
    candidates = candidates[candidates.native_compound_id.isin(ids)]
    audits["publisher_native_ids"] = int(candidates.native_compound_id.nunique())

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        for native, options in candidates.groupby("native_compound_id", sort=True):
            if (not set(options.NAME.map(alias_token)) <= names.get(native, set())
                    or options.NAME.str.contains(MIXTURE).any()):
                audits["name_conflicts"].append(native)
                continue
            smiles = set(options.SMILES) | set(options.canSMILES)
            structures = [unambiguous_parent(item) for item in sorted(smiles - {""})]
            if not structures or any(x is None for x in structures) or len({x[1] for x in structures}) > 1:
                audits["structure_conflicts"].append(native)
                continue
            structure, exact, connectivity, scaffold = min(structures)
            result.append({"native_compound_id": native, "compound_id": exact,
                           "isomeric_smiles": structure, "connectivity_group": connectivity,
                           "scaffold_group": scaffold, "publisher_drug_label": options.NAME.min()})
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    return pd.DataFrame(result), audits


def run(args):
    _, clean, _, _ = registered_inputs(args)
    if sha256(args.drug_info) != DRUG_INFO_SHA256:
        raise ValueError("Unexpected extracted IMPROVE structure catalogue")
    rece = json.loads((args.original / "receipt.json").read_text())["inputs"]
    source_ids = ("gdsc2", "gdsc_labels", "depmap_models")
    for name in source_ids:
        if sha256(Path(rece[name]["path"])) != rece[name]["sha256"]:
            raise ValueError(f"Input changed for {name}")
    catalogue = pd.read_csv(args.drug_info, sep="\t", dtype=str, keep_default_na=False)
    paths = {name: rece[name]["path"] for name in source_ids}
    official = read_raw(paths["gdsc_labels"], ["DRUG_ID", "DRUG_NAME", "SYNONYMS", "TARGET"])
    ids = set(read_raw(paths["gdsc2"], ["DRUG_ID"]).DRUG_ID)
    mapped, exclusions = matched_structures(catalogue, official, ids)

    current = clean[clean.source_id.eq("GDSC2")][["native_compound_id", "compound_id"]].drop_duplicates()
    agreement = mapped[["native_compound_id", "compound_id"]].merge(
        current, on="native_compound_id", suffixes=("_mapped", "_registered"), validate="one_to_one",
    )
    conflict = agreement[agreement.compound_id_mapped.ne(agreement.compound_id_registered)]
    # A disagreement in an older native ID cannot introduce an alternative parent to the library.
    new = mapped[~mapped.native_compound_id.isin(current.native_compound_id) &
                 ~mapped.compound_id.isin(current.compound_id)].copy()

    raw = read_raw(paths["gdsc2"], ["CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID"])
    roster = roster_table(raw, "GDSC2", "CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID")
    roster = roster_maps(roster, read_raw(paths["depmap_models"], MODEL_COLS))
    roster = roster[roster.cell_id.isin(clean.cell_id)]
    observations = gdsc_rows(paths["gdsc2"], roster, new[["native_compound_id", "compound_id"]])
    single, _ = response_rows(observations, "GDSC2", "CellTiterGlo_72h", 72,
                              "Sanger_GDSC2_8.5_fitted_curve", ["dose_min_uM", "dose_max_uM"])
    cells = clean[["cell_id", "lineage_group", "source_cell_label", "tissue_origin", "lineage_role"]]
    cells = cells.drop_duplicates().set_index("cell_id", verify_integrity=True)
    targets = (official.groupby("DRUG_ID").TARGET.agg(lambda labels: ";".join(sorted(set(labels) - {""})))
               .rename("drug_target_source_labels"))
    single = single.join(cells, on="cell_id", validate="many_to_one").merge(
        new, on=["native_compound_id", "compound_id"], validate="many_to_one",
    ).join(targets, on="native_compound_id", validate="many_to_one")
    single = single.rename(columns={"publisher_drug_label": "source_compound_label"})
    single["row_id"] = (single.source_id + "|" + single.cell_id + "|" + single.compound_id +
                        "|" + single.endpoint_id)
    hcc = single[single.lineage_role.eq("adult_hcc")]
    if single.empty or not single.row_id.is_unique or set(single.row_id) & set(clean.row_id):
        raise ValueError("GDSC addition requires distinctive measurements on registered cells")
    with new_output(args.out) as tmp:
        paths_out = {
            "additional_rows.parquet": single.sort_values("row_id"),
            "structure_registry.parquet": mapped.sort_values("native_compound_id"),
        }
        for name, rows in paths_out.items():
            rows.to_parquet(tmp / name, index=False)
        publish_json(tmp / "receipt.json", {
            "core_bundle_sha256": RECEIPTS["core"], "publisher_catalogue_sha256": DRUG_INFO_SHA256,
            "publisher_doi": "10.5281/zenodo.22920452", "publisher_license": "CC-BY-4.0",
            "source_sha256": {name: rece[name]["sha256"] for name in source_ids},
            "source_native_ids": len(ids), "exact_join_native_ids": len(mapped),
            "previous_verified_native_ids": len(current), "matched_previous_native_ids": len(agreement),
            "conflicting_previous_native_ids": conflict.to_dict("records"),
            "new_source_drugs": int(single.compound_id.nunique()), "new_source_rows": len(single),
            "new_source_drugs_already_in_core_through_other_sources": int(
                single.loc[single.compound_id.isin(clean.compound_id), "compound_id"].nunique()
            ),
            "new_source_cells": int(single.cell_id.nunique()),
            "hcc_new_source_rows": len(hcc), "hcc_new_source_drugs": int(hcc.compound_id.nunique()),
            "hcc_cell_models": int(hcc.cell_id.nunique()), "excluded": exclusions,
            "artifacts": {name: sha256(tmp / name) for name in paths_out},
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "core", "drug-info", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
