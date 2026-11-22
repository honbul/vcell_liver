"""Audit and prepare public-source response training and source-new adult LIMORE queries."""

from collections import defaultdict
import gzip
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from rdkit import RDLogger

from .function_data import alias_token, canonical_chemistry, file_sha256, new_output, read_tsv, verify_inputs, write_json


MODEL_COLS = [
    "ModelID", "PatientID", "RRID", "CellLineName", "StrippedCellLineName", "CCLEName",
    "SangerModelID", "COSMICID", "ModelIDAlias", "OncotreeLineage", "OncotreePrimaryDisease",
    "OncotreeSubtype", "Age",
]
MODEL_KEYS = ["ModelID", "RRID", "SangerModelID", "COSMICID", "ModelIDAlias", "PatientID"]
MODEL_NAMES = ["CellLineName", "StrippedCellLineName", "CCLEName"]


def read_raw(path, columns, sep=","):
    return pd.read_csv(path, sep=sep, usecols=columns, dtype=str, keep_default_na=False)


def roster_table(frame, source, names, first_id, second_id=None):
    """Deduplicate source identities without selecting on a response value."""
    records = pd.DataFrame({
        "source_id": source, "name": frame[names], "source_key": frame[first_id],
        "source_key_2": frame[second_id] if second_id else "",
    })
    return records.drop_duplicates(ignore_index=True)


def parse_cellosaurus_identity(path):
    """Read only identity, cross-references and same lineage links from the public release."""
    profiles, secondary = {}, {}
    fields = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("//"):
                if fields["AC"]:
                    acc = fields["AC"][0]
                    names = fields["ID"] + [alias.strip() for aliases in fields["SY"] for alias in aliases.split(";")]
                    references = [part.split(";", 1)[-1].strip() for part in fields["DR"]
                                  if part.split(";", 1)[0] in {"DepMap", "Cosmic", "Cell_Model_Passports"}]
                    related = {
                        value.split(" ! ", 1)[0] for key in ("HI", "OI") for value in fields[key]
                    }
                    profiles[acc] = (names, references, related)
                    secondary.update({alias.strip(): acc for part in fields["AS"] for alias in part.split(";")})
                fields.clear()
            elif line[:2] in {"AC", "AS", "ID", "SY", "DR", "HI", "OI"} and line[2:5] == "   ":
                fields[line[:2]].append(line[5:].strip())
    neighbors = defaultdict(set)
    for acc, (_, _, relationships) in profiles.items():
        for similar in relationships:
            similar = secondary.get(similar, similar)
            neighbors[acc].add(similar)
            neighbors[similar].add(acc)
    return profiles, secondary, neighbors


def related_cellosaurus(seed, profiles, secondary, neighbors):
    remaining = {secondary.get(acc, acc) for acc in seed if acc}
    known = set()
    for acc in remaining:
        if acc not in profiles:
            raise ValueError(f"LIMORE registry accession {acc} absent from pinned Cellosaurus release")
    while remaining:
        acc = remaining.pop()
        known.add(acc)
        remaining.update(neighbors[acc] - known)
    names = {name for acc in known if acc in profiles for name in profiles[acc][0]}
    keys = {key for acc in known if acc in profiles for key in profiles[acc][1]}
    keys.update(alias for alias, canonical in secondary.items() if canonical in known)
    return known, keys, names


def protected_adult_families(limore_cells, db, models, roster, max_families):
    """Choose groups absent from all registered source rosters using identity metadata only."""
    report_rows = []
    adult = limore_cells[limore_cells.included.eq("True")]
    roster = roster.assign(name_token=roster.name.map(alias_token))
    for group, cells in adult.groupby("family_id", sort=True):
        seed = set(cells.cellosaurus_id) | set(cells.root_ancestor)
        for relations in cells.same_individual:
            seed.update(item.strip() for item in relations.split(",") if item.strip())
        accessions, source_ids, aliases = related_cellosaurus(seed, *db)
        # Include lines sharing a known DepMap source patient even with a distinct cell alias.
        direct = models[
            models.RRID.isin(accessions) | models.ModelID.isin((set(cells.depmap_model_id) | source_ids) - {""})
        ]
        known_patients = (set(cells.depmap_patient_id) | set(direct.PatientID)) - {""}
        relatives = models.loc[models.RRID.isin(accessions) | models.PatientID.isin(known_patients)
                               | models.ModelID.isin(set(direct.ModelID))]
        exact_ids = set(cells.depmap_model_id) | accessions | source_ids
        exact_ids.update(value for column in MODEL_KEYS for value in relatives[column] if value)
        exact_ids.discard("")
        names = set(cells.source_name) | set(cells.cellosaurus_name) | aliases
        names.update(value for column in MODEL_NAMES for value in relatives[column] if value)
        names = set(map(alias_token, names)) - {""}
        matching = roster[
            roster.source_key.isin(exact_ids) | roster.source_key_2.isin(exact_ids) | roster.name_token.isin(names)
        ]
        exposed = sorted(matching.source_id.unique())
        model_ids = ";".join(sorted(set(relatives.ModelID)))
        siblings = ";".join(sorted(accessions))
        for cell in cells.sort_values("model_id").itertuples(index=False):
            report_rows.append({
                "cell_id": cell.model_id, "cell_group": group, "source_cell_label": cell.source_name,
                "cellosaurus_id": cell.cellosaurus_id, "table_s1_age_years": cell.table_s1_age_years,
                "accessions_audited": siblings, "related_depmap_models": model_ids,
                "roster_sources": ";".join(exposed),
                "roster_labels": ";".join(sorted(set(matching.source_id + ":" + matching.name))),
                "source_new": not exposed,
            })
    audit = pd.DataFrame(report_rows).sort_values("cell_id", ignore_index=True)
    candidates = audit[audit.source_new].drop_duplicates("cell_group", keep="first")
    audit["confirmation_query"] = audit.cell_id.isin(candidates.cell_id.head(max_families))
    audit["prior_project_exposure"] = "LIMORE outcomes previously analyzed; current cross fit excludes every LIMORE outcome"
    if audit.confirmation_query.sum() < max_families:
        raise ValueError(f"Only {len(candidates)} source-new independent adult-HCC groups; required {max_families}")
    return audit


def unique_lookup(rows, key_columns, model_column="ModelID", normalize=str):
    candidates = defaultdict(set)
    for key in key_columns:
        for value, model in zip(rows[key], rows[model_column]):
            if value:
                candidates[normalize(value)].add(model)
    return candidates


def resolve_model(row, by_ids, by_names):
    id_sets = [by_ids[value] for value in (row.source_key, row.source_key_2) if value in by_ids]
    candidates = set.intersection(*id_sets) if id_sets else by_names.get(alias_token(row.name), set())
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def roster_maps(roster, models):
    by_ids = unique_lookup(models, MODEL_KEYS[:5])
    by_names = unique_lookup(models, MODEL_NAMES, normalize=alias_token)
    # GDSC/PRISM stable IDs take precedence; CTRP numeric master IDs have no relation to COSMIC IDs.
    table = roster.copy()
    known_ids = ~table.source_id.eq("CTRPv2")
    table["cell_id"] = [
        resolve_model(row, by_ids if is_stable else {}, by_names)
        for row, is_stable in zip(table.itertuples(index=False), known_ids)
    ]
    return table


def source_structures(metadata, id_column, name_column, smiles_column, limore_panel):
    """Require a single exact standardized parent for each source chemical identifier."""
    structures = metadata[[id_column, name_column, smiles_column]].drop_duplicates().copy()
    lookup = {}
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        for smiles in structures[smiles_column].unique():
            try:
                lookup[smiles] = canonical_chemistry(smiles)[1] if smiles else None
            except (ValueError, RuntimeError):
                lookup[smiles] = None
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")
    structures["compound_id"] = structures[smiles_column].map(lookup)
    groups = structures.groupby(id_column, dropna=False).compound_id.nunique(dropna=False)
    ambiguous = set(groups[groups.ne(1)].index)
    matched = structures[structures.compound_id.isin(limore_panel.compound_id) & ~structures[id_column].isin(ambiguous)]
    return (matched[[id_column, name_column, "compound_id"]]
            .drop_duplicates(id_column).rename(columns={id_column: "native_compound_id", name_column: "native_drug_label"}))


def select_run_identity(rows):
    """Use the largest coverage identity per compound and pick one published protocol before outcome analysis."""
    counts = (rows.groupby(["compound_id", "native_compound_id", "source_run"], dropna=False)
              .cell_id.nunique().rename("n_cells").reset_index())
    counts["preferred_run"] = counts.source_run.eq("MTS010")
    counts = counts.sort_values(["compound_id", "preferred_run", "n_cells", "native_compound_id", "source_run"],
                                ascending=[True, False, False, True, True])
    chosen = counts.drop_duplicates("compound_id")
    keys = ["compound_id", "native_compound_id", "source_run"]
    return rows.merge(chosen[keys], on=keys, validate="m:1"), chosen


def response_rows(rows, source_id, assay_id, time_h, protocol, dose_columns):
    value = pd.to_numeric(rows.endpoint_value, errors="coerce")
    rows = rows[value.notna()].copy()
    rows["endpoint_value"] = value[value.notna()]
    rows, selected = select_run_identity(rows)
    fields = ["cell_id", "compound_id", "native_compound_id", "native_drug_label", "source_run"] + dose_columns
    # Curves for multiple experiments of one registered protocol are repeat measurements.
    observations = rows.groupby(fields, dropna=False, sort=True).endpoint_value.agg(["median", "size"]).reset_index()
    observations = observations.rename(columns={"median": "endpoint_value", "size": "n_source_records"})
    if observations.duplicated(["cell_id", "compound_id"]).any():
        raise ValueError(f"{source_id} has repeated dose-window protocols for one source cell-compound")
    observations = observations.assign(
        source_id=source_id, assay_id=assay_id, source_protocol=protocol,
        endpoint_id=f"{source_id}_native_AUC", endpoint_unit="dimensionless_source_auc",
        endpoint_direction="lower_is_more_sensitive", endpoint_contract="curve_summary",
        supported_dose_response=False, baseline_is_proxy=True, time_h=time_h, dose_uM=None,
        rna_assay="bulk_rnaseq_log2_tpm_plus_one", culture_format="2D",
    )
    selected["source_id"] = source_id
    return observations, selected


def gdsc_rows(path, lookup, match):
    gdsc = read_raw(path, ["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME",
                           "MIN_CONC", "MAX_CONC", "AUC"])
    gdsc = gdsc.rename(columns={"DRUG_ID": "native_compound_id", "DRUG_NAME": "native_drug_label",
                                "AUC": "endpoint_value", "MIN_CONC": "dose_min_uM", "MAX_CONC": "dose_max_uM"})
    rows = gdsc.merge(match, on="native_compound_id", validate="m:1", how="inner")
    cells = lookup.rename(columns={"source_key": "SANGER_MODEL_ID", "source_key_2": "COSMIC_ID",
                                   "name": "CELL_LINE_NAME"})
    rows = rows.merge(cells[["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "cell_id"]],
                      on=["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME"], validate="m:1")
    rows["source_run"] = "GDSC2_" + rows.dose_min_uM + "_" + rows.dose_max_uM
    for column in ("dose_min_uM", "dose_max_uM"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    return rows


def ctrp_rows(sources, lookup, limore_panel, *, native_ids=None):
    compounds = read_raw(sources["ctrp_compounds"]["path"],
                         ["master_cpd_id", "cpd_name", "cpd_smiles", "top_test_conc_umol"], sep="\t")
    mapping = source_structures(compounds, "master_cpd_id", "cpd_name", "cpd_smiles", limore_panel)
    if native_ids is not None:
        mapping = mapping[mapping.native_compound_id.isin(native_ids)]
    dose = compounds[["master_cpd_id", "top_test_conc_umol"]].rename(columns={
        "master_cpd_id": "native_compound_id", "top_test_conc_umol": "dose_max_uM"})
    mapping = mapping.merge(dose, on="native_compound_id", validate="1:1")
    experiments = read_raw(sources["ctrp_experiments"]["path"], ["experiment_id", "master_ccl_id", "snp_fp_status"], sep="\t")
    # Explicit failed fingerprint records cannot broaden the assay's cell identity support.
    experiments = experiments[experiments.snp_fp_status.eq("SNP-matched-reference")]
    experiments = experiments[["experiment_id", "master_ccl_id"]].drop_duplicates()
    if not experiments.experiment_id.is_unique:
        raise ValueError("Different cells annotated to one CTRP experiment identifier")
    curves = read_raw(sources["ctrp_curves"]["path"], ["experiment_id", "master_cpd_id", "area_under_curve"], sep="\t")
    curves = curves.rename(columns={"master_cpd_id": "native_compound_id", "area_under_curve": "endpoint_value"})
    rows = curves.merge(mapping, on="native_compound_id", validate="m:1").merge(experiments, on="experiment_id", validate="m:1")
    cells = lookup[["source_key", "cell_id"]].rename(columns={"source_key": "master_ccl_id"})
    rows = rows.merge(cells, on="master_ccl_id", validate="m:1")
    rows["source_run"] = "CTRPv2"
    rows["dose_min_uM"] = np.nan
    rows["dose_max_uM"] = pd.to_numeric(rows["dose_max_uM"], errors="raise")
    return rows


def prism_rows(sources, lookup, limore_panel, *, native_ids=None):
    raw = read_raw(sources["prism_curves"]["path"],
                   ["broad_id", "depmap_id", "ccle_name", "screen_id", "auc", "passed_str_profiling", "name", "smiles"])
    mapping = source_structures(raw, "broad_id", "name", "smiles", limore_panel).drop(columns="native_drug_label")
    if native_ids is not None:
        mapping = mapping[mapping.native_compound_id.isin(native_ids)]
    rows = raw[raw.passed_str_profiling.str.upper().eq("TRUE")].rename(columns={
        "broad_id": "native_compound_id", "depmap_id": "source_key", "ccle_name": "name",
        "name": "native_drug_label", "screen_id": "source_run", "auc": "endpoint_value",
    })
    rows = rows.merge(mapping, on="native_compound_id", validate="m:1")
    rows = rows.merge(lookup[["source_key", "name", "cell_id"]], on=["source_key", "name"], validate="m:1")
    rows["dose_min_uM"] = np.nan
    rows["dose_max_uM"] = np.nan
    return rows


def read_depmap_rna(path, selected_models, axis):
    header = pd.read_csv(path, nrows=0).columns
    symbol_columns = defaultdict(list)
    for name in header:
        match = re.fullmatch(r"(.+) \(\d+\)", name)
        if match:
            symbol_columns[match.group(1)].append(name)
    unique = {symbol: columns[0] for symbol, columns in symbol_columns.items() if len(columns) == 1}
    axis = axis[axis.isin(unique)]
    columns = {unique[symbol]: symbol for symbol in axis}
    pieces = []
    for chunk in pd.read_csv(path, usecols=["ModelID", "IsDefaultEntryForModel", *columns], chunksize=128):
        kept = chunk[chunk.ModelID.isin(selected_models) & chunk.IsDefaultEntryForModel.eq("Yes")]
        pieces.append(kept.drop(columns="IsDefaultEntryForModel").set_index("ModelID").rename(columns=columns))
    rna = pd.concat(pieces).reindex(columns=axis).astype("float32").sort_index()
    rna.index.name = "cell_id"
    if not rna.index.is_unique or not np.isfinite(rna).to_numpy().all():
        raise ValueError("Duplicate or invalid DepMap default RNA profile")
    return rna


def prep_cross(settings, limore_path, output):
    cfg = settings["cross"]
    sources = cfg["sources"]
    receipt_inputs = verify_inputs(sources)
    limore_path = Path(limore_path)
    receipt_inputs["limore_bundle"] = json.loads((limore_path / "receipt.json").read_text())
    gdsc_lookup = Path(cfg["gdsc_pug_match"])
    receipt_inputs["gdsc_pug_match"] = {
        "path": str(gdsc_lookup), "sha256": file_sha256(gdsc_lookup), "bytes": gdsc_lookup.stat().st_size,
    }
    locked_roster = verify_inputs({"locked_roster": cfg["locked_roster"]})["locked_roster"]
    limore_cells = read_tsv(limore_path / "cell_registry.tsv", dtype=str, keep_default_na=False)
    cellosaurus = parse_cellosaurus_identity(sources["cellosaurus"]["path"])
    depmap = read_raw(sources["depmap_models"]["path"], MODEL_COLS)
    if not depmap.ModelID.is_unique:
        raise ValueError("Duplicate DepMap model identifiers")

    gdsc_cells = [
        roster_table(read_raw(sources[db]["path"], ["CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID"]),
                     db.upper(), "CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID")
        for db in ("gdsc1", "gdsc2")
    ]
    others = [
        roster_table(read_raw(sources["ctrp_cells"]["path"], ["ccl_name", "master_ccl_id"], sep="\t"),
                     "CTRPv2", "ccl_name", "master_ccl_id"),
        roster_table(read_raw(sources["prism_cells"]["path"], ["ccle_name", "depmap_id"]),
                     "PRISM20Q2", "ccle_name", "depmap_id"),
    ]
    roster = pd.concat([*gdsc_cells, *others], ignore_index=True)
    cohort = protected_adult_families(limore_cells, cellosaurus, depmap, roster, cfg["clean_families"])
    observed_tsv = cohort.to_csv(sep="\t", index=False).encode("utf-8")
    if observed_tsv != Path(locked_roster["path"]).read_bytes():
        raise ValueError("Recomputed query roster disagrees with the preregistered identity-only roster")
    mapping = roster_maps(roster, depmap)
    candidates = cohort[cohort.confirmation_query]
    withheld_depmap = set(";".join(candidates.related_depmap_models).split(";")) - {""}
    fit_map = mapping[mapping.cell_id.notna() & ~mapping.cell_id.isin(withheld_depmap)]

    panel = read_tsv(limore_path / "compound_registry.tsv", dtype=str).dropna(subset=["compound_id"])
    gdsc_chem = read_tsv(cfg["gdsc_pug_match"], dtype=str).query("match_status == 'matched'")
    gdsc_chem = gdsc_chem[gdsc_chem.compound_id.isin(panel.compound_id)]
    gdsc_chem = gdsc_chem.rename(columns={"gdsc_drug_id": "native_compound_id"})
    gdsc_chem = gdsc_chem[["native_compound_id", "compound_id"]]
    inputs = [
        (gdsc_rows(sources["gdsc2"]["path"], fit_map[fit_map.source_id.eq("GDSC2")], gdsc_chem),
         "GDSC2", "CellTiterGlo_72h", 72, "Sanger_GDSC2_8.5_fitted_curve"),
        (ctrp_rows(sources, fit_map[fit_map.source_id.eq("CTRPv2")], panel),
         "CTRPv2", "CellTiterGlo_72h", 72, "Broad_CTD2_CTRPv2.0_2015_postQC_curve"),
        (prism_rows(sources, fit_map[fit_map.source_id.eq("PRISM20Q2")], panel),
         "PRISM20Q2", "PRISM_barcoded_viability", 120, "Broad_PRISM_20Q2_secondary_curve"),
    ]
    dose_columns = ["dose_min_uM", "dose_max_uM"]
    results = [response_rows(*source, dose_columns) for source in inputs]
    train = pd.concat([table for table, _ in results], ignore_index=True)
    chosen_protocols = pd.concat([reg for _, reg in results], ignore_index=True)

    selected_models = set(train.cell_id)
    limore_rna = pd.read_parquet(limore_path / "rna.parquet")
    fit_rna = read_depmap_rna(sources["depmap_rna"]["path"], selected_models, limore_rna.columns)
    train = train[train.cell_id.isin(fit_rna.index)]
    panel_columns = ["compound_id", "source_compound_label", "isomeric_smiles", "connectivity_group",
                     "scaffold_group", "drug_target_source_labels"]
    train = train.merge(panel[panel_columns], on="compound_id", validate="m:1")
    train = train.merge(depmap, left_on="cell_id", right_on="ModelID", validate="m:1")
    train["cell_group"] = train.PatientID.where(train.PatientID.ne(""), train.RRID.where(train.RRID.ne(""), train.cell_id))
    train["source_cell_label"] = train.CellLineName
    train["tissue_origin"] = train.OncotreeLineage.str.lower()
    hcc = train.OncotreeSubtype.str.contains("hepatocellular carcinoma", case=False)
    adult = pd.to_numeric(train.Age, errors="coerce").ge(18)
    train["lineage_role"] = np.select([hcc & adult, hcc], ["adult_hcc", "hcc_age_other_or_unknown"], default="other")
    train["record_id"] = train.source_id + "|" + train.cell_id + "|" + train.compound_id + "|" + train.endpoint_id
    train = train.drop(columns=MODEL_COLS).sort_values("record_id", ignore_index=True)
    fit_rna = fit_rna.loc[sorted(train.cell_id.unique())]

    # Compound support, query roster and gene axis depend on identities and assay coverage.
    # These are fixed before opening query outcomes for the distinct scoring artifact.
    supported = train.groupby("compound_id").source_id.nunique().rename("n_train_sources")
    query = pd.read_parquet(limore_path / "rows.parquet")
    query = query[query.cell_id.isin(candidates.cell_id) & query.compound_id.isin(supported.index)]
    if query.cell_id.nunique() != cfg["clean_families"] or query.cell_group.nunique() != cfg["clean_families"]:
        raise ValueError("Incomplete coverage for the source-new confirmation roster")
    query = query.join(supported, on="compound_id", validate="m:1").sort_values("record_id", ignore_index=True)
    labels = query[["record_id", "endpoint_id", "endpoint_unit", "endpoint_value"]]
    query = query.drop(columns="endpoint_value")
    query_rna = limore_rna.loc[sorted(query.cell_id.unique()), fit_rna.columns]
    if set(train.cell_id) & (withheld_depmap | set(query.cell_id)):
        raise ValueError("Overlap between fit response inputs and withheld cells")
    if train.record_id.duplicated().any() or query.record_id.duplicated().any():
        raise ValueError("Duplicate record identity")
    audit = {
        "inputs": receipt_inputs, "query_cells": int(query.cell_id.nunique()), "query_rows": int(len(query)),
        "query_compounds": int(query.compound_id.nunique()), "query_n_sources": sorted(query.n_train_sources.unique().tolist()),
        "common_all_three_compounds": int(supported.eq(3).sum()), "fixed_common_gene_symbols": int(fit_rna.shape[1]),
        "adult_age_eligible_models": int(len(cohort)), "source_new_models": int(cohort.source_new.sum()),
        "source_new_families": int(cohort.loc[cohort.source_new, "cell_group"].nunique()),
        "train_rows": int(len(train)), "train_cells": int(train.cell_id.nunique()), "train_cell_groups": int(train.cell_group.nunique()),
        "train_by_source": {
            source: {"rows": int(len(part)), "cells": int(part.cell_id.nunique()),
                     "compounds": int(part.compound_id.nunique()),
                     "adult_hcc_cells": int(part.loc[part.lineage_role.eq("adult_hcc"), "cell_id"].nunique())}
            for source, part in train.groupby("source_id", sort=True)
        },
        "train_rna_scale": "DepMap Public 25Q3 default log2(TPM+1)", "query_rna_scale": "GEO LIMORE log2(FPKM+1)",
        "endpoints": "Native fit AUCs (lower=more sensitive); LIMORE AA (higher=more sensitive). No endpoint-equivalence claim.",
        "fit_role": "External-source response only. No LIMORE response appears in fit, hyperparameter selection or scaling.",
        "test_role": "Source-new relative to these public screens; historically exposed LIMORE development resource.",
        "roster_rule": "All published GDSC1/2, CTRPv2 and PRISM20Q2 source registries audited before exact compound matching.",
        "query_selection": "One model per source-new family, first 50 unique families by lexical LIMORE cell ID.",
        "replicates": "One source-native dose window/compound. MTS010 first for PRISM then widest observed model coverage. Median collapse within protocol.",
    }
    with new_output(output) as path:
        for directory in ("train", "test_inputs", "test_labels", "audit"):
            (path / directory).mkdir()
        train.to_parquet(path / "train" / "rows.parquet", index=False)
        fit_rna.to_parquet(path / "train" / "rna.parquet")
        query.to_parquet(path / "test_inputs" / "rows.parquet", index=False)
        query_rna.to_parquet(path / "test_inputs" / "rna.parquet")
        labels.to_parquet(path / "test_labels" / "outcomes.parquet", index=False)
        cohort.to_csv(path / "audit" / "holdout_roster.tsv", sep="\t", index=False)
        chosen_protocols.to_csv(path / "audit" / "source_protocols.tsv", sep="\t", index=False)
        audit["artifacts"] = {
            str(file.relative_to(path)): {"sha256": file_sha256(file), "bytes": file.stat().st_size}
            for file in sorted(path.rglob("*")) if file.is_file()
        }
        write_json(path / "receipt.json", audit)
    return audit
