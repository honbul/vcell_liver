"""Register H2 feature inputs and a response-blind GDSC1 query roster."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy import sparse
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

from vc_build.cross_source_data import MODEL_COLS, read_raw, roster_maps, roster_table
from vc_build.function_data import alias_token, canonical_chemistry
from vc_build.improve_gdsc_registry import DRUG_INFO_SHA256, matched_structures
from vc_build.io import publish_json, sha256
from vc_build.licob_cold import checked
from vc_build.v0plus_cv import molecular_inputs

STORAGE = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong")
PUBLIC = Path("/mnt/S1/sdata/pubdata")
VC = STORAGE / "vc_build_v1"
COLD = VC / "v0plus-runs/licob-cold-5bf53fc/inputs"
LIMORE = VC / "function/cross-v1"
PRIOR = VC / "v0plus-public-priors"
HALLMARK = PUBLIC / "Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt"
MODEL = PUBLIC / "DepMap/25Q3/Model.csv"
TARGET = VC / "external-code-v3/drug_targets.json"
AUXILIARY = VC / "v0plus-runs/depmap-25q3-bb8a47b"
PAINTING = STORAGE / "liver_dili_v1/assays-v1/painting.parquet"
LINCS = PUBLIC / "LINCS/LINCS2020"
GCTX = LINCS / "level5_beta_trt_cp_n720216x12328.gctx"
GDSC = PUBLIC / "DepMap/DrugScreens/GDSC_rel8.5/GDSC1_fitted_dose_response_27Oct23.csv"
BRIDGE = VC / "v0plus-runs/gdsc1-bridge-1ba7f80/inputs"
SUPPLEMENT = VC / "licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
KEY = "compound_id"
SEED = 18407

# Each alternative is a predetermined per-compound rule. Learned normalization follows in CV.
RECIPES = {
    "mean24": ((24,), False, 1, False),
    "modz24": ((24,), True, 1, False),
    "modz624": ((6, 24), True, 1, False),
    "strict1": ((6, 24), True, 1, True),
    "strict2": ((6, 24), True, 2, True),
}


def parquet_lock(path, digest, **kwargs):
    if sha256(path) != digest:
        raise ValueError(f"Changed source {path}")
    return pd.read_parquet(path, **kwargs)


def canonical_registry(rows):
    unique = rows[[KEY, "isomeric_smiles"]].drop_duplicates()
    result = unique.isomeric_smiles.map(canonical_chemistry)
    canonical = pd.DataFrame(result.tolist(), columns=["smiles", KEY, "connectivity", "scaffold"], index=unique.index)
    if not canonical[KEY].eq(unique[KEY]).all():
        raise ValueError("Registered exact parent differs from canonical identity")
    aliases = canonical.groupby(KEY).scaffold.agg(lambda v: "|".join(sorted(set(v))))
    return (canonical.sort_values([KEY, "smiles"]).drop_duplicates(KEY).set_index(KEY)
            .join(aliases.rename("scaffold_aliases")))


def linked_groups(registry):
    """Transitive closure of shared exact connectivity and Murcko scaffold."""
    labels = {key: key for key in registry.index}

    def root(key):
        while labels[key] != key:
            key = labels[key]
        return key

    aliases = registry.scaffold_aliases.str.split("|").explode().rename("scaffold")
    for values in (registry.connectivity, aliases):
        for _, group in values.groupby(values):
            names = group.index
            for key in names[1:]:
                a, b = sorted((root(names[0]), root(key)))
                labels[b] = a
    return pd.Series({key: root(key) for key in labels}, name="scaffold_block")


def fold_ids(blocks):
    groups = blocks.sort_index()
    out = pd.Series(-1, index=groups.index, dtype="int8")
    for fold, (_, test) in enumerate(GroupKFold(3).split(groups, groups=groups)):
        out.iloc[test] = fold
    return out


def folds(registry):
    result = registry.copy()
    result["scaffold_block"] = linked_groups(result)
    result["outer"] = fold_ids(result.scaffold_block)
    for outer in range(3):
        eligible = result[result.outer.ne(outer)]
        result[f"inner{outer}"] = fold_ids(eligible.scaffold_block).reindex(result.index, fill_value=-1)
    for grouping in ("connectivity", "scaffold", "scaffold_block"):
        if result.groupby(grouping).outer.nunique().max() != 1:
            raise AssertionError(f"Split crosses {grouping}")
    return result


def gdsc_roster(registry, cells):
    evidence = json.loads((BRIDGE / "receipt.json").read_text())
    official_sources = evidence["official_data"]
    for name, spec in official_sources.items():
        if sha256(Path(spec["path"])) != spec["sha256"]:
            raise ValueError(f"Changed GDSC source {name}")
    label_source = official_sources["gdsc_labels"]["path"]
    raw = read_raw(GDSC, ["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME"])
    official = read_raw(label_source, ["DRUG_ID", "DRUG_NAME", "SYNONYMS", "TARGET"])
    names = (official.assign(aliases=official.DRUG_NAME + "," + official.SYNONYMS)
             .groupby("DRUG_ID").aliases.agg(",".join).str.split(",").map(lambda v: set(map(alias_token, v))))
    observed = raw[["DRUG_ID", "DRUG_NAME"]].drop_duplicates().groupby("DRUG_ID").DRUG_NAME.agg(set)
    native = {key for key, values in observed.items() if set(map(alias_token, values)) <= names.get(key, set())}
    path = VC / "licob-v0plus/improve-pdo-22920452/drug_info.tsv"
    if sha256(path) != DRUG_INFO_SHA256:
        raise ValueError("Changed IMPROVE catalogue")
    published = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    mapped, audit = matched_structures(published, official, native)
    exact = mapped.loc[~mapped.isomeric_smiles.map(
        lambda text: any(site.specified != Chem.StereoSpecified.Specified
                         for site in Chem.FindPotentialStereo(Chem.MolFromSmiles(text))))].copy()
    prior = parquet_lock(BRIDGE / "observations.parquet", evidence["observation_sha256"],
                         columns=[KEY, "native_compound_id"])
    connectivity = registry.connectivity
    scaffold = set(registry.scaffold_aliases.str.split("|").explode())
    fresh = exact[~exact[KEY].isin(registry.index) & ~exact[KEY].isin(prior[KEY])
                  & ~exact.native_compound_id.isin(prior.native_compound_id)
                  & ~exact.connectivity_group.isin(connectivity)
                  & ~exact.scaffold_group.isin(scaffold)].copy()
    models = read_raw(MODEL, MODEL_COLS)
    lookup = roster_maps(roster_table(raw, "GDSC1", "CELL_LINE_NAME", "SANGER_MODEL_ID", "COSMIC_ID"), models)
    stable = models.set_index("ModelID").SangerModelID
    lookup = lookup[lookup.cell_id.isin(cells.index) & lookup.source_key.ne("")
                    & lookup.cell_id.map(stable).eq(lookup.source_key)]
    sel = raw.merge(lookup.rename(columns={"source_key": "SANGER_MODEL_ID", "source_key_2": "COSMIC_ID",
                    "name": "CELL_LINE_NAME"})[["SANGER_MODEL_ID", "COSMIC_ID", "CELL_LINE_NAME", "cell_id"]],
                    validate="many_to_one").drop_duplicates(["cell_id", "DRUG_ID"])
    selected = sel.merge(fresh, left_on="DRUG_ID", right_on="native_compound_id", validate="many_to_one")
    targets = official.groupby("DRUG_ID").TARGET.agg(lambda v: ";".join(sorted(set(v) - {""})))
    selected["drug_target_source_labels"] = selected.DRUG_ID.map(targets)
    selected["source_compound_label"] = selected.DRUG_NAME
    selected = selected.join(cells[["lineage_group"]], on="cell_id", validate="many_to_one")
    selected = selected.assign(source_id="GDSC1", endpoint_id="AUC",
                               row_id="GDSC1|" + selected.cell_id + "|" + selected[KEY])
    # Multiple native aliases can point to a single approved parent. Query once and retain aliases for label aggregation.
    meta = {"catalogue_match_audit": audit, "native_drugs": int(raw.DRUG_ID.nunique()),
            "stereo_resolved_native": len(exact), "prior_bridge_parents": int(prior[KEY].nunique()),
            "query_parents": int(selected[KEY].nunique()), "query_native": int(selected.DRUG_ID.nunique()),
            "query_scaffolds": int(selected.scaffold_group.nunique()), "query_cells": int(selected.cell_id.nunique()),
            "query_pairs": int(selected.row_id.nunique()), "raw_sha256": official_sources["gdsc1"]["sha256"],
            "query_outcome_columns_loaded": []}
    return selected, meta


def lincs_metadata(registry, model, limore):
    info = pd.read_csv(LINCS / "compoundinfo_beta.txt", sep="\t").fillna("")
    # Consider deposited exact keys plus normalized SMILES. Conflicts and ambiguous pert IDs are removed.
    keep = info[info.inchi_key.str[:14].isin(set(registry.connectivity))
                | info.inchi_key.eq("")].copy()
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        def exact(smiles):
            try:
                return canonical_chemistry(smiles)[1]
            except (ValueError, RuntimeError):
                return ""
        key = {smiles: exact(smiles) for smiles in keep.canonical_smiles.unique()}
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")
    derived = keep.canonical_smiles.map(key)
    # Identify canonical exact parents from provided structure and reject a discordant connectivity annotation.
    good = keep.inchi_key.eq("") | keep.inchi_key.str[:14].eq(derived.str[:14])
    keep[KEY] = derived.where(good, "")
    mapped = keep.loc[keep[KEY].ne(""), ["pert_id", KEY, "moa", "target"]].drop_duplicates()
    conflicts = mapped.groupby("pert_id")[KEY].nunique().gt(1)
    mapped = mapped[~mapped.pert_id.isin(conflicts[conflicts].index)]
    mapped = mapped[mapped[KEY].isin(registry.index)].drop_duplicates("pert_id")
    cell = pd.read_csv(LINCS / "cellinfo_beta.txt", sep="\t").fillna("")
    protected = set(limore.cellosaurus_id.dropna()) - {""}
    protected_name = set(limore.source_cell_label.map(alias_token))
    declared = cell.cell_lineage.ne("liver") & ~cell.cell_lineage.isin(["", "unknown"])
    safe = cell[declared & ~cell.cellosaurus_id.isin(protected)
                & ~cell.cell_iname.map(alias_token).isin(protected_name)
                & ~cell.cell_alias.map(alias_token).isin(protected_name)]
    model_ids = set(model.RRID) - {""}
    shared_cells = sorted(safe.loc[safe.cellosaurus_id.isin(model_ids), "cell_iname"])
    cols = ["sig_id", "pert_id", "pert_type", "cell_iname", "pert_time", "pert_time_unit",
            "pert_dose", "pert_dose_unit", "nsample", "cc_q75", "qc_pass", "det_plates",
            "det_wells", "distil_ids", "bead_batch", "project_code", "is_hiq"]
    sig = pd.read_csv(LINCS / "siginfo_beta.txt", sep="\t", usecols=cols, low_memory=False)
    sig = sig[sig.pert_type.eq("trt_cp")].merge(mapped, on="pert_id", validate="many_to_one")
    pub = sig.copy()
    sig = sig[sig.cell_iname.isin(safe.cell_iname)].copy()
    non_liver = sig.copy()
    sig["time_h"] = pd.to_numeric(sig.pert_time, errors="coerce")
    sig["dose_um"] = pd.to_numeric(sig.pert_dose, errors="coerce")
    sig["replicates"] = pd.to_numeric(sig.nsample, errors="coerce").fillna(0)
    sig["cc"] = pd.to_numeric(sig.cc_q75, errors="coerce")
    units = sig.pert_dose_unit.str.lower().str.replace("µ", "u").str.replace("μ", "u")
    good = (sig.pert_time_unit.eq("h") & sig.time_h.isin([6, 24]) & units.eq("um")
            & sig.dose_um.between(.5, 20) & sig.replicates.ge(1))
    candidate = sig[good].copy()
    if not candidate.sig_id.is_unique:
        raise ValueError("Signature IDs are ambiguous")
    source = {"catalogue_parents": int(mapped[KEY].nunique()), "all_l5_cp_parents": int(pub[KEY].nunique()),
              "non_liver_l5_cp_parents": int(non_liver[KEY].nunique()), "selected_window_profiles": len(candidate),
              "eligible_cell_lines": sorted(safe.cell_iname), "observed_cell_lines": sorted(candidate.cell_iname.unique()),
              "observed_depmap_rrid_overlaps": sorted(set(shared_cells) & set(candidate.cell_iname)),
              "excluded_liver_lines": sorted(cell[cell.cell_lineage.eq("liver")].cell_iname),
              "unknown_lineage_cells_excluded": int(cell.cell_lineage.isin(["", "unknown"]).sum()),
              "published_signature_replicate_histogram": candidate.replicates.value_counts().sort_index().to_dict(),
              "project_counts": candidate.project_code.value_counts().to_dict(),
              "bead_batch_counts": candidate.bead_batch.value_counts().to_dict(),
              "dose_counts": candidate.dose_um.value_counts().sort_index().to_dict(),
              "time_counts": candidate.time_h.value_counts().to_dict(),
              "unique_plates": len(set("|".join(candidate.det_plates.dropna()).split("|")) - {""})}
    upstream = pd.DataFrame({"mapped": registry.index.isin(mapped[KEY]),
                             "any_l5": registry.index.isin(pub[KEY]),
                             "non_liver_l5": registry.index.isin(non_liver[KEY])}, index=registry.index)
    return candidate, source, upstream


def pathway_weights(genes, edges):
    subset = edges[edges.gene.isin(genes) & edges.signature.str.startswith(("PROGENY_", "HALLMARK_"))]
    mat = subset.pivot_table(index="gene", columns="signature", values="weight", aggfunc="sum").reindex(genes).fillna(0)
    support = mat.ne(0).sum().rename("landmarks")
    # Fourteen PROGENy footprints and thirty Hallmarks with the highest landmark coverage.
    hallmark = support[support.index.str.startswith("HALLMARK_")].sort_index().sort_values(ascending=False, kind="stable").head(30)
    selected = sorted(hallmark.index.union(support[support.index.str.startswith("PROGENY_")].index))
    if len(selected) != 44 or support[selected].min() < 5:
        raise ValueError("Expected 44 supported fixed pathway footprints")
    mat = mat[selected]
    return mat / np.sqrt(mat.pow(2).sum(axis=0))


def modz_weights(values):
    if len(values) < 2:
        return np.ones(len(values))
    ranks = rankdata(values, axis=1)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    agree = (unit @ unit.T).clip(min=.01)
    np.fill_diagonal(agree, 0)
    weights = agree.sum(axis=1)
    return weights / weights.sum()


def lincs_profiles(sig, edges):
    import h5py

    genes = pd.read_csv(LINCS / "geneinfo_beta.txt", sep="\t", dtype={"gene_id": str})
    genes = genes.set_index("gene_id", verify_integrity=True)
    landmarks = genes[genes.feature_space.eq("landmark")]
    bing = genes[genes.feature_space.isin(["landmark", "best inferred"])].drop_duplicates("gene_symbol")
    weights = pathway_weights(bing.gene_symbol.to_list(), edges)
    weights = weights.loc[weights.ne(0).any(axis=1)]
    positions = bing.reset_index().set_index("gene_symbol").loc[weights.index, "gene_id"]
    sparse_weights = sparse.csr_matrix(weights.to_numpy(dtype=np.float32))
    if len(landmarks) != 978 or not landmarks.gene_symbol.is_unique:
        raise ValueError("Unexpected L1000 landmark axis")
    with h5py.File(GCTX, "r") as file:
        row_ids = pd.Index(file["0/META/ROW/id"][:].astype(str))
        gp, pp = row_ids.get_indexer(landmarks.index), row_ids.get_indexer(positions)
        rp = pd.Index(file["0/META/COL/id"][:].astype(str)).get_indexer(sig.sig_id)
        if min(gp) < 0 or min(pp) < 0 or min(rp) < 0:
            raise ValueError("Missing registered signature/landmark")
        order = np.argsort(rp)
        sig = sig.iloc[order].reset_index(drop=True)
        matrix = file["0/DATA/0/matrix"]
        values = np.empty((len(rp), len(gp) + len(weights.columns)), np.float32)
        for pos in range(0, len(rp), 1000):
            indices = rp[order[pos:pos + 1000]]
            block = matrix[indices, :]
            values[pos:pos + len(indices), :len(gp)] = block[:, gp]
            values[pos:pos + len(indices), len(gp):] = block[:, pp] @ sparse_weights
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite landmark values")
    for name, (times, weighted, cell_min, strict) in RECIPES.items():
        active = sig[sig.time_h.isin(times)]
        if strict:
            active = active[active.cc.ge(.2) & active.replicates.ge(2) & active.qc_pass.eq(1)]
        support = active.groupby(KEY).cell_iname.nunique()
        active = active[active[KEY].isin(support[support.ge(cell_min)].index)]
        # First collapse between-plate signatures at fixed contexts, then doses and times;
        # cell lines have equal weight so heavily profiled lines cannot dominate the consensus.
        contexts = {}
        for key, part in active.groupby([KEY, "cell_iname", "time_h", "dose_um"], sort=True):
            block = values[part.index]
            contexts[key] = modz_weights(block[:, :len(gp)]) @ block if weighted else block.mean(axis=0)
        indexed = pd.MultiIndex.from_tuples(contexts, names=[KEY, "cell", "time", "dose"])
        context = pd.DataFrame(list(contexts.values()), index=indexed, columns=landmarks.gene_symbol.to_list() + list(weights.columns))
        time = context.groupby(level=[0, 1, 2]).mean()
        cell = time.groupby(level=[0, 1]).mean()
        profile = cell.groupby(level=0).mean()
        yield name, profile.iloc[:, :len(gp)], profile.iloc[:, len(gp):], weights, {"parents": len(profile), "signatures": len(active), "contexts": len(context),
                             "cells": int(active.cell_iname.nunique()), "time_h": list(times),
                             "minimum_cell_lines": cell_min, "strict_qc": strict}


def register(out):
    checked(COLD, ("train", "query", "rna"))
    out.mkdir(parents=True, exist_ok=False)
    columns = [KEY, "isomeric_smiles", "lineage_group", "cell_id", "source_id", "endpoint_id", "core_rows"]
    train = pd.read_parquet(COLD / "train.parquet", columns=columns)
    registry = folds(canonical_registry(train))
    if len(train) != 670830 or len(registry) != 1186:
        raise ValueError("Unexpected 2D response roster")
    cells = train[["lineage_group", "cell_id"]].drop_duplicates().set_index("cell_id", verify_integrity=True)
    model = pd.read_csv(MODEL, dtype=str).fillna("").set_index("ModelID").reindex(cells.index).fillna("")
    cells["tissue"] = model.OncotreeLineage.mask(model.OncotreeLineage.eq(""), "Unresolved")
    limore_receipt = json.loads((LIMORE / "receipt.json").read_text())
    limore = parquet_lock(LIMORE / "test_inputs/rows.parquet",
                         limore_receipt["artifacts"]["test_inputs/rows.parquet"]["sha256"])
    licob = pd.read_parquet(COLD / "query.parquet")
    gdsc, gdsc_meta = gdsc_roster(registry, cells)
    all_drugs = canonical_registry(pd.concat([train, limore, licob, gdsc], ignore_index=True))
    painting = pd.read_parquet(PAINTING)
    if not painting.index.is_unique:
        raise ValueError("Duplicate cached Cell Painting parent")
    paint = painting.loc[painting.index.intersection(all_drugs.index)]
    paint = paint[paint.notna().mean(axis=1).ge(.75)]
    paint.to_parquet(out / "painting.parquet")
    cells.to_parquet(out / "cells.parquet")
    gdsc.to_parquet(out / "gdsc_query.parquet", index=False)
    sig, source, coverage = lincs_metadata(all_drugs, model, limore)
    sig.to_parquet(out / "lincs_selected_signatures.parquet", index=False)
    coverage["painting"] = coverage.index.isin(paint.index)
    edges, _, prior = molecular_inputs(PRIOR, HALLMARK, TARGET)
    recipes = {}
    for name, profile, path_profile, weights, info in lincs_profiles(sig, edges):
        profile.to_parquet(out / f"lincs_{name}.parquet")
        path_profile.to_parquet(out / f"paths_{name}.parquet")
        coverage[f"lincs_{name}"] = coverage.index.isin(profile.index)
        recipes[name] = info
    weights.to_parquet(out / "pathway_weights.parquet")
    coverage.to_parquet(out / "coverage.parquet")
    registry.to_parquet(out / "folds.parquet")
    cohort = [("DRP", registry.index), ("LIMORE", pd.Index(limore[KEY].unique())),
              ("LICOB", pd.Index(licob[KEY].unique())), ("LICOB18", pd.Index(licob.loc[
                  ~licob.in_source & licob.remaining_unspecified_stereo.eq(0), KEY].unique())),
              ("GDSC1", pd.Index(gdsc[KEY].unique()))]
    groups = cohort + [(f"outer{f}", registry.index[registry.outer.eq(f)]) for f in range(3)]
    groups += [(f"outer{f}/inner{i}", registry.index[registry[f"inner{f}"].eq(i)]) for f in range(3) for i in range(3)]
    counts = []
    for label, names in groups:
        roster = coverage.loc[names]
        for modality in ["total", *roster.columns, "painting&lincs_modz624"]:
            if modality == "total":
                supported = names
            elif modality == "painting&lincs_modz624":
                supported = roster.index[roster.painting & roster.lincs_modz624]
            else:
                supported = roster.index[roster[modality]]
            counts.append((label, modality, len(supported), all_drugs.loc[supported, "scaffold"].nunique()))
    pd.DataFrame(counts, columns=["cohort", "view", "parents", "scaffolds"]).to_csv(
        out / "support.tsv", sep="\t", index=False)
    publish_json(out / "data_receipt.json", {
        "source_registration_sha256": sha256(COLD / "receipt.json"), "limore_registration_sha256": sha256(LIMORE / "receipt.json"),
        "source_rows": len(train), "source_parents": len(registry), "source_by_screen": train.source_id.value_counts().to_dict(),
        "source_cells": len(cells), "source_scaffold_components": int(registry.scaffold_block.nunique()),
        "painting_input": str(PAINTING), "painting_shape": list(painting.shape), "painting_sha256": sha256(PAINTING),
        "painting_profile_source": json.loads((PAINTING.parent / "receipt.json").read_text()).get("phenotype", {}),
        "lincs": source, "recipes": recipes, "gdsc": gdsc_meta, "pathways": list(weights.columns), "prior": prior,
        "lincs_metadata_sha256": {f.name: sha256(f) for f in sorted(LINCS.glob("*_beta.txt"))},
        "gctx_path": str(GCTX), "gctx_bytes": GCTX.stat().st_size,
        "files": {f.name: sha256(f) for f in sorted(out.iterdir())},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    register(parser.parse_args().out)
