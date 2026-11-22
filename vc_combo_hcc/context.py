"""Register exact target claims, matrix-independent mono assays and DepMap context."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from .identity import CompoundRegistry, exact_token, unique_map
from .splits import record


GLYPH = re.compile(r"^([^\s]+)\s+\((\d+)\)$")
CLAIM_SEPARATOR = re.compile(r"[,;|/]+")
MOL_FIELDS = ["id", "smiles", "connectivity", "scaffold"]


def verified(directory, key):
    path = directory / key
    receipt = json.loads((directory / "receipt.json").read_text())
    if record(path) != receipt["files"][key]:
        raise ValueError(f"Input changed after freezing at {path}")
    return path


def gene_catalog(path):
    fields = ["symbol", "entrez_id", "prev_symbol", "alias_symbol"]
    names = pd.read_csv(path, usecols=fields, dtype=str).dropna(subset=["symbol", "entrez_id"])
    names["entrez_id"] = names.entrez_id.str.extract(r"^(\d+)(?:\.0+)?$", expand=False)
    names = names.dropna(subset=["entrez_id"])
    names = names.drop_duplicates(subset=["symbol", "entrez_id"])
    names = names[~names.symbol.duplicated(keep=False)]
    approved = names.set_index("symbol").entrez_id.to_dict()
    candidates = {}
    for source in names.itertuples(index=False):
        aliases = [p for field in (source.prev_symbol, source.alias_symbol)
                   if pd.notna(field) for p in field.split("|")]
        for old in aliases:
            candidates.setdefault(old.upper(), set()).add(source.symbol)
    aliases = {alias: next(iter(target)) for alias, target in candidates.items()
               if len(target) == 1 and alias not in approved}
    return approved, aliases


def exact_targets(mention, approved, aliases):
    if not isinstance(mention, str):
        return set()
    result = set()
    for token in CLAIM_SEPARATOR.split(mention.upper()):
        key = token.strip().strip('"')
        value = key if key in approved else aliases.get(key)
        if value:
            result.add(value)
    return result


def by_gene_columns(columns, approved):
    matches = {}
    for column in columns:
        hit = GLYPH.fullmatch(column)
        if hit and approved.get(hit[1]) == hit[2]:
            if hit[1] in matches:
                raise ValueError(f"Repeated exact target feature {hit[1]}")
            matches[hit[1]] = column
    return matches


def expression_panels(rna_path, effect_path, cells, approved):
    rna = pd.read_csv(rna_path, low_memory=False)
    defaults = rna.IsDefaultEntryForModel.astype(str).str.strip().str.lower().isin(("true", "yes", "1"))
    mask = defaults & rna.ModelID.isin(cells)
    genes = by_gene_columns(rna.columns, approved)
    expr = rna.loc[mask].set_index("ModelID", verify_integrity=True)[list(genes.values())]
    expr.columns = pd.Index(genes, name="symbol")
    expr = expr.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if len(expr) < 30 or expr.shape[1] < 10000:
        raise ValueError("Independent cell RNA coverage is unexpectedly low")
    del rna
    effect = pd.read_csv(effect_path, low_memory=False, index_col=0)
    effect = effect.loc[effect.index.intersection(cells)]
    selected = by_gene_columns(effect.columns, approved)
    if not effect.index.is_unique or len(selected) < 12000:
        raise ValueError("Mismatched DepMap Chronos gene-effect release")
    effect = effect[list(selected.values())].replace([np.inf, -np.inf], np.nan)
    effect.columns = pd.Index(selected, name="symbol")
    return expr.astype(np.float32), effect.astype(np.float32)


def known_drugs(a, molecules):
    # Only use high-specificity, exact sources. No mining of response data
    # or guesses based on a similarly named compound.
    snapshot = a.sanger_structures / "sanger_structures.tsv"
    if record(snapshot) != json.loads((a.sanger_structures / "receipt.json").read_text())["files"][snapshot.name]:
        raise ValueError("Sanger drug ids require the frozen public crosswalk")
    sanger = pd.read_csv(snapshot, sep="\t", dtype=str)
    drugcomb = pd.read_excel(a.drugcomb_metadata, engine="openpyxl", dtype=str, sheet_name="drug")
    registry = CompoundRegistry(a.drugcomb_metadata)
    drugcomb["canonical_id"] = registry.lookup(drugcomb.dname)
    targets_path = verified(a.bridge_targets, "targets.tsv")
    bridge = pd.read_csv(targets_path, sep="\t", dtype=str)
    sources = []
    for d in sorted(a.sanger_matrices):
        fields = [f"lib{i}_{feature}" for i in (1, 2) for feature in ("ID", "name", "target", "pathway")]
        annotations = pd.read_csv(d, usecols=fields, dtype=str).drop_duplicates()
        for arm in (1, 2):
            x = annotations[[f"lib{arm}_{col}" for col in ("ID", "target", "pathway")]].copy()
            x.columns = ["drug_id", "claim", "pathway"]
            sources.append(x.drop_duplicates())
    exact = pd.concat(sources).drop_duplicates().merge(sanger[["drug_id", "id"]], on="drug_id", validate="many_to_one")
    return drugcomb[drugcomb.canonical_id.isin(molecules)], exact, bridge, sanger


def pharmacology(a, molecules, approved, aliases):
    eligible_ids = set(molecules.id)
    dc, matrix, bridge, sanger = known_drugs(a, eligible_ids)
    records = []
    mentions = Counter()
    for origin, table, id_key, target_key in [
        ("DrugComb metadata", dc, "canonical_id", "target_name"),
        ("Sanger matrix metadata", matrix, "id", "claim"),
        ("ChEMBL37 and LIMORE exact bridge", bridge, "drug_id", "symbol"),
    ]:
        for drug, target in table[[id_key, target_key]].dropna().drop_duplicates().itertuples(index=False, name=None):
            if drug not in eligible_ids:
                continue
            mentioned = exact_targets(target, approved, aliases)
            for symbol in sorted(mentioned):
                records.append((drug, symbol, origin, target))
            if not mentioned:
                mentions[origin] += 1
    exact_targets_table = pd.DataFrame(records, columns=["id", "symbol", "source", "claim"]).drop_duplicates()
    meta = dc[["canonical_id", "dname", "clinical_phase", "drugbank_id"]].drop_duplicates()
    meta = meta.rename(columns={"canonical_id": "id"})
    pathways = matrix.loc[matrix.id.isin(molecules.id), ["id", "pathway"]].dropna().drop_duplicates()
    return exact_targets_table, meta, sanger, pathways, dict(mentions)


def monotherapies(raw, sanger, models, target_compounds):
    data = pd.read_csv(raw, usecols=["DRUG_ID", "SANGER_MODEL_ID", "COSMIC_ID",
                                    "CELL_LINE_NAME", "DRUG_NAME", "AUC"], dtype={"DRUG_ID": str, "COSMIC_ID": str})
    crosswalk = models.reset_index().loc[:, ["cell_id", "SangerModelID"]].dropna()
    sx = crosswalk.drop_duplicates()
    sx = sx[~sx.SangerModelID.duplicated(keep=False)]
    sm = data.merge(sx, left_on="SANGER_MODEL_ID", right_on="SangerModelID", validate="many_to_one")
    name_map = unique_map((exact_token(r.name), r.id) for r in sanger.itertuples())
    preferred = sm.DRUG_ID.map(sanger.set_index("drug_id").id)
    agreement = sm.DRUG_NAME.map(exact_token).map(name_map)
    sm["id"] = preferred.where(preferred.eq(agreement) | agreement.isna(), other=None)
    sm["id"] = sm.id.fillna(agreement.where(preferred.isna()))
    qualified = sm.dropna(subset=["id"])
    # Dataset and same drug id are verified by exact cross-source stable
    # structures. The GDSC2 AUC is a distinct single-agent experiment.
    valid = qualified.id.isin(target_compounds) & qualified.AUC.between(-.10, 1.10)
    mono = qualified.loc[valid, ["cell_id", "id", "AUC"]].groupby(["cell_id", "id"]).AUC.median()
    return (1 - mono).clip(0, 1).rename("inhibition").reset_index()


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    assay_receipt = a.assays / "receipt.json"
    molecules = pd.read_csv(verified(a.assays, "molecules.tsv"), sep="\t", dtype=str)
    if set(molecules.columns) != set(MOL_FIELDS) or not molecules.id.is_unique:
        raise ValueError("Invalid registered molecular contexts")
    cells = pd.read_csv(verified(a.assays, "cells.tsv"), sep="\t", dtype=str, index_col="cell_id")
    registry = CompoundRegistry(a.drugcomb_metadata)
    anchors = registry.lookup(pd.Series(["Sorafenib", "Lenvatinib"]))
    if anchors.isna().any() or anchors.nunique() != 2:
        raise ValueError("Could not verify the two screening anchor molecules")
    extra = registry.keys.loc[anchors[~anchors.isin(molecules.id)]].reset_index()
    molecules = pd.concat([molecules, extra[MOL_FIELDS]], ignore_index=True)
    approved, aliases = gene_catalog(a.genes)
    rna, effect = expression_panels(a.rna, a.chronos, cells.index, approved)
    targets, compounds, sanger, paths, misses = pharmacology(a, molecules, approved, aliases)
    if targets.id.nunique() < 20:
        raise ValueError("Insufficient independently verified pharmacologic targets")
    if "SangerModelID" not in cells:
        cells = cells.join(pd.read_csv(a.models, usecols=["ModelID", "SangerModelID"],
                                       dtype=str).set_index("ModelID"), how="left")
    mono = monotherapies(a.gdsc_monotherapy, sanger, cells, set(molecules.id))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as work:
        out = Path(work)
        molecules[["id", "smiles"]].to_csv(out / "molecules.tsv", sep="\t", index=False)
        rna.to_parquet(out / "rna.parquet")
        effect.to_parquet(out / "chronos.parquet")
        mono.to_parquet(out / "single_agent.parquet", index=False)
        compounds.to_csv(out / "drug_metadata.tsv", sep="\t", index=False)
        targets.to_csv(out / "target_registry.tsv", sep="\t", index=False)
        paths.to_csv(out / "curated_pathways.tsv", sep="\t", index=False)
        (out / "receipt.json").write_text(json.dumps({
            "assays": record(assay_receipt),
            "public_inputs": {key: record(p) for key, p in vars(a).items()
                              if key != "output" and isinstance(p, Path) and p.is_file()},
            "targets": {"claims": len(targets), "molecules": targets.id.nunique(),
                        "genes": targets.symbol.nunique(), "unresolved_claim_counts": misses},
            "cell_context": {"rna": len(rna), "chronos": len(effect), "assay_cells": len(cells)},
            "independent_gdsc2_monotherapies": len(mono),
            "files": {p.name: record(p) for p in sorted(out.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        out.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assays", "rna", "chronos", "genes", "models", "sanger-structures", "bridge-targets",
                 "drugcomb-metadata", "gdsc-monotherapy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sanger-matrices", nargs="+", type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
