"""Conservative public compound-to-target joins keyed by exact InChIKey."""
import argparse
import json
from pathlib import Path
import re
import tempfile

import pandas as pd

from .gene_features import unambiguous_symbol_map
from .perturbseq import record
from .predict import checked, write_json


# Resolve the LIMORE assay's published biochemical shorthand to conservative
# protein groups. A family stays a group for both the split and prediction.
FAMILIES = {
    "ABL": "ABL1 ABL2", "ACAT1": "SOAT1", "AURORA A": "AURKA", "AURORA B": "AURKB",
    "BET": "BRD2 BRD3 BRD4 BRDT", "BCL2": "BCL2 BCL2L1 BCL2L2", "C-KIT": "KIT",
    "CHK1": "CHEK1", "CK2": "CSNK2A1 CSNK2A2 CSNK2B", "DNA METHYLATION": "DNMT1 DNMT3A DNMT3B",
    "DNAPK": "PRKDC", "FAK": "PTK2 PTK2B", "FARNESYL TRANSFERASE": "FNTA FNTB",
    "FGFR": "FGFR1 FGFR2 FGFR3 FGFR4", "GLUTAMINASE": "GLS GLS2", "HER2": "ERBB2",
    "HIF": "EGLN1 EGLN2 EGLN3", "HSP90": "HSP90AA1 HSP90AB1 HSP90B1 TRAP1",
    "HDAC": "HDAC1 HDAC2 HDAC3 HDAC4 HDAC5 HDAC6 HDAC7 HDAC8 HDAC9 HDAC10 HDAC11",
    "IAP": "BIRC2 BIRC3 XIAP", "IGF-1R": "IGF1R", "JAK": "JAK1 JAK2 JAK3 TYK2",
    "KSP": "KIF11", "MEK": "MAP2K1 MAP2K2",
    "MICROTUBULE": "TUBA1A TUBA1B TUBA1C TUBA3C TUBA3D TUBA3E TUBA4A TUBA8 TUBB TUBB1 TUBB2A TUBB2B TUBB3 TUBB4A TUBB4B TUBB6 TUBB8",
    "MTORC1": "MTOR RPTOR", "PARP": "PARP1 PARP2", "PDGFR": "PDGFRA PDGFRB",
    "PDK1": "PDPK1", "PI3K": "PIK3CA PIK3CB PIK3CD PIK3CG", "PIM": "PIM1 PIM2 PIM3",
    "PKC": "PRKCA PRKCB PRKCD PRKCE PRKCG PRKCH PRKCI PRKCQ PRKCZ",
    "PROTEASOME": "PSMB1 PSMB2 PSMB3 PSMB4 PSMB5 PSMB6 PSMB7 PSMB8 PSMB9 PSMB10",
    "RAF": "ARAF BRAF RAF1", "SURVIVIN": "BIRC5", "TOPOISOMERASE I": "TOP1",
    "TOPOISOMERASE II": "TOP2A TOP2B", "VEGFR": "FLT1 KDR FLT4", "VEFGR": "FLT1 KDR FLT4",
}
NON_PROTEIN_DISCLAIMS = {"MULTI-TARGETED", "DNA CROSSLINKER", "DNA ANTIMETABOLITE",
                        "FERROPTOSIS ACTIVATOR", "ROS", "RNA HELICASE"}


def expand_token(label, symbol_map):
    if label in FAMILIES:
        return FAMILIES[label].split()
    match = re.fullmatch(r"([A-Z]+)(\d+(?:/\d+)+)", label)
    if match:
        return [match[1] + suffix for suffix in match[2].split("/")]
    # Historical symbols are only accepted via a separate verified alias
    # crosswalk. Avoid converting arbitrary pathway names to exact targets.
    return [label] if label in symbol_map else []


def exact_targets(annotation, snapshot, hgnc):
    registered = pd.read_csv(annotation, sep="\t", dtype=str)
    registered = registered.dropna(subset=["compound_id"])
    if not registered.compound_id.is_unique:
        raise ValueError("One registered InChIKey per structurally supported assay compound is required")
    receipt = json.loads((snapshot / "receipt.json").read_text())
    if record(annotation) != receipt["registry"]:
        raise ValueError("Target claims were downloaded for a different chemical registry")
    entries = {key: json.loads(checked(snapshot, f"{key}.json", receipt).read_text())
               for key in ("molecules", "mechanisms", "targets")}
    symbol_map = unambiguous_symbol_map(hgnc)
    accessions = pd.read_csv(hgnc, usecols=["ensembl_gene_id", "uniprot_ids"], dtype=str).dropna()
    accessions = accessions[accessions.ensembl_gene_id.isin(symbol_map.index)].copy()
    accessions["accession"] = accessions.uniprot_ids.str.split("|", regex=False)
    accessions = accessions.explode("accession")[["accession", "ensembl_gene_id"]].drop_duplicates()
    accessions["accession"] = accessions.accession.str.strip()
    accessions = accessions[~accessions.accession.duplicated(keep=False)].set_index("accession").ensembl_gene_id
    genes = pd.Series(symbol_map.index, index=symbol_map.str.upper()).to_dict()
    reference = registered.set_index("compound_id")
    parent_to_compounds = {}
    for molecule in entries["molecules"]:
        key = molecule["molecule_structures"]["standard_inchi_key"]
        if key not in reference.index:
            raise ValueError("ChEMBL returned a non-exact chemical match")
        hierarchy = molecule.get("molecule_hierarchy") or {}
        parent = hierarchy.get("parent_chembl_id") or molecule["molecule_chembl_id"]
        parent_to_compounds.setdefault(parent, set()).add(key)
    targets = {row["target_chembl_id"]: row for row in entries["targets"]}
    associations = []
    unmapped = []
    for key, description in reference.drug_target_source_labels.fillna("").items():
        for field in description.split(","):
            phrase = field.strip().upper()
            if not phrase or phrase in NON_PROTEIN_DISCLAIMS:
                continue
            symbols = expand_token(phrase, genes)
            if not symbols or any(sym not in genes for sym in symbols):
                unmapped.append({"compound": key, "phrase": phrase, "source": "LIMORE"})
            for symbol in symbols:
                if symbol in genes:
                    associations.append((key, genes[symbol], "LIMORE", phrase, "assay_annotation"))
    for claim in entries["mechanisms"]:
        target_id = claim.get("target_chembl_id")
        target = targets.get(target_id)
        parents = parent_to_compounds.get(claim["parent_molecule_chembl_id"], set())
        if not target or claim.get("direct_interaction") != 1 or target["organism"] != "Homo sapiens":
            continue
        for subunit in target["target_components"]:
            accession = subunit.get("accession")
            if accession not in accessions:
                for key in parents:
                    unmapped.append({"compound": key, "phrase": f"{target_id} / {accession}", "source": "ChEMBL"})
                continue
            for key in parents:
                associations.append((key, accessions[accession], receipt["release"], target_id,
                                     claim.get("action_type") or "unspecified"))
    result = pd.DataFrame(associations, columns=["drug_id", "gene", "source", "claim", "action"]).drop_duplicates()
    result = result.sort_values(list(result.columns)).reset_index(drop=True)
    return result, unmapped, parent_to_compounds, symbol_map


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    targets, unmapped, parent, symbols = exact_targets(a.registry, a.snapshot, a.hgnc)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        staged = Path(directory)
        targets["symbol"] = targets.gene.map(symbols)
        targets.to_csv(staged / "targets.tsv", sep="\t", index=False)
        write_json(staged / "unmapped_claims.json", unmapped)
        write_json(staged / "receipt.json", {
            "inputs": {"registry": record(a.registry), "snapshot": record(a.snapshot / "receipt.json"),
                       "hgnc": record(a.hgnc)},
            "unique_targets": targets.gene.nunique(), "mapped_compounds": targets.drug_id.nunique(),
            "mapped_chembl_parents": len(parent), "unmapped_claims": len(unmapped),
            "holdout_semantics": "Unseen registered protein targets. Undocumented secondary targets cannot be excluded.",
            "files": {f.name: record(f) for f in sorted(staged.iterdir())},
        })
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("registry", "snapshot", "hgnc", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
