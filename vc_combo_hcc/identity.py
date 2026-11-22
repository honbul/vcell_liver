"""Resolve compound parents and biological lineages using public registered IDs."""
import re

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


def exact_token(name):
    return re.sub(r"[^A-Z0-9]", "", str(name).strip().upper())


def unique_map(pairs):
    values = {}
    for key, value in pairs:
        if key and pd.notna(value):
            values.setdefault(key, set()).add(value)
    return {k: next(iter(v)) for k, v in values.items() if len(v) == 1}


def numeric_id(value):
    if pd.isna(value):
        return ""
    match = re.fullmatch(r"(\d+)(?:\.0+)?", str(value))
    return match[1] if match else ""


class CellRegistry:
    def __init__(self, depmap, drugcomb_cell):
        fields = [
            "ModelID", "PatientID", "CellLineName", "StrippedCellLineName", "OncotreeLineage",
            "OncotreeCode", "ModelType", "AgeCategory", "RRID",
        ]
        models = pd.read_csv(depmap, dtype=str)
        other = {exact_token(col): col for col in models.columns}
        cosmic = other.get("COSMICID")
        sanger = other.get("SANGERMODELID")
        models = models.loc[models.ModelType.eq("Cell Line"), fields + [col for col in (cosmic, sanger) if col]]
        models = models.dropna(subset=["ModelID", "PatientID", "OncotreeLineage"]).copy()
        if not models.ModelID.is_unique:
            raise ValueError("Ambiguous DepMap identities")
        pubs = pd.read_excel(drugcomb_cell, sheet_name="cell_line", dtype=str, engine="openpyxl")
        pubs = pubs[pubs.depmap_id.isin(models.ModelID)].copy()
        names = []
        for row in models.itertuples(index=False):
            names.extend((exact_token(k), row.ModelID) for k in (row.CellLineName, row.StrippedCellLineName))
        for row in pubs.itertuples(index=False):
            names.append((exact_token(row.name), row.depmap_id))
        self.names = unique_map(names)
        self.cosmic = unique_map(
            [(numeric_id(c), i) for c, i in zip(pubs.cosmic_id, pubs.depmap_id)]
            + ([(numeric_id(c), i) for c, i in zip(models[cosmic], models.ModelID)] if cosmic else [])
        )
        self.sanger = unique_map(
            [(c, i) for c, i in zip(pubs.cell_model_passport_id, pubs.depmap_id) if pd.notna(c)]
            + ([(c, i) for c, i in zip(models[sanger], models.ModelID) if pd.notna(c)] if sanger else [])
        )
        known_liver = set(models[models.OncotreeLineage.eq("Liver")].PatientID)
        models["liver"] = models.PatientID.isin(known_liver)
        models["adult_hcc"] = (models.OncotreeLineage.eq("Liver") & models.OncotreeCode.eq("HCC")
                               & models.AgeCategory.eq("Adult") & models.ModelID.ne("ACH-000361"))
        self.keys = models.rename(columns={"ModelID": "cell_id", "PatientID": "cell_group"}).set_index("cell_id")

    def lookup(self, name, sanger=None, cosmic=None):
        if sanger is None:
            return name.map(lambda n: self.names.get(exact_token(n)))
        merged = pd.DataFrame({
            "sanger": sanger.map(self.sanger),
            "cosmic": cosmic.map(lambda v: self.cosmic.get(numeric_id(v))),
            "exact_name": name.map(lambda n: self.names.get(exact_token(n))),
        })
        # Stable assay ID is authoritative. A second stable source ID may
        # corroborate it. Conflicting source IDs receive no model features.
        ambiguous = (merged.sanger.notna() & merged.cosmic.notna()
                     & merged.sanger.ne(merged.cosmic))
        choose = merged.sanger.fillna(merged.cosmic).fillna(merged.exact_name)
        return choose.mask(ambiguous)


def structural_identity(smiles, published_key):
    if pd.isna(smiles) or pd.isna(published_key):
        return None
    allowed = {5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None or Chem.MolToInchiKey(mol) != published_key:
                return None
            parent = rdMolStandardize.FragmentParent(mol)
            parent = rdMolStandardize.Uncharger().uncharge(parent)
            Chem.SanitizeMol(parent)
        except (ValueError, RuntimeError):
            return None
    if not parent.GetNumAtoms() or {atom.GetAtomicNum() for atom in parent.GetAtoms()} - allowed:
        return None
    with rdBase.BlockLogs():
        # Resolve equivalent author files through the same standard InChI
        # before computing a topology. Tautomeric inputs can share a key
        # while RDKit sees different ring topology in the original SMILES.
        formula = Chem.MolToInchi(parent)
        exact = Chem.InchiToInchiKey(formula) if formula else ""
        canonical_parent = Chem.MolFromInchi(formula) if formula else None
        if not exact or canonical_parent is None or canonical_parent.GetNumHeavyAtoms() > 256:
            return None
        if Chem.MolToInchiKey(canonical_parent) != exact:
            return None
    parent = canonical_parent
    connectivity = exact.split("-")[0]
    scaffold = MurckoScaffold.GetScaffoldForMol(parent)
    group = ("acyclic:" + connectivity if scaffold.GetNumAtoms() == 0
             else Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold), isomericSmiles=False))
    return exact, connectivity, group, Chem.MolToSmiles(parent, isomericSmiles=True)


class CompoundRegistry:
    def __init__(self, path):
        raw = pd.read_excel(path, sheet_name="drug", engine="openpyxl", dtype=str)
        # Stable exact names take precedence. Synonyms apply only when
        # independently associated with one standardized parent identity.
        primary, aliases, identities = [], [], {}
        self.rejected = 0
        for row in raw.itertuples(index=False):
            block = structural_identity(row.smiles, row.inchikey)
            if block is None:
                self.rejected += 1
                continue
            identity, parent, scaffold, smiles = block
            if identity in identities and identities[identity][:2] != (parent, scaffold):
                raise ValueError(f"Inconsistent recorded structure for {identity}")
            identities[identity] = (parent, scaffold, smiles)
            primary.append((exact_token(row.dname), identity))
            for name in str(row.synonyms).split(";") if pd.notna(row.synonyms) else []:
                aliases.append((exact_token(name), identity))
        self.primary = unique_map(primary)
        self.aliases = unique_map(primary + aliases)
        self.known_primary = {key for key, _ in primary}
        self.keys = pd.DataFrame.from_dict(identities, orient="index", columns=["connectivity", "scaffold", "smiles"])
        self.keys.index.name = "id"

    def lookup(self, drug):
        aliases = drug.map(exact_token)
        primary = aliases.map(self.primary)
        fallback = aliases.mask(aliases.isin(self.known_primary)).map(self.aliases)
        return primary.fillna(fallback)

    def add_registered_sanger(self, annotated):
        self.sanger = annotated.set_index("drug_id", verify_integrity=True)
        unique = annotated.set_index("id")[["connectivity", "scaffold", "smiles"]]
        unique = unique[~unique.index.duplicated(keep="first")]
        overlap = sorted(set(unique.index) & set(self.keys.index))
        if not unique.loc[overlap, ["connectivity", "scaffold"]].equals(
            self.keys.loc[overlap, ["connectivity", "scaffold"]],
        ):
            raise ValueError("The source compound keys disagree on scaffold identity")
        self.keys = pd.concat((self.keys, unique.loc[~unique.index.isin(self.keys.index)]))

    def lookup_sanger(self, names, drug_ids):
        annotated = self.sanger.reindex(drug_ids.to_numpy())
        observed = annotated.name.fillna("").map(exact_token).to_numpy()
        wanted = names.map(exact_token).to_numpy()
        different = annotated.id.notna().to_numpy() & (observed != wanted)
        if different.any():
            raise ValueError("A Sanger library drug name differs from its registered ID")
        preferred = pd.Series(annotated.id.to_numpy(), index=names.index)
        return preferred.fillna(self.lookup(names))
