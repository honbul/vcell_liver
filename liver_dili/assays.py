"""Harmonize public liver RNA contrasts and verified Cell Painting profiles."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .chemistry import identity
from .splits import digest, receipt


WP = {
    "WP408": "oxidative_stress", "WP4925": "unfolded_protein_response",
    "WP4324": "mitochondrial_complex_I", "WP5238": "cholestasis",
    "WP4389": "bile_acid_synthesis", "WP368": "mitochondrial_fatty_acid_oxidation",
    "WP2011": "lipid_homeostasis", "WP254": "apoptosis", "WP453": "inflammation",
}
HALLMARK = {
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY": "oxidative_stress",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "er_stress",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "mitochondrial_oxphos",
    "HALLMARK_BILE_ACID_METABOLISM": "bile_acid_metabolism",
    "HALLMARK_FATTY_ACID_METABOLISM": "fatty_acid_metabolism",
    "HALLMARK_XENOBIOTIC_METABOLISM": "xenobiotic_cyp_metabolism",
    "HALLMARK_APOPTOSIS": "apoptosis",
    "HALLMARK_INFLAMMATORY_RESPONSE": "inflammation",
}


def valid(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        return identity(smiles)
    except (ValueError, RuntimeError):
        return None


def heparg(folder, name, hgnc):
    rows = pd.read_csv(folder / "rows.tsv", sep="\t", low_memory=False)
    genes = pd.read_csv(folder / "genes.tsv", sep="\t").ensembl_id.map(hgnc)
    values = np.load(folder / "delta.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != (len(rows), len(genes)):
        raise ValueError("HepaRG input registry dimensions changed")
    field = rows.smiles.map(valid)
    good = field.notna() & rows.time_h.eq(24) & rows.dose_uM.gt(0)
    rows = rows.loc[good, ["compound", "dose_uM", "plate_id", "culture"]].copy()
    rows["smiles"] = field[good].map(lambda x: x[0])
    rows["inchikey"] = field[good].map(lambda x: x[1])
    rows["connectivity"] = field[good].map(lambda x: x[2])
    rows["assay"] = name
    genes_ok = genes.notna() & ~genes.duplicated(keep=False)
    return rows.reset_index(drop=True), pd.DataFrame(values[good.to_numpy(), :][:, genes_ok], columns=genes[genes_ok])


def phh(folder, structures):
    table = pd.read_csv(folder / "rows.tsv", sep="\t")
    genes = pd.read_csv(folder / "genes.tsv", sep="\t").symbol
    values = np.memmap(folder / "delta.f32", mode="r", dtype="<f4", shape=(len(table), len(genes)))
    candidates = structures.dropna(subset=["inchikey"]).reset_index()
    candidates["key"] = candidates.name.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    # Disallow a collision even if the spelling normalization looks harmless.
    unique = candidates.groupby("key").inchikey.transform("nunique").eq(1)
    candidates = candidates.loc[unique].drop_duplicates("key").set_index("key")
    keys = table.compound.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    for field in ("smiles", "inchikey", "connectivity"):
        table[field] = keys.map(candidates[field])
    table["assay"] = "TGGATEs_PHH"
    good = table.inchikey.notna()
    return table.loc[good].reset_index(drop=True), pd.DataFrame(values[good.to_numpy()], columns=genes)


def register_sets(raw, entrez):
    definitions = {}
    for line in (raw / "wikipathways.gmt").read_text().splitlines():
        key, _, *genes = line.split("\t")
        wp = key.split("%")[-2]
        if wp in WP:
            definitions["wikipathways__" + WP[wp]] = dict.fromkeys(
                (entrez[g] for g in genes if g in entrez), 1.0)
    if len(definitions) != len(WP):
        raise ValueError("Missing registered WikiPathways model")
    for path in sorted(raw.glob("HALLMARK_*.gmt")):
        key, _, *genes = path.read_text().strip().split("\t")
        definitions["hallmark__" + HALLMARK[key]] = dict.fromkeys(genes, 1.0)
    if sum(name.startswith("hallmark__") for name in definitions) != len(HALLMARK):
        raise ValueError("Missing registered Hallmark sets")
    progeny = pd.read_csv(raw / "progeny.tsv", sep="\t").sort_values(["pathway", "p.value", "gene"])
    for name, group in progeny.groupby("pathway"):
        block = group.drop_duplicates("gene").head(100).set_index("gene")
        definitions[f"progeny__{name}"] = block.weight.to_dict()
    return definitions


def pathway_matrix(expression, definitions):
    arr = expression.to_numpy()
    rank = rankdata(arr, axis=1)
    rank = ((rank - (arr.shape[1] + 1)/2) / (arr.shape[1] / np.sqrt(12))).astype(np.float32)
    # Preserve the zero vector in contrast data, including tied zero-valued genes.
    coverage, values = {}, {}
    for name, definition in definitions.items():
        overlap = sorted(set(definition) & set(expression))
        coverage[name] = {"matched": len(overlap), "registered": len(definition)}
        if len(overlap) < max(5, len(definition) * .33):
            continue
        coefficients = np.array([definition[g] for g in overlap])
        values[name] = (rank[:, expression.columns.get_indexer(overlap)] @ coefficients /
                        np.linalg.norm(coefficients))
    return pd.DataFrame(values, index=expression.index), coverage


def paint(cpg, lincs, prism):
    cids = pd.read_csv(lincs, sep="\t", usecols=["pert_id", "canonical_smiles"]).rename(
        columns={"pert_id": "id", "canonical_smiles": "smiles"})
    pids = pd.read_csv(prism, usecols=["broad_id", "smiles"]).rename(columns={"broad_id": "id"})
    mapping = pd.concat([cids, pids], ignore_index=True).drop_duplicates()
    mapping["identity"] = mapping.smiles.map(valid)
    mapping = mapping.dropna(subset=["identity"])
    mapping["inchikey"] = mapping.identity.map(lambda x: x[1])
    mapping["parent"] = mapping.id.str.extract(r"^(BRD-[AK]\d{8})", expand=False)
    sample_map = mapping.loc[mapping.groupby("id").inchikey.transform("nunique").eq(1)].drop_duplicates("id")
    parent_map = mapping.loc[mapping.groupby("parent").inchikey.transform("nunique").eq(1)].drop_duplicates("parent")
    sample_map = sample_map.set_index("id").inchikey
    parent_map = parent_map.set_index("parent").inchikey
    proven = pd.read_csv(cpg / "broad_id_map.csv")
    columns = [col for col in proven.columns if re.fullmatch(r"broad_id_\d{8}", col)]
    cross = proven.melt(id_vars=["InChIKey14"], value_vars=columns, value_name="parent").dropna(subset=["parent"])
    orthogonal = cross.groupby("parent").InChIKey14.agg(lambda v: next(iter(set(v))) if v.nunique() == 1 else "")
    frames, counts = [], {}
    for path in sorted(cpg.glob("*_consensus_median_dmso.csv.gz")):
        raw = pd.read_csv(path, low_memory=False)
        fields = sorted(field for field in raw if field.startswith(("Cells_", "Nuclei_", "Cytoplasm_")))
        select = (raw.Metadata_cell_id.eq("A549") & raw.Metadata_time_point.eq("48H")
                  & pd.to_numeric(raw.Metadata_dose_recode, errors="coerce").eq(6))
        subset = raw.loc[select].copy()
        ids = subset.Metadata_broad_sample
        parents = ids.str.extract(r"^(BRD-[AK]\d{8})", expand=False)
        keys = ids.map(sample_map).fillna(parents.map(parent_map))
        agrees = parents.map(orthogonal)
        keys = keys.where(agrees.isna() | agrees.eq(keys.str.slice(0, 14)))
        counts[path.name] = {"dose_context": len(subset), "stereo_resolved": int(keys.notna().sum())}
        profiles = subset[fields].copy()
        profiles["inchikey"] = keys
        frames.append(profiles.dropna(subset=["inchikey"]).groupby("inchikey").median())
    return pd.concat(frames).groupby(level="inchikey").median(), counts


def prepare(args):
    arrays, measurements = {}, {}
    hgnc = pd.read_csv(args.hgnc, sep="\t", usecols=["ensembl_gene_id", "entrez_id", "symbol"], dtype=str)
    entrez = hgnc.dropna(subset="entrez_id").drop_duplicates("entrez_id", keep=False).set_index("entrez_id").symbol
    hgnc = hgnc.dropna(subset="ensembl_gene_id").drop_duplicates("ensembl_gene_id", keep=False).set_index("ensembl_gene_id").symbol
    compounds = pd.read_csv(args.structures, sep="\t", index_col="name")
    for name, path in (("GSE284321", args.heparg_284321), ("GSE296254", args.heparg_296254)):
        measurements[name], arrays[name] = heparg(path, name, hgnc)
    measurements["TGGATEs_PHH"], arrays["TGGATEs_PHH"] = phh(args.phh, compounds)
    shared = sorted(set.intersection(*(set(frame) for frame in arrays.values())))
    if len(shared) < 2500:
        raise ValueError("Unexpectedly small cross-assay gene intersection")
    definitions = register_sets(args.raw, entrez)
    args.out.mkdir(parents=True, exist_ok=False)
    outputs, coverage = [], {}
    for name, meta in measurements.items():
        values = arrays[name].loc[:, shared]
        scores, coverage[name] = pathway_matrix(values, definitions)
        molecular = meta[["assay", "compound", "dose_uM", "smiles", "inchikey", "connectivity"]]
        rows = pd.concat([molecular, scores], axis=1)
        rows.to_parquet(args.out / f"{name}.parquet", index=False)
        values.to_parquet(args.out / f"{name}-genes.parquet", index=False)
        outputs.append(dict(assay=name, rows=len(meta), stereo=meta.inchikey.nunique(),
                            parents=meta.connectivity.nunique(), genes=len(shared)))
    painting, paint_support = paint(args.cpg, args.lincs, args.prism)
    painting.to_parquet(args.out / "painting.parquet")
    for col in [col for col in painting.columns if painting[col].notna().any()]:
        if not pd.api.types.is_numeric_dtype(painting[col]):
            raise ValueError("Cell Painting admitted metadata among numerical features")
    fda = pd.read_csv(args.fda, sep="\t").dropna(subset=["inchikey"])
    overlaps = {}
    for name, meta in measurements.items():
        overlaps[name] = {"exact_stereo": len(set(meta.inchikey) & set(fda.inchikey)),
                          "connectivity": len(set(meta.connectivity) & set(fda.connectivity))}
    overlaps["Cell_Painting"] = {"exact_stereo": len(set(painting.index) & set(fda.inchikey)),
                                "connectivity": len(set(painting.index.str[:14]) & set(fda.connectivity))}
    loaded = [args.structures, args.fda, args.hgnc, args.lincs, args.prism, *args.cpg.glob("*"),
              *args.phh.glob("*"), *args.raw.glob("*.gmt"), args.raw / "progeny.tsv",
              *(path / file for path in (args.heparg_284321, args.heparg_296254)
                for file in ("rows.tsv", "genes.tsv", "delta.npy", "receipt.json"))]
    receipt(args.out / "receipt.json", dict(assays=outputs, common_genes=len(shared), phenotype=paint_support,
        fda_intersections=overlaps, coverage=coverage, painting_compounds=len(painting), painting_features=painting.shape[1],
        inputs={str(path): digest(path) for path in loaded if path.is_file()},
        files={p.name: digest(p) for p in sorted(args.out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("heparg-284321", "heparg-296254", "phh", "hgnc", "structures", "fda",
                  "raw", "cpg", "lincs", "prism", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    prepare(parser.parse_args())
