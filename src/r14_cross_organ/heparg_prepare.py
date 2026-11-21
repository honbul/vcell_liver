"""Build HepaRG well caches and a shared structural split without reading model scores."""

import argparse
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import read_json, sha256, write_json

PRIMARY = "GSE296254"
SECONDARY = "GSE284321"
DSSTOX = "https://clowder.edap-cluster.com/api/files/6114f5ffe4b0856fdc658625/blob"
REQUIRED = {
    "MAPPED_DTXSID": "dtxsid",
    "PREFERRED_NAME_DTXSID": "name",
    "CASRN_DTXSID": "cas",
    "SMILES_MAPPED_DTXSID": "smiles",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def name_key(text):
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def parent(smiles):
    normal = canonical_parent(smiles) if smiles and smiles != "-" else None
    if not normal:
        return None
    key = Chem.MolToInchiKey(Chem.MolFromSmiles(normal[1])).split("-")[0]
    return (key, normal[1]) if len(key) == 14 else None


def source_index(base, dataset):
    rows = pd.read_csv(base / f"{dataset}_bulk_sample_index.tsv", sep="\t", keep_default_na=False)
    if rows.library.duplicated().any() or not rows.in_count_header.all():
        raise ValueError("Unmatched or duplicate HepaRG library")
    rows = rows.loc[rows.time.eq("24h")].copy()
    rows["dataset"] = dataset
    rows["position"] = rows.library.str.extract(r"_([A-P](?:0[1-9]|1[0-9]|2[0-4]))$")
    if rows.position.isna().any() or not (rows.plate + "_" + rows.position).eq(rows.library).all():
        raise ValueError("The released plate/well fields disagree")
    rows["vehicle"] = rows.treatment.str.contains(r"\bvehicle control\b", case=False, regex=True)
    rows["culture"] = rows.batch.str.extract(r"(?:^|\s)culture:(\S+)").fillna("")
    rows["dose_uM"] = pd.to_numeric(rows.concentration.str.extract(r"^([-+\d.eE]+)\s*uM$", expand=False), errors="coerce")
    rows = rows.loc[rows.vehicle | (rows.dose_uM.gt(0) & ~rows.treatment.str.contains("viability", case=False))]
    rows["name_key"] = rows.chemical.map(name_key)
    rows["scope"] = np.where(rows.vehicle, "vehicle", "donor")
    rows["physical_well_id"] = rows.library
    rows["plate_id"] = rows.plate
    rows["compound"] = rows.chemical
    rows["cell_line_id"] = "HepaRG"
    rows["time_h"] = 24
    rows["n_cells"] = 1
    return rows.reset_index(drop=True)


def chemistry(args):
    direct = pd.concat([source_index(args.index, name) for name in (PRIMARY, SECONDARY)])
    direct = direct.loc[~direct.vehicle, ["dataset", "identity", "chemical", "name_key"]].drop_duplicates()
    dss = pd.read_csv(args.dsstox, usecols=list(REQUIRED), dtype=str, keep_default_na=False,
                      encoding="latin-1").rename(columns=REQUIRED)
    dss["name_key"] = dss.name.map(name_key)
    dss = dss.loc[dss.dtxsid.isin(direct.identity) | dss.name_key.isin(direct.name_key)].copy()
    dss["structure"] = dss.smiles.map(parent)
    dss = dss.dropna(subset=["structure"])
    by_id = {n: f for n, f in dss.groupby("dtxsid")}
    by_name = {n: f for n, f in dss.groupby("name_key")}
    fold_record = read_json(args.tahoe_partition / "split.json")["outer"]
    tahoe = pd.read_csv(args.tahoe_partition / "source_rows.tsv", sep="\t",
                       usecols=["compound_group_id", "compound_entity_id", "smiles"]).drop_duplicates()
    tahoe["structure"] = tahoe.smiles.fillna("").map(parent)
    tahoe["parent_key"] = tahoe.structure.map(lambda x: x[0] if x else "")
    tahoe["outer_fold"] = tahoe.compound_group_id.astype(str).map(fold_record)
    hist = tahoe.loc[tahoe.parent_key.ne("")].groupby("parent_key").outer_fold.agg(lambda x: sorted(set(x)))
    if hist.map(len).gt(1).any():
        raise ValueError("Historical Tahoe has connectivity collisions across folds")
    historical = hist.map(lambda x: x[0]).to_dict()
    out = []
    for row in direct.itertuples(index=False):
        candidates = by_id.get(row.identity)
        evidence = "DSSTox DTXSID"
        if candidates is None:
            candidates = by_name.get(row.name_key)
            evidence = "DSSTox exact normalized preferred name"
        parents = set(candidates.structure) if candidates is not None else set()
        if len({p[0] for p in parents}) == 1:
            key, smiles = sorted(parents)[0]
            cds = candidates.loc[candidates.structure.map(lambda p: p[0] == key)]
            ids = ",".join(sorted(set(cds.dtxsid)))
            cas = ",".join(sorted(set(cds.cas) - {"", "-"}))
        else:
            key = smiles = ids = cas = ""
            evidence = "conflicting structure" if parents else "unresolved structure"
        out.append(dict(dataset=row.dataset, identity=row.identity, chemical=row.chemical,
                        name_key=row.name_key, parent_key=key, smiles=smiles, dtxsid=ids,
                        cas=cas, identity_evidence=evidence))
    result = pd.DataFrame(out)
    unknown = sorted(set(result.parent_key) - {""} - historical.keys(), key=digest)
    mapping = {**historical, **{key: i % 3 for i, key in enumerate(unknown)}}
    result["outer_fold"] = result.parent_key.map(mapping).fillna(-1).astype(int)
    result["repeat_group"] = np.where(result.parent_key.ne(""), result.parent_key,
                                       result.dataset + "|" + result.identity.where(result.identity.ne("NA"), result.name_key))
    result["repeat_fold"] = [f if f >= 0 else int(digest(g), 16) % 3
                             for g, f in zip(result.repeat_group, result.outer_fold)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, sep="\t", index=False, mode="x")
    write_json(args.out.with_suffix(".json"), {
        "files": {str(p): sha256(p) for p in [args.out, args.dsstox, args.tahoe_partition / "split.json"]},
        "rows": len(result), "source": DSSTOX,
        "coverage": {name: {"records": len(part), "mapped": int(part.parent_key.ne("").sum()),
                            "distinct_parents": int(part.loc[part.parent_key.ne(""), "parent_key"].nunique())}
                     for name, part in result.groupby("dataset")},
    })


def known_probes(primary_file, gene_file):
    skim = pd.read_csv(primary_file, sep="\t", usecols=["probe_id", "gene"], dtype=str,
                       keep_default_na=False)
    tahoe = pd.read_csv(gene_file, sep="\t", usecols=["gene_symbol", "ensembl_id"], dtype=str,
                       keep_default_na=False)
    tahoe = tahoe[tahoe.ensembl_id.str.fullmatch(r"ENSG\d+")].drop_duplicates()
    unique = tahoe.loc[~tahoe.gene_symbol.duplicated(keep=False)].set_index("gene_symbol").ensembl_id
    skim["ensembl_id"] = skim.gene.map(unique)
    skim = skim.dropna(subset=["ensembl_id"])[["probe_id", "ensembl_id"]].drop_duplicates()
    return skim.loc[~skim.probe_id.duplicated(keep=False)].set_index("probe_id").ensembl_id


def read_counts(file, libraries, probes, genes):
    library_names = libraries.tolist()
    raw = np.zeros((len(genes), len(libraries)), dtype=np.int32)
    detected = np.zeros(len(libraries), dtype=np.int32)
    seen = set()
    available = np.zeros(len(genes), dtype=bool)
    types = {name: np.int32 for name in library_names}
    types["probe_id"] = str
    sep = "," if file.name.endswith(".csv.gz") else "\t"
    for chunk in pd.read_csv(file, sep=sep, usecols=["probe_id", *library_names], dtype=types, chunksize=512):
        if chunk.probe_id.duplicated().any() or seen.intersection(chunk.probe_id):
            raise ValueError(f"Duplicate probe: {file}")
        seen.update(chunk.probe_id)
        mapped = chunk.probe_id.map(probes)
        valid = mapped.notna().to_numpy()
        counts = chunk.loc[valid, library_names].to_numpy(dtype=np.int32)
        if np.any(counts < 0):
            raise ValueError(f"Negative count: {file}")
        gids = genes.get_indexer(mapped.loc[valid])
        available[gids] = True
        np.add.at(raw, gids, counts)
        detected += np.count_nonzero(counts, axis=0)
    totals = raw.sum(axis=0)
    normalized = np.log1p(raw.astype(np.float32) * (10000 / totals.clip(1)).astype(np.float32)).T
    return normalized, available, totals, detected


def prepare(args):
    rows = source_index(args.index, args.dataset)
    chemistry = pd.read_csv(args.manifest, sep="\t", dtype={"identity": str}, keep_default_na=False)
    rows = rows.merge(chemistry.drop(columns="name_key"), on=["dataset", "identity", "chemical"],
                      how="left", validate="many_to_one", sort=False)
    rows["parent_key"] = rows.parent_key.fillna("")
    rows["smiles"] = rows.smiles.fillna("")
    rows["repeat_group"] = rows.repeat_group.fillna("")
    rows["outer_fold"] = rows.outer_fold.fillna(-1).astype(int)
    rows["repeat_fold"] = rows.repeat_fold.fillna(-1).astype(int)
    doses = rows.loc[~rows.vehicle, ["repeat_group", "dose_uM"]].drop_duplicates().sort_values("dose_uM")
    doses["dose_rank"] = doses.groupby("repeat_group").cumcount() + 1
    rows = rows.merge(doses, on=["repeat_group", "dose_uM"], how="left", validate="many_to_one", sort=False)
    rows["dose_rank"] = rows.dose_rank.fillna(0).astype(int)
    probes = known_probes(args.primary_file, args.genes)
    genes = pd.Index(sorted(probes.unique()))
    args.out.mkdir(parents=True, exist_ok=False)
    scratch = args.out / "log.scratch.npy"
    y = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32, shape=(len(rows), len(genes)))
    mapped = np.ones(len(genes), dtype=bool)
    for file, ids in rows.groupby("count_file", sort=True).indices.items():
        z, available, totals, detected = read_counts(args.raw / file, rows.library.iloc[ids], probes, genes)
        y[ids] = z
        mapped &= available
        rows.loc[ids, "mapped_library"] = totals
        rows.loc[ids, "detected_probes"] = detected
    good = (rows.mapped_library >= 100000) & (rows.detected_probes >= 1000)
    across_plate = {}
    qc_controls = rows.vehicle & good
    v_index = np.flatnonzero(qc_controls)
    for plate, vd in rows.loc[qc_controls].groupby("plate"):
        if len(vd) >= 2:
            across_plate[plate] = y[vd.index][:, mapped].mean(axis=0)
    valid = good & ~rows.vehicle & rows.plate.isin(across_plate)
    reached = rows.loc[valid].reset_index(drop=True)
    pd.DataFrame({"ensembl_id": genes[mapped]}).to_csv(args.out / "genes.tsv", sep="\t", index=False)
    reached.to_csv(args.out / "rows.tsv", sep="\t", index=False)
    rows.loc[v_index].to_csv(args.out / "vehicles.tsv", sep="\t", index=False)
    np.save(args.out / "vehicles.npy", y[v_index][:, mapped])
    out_shape = (len(reached), int(mapped.sum()))
    response = np.lib.format.open_memmap(args.out / "delta.npy", mode="w+", dtype=np.float32, shape=out_shape)
    controls = np.lib.format.open_memmap(args.out / "control.npy", mode="w+", dtype=np.float32, shape=out_shape)
    selected = np.flatnonzero(valid)
    for plate, indices in reached.groupby("plate").indices.items():
        controls[indices] = across_plate[plate]
        response[indices] = y[selected[indices]][:, mapped] - across_plate[plate]
    del controls, response, y
    scratch.unlink()
    source_md = args.index / f"{args.dataset}_bulk_sample_index.tsv"
    write_json(args.out / "receipt.json", {
        "dataset": args.dataset, "protocol_sha256": sha256(args.protocol),
        "manifest_sha256": sha256(args.manifest), "metadata_sha256": sha256(source_md),
        "axes": {"rows": len(reached), "genes": int(mapped.sum()), "vehicles": len(v_index),
                 "structurally_resolved_wells": int(reached.parent_key.ne("").sum()),
                 "protein_assay_probe_matches": len(probes)},
        "files": {p.name: sha256(p) for p in sorted(args.out.iterdir())},
        "well_evidence": "physical plate and A01-P24 coordinate from original GEO library ID",
        "culture_field": "release batch culture token" if args.dataset == PRIMARY else "unavailable",
        "response": "log1p CP10k of mapped collapsed genes minus same-plate QC-vehicle mean",
    })


def main():
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, args, entry in (
        ("chemistry", ["index", "dsstox", "tahoe-partition", "out"], chemistry),
        ("prepare", ["index", "manifest", "genes", "primary-file", "raw", "protocol", "out"], prepare),
    ):
        sub = stages.add_parser(name)
        for arg in args:
            sub.add_argument(f"--{arg}", type=Path, required=True)
        if name == "prepare":
            sub.add_argument("--dataset", choices=[PRIMARY, SECONDARY], required=True)
        sub.set_defaults(entry=entry)
    args = parser.parse_args()
    args.entry(args)


if __name__ == "__main__":
    main()
