"""Audit optional public assays and identity coverage without exposing clinical PK values."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import h5py
import pandas as pd
from sklearn.model_selection import GroupKFold

from .splits import data_groups, digest, receipt
from .transfer import heparg_plates, match


def obs_codes(obs, field):
    item = obs[field]
    if isinstance(item, h5py.Dataset):
        return [x.decode() if isinstance(x, bytes) else x for x in item[:]]
    categories = [x.decode() if isinstance(x, bytes) else x for x in item["categories"][:]]
    return [categories[i] if i >= 0 else None for i in item["codes"][:]]


def plate_support(source, phh):
    coordinates = heparg_plates(source)
    pairs = match(coordinates, phh)
    used = coordinates.iloc[pairs.index_a].reset_index(drop=True)
    groups = data_groups(used, None, "murcko")
    by_fold = []
    for fold, (train, test) in enumerate(GroupKFold(4).split(used, groups=groups)):
        holdout = set().union(*used.plate_id.iloc[test])
        source = set().union(*used.plate_id.iloc[train])
        purged = used.plate_id.iloc[train].map(lambda x: not bool(x & holdout))
        by_fold.append(dict(fold=fold, train=len(train), held=len(test),
                            shared_plates=len(source & holdout), plate_purged_train=int(purged.sum())))
    return by_fold


def audit(args):
    files = sorted(args.dilimap.glob("*.h5ad"))
    if len(files) != 2:
        raise ValueError("Expected the published train and validation WikiPathways matrices")
    fda = pd.read_csv(args.fda, sep="\t", usecols=["fda_id", "release_increment"])
    newer = set(fda.loc[fda.release_increment, "fda_id"])
    findings = {}
    for path in files:
        with h5py.File(path) as packed:
            observations, pathways = packed["X"].shape
            names = set(obs_codes(packed["obs"], "compound_name")) - {None}
            identifiers = set(obs_codes(packed["obs"], "LTKBID")) & set(fda.fda_id)
            findings[path.name] = dict(observations=observations, pathways=pathways, compounds=len(names),
                fda_id_overlap=len(identifiers), release_increment_overlap=len(identifiers & newer),
                sha256=digest(path))
    viability = {}
    for name in ("open_tggates_cell.zip", "open_tggates_cell_viability.zip"):
        path = args.tggates / name
        with ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as stream:
                data = pd.read_csv(stream, encoding="latin1", dtype=str)
        subset = data.loc[data.SPECIES.eq("Human")]
        viability[name] = dict(human_rows=len(subset), compounds=subset.COMPOUND_NAME.nunique(), sha256=digest(path))
    prep = pd.read_csv(args.phh / "rows.tsv", sep="\t")
    physical_support = {}
    for field in ("treated_barcodes", "control_barcodes", "experiment"):
        keys = prep.assign(key=prep[field].astype(str).str.split(",")).explode("key")
        compounds_per_key = keys.groupby("key").compound.nunique()
        physical_support[field] = dict(unique=len(compounds_per_key),
                                      shared_between_compounds=int(compounds_per_key.gt(1).sum()))
    target = {}
    for path in (args.toxcast_assay, args.toxcast_mapping):
        sheets = pd.read_excel(path, sheet_name=None)
        target[path.name] = dict(sha256=digest(path), sheets={})
        for name, table in sheets.items():
            text = table.fillna("").astype(str).agg(" ".join, axis=1)
            target[path.name]["sheets"][name] = dict(total_rows=len(table),
                bsep_rows=int(text.str.contains(r"\bABCB11\b|\bBSEP\b|bile salt export pump", case=False).sum()),
                cyp_rows=int(text.str.contains(r"\bCYP[123][A-Z]\d+\b|cytochrome P450", case=False).sum()),
                mitochondrial_rows=int(text.str.contains("mitochondri", case=False).sum()))
    phh_conditions = pd.read_parquet(args.assays / "TGGATEs_PHH.parquet", columns=["inchikey", "dose_uM"])
    phh_conditions = phh_conditions.loc[phh_conditions.dose_uM.between(1, 1000)].drop_duplicates().reset_index(drop=True)
    plates = {source.parent.name: plate_support(source, phh_conditions) for source in args.heparg}
    receipt(args.out, dict(dilimap=findings, tggates_viability=viability, toxcast_target_keyword_audit=target,
        phh_prepared_rows=len(prep), phh_prepared_compound_names=prep.compound.nunique(),
        phh_rows_with_dna=int(prep.dna_percent.notna().sum()), phh_rows_with_ldh=int(prep.ldh_percent.notna().sum()),
        phh_physical_support=physical_support,
        heparg_matched_transfer_plate_support=plates,
        excluded_input_fields=["Cmax", "postmarketing daily dose", "indication", "therapeutic class", "clinical warning"],
        clinical_pk_column_values_read=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dilimap", "fda", "tggates", "phh", "toxcast-assay", "toxcast-mapping", "assays", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--heparg", type=Path, nargs="+", required=True, help="R14 sample metadata tables")
    audit(parser.parse_args())
