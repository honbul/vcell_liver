"""Compare frozen FDA identities to the authors' released PubChem18 roster and assay split."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vc_fm_bench.artifacts import digest, new_directory, write_json
from .data import G1, SPLITS, Registered, legacy_rosters, read
from .molecular import molecule


def unpack(archive, destination):
    with ZipFile(archive) as zipped:
        names = {Path(p).name: p for p in zipped.namelist()}
        members = ("compound_names.parquet", "compound_smiles.parquet", "assay_names.parquet", "activity.parquet")
        paths = {name: destination / name for name in members}
        for name, path in paths.items():
            with zipped.open(names[name]) as source, path.open("wb") as out:
                shutil.copyfileobj(source, out)
    return paths


def matched_index(compounds, smiles, metadata):
    if not compounds.CID.is_unique or not smiles.CID.is_unique or len(compounds) != len(smiles):
        raise ValueError("Author compound roster identity is ambiguous")
    ids = pd.Series(compounds.index.to_numpy(), index=compounds.CID)
    available_smiles = smiles.set_index("CID").CanonicalSMILES
    target_smiles = defaultdict(set)
    matched = {}
    for key, row in metadata.iterrows():
        cid_num = pd.to_numeric(row.pubchem_cid, errors="coerce")
        cids = {int(cid_num)} if pd.notna(cid_num) and int(cid_num) in ids else set()
        matched[key] = dict(supplied_cid=cids.copy(), candidates=cids)
        for value in (row.smiles, row.supplied_smiles, molecule(row.smiles)[0]):
            if isinstance(value, str) and value:
                target_smiles[value].add(key)
    for cid, value in available_smiles.loc[available_smiles.isin(target_smiles)].items():
        for key in target_smiles[value]:
            matched[key]["candidates"].add(cid)
    result = {}
    for key, record in matched.items():
        target = molecule(metadata.at[key, "smiles"])[1]
        exact, connectivity = set(), set()
        for candidate in record["candidates"]:
            value = available_smiles.at[candidate]
            try:
                observed = molecule(value)[1]
            except (ValueError, TypeError):
                continue
            if observed == target:
                exact.add(candidate)
            if observed.split("-")[0] == target.split("-")[0]:
                connectivity.add(candidate)
        result[key] = dict(**record, exact=exact, connectivity=connectivity)
    return result, ids


def train_support(parquet, size, eligible_indices, assay_cut):
    # Read only membership and binary activity columns, with bounded memory.
    mask = np.zeros(size, dtype=bool)
    mask[eligible_indices] = True
    any_early, active_early = set(), set()
    file = pq.ParquetFile(parquet)
    for block in file.iter_batches(batch_size=750_000, use_threads=True,
                                   columns=["compound_idx", "assay_idx", "activity"]):
        compound, assay, activity = [column.to_numpy() for column in block.columns]
        keep = mask[compound] & (assay < assay_cut)
        any_early.update(np.unique(compound[keep]).tolist())
        active_early.update(np.unique(compound[keep & (activity == 1)]).tolist())
    return any_early, active_early, file.metadata.num_rows


def audit(options):
    reg = Registered()
    legacy = legacy_rosters()
    scopes = {
        "scaffold_887": set().union(*(reg.query(f"murcko-{fold}").index for fold in range(5))),
        "release_155": set(reg.query("release-0").index),
        "legacy_release_147": set(legacy["release"][1].index),
        "legacy_scaffold_858": set(legacy["murcko"][1].index),
    }
    fda = G1 / "liver_dili_v1/chemistry-v1/fda_structures.tsv"
    if digest(fda) != read(SPLITS / "receipt.json")["source_sha256"]:
        raise ValueError("FDA structure registry no longer matches the frozen split")
    columns = ["fda_id", "pubchem_cid", "smiles", "supplied_smiles", "inchikey", "connectivity"]
    metadata = pd.read_csv(fda, sep="\t", usecols=columns).set_index("fda_id", verify_integrity=True)
    metadata = metadata.loc[sorted(set().union(*scopes.values()))]
    with new_directory(options.out) as out:
        cache = options.cache
        cache.mkdir(exist_ok=False, parents=True)
        files = unpack(options.archive, cache)
        cids = pd.read_parquet(files["compound_names.parquet"])
        smiles = pd.read_parquet(files["compound_smiles.parquet"])
        assays = pd.read_parquet(files["assay_names.parquet"], columns=["AID"])
        if len(cids) != 2120811 or len(assays) != 21002:
            raise ValueError("The public archive does not match the CLAMP paper metadata")
        candidates, lookup = matched_index(cids, smiles, metadata)
        del cids, smiles
        indices = {lookup.at[cid] for record in candidates.values() for cid in record["candidates"]}
        cutoff = round(.60 * len(assays))
        any_early, active_early, rows = train_support(
            files["activity.parquet"], len(lookup), list(indices), cutoff)
        if rows != 223219241:
            raise ValueError("Unexpected PubChem18 activity denominator")
        results = {}
        for scope, keys in scopes.items():
            subset = []
            for key in sorted(keys):
                record = candidates[key]
                clean = record["connectivity"]
                exact = record["exact"]
                subset.append(dict(fda_id=key, pubchem_cid_list_match=bool(record["supplied_cid"]),
                                   archive_exact_parent=bool(exact), archive_connectivity=bool(clean),
                                   source_has_any_early_assay=any(lookup.at[cid] in any_early for cid in clean),
                                   source_has_active_early_assay=any(lookup.at[cid] in active_early for cid in clean),
                                   exact_parent_active_early_assay=any(lookup.at[cid] in active_early for cid in exact)))
            frame = pd.DataFrame(subset)
            results[scope] = dict(n=len(frame), **{k: int(v) for k, v in frame.drop(columns="fda_id").sum().items()})
        write_json(out / "exposure_receipt.json", dict(
            scopes=results, pubchem18_sha256=digest(options.archive), compound_rows=len(lookup),
            assays=len(assays), public_activity_rows=rows, early_assay_count=cutoff,
            last_early_assay_aid=int(assays.AID.iloc[cutoff-1]),
            membership_rule="registered FDA PubChem CID or exact archived SMILES string, then RDKit ChargeParent identity",
            checkpoint_training_manifest_available=False,
            checkpoint_training_reconstruction="author public PubChem18 plus public time_a oldest 60% and active-only filter; exact checkpoint training roster not documented"))
        print(results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "cache", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    audit(parser.parse_args())
