"""Rebuild authenticated, treatment-matched LINCS Level 5 landmark priors."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, normalized_name, structural_keys
from src.r13_optimization.transfer import unit_norm

PROTOCOL_SHA256 = "d0680c422b41c63f895b8c1efc0fdedbcbc7965fdc1e35a40e6eb7bde6da8693"
STUDIES = ("GSE92742", "GSE70138")
PROFILES = ("pan_near10", "pan_anydose", "hepatic_near10", "hepg2_near10", "phh_near10")
ENTREZ_COUNT = 978
# Study metadata uses these verified labels for HepG2 or primary hepatocytes.
HEPATIC = frozenset({"HEPG2", "PHH"})


def find_file(folder, glob):
    paths = sorted(folder.glob(glob))
    if len(paths) != 1:
        raise ValueError(f"Need exactly one original file {folder}/{glob}: {paths}")
    return paths[0]


def asset_paths(folder, study):
    phase2 = study == "GSE70138"
    tail = "_2017-03-06.txt.gz" if phase2 else ".txt"
    return {
        "gctx": find_file(folder, "*Level5*.gctx"),
        "gene": find_file(folder, f"*gene_info{tail}"),
        "pert": find_file(folder, f"*pert_info{tail}"),
        "sig": find_file(folder, f"*sig_info{tail}"),
        "cell": find_file(folder, "*cell_info_2017-04-28.txt.gz" if phase2 else "*cell_info.txt"),
    }


@dataclass(frozen=True)
class Identity:
    chemistry: frozenset
    full: frozenset
    name: frozenset
    cid: frozenset


def identity(smiles, names, inchi="", cid=""):
    smiles = clean(smiles)
    chemistry, _ = structural_keys(smiles)
    full = set()
    with rdBase.BlockLogs():
        if parent := canonical_parent(smiles) if smiles else None:
            mol = Chem.MolFromSmiles(parent[0])
            if mol is not None:
                if matched := Chem.MolToInchiKey(mol):
                    full.add(matched)
    key = clean(inchi).upper()
    if re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key):
        full.add(key)
        chemistry.add(("connectivity", key.split("-")[0]))
    names = {normalized_name(n) for n in names if clean(n)}
    identifier = clean(cid).split(".")[0]
    return Identity(frozenset(chemistry), frozenset(full), frozenset(names - {""}),
                    frozenset({identifier} if identifier and identifier.isdecimal() else ()))


def collect_identity(rows):
    parts = []
    for row in rows.itertuples(index=False):
        names = getattr(row, "names", getattr(row, "pert_iname", "")).split("|")
        smiles = getattr(row, "smiles", getattr(row, "canonical_smiles", ""))
        inchi = getattr(row, "inchi_key", "")
        cid = getattr(row, "cid", getattr(row, "pubchem_cid", ""))
        parts.append(identity(smiles, names, inchi, cid))
    return Identity(*(frozenset.union(frozenset(), *(getattr(p, field) for p in parts))
                      for field in Identity.__dataclass_fields__))


def read_table(path, **kwargs):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def tahoe_targets(entities, split):
    parent_set = set(read_json(split)["outer"])
    if len(parent_set) != 203:
        raise ValueError("Reference Tahoe split is not the 203-parent split")
    records = read_table(entities)
    records = records.loc[records.compound_group_id.isin(parent_set)].copy()
    if set(records.compound_group_id) != parent_set:
        raise ValueError("A frozen Tahoe parent was not recovered from metadata")
    return records.rename(columns={"compound_group_id": "parent_key", "compound_entity_id": "names"})


def catalogue(rows):
    return {str(parent): collect_identity(subset)
            for parent, subset in rows.groupby("parent_key", sort=True)}


def conflict(left, right):
    # Uncharged fragment and tautomer equivalence can reconcile salt forms.
    return bool(left.chemistry and right.chemistry and not left.chemistry & right.chemistry
                and not left.full & right.full)


def match_study(target, refs, study, cohort):
    fields = ("full", "chemistry", "cid", "name")
    indexes = {field: defaultdict(set) for field in fields}
    for key, row in target.items():
        for field in fields:
            for token in getattr(row, field):
                indexes[field][token].add(key)
    matched = {}
    audit = []
    for key, row in refs.items():
        by_field = {}
        for field in fields:
            by_field[field] = {candidate for token in getattr(row, field)
                               for candidate in indexes[field].get(token, ())}
        highest = next((f for f in fields if by_field[f]), None)
        if highest is None:
            continue
        options = by_field[highest]
        if len(options) > 1 and highest in {"full", "chemistry", "cid"} and len(options & by_field["name"]) == 1:
            options = options & by_field["name"]
        confirmed = {p for p in options if not conflict(target[p], row)}
        flags = []
        if len(confirmed) == 1 and (len(options) == 1 or highest == "full"):
            parent = next(iter(confirmed))
            matched[key] = parent
            for field in fields:
                if by_field[field] and parent not in by_field[field]:
                    flags.append(f"other_{field}_match")
            audit.append((cohort, study, key, parent, highest, "matched", ",".join(flags)))
        else:
            reason = "structure_conflict" if not confirmed else "ambiguous"
            for parent in sorted(options):
                audit.append((cohort, study, key, parent, highest, reason, ""))
    # Conservative fallback: a name/CID-only LINCS ID that could also refer to a
    # conflicting structure in the same original GEO release is removed.
    by_lincs_name = defaultdict(set)
    by_lincs_cid = defaultdict(set)
    for key, row in refs.items():
        for name in row.name:
            by_lincs_name[name].add(key)
        for cid in row.cid:
            by_lincs_cid[cid].add(key)
    final_audit = []
    for item in audit:
        c, s, key, parent, tier, status, flags = item
        if tier in {"name", "cid"} and status == "matched":
            row = refs[key]
            other = {j for n in row.name for j in by_lincs_name[n]} if tier == "name" else {
                j for n in row.cid for j in by_lincs_cid[n]
            }
            if any(conflict(target[parent], refs[j]) for j in other):
                matched.pop(key)
                status = "source_alias_conflict"
        final_audit.append((c, s, key, parent, tier, status, flags))
    return matched, final_audit


def dose_uM(rows):
    parsed = rows.pert_idose.str.extract(r"^\s*([-+\d.eE]+)\s*([^\d\s]+)\s*$")
    amounts = pd.to_numeric(parsed[0], errors="coerce")
    units = parsed[1].str.replace("μ", "u", regex=False).str.replace("µ", "u", regex=False)
    scales = units.str.lower().map({"um": 1., "mm": 1e3, "nm": 1e-3, "pm": 1e-6, "m": 1e6})
    return amounts * scales


def study_metadata(assets):
    sig = read_table(assets["sig"])
    sig = sig.loc[sig.pert_type.eq("trt_cp") & sig.pert_itime.str.match(r"^24(?:\.0+)?\s*h$", case=False)].copy()
    sig["dose_uM"] = dose_uM(sig)
    sig = sig.loc[np.isfinite(sig.dose_uM) & sig.dose_uM.gt(0)].copy()
    sig["cell_id"] = sig.cell_id.str.strip().str.upper()
    sig = sig.loc[sig.cell_id.ne("") & sig.cell_id.ne("-666")]
    if sig.sig_id.duplicated().any():
        raise ValueError("Duplicate qualifying LINCS signature IDs")
    gene = read_table(assets["gene"]).loc[lambda x: x.pr_is_lm.eq("1")]
    if len(gene) != ENTREZ_COUNT or not gene.pr_gene_id.is_unique:
        raise ValueError("Unexpected Level 5 landmark gene axis")
    gene = gene.assign(pr_gene_id=gene.pr_gene_id.astype(int)).set_index("pr_gene_id").sort_index()
    return sig, gene.pr_gene_symbol


def read_values(path, metadata, gene_ids):
    with h5py.File(path, "r") as f:
        profile_index = pd.Index(f["0/META/COL/id"][:].astype(str))
        positions = profile_index.get_indexer(metadata.sig_id)
        if not profile_index.is_unique or np.any(positions < 0):
            raise ValueError("Incorrect original LINCS signature IDs")
        gene_index = pd.Index(f["0/META/ROW/id"][:].astype(int))
        columns = gene_index.get_indexer(gene_ids)
        if not gene_index.is_unique or np.any(columns < 0):
            raise ValueError("Incorrect original LINCS landmark axis")
        matrix = f["0/DATA/0/matrix"]
        if matrix.shape != (len(profile_index), len(gene_index)):
            raise ValueError("Unexpected GCTX axis order or dimensions")
        # Read adjacent HDF5 slabs once, then select measured landmarks in memory.
        order = np.argsort(positions)
        sorted_rows = positions[order]
        values = np.empty((len(positions), len(columns)), dtype=np.float32)
        offsets = sorted_rows // 2048
        checks = np.r_[0, np.flatnonzero(np.diff(offsets)) + 1, len(sorted_rows)]
        for start, end in zip(checks[:-1], checks[1:]):
            ix = sorted_rows[start:end]
            block = matrix[ix[0]:ix[-1] + 1]
            values[order[start:end]] = block[np.ix_(ix - ix[0], columns)]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite Level 5 expression")
    values -= values.mean(axis=1, keepdims=True)
    return unit_norm(values)


def consensus(profiles, responses, matched, keys):
    profiles = profiles.loc[profiles.pert_id.isin(matched)].copy()
    profiles["key"] = profiles.pert_id.map(matched)
    profiles["matrow"] = profiles.index
    result = {}
    support = {}
    # Average duplicate records within each nominal dose, then nominal doses,
    # then cell lines and finally GEO studies. Renormalize the resulting direction.
    groups = ("study", "key", "cell_id", "dose_uM")
    summary = {}
    for label, subset in profiles.groupby(list(groups), sort=True):
        summary[label] = responses[subset.matrow.to_numpy()].mean(axis=0, dtype=np.float64).astype(np.float32)
    component = pd.DataFrame(list(summary), columns=list(groups))
    if not summary:
        for family in PROFILES:
            result[family] = np.zeros((len(keys), responses.shape[1]), dtype=np.float32)
            support[family] = np.zeros(len(keys), dtype=bool)
        return result, support
    component["vec"] = np.arange(len(component))
    patterns = np.stack(list(summary.values()))
    for family in PROFILES:
        valid = component
        if family != "pan_anydose":
            valid = valid.loc[valid.dose_uM.between(5, 20)].copy()
            valid["distance"] = np.abs(np.log2(valid.dose_uM / 10.))
            closest = valid.groupby(["study", "key", "cell_id"], sort=True).distance.transform("min")
            valid = valid.loc[np.isclose(closest, valid.distance, rtol=0, atol=1e-12)]
        if not family.startswith("pan"):
            types = HEPATIC if family == "hepatic_near10" else {family.split("_")[0].upper()}
            valid = valid.loc[valid.cell_id.isin(types)]
        by_parent = defaultdict(list)
        for (_, parent), study in valid.groupby(["study", "key"], sort=True):
            line = [patterns[part.vec].mean(axis=0) for _, part in study.groupby("cell_id", sort=True)]
            by_parent[parent].append(np.mean(line, axis=0))
        vectors = np.stack([np.mean(by_parent[key], axis=0) if key in by_parent else
                            np.zeros(responses.shape[1], dtype=np.float32) for key in keys])
        result[family] = unit_norm(vectors)
        support[family] = np.any(result[family] != 0, axis=1)
    return result, support


def build(args):
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("P1–P3 protocol differs from the prespecified bytes")
    target_path = {"entities": args.tahoe_entities, "split": args.tahoe_split, "heparg": args.heparg}
    targets = {"tahoe": catalogue(tahoe_targets(args.tahoe_entities, args.tahoe_split)),
               "heparg": catalogue(read_table(args.heparg))}
    args.out.mkdir(parents=True, exist_ok=False)
    all_values, all_meta, all_audit = [], [], []
    references = {}
    gene_axis = None
    for study in STUDIES:
        assets = asset_paths(args.lincs / study, study)
        refs = read_table(assets["pert"]).loc[lambda x: x.pert_type.eq("trt_cp")]
        by_meta = {code: collect_identity(sub) for code, sub in refs.groupby("pert_id", sort=True)}
        metadata, genes = study_metadata(assets)
        if gene_axis is None:
            gene_axis = genes
        elif not gene_axis.equals(genes):
            raise ValueError("Original LINCS landmark axes differ")
        known_ids = set()
        for cohort, target in targets.items():
            matched, audit = match_study(target, by_meta, study, cohort)
            known_ids.update(matched)
            all_audit.extend(audit)
        metadata = metadata.loc[metadata.pert_id.isin(known_ids)].reset_index(drop=True)
        metadata["study"] = study
        if not len(metadata):
            raise ValueError(f"No matched signature for {study}")
        print(f"{study}: {len(known_ids)} chemical IDs, {len(metadata)} qualifying 24 h profiles", flush=True)
        values = read_values(assets["gctx"], metadata, gene_axis.index)
        all_values.append(values)
        all_meta.append(metadata[["study", "sig_id", "pert_id", "cell_id", "dose_uM"]])
        references[study] = {tag: {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
                             for tag, path in assets.items()}
    metadata = pd.concat(all_meta, ignore_index=True)
    values = np.concatenate(all_values, axis=0)
    audit = pd.DataFrame(all_audit, columns=("cohort", "study", "pert_id", "parent_key", "method", "decision", "flags"))
    audit.to_csv(args.out / "match_audit.tsv", sep="\t", index=False)
    metadata.to_csv(args.out / "profile_metadata.tsv", sep="\t", index=False)
    counts = {}
    for cohort, target in targets.items():
        keys = sorted(target, key=int if cohort == "tahoe" else str)
        rows = audit.loc[audit.cohort.eq(cohort) & audit.decision.eq("matched")]
        maps = rows.set_index(["study", "pert_id"]).parent_key
        if not maps.index.is_unique:
            raise ValueError("A study LINCS ID was assigned to multiple parent groups")
        dataset = metadata.copy()
        dataset["link"] = list(zip(dataset.study, dataset.pert_id))
        vec, support = consensus(dataset.rename(columns={"pert_id": "code", "link": "pert_id"}),
                                 values, maps.to_dict(), keys)
        np.savez_compressed(args.out / f"lincs_{cohort}.npz",
                            keys=np.asarray(keys, dtype=str),
                            entrez=gene_axis.index.to_numpy(dtype=np.int64),
                            symbols=gene_axis.to_numpy(dtype=str), **vec,
                            **{f"has_{name}": flag for name, flag in support.items()})
        cov = pd.DataFrame({"group" if cohort == "tahoe" else "parent_key": keys})
        cov["matched_BRIDs"] = cov.iloc[:, 0].map(rows.groupby("parent_key").pert_id.agg(
            lambda x: "|".join(sorted(set(x))))).fillna("")
        cov["match_methods"] = cov.iloc[:, 0].map(rows.groupby("parent_key").method.agg(
            lambda x: "|".join(sorted(set(x))))).fillna("")
        for name, flag in support.items():
            cov[f"has_{name}"] = flag
        cov.to_csv(args.out / f"{cohort}_coverage.tsv", sep="\t", index=False)
        counts[cohort] = {name: int(flag.sum()) for name, flag in support.items()}
        print(cohort, counts[cohort], flush=True)
    write_json(args.out / "receipt.json", {
        "protocol_sha256": PROTOCOL_SHA256, "n_genes": len(gene_axis), "prior_counts": counts,
        "source_files": references, "target_inputs": {k: sha256(p) for k, p in target_path.items()},
        "code_sha256": sha256(__file__),
        "files": {p.name: sha256(p) for p in sorted(args.out.iterdir())},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("heparg", "tahoe-entities", "tahoe-split", "lincs", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
