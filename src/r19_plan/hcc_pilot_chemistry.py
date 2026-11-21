"""Audit and draft a metadata-only chemistry slate for the adult HCC pilot."""

import argparse
from collections import defaultdict
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.goal_tracks.t1_score import registry
from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP
from src.r13_optimization.lincs_transfer import aliases, clean, identifiers
from src.r15_lincs.data import STUDIES, asset, table
from src.r19_plan.hcc_pilot_layout import SLOTS, verify
from src.r19_plan.i_audit import base_brd
from src.r19_plan.r18_preflight import checked_tables


PHASES = ("Launched", "Phase 3", "Phase 2")
BANDS = ("below_0.35", "0.35_to_below_0.55", "0.55_or_more")
SEED = 191496


def rank_key(parent):
    return hashlib.sha256(f"{SEED}|{parent}".encode()).hexdigest()


def aggregate_keys(frame, name, smiles, broad=None, inchis=None, cid=None, extra=None):
    keys = set()
    fields = [col for col in (name, smiles, broad, inchis, cid, extra) if col is not None]
    for record in frame[fields].drop_duplicates().to_dict("records"):
        titles = [record.get(name, "")]
        if extra:
            titles.extend(aliases(record[extra]))
        raw_id = base_brd(record.get(broad, ""))
        relevant, _ = identifiers(titles, record.get(smiles, ""), record.get(inchis, ""), record.get(cid, ""))
        if raw_id:
            relevant.add(("brd", raw_id))
        keys.update(relevant)
    return keys


def read_tahoe(path):
    meta = pd.read_parquet(path)
    structures = {"canonical_smiles", "canonical-smiles"} & set(meta.columns)
    if len(structures) != 1:
        raise ValueError(f"Expected a single drug structure column in {path}")
    cid = next((column for column in ("pubchem_cid", "pubchem-cid") if column in meta), None)
    return aggregate_keys(meta, "drug", structures.pop(), cid=cid)


def exposures(args):
    fit_root = args.base/"vcell_liver_goal_tracks/v1/t1-cache"
    raw = args.base/"vcell_liver_r14_cross_organ"
    inputs, sets = {}, {}

    for study in STUDIES:
        path = asset(args.lincs, study, "pert_info")
        df = table(path)
        df = df.loc[df.pert_type.eq("trt_cp")]
        inputs[study] = path
        cid = "pubchem_cid" if "pubchem_cid" in df else None
        sets[study] = aggregate_keys(df, "pert_iname", "canonical_smiles", "pert_id", "inchi_key", cid)
    inputs["CMap2020"] = args.lincs/"LINCS2020/compoundinfo_beta.txt"
    sets["CMap2020"] = aggregate_keys(table(inputs["CMap2020"]), "cmap_name", "canonical_smiles", "pert_id",
                                     "inchi_key", extra="compound_aliases")
    for project, root in (("Tahoe100M", args.tahoe), ("EmeraldBay", args.emerald)):
        inputs[project] = root/"metadata/drug_metadata.parquet"
        sets[project] = read_tahoe(inputs[project])
    inputs["SciPlex3"] = raw/"metadata/v4/sciplex3_wells_24h.tsv"
    sets["SciPlex3"] = aggregate_keys(table(inputs["SciPlex3"]), "name", "SMILES")
    inputs["HepaRG"] = raw/"execution-v1/data/heparg_compound_folds.tsv"
    sets["HepaRG"] = aggregate_keys(table(inputs["HepaRG"]), "chemical", "smiles")
    inputs["MIX-seq"] = raw/"metadata/v4/mix_seq_liver_assignments.tsv"
    mixes = table(inputs["MIX-seq"]).archive.str.replace(r"_expt\d+.*$", "", regex=True)
    sets["MIX-seq"] = aggregate_keys(pd.DataFrame({"drug": mixes}), "drug", None)

    checked_tables(args.r18, args.archive/"r19_plan/data/r18-support-receipt.json")
    for name, file in (("R18_reserved", "r18_reserved_manifest.tsv"), ("R18_donor", "r18_B_donor_exceptions.tsv")):
        inputs[name] = args.r18/file
        scoped = table(inputs[name], usecols=["pert_id", "parent"])
        sets[name] = aggregate_keys(scoped, None, "parent", "pert_id")
    axis = fit_root/"split.json"
    if sha256(axis) != read_json(fit_root/"receipt.json")["files"]["split.json"]:
        raise ValueError("P4 chemical registry changed")
    inputs["P4_split"] = axis
    inputs["P4_entities"] = args.base/"vcell_liver_r11_chemical_probe/assets/features/entities.tsv"
    pinned = read_json(args.archive/"r11_chemical_probe/manifest/feature_receipt.json")["outputs"]["entities.tsv"]
    if sha256(inputs["P4_entities"]) != pinned:
        raise ValueError("P4 structure support changed")
    pool = set(registry(fit_root))
    found = pd.read_csv(inputs["P4_entities"], sep="\t")
    reference = found.loc[found[GROUP].isin(pool)].sort_values([GROUP, ENTITY]).drop_duplicates(GROUP)
    if len(reference) != 203 or set(reference[GROUP]) != pool:
        raise ValueError("The 203-parent P4 structure reference changed")
    return sets, reference, {name: dict(file=str(path), sha256=sha256(path)) for name, path in inputs.items()}


def source_candidates(path):
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = raw[["broad_id", "name", "smiles", "phase"]].drop_duplicates().rename(columns={"name": "drug"})
    frame["brd"] = frame.broad_id.map(base_brd)
    frame = frame.loc[frame.brd.ne("")]
    with rdBase.BlockLogs():
        structures = {s:canonical_parent(s) for s in set(frame.smiles) if clean(s)}
    frame["isomer"] = frame.smiles.map({k:v[0] for k,v in structures.items() if v})
    frame = frame.loc[frame.isomer.notna()].copy()
    frame["parent"] = frame.smiles.map(lambda s: identifiers([],s)[1])
    frame = frame.loc[frame.parent.ne("")]
    conflicting = (frame.groupby("brd").parent.transform("nunique").gt(1)
                   | frame.groupby("parent").isomer.transform("nunique").gt(1))
    dropped = frame.loc[conflicting, "parent"].nunique()
    frame = frame.loc[~conflicting].copy()
    matches = frame.groupby("parent").agg(aliases=("drug", lambda s: tuple(sorted(set(s)))),
                                           brd_aliases=("brd", lambda s: tuple(sorted(set(s)))))
    preferred = frame.phase.map({phase:i for i,phase in enumerate(PHASES)}).fillna(len(PHASES))
    frame = frame.assign(priority=preferred).sort_values(["priority","brd","drug","broad_id"]).drop_duplicates("parent")
    return frame.drop(columns="priority").join(matches,on="parent"), dict(
        source_records=len(raw), usable_parents=len(frame), ambiguous_structure_parents=int(dropped))


def annotate(frame, exposure, reference):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in reference.smiles]
    trained_scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in reference.smiles} - {""}
    recognized = []
    for row in frame.itertuples(index=False):
        keys, parent = identifiers(row.aliases,row.smiles)
        keys |= {("brd", value) for value in row.brd_aliases}
        blocked = sorted(name for name, seen in exposure.items() if keys & seen)
        mol = Chem.MolFromSmiles(parent)
        distances = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), fps)
        context = int(np.argmax(distances))
        similarity = distances[context]
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        recognized.append(dict(parent=parent, brd_id=row.broad_id, drug=row.drug, phase=row.phase,
                               raw_smiles=row.smiles, isomeric_parent=row.isomer, exclusions="|".join(blocked),
                               eligible=not blocked and row.phase in PHASES, accessible_catalog=not blocked,
                               max_tanimoto=similarity, closest_P4_group=int(reference.iloc[context][GROUP]),
                               scaffold=scaffold, P4_scaffold_seen=scaffold in trained_scaffolds,
                               similarity_band=BANDS[int(similarity >= .35)+int(similarity >= .55)], order=rank_key(parent)))
    return pd.DataFrame(recognized).sort_values("parent").reset_index(drop=True)


def choose(table):
    pool = table.loc[table.eligible].sort_values("order")
    candidates, seen = {}, set()

    def add(rows, size):
        accepted = 0
        for idx, row in rows.iterrows():
            key = row.scaffold or row.parent
            if key in seen:
                continue
            candidates[idx] = row
            seen.add(key)
            accepted += 1
            if accepted == size or len(candidates) == len(SLOTS):
                break

    for band in reversed(BANDS):
        add(pool.loc[pool.similarity_band.eq(band)], 32)
    if len(candidates) < len(SLOTS):
        for band in reversed(BANDS):
            if len(candidates) == len(SLOTS):
                break
            add(pool.loc[pool.similarity_band.eq(band)], len(SLOTS)-len(candidates))
    choices = table.loc[list(candidates)].copy()
    order = choices.parent.map(lambda s: rank_key("slots:"+s)).sort_values().index
    slots = choices.loc[order].assign(parent_slot=SLOTS[:len(order)])
    return slots.drop(columns="order")


def run(args):
    stage = read_json(args.archive/"r19_plan/data/i_views/public_source_manifest.json")
    metadata = next(row for row in stage if row["path"]=="prism/primary-screen-replicate-collapsed-treatment-info.csv")
    if sha256(args.prism)!=metadata["sha256"]:
        raise ValueError("PRISM treatment metadata differ from the approved I4 catalog")
    original = read_json(args.layout.parent/"receipt.json")
    if sha256(args.layout)!=original["files"][args.layout.name]:
        raise ValueError("The original anonymous pilot layout changed")
    layout = pd.read_csv(args.layout,sep="\t")
    verify(layout)
    exclusion, reference, inputs = exposures(args)
    frame, diagnostics = source_candidates(args.prism)
    audit = annotate(frame, exclusion, reference)
    slots = choose(audit)
    assigned = layout.loc[layout.kind.eq("treated")].merge(slots[["parent_slot","parent"]], on="parent_slot",
                                                           how="inner", validate="many_to_one")
    if (len(assigned)!=6*len(slots) or not assigned.groupby("parent").well.nunique().eq(6).all()
            or not assigned.groupby(["cell","parent"]).culture_block_slot.nunique().eq(3).all()):
        raise ValueError("The draft fails the original independent repeat/spatial layout")
    summary = audit.groupby(["phase","similarity_band"]).agg(
        registry=("parent","size"), never_in_joined_catalogs=("accessible_catalog","sum"), eligible=("eligible","sum")).reset_index()
    sizes = (audit.exclusions.str.split("|").explode().loc[lambda s:s.ne("")].value_counts()
             .rename_axis("source").rename("matched_prism_parent").reset_index())
    with TemporaryDirectory(dir=args.out.parent) as temp:
        output = Path(temp)
        for name, frame in (("candidate_audit",audit.drop(columns="order")),("draft_slots",slots),("support",summary),
                            ("exclusion_counts",sizes),("draft_treatment_coordinates",assigned)):
            frame.to_csv(output/(name+".tsv"),sep="\t",index=False,na_rep="NA")
        write_json(output/"receipt.json",dict(metadata_only=True, expression_read=False, measured=False, acquired_compounds=False,
            source_prism=metadata, exclusion_sources=inputs, protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            layout_sha256=sha256(args.layout), seed=SEED, selected=len(slots), selection_diagnostic=diagnostics,
            files={p.name:sha256(p) for p in output.iterdir()}))
        if args.out.exists():
            raise FileExistsError(args.out)
        output.rename(args.out)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("base","archive","lincs","tahoe","emerald","r18","prism","layout","protocol","out"):
        cli.add_argument("--"+name,type=Path,required=True)
    run(cli.parse_args())
