"""Bind the new CMap state development data to the audited physical-instance allowlist."""

import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.r18_input_bundle import extract_matrix
from src.r19_plan.r18_matrix_index import metadata_index

ADULT = {"JHH5": "CVCL_0364", "JHH7": "CVCL_2805"}
DONORS = {
    "A375": "CVCL_0132", "A549": "CVCL_0023", "HT29": "CVCL_0320",
    "MCF7": "CVCL_0031", "PC3": "CVCL_0035", "U2OS": "CVCL_0042",
}
CONTEXT = ["cell_id", "rna_plate"]
DEFAULT_PROFILE = ["SequencingID", "ModelID", "ModelConditionID", "IsDefaultEntryForModel"]


def table(path, **kwargs):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def pinned(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"State source hash changed: {path}")
    return path


def unique_wells(rows):
    if (rows.empty or rows.inst_id.duplicated().any() or rows.duplicated(["rna_plate", "rna_well"]).any()
            or not rows.rna_well.str.fullmatch(r"[A-Z]+[0-9]{2}").all()):
        raise ValueError("The state cohort has a missing or ambiguous physical well")
    return rows


def exact_structure(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "", ""
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if sum(any(atom.GetAtomicNum() == 6 for atom in piece.GetAtoms()) for piece in fragments) != 1:
            return "", ""
        parent = rdMolStandardize.ChargeParent(mol)
        compound = Chem.MolToSmiles(parent, isomericSmiles=True)
        key = Chem.MolToInchiKey(parent)
    return (compound, key.split("-")[0]) if key else ("", "")


def chemistry(path, chosen):
    catalog = table(path, usecols=["pert_id", "canonical_smiles"])
    catalog = catalog.loc[catalog.pert_id.isin(chosen)].drop_duplicates()
    pairs = {s: exact_structure(s) for s in catalog.canonical_smiles.unique()}
    cols = ["smiles", "connectivity_group"]
    catalog[cols] = pd.DataFrame(catalog.canonical_smiles.map(pairs).tolist(), index=catalog.index)
    catalog = catalog.loc[catalog.smiles.ne(""), ["pert_id", *cols]].drop_duplicates()
    catalog = catalog.loc[~catalog.pert_id.duplicated(keep=False)].copy()
    catalog["compound_id"] = catalog.smiles
    return catalog


def default(values):
    return values.str.lower().isin(["true", "yes", "1"])


def basal_identity(depot):
    models = pd.read_csv(depot / "Model.csv", dtype=str, keep_default_na=False)
    profiles = pd.read_csv(depot / "OmicsProfiles.csv", dtype=str, keep_default_na=False)
    profiles = profiles.loc[
        profiles.DataType.eq("rna") & ~default(profiles.Stranded) & default(profiles.IsDefaultEntryForModel),
        DEFAULT_PROFILE,
    ].drop_duplicates()
    allowed = ADULT | DONORS
    models = models.loc[models.RRID.isin(allowed.values())]
    models = models.loc[models.StrippedCellLineName.isin(allowed)]
    if models.RRID.duplicated().any() or dict(zip(models.StrippedCellLineName, models.RRID)) != allowed:
        raise ValueError("Ambiguous State cell identity in DepMap")
    if not models.loc[models.RRID.isin(ADULT.values()), "OncotreePrimaryDisease"].eq("Hepatocellular Carcinoma").all():
        raise ValueError("An adult HCC group's cell classification changed")
    joined = models[["StrippedCellLineName", "RRID", "ModelID", "OncotreePrimaryDisease"]].merge(
        profiles, on="ModelID", validate="one_to_many",
    )
    joined = joined.rename(columns={"StrippedCellLineName": "cell_id"})
    if len(joined) != len(allowed) or not joined.cell_id.is_unique or joined.SequencingID.duplicated().any():
        raise ValueError("Expected an unambiguous default DepMap profile for every CMap state line")
    return joined


def basal_expression(path, basal, genes):
    columns_by_gene = {}
    header = pd.read_csv(path, nrows=0).columns
    gene_counts = Counter(int(m[1]) for col in header if (m := re.search(r"\(([0-9]+)\)$", col)))
    columns_by_gene.update({
        int(m[1]): col for col in header if (m := re.search(r"\(([0-9]+)\)$", col))
        and gene_counts[int(m[1])] == 1 and int(m[1]) in set(genes)
    })
    if len(columns_by_gene) != len(genes):
        raise ValueError("DepMap basal proxy has missing or ambiguous landmark genes")
    columns = [columns_by_gene[g] for g in genes]
    chunks = (chunk.loc[chunk.SequencingID.isin(basal.SequencingID)] for chunk in pd.read_csv(
        path, usecols=[*DEFAULT_PROFILE, *columns], chunksize=64, dtype={c: str for c in DEFAULT_PROFILE},
    ))
    hit = pd.concat(chunks).set_index("SequencingID", verify_integrity=True).loc[basal.SequencingID]
    if (not np.array_equal(basal.ModelID, hit.ModelID)
            or not np.array_equal(basal.ModelConditionID, hit.ModelConditionID)
            or not default(hit.IsDefaultEntryForModel).all()):
        raise ValueError("State basal sequencing rows have unexpected metadata")
    values = hit[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite state basal feature")
    return pd.DataFrame(values, index=pd.Index(basal.cell_id, name="cell_id"), columns=genes.astype(str))


def audited_metadata(args):
    input_receipt = pinned(args.audit / "receipt.json", sha256(args.audit_reference))
    receipt = read_json(input_receipt)
    if not receipt["metadata_only"] or receipt["expression_read"] or not receipt["registered_control_block"]:
        raise ValueError("The CMap source must be pre-screened against the registered protected domains")
    allowed = {
        name: table(pinned(args.audit / f"{name}.tsv", receipt["files"][f"{name}.tsv"]))
        for name in ("targets", "donors", "controls")
    }
    targets = allowed["targets"]
    if set(targets.cell_id) != set(ADULT) or not targets.b1_seen.eq("True").all():
        raise ValueError("CMap target support or prior exposure differs from the registry")
    donors = allowed["donors"]
    donors = donors.loc[donors.view.eq("nonhepatic") & donors.cell_id.isin(DONORS)]
    rows = pd.concat((targets, donors), ignore_index=True)
    if set(rows.study) != {"CMap2020"} or not rows.actual_dose_uM.eq("10").all():
        raise ValueError("State adapter received records outside the fixed CMap 24 h / 10 µM scope")
    chem_path = pinned(args.chemistry, receipt["source_metadata_sha256"]["identity"])
    rows = rows.merge(chemistry(chem_path, rows.pert_id), on="pert_id", validate="many_to_one")
    targets = targets.groupby("cell_id").size().to_dict()
    identity = basal_identity(args.depmap)
    rows = rows.merge(identity[["cell_id", "RRID", "SequencingID"]], on="cell_id", validate="many_to_one")
    if set(rows.cell_id) != set(ADULT | DONORS):
        raise ValueError("The audited state inputs lost an entire declared cell context")
    controls = allowed["controls"].merge(rows[CONTEXT].drop_duplicates(), on=CONTEXT, validate="many_to_one")
    unique_wells(pd.concat((rows, controls)))
    used = pd.MultiIndex.from_frame(rows[CONTEXT])
    counts = controls.groupby(CONTEXT).size().reindex(used)
    if counts.hasnans or counts.min() < 3:
        raise ValueError("State adapter expected at least three independently registered physical DMSO wells per plate")
    rows = rows.assign(
        lineage_group=rows.RRID, basal_rna_id=rows.SequencingID, dose_uM=10., time_h=24,
        plate_id=rows.rna_plate, position=rows.rna_well, sample_id=rows.inst_id, baseline_is_proxy=True,
        context_role=np.where(rows.cell_id.isin(ADULT), "adult_hcc", "nonhepatic_auxiliary"),
    )
    fields = ["study", "sample_id", "inst_id", "cell_id", "lineage_group", "basal_rna_id", "baseline_is_proxy",
              "context_role", "pert_id", "compound_id", "smiles", "connectivity_group", "dose_uM", "time_h",
              "plate_id", "position", "rna_plate", "rna_well", "det_plate", "det_well"]
    return rows[fields], controls, identity, targets


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rows, controls, identity, sizes = audited_metadata(args)
    source = read_json(args.matrix_reference)
    if (source["study"] != "CMap2020" or not source["metadata_only"] or source["expression_read"]
            or sha256(args.landmarks) != source["landmark_axis_sha256"]):
        raise ValueError("CMap Level 3 original matrix axes or landmark axis changed")
    genes = pd.read_csv(args.landmarks, sep="\t").pr_gene_id.to_numpy(dtype=np.int64)
    arrays = {}
    for kind, members in (("treated", rows), ("controls", controls)):
        record = source["matrix_records"][kind]
        indexed, axis, checked = metadata_index(Path(record["path"]), members, genes)
        if checked != record:
            raise ValueError("Source CMap coordinates changed")
        indexed.to_csv(args.out / f"{kind}.tsv", sep="\t", index=False)
        arrays[kind] = axis.level3_gene_index.tolist()

    identity.to_csv(args.out / "basal_identity.tsv", sep="\t", index=False)
    basal_path = args.depmap / "OmicsExpressionTPMLogp1HumanAllGenes.csv"
    basal_expression(basal_path, identity, genes).to_csv(args.out / "basal.tsv", sep="\t")
    pd.DataFrame({"gene_id": genes}).to_csv(args.out / "genes.tsv", sep="\t", index=False)
    stats = rows.groupby(["context_role", "cell_id"]).agg(
        wells=("inst_id", "size"), plates=("plate_id", "nunique"),
        exact_compounds=("compound_id", "nunique"), connectivity_groups=("connectivity_group", "nunique"),
    ).reset_index()
    stats["raw_adult_manifest_wells"] = stats.cell_id.map(sizes).fillna(0).astype(int)
    stats.to_csv(args.out / "support.tsv", sep="\t", index=False)
    roles = rows.loc[rows.cell_id.isin(ADULT)].groupby("compound_id").cell_id.nunique()
    write_json(args.out / "receipt.json", {
        "upstream_audit_sha256": sha256(args.audit_reference), "matrix_reference_sha256": sha256(args.matrix_reference),
        "protocol_sha256": sha256(args.protocol), "code_sha256": sha256(__file__),
        "sources": {str(p): sha256(p) for p in (args.chemistry, args.landmarks, basal_path,
                    args.depmap / "Model.csv", args.depmap / "OmicsProfiles.csv")},
        "treated_or_vehicle_expression_read": False, "forbidden_coordinates_excluded_by_pinned_audit": True,
        "shared_adult_exact_compounds": int(roles.eq(len(ADULT)).sum()),
        "matrix_records": source["matrix_records"], "indices": arrays,
        "basal_units": "log2(TPM+1)", "delta_units": "L1000 Level 3 landmark expression units",
        "source_version": "CMap 2020 Beta and DepMap Public 25Q3",
        "files": {f.name: sha256(f) for f in sorted(args.out.iterdir())},
    })


def open_scope(root):
    stamp = read_json(root / "receipt.json")
    for name, digest in stamp["files"].items():
        pinned(root / name, digest)
    if stamp["treated_or_vehicle_expression_read"] or not stamp["forbidden_coordinates_excluded_by_pinned_audit"]:
        raise ValueError("Can only use state metadata pinned before reading expression")
    return (stamp, table(root / "treated.tsv"), table(root / "controls.tsv"),
            table(root / "basal.tsv").set_index("cell_id").astype(float),
            table(root / "genes.tsv").gene_id.to_numpy(dtype=np.int64))


def response(rows, controls, genes, stamp):
    """Extract only the registered response wells and matching vehicle, in input order."""
    control = controls.merge(rows[CONTEXT].drop_duplicates(), on=CONTEXT, validate="many_to_one")
    unique_wells(pd.concat((rows, control)))
    read = {
        kind: extract_matrix(stamp["matrix_records"][kind], members, np.array(stamp["indices"][kind]), genes)
        for kind, members in (("treated", rows), ("controls", control))
    }
    vehicle = pd.DataFrame(read["controls"], index=pd.MultiIndex.from_frame(control[CONTEXT]))
    averages = vehicle.groupby(level=[0, 1]).mean().reindex(pd.MultiIndex.from_frame(rows[CONTEXT]))
    delta = read["treated"] - averages.to_numpy()
    if len(delta) != len(rows) or not np.isfinite(delta).all():
        raise ValueError("Absent vehicle or invalid L1000 landmark contrast")
    return delta.astype(np.float32), control


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "audit-reference", "matrix-reference", "chemistry", "landmarks", "depmap", "protocol", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())
