"""Audit the previous LINCS inputs using registered metadata, without expression."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys

from .data import LIVER, STUDIES, hepatic_lines, instances, signatures, table


def audit(args):
    base, root = args.base, args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    b1 = base / "vcell_liver_transigen_adaptation_b1/assets"
    processed = b1 / "processed/cmap2020_v1"
    sources = {
        "B1_rows": processed / "rows.tsv", "B1_drugs": processed / "drugs.tsv",
        "B1_instances": b1 / "raw/cmap2020/instinfo_beta.txt",
        "R8_units": base / "vcell_liver_replicate_calibrated_r8/assets/processed/store_v1/units.tsv",
        "R13_identity": base / "vcell-r13-lincs-cid-study.7NGzdyc8/donor/identity.tsv",
        "R13_receipt": base / "vcell-r13-lincs-cid-study.7NGzdyc8/donor/signature_receipt.json",
        "P23_match": base / "vcell_liver_p123/v1/lincs/match_audit.tsv",
        "P23_signatures": base / "vcell_liver_p123/v1/lincs/profile_metadata.tsv",
    }
    manifest = read_json(processed / "build_manifest.json")
    for key, filename in (("B1_rows", "rows.tsv"), ("B1_drugs", "drugs.tsv")):
        if sha256(sources[key]) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"The B1 data contract changed for {key}")
    b = table(sources["B1_rows"])
    identity = table(sources["B1_drugs"]).set_index("pert_id")
    derived = identity.smiles_used.map(lambda v: structural_keys(clean(v))[1])
    b["parent"] = b.pert_id.map(derived)
    b1_groups = b[["cell", "drug_group", "pert_id", "det_plate", "n_trt_wells", "parent", "role"]]
    cols = ["pert_id", "cell_iname", "det_plate", "det_well", "rna_plate", "rna_well", "sample_id", "pert_type",
            "nearest_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "qc_pass"]
    raw = table(sources["B1_instances"], usecols=cols)
    raw = raw.loc[raw.pert_type.eq("trt_cp") & pd.to_numeric(raw.pert_time, errors="coerce").eq(24) & raw.pert_time_unit.eq("h") &
                  pd.to_numeric(raw.nearest_dose, errors="coerce").eq(10) & raw.pert_dose_unit.eq("uM") & raw.qc_pass.eq("1")]
    raw = raw.rename(columns={"cell_iname": "cell"}).merge(
        b1_groups, on=["cell", "pert_id", "det_plate"], validate="many_to_one")
    if len(raw) != b.n_trt_wells.astype(int).sum():
        raise ValueError("Reconstructed B1 wells do not match its archived row counts")
    r8 = table(sources["R8_units"])
    held = r8[["cell", "drug_group", "plates"]].assign(det_plate=lambda x: x.plates.str.split("|")).explode("det_plate")
    keep8 = raw.merge(held.drop(columns="plates"), on=["cell", "drug_group", "det_plate"], validate="many_to_one")
    keep8 = pd.concat((keep8, raw.loc[raw.role.eq("target")])).drop_duplicates("sample_id")
    r13 = table(sources["R13_identity"])
    keep13 = raw.loc[raw.role.eq("source") & raw.pert_id.isin(r13.loc[r13.retained.eq("True"),"pert_id"])]
    ids_and_cells = {"B1": (raw, b), "R8": (keep8, r8), "R13": (keep13, b.loc[b.role.eq("source") & b.pert_id.isin(keep13.pert_id)])}

    match = table(sources["P23_match"])
    monitored = table(sources["P23_signatures"])
    matched = match.loc[match.decision.eq("matched"), ["study", "pert_id", "cohort"]].drop_duplicates()
    monitored = monitored.merge(matched, on=["study", "pert_id"], validate="many_to_many")
    excluded = set.union(*(hepatic_lines(root, s) for s in STUDIES))
    counts, overlaps = [], []
    for name, (inst_used, table_used) in ids_and_cells.items():
        for scope, allowed in (("HEPG2", inst_used.cell.eq("HEPG2")), ("HUH7", inst_used.cell.eq("HUH7")),
                               ("PHH", inst_used.cell.eq("PHH")), ("non_hepatic", ~inst_used.cell.isin(excluded))):
            subset = inst_used.loc[allowed]
            counts.append(dict(round=name, scope=scope, measured=len(subset), ids=subset.pert_id.nunique(),
                               parents=subset.parent.nunique(), lines=subset.cell.nunique(), unit="Level3_instance"))
    for name, key in (("P2", "tahoe"), ("P3", "heparg")):
        used = monitored.loc[monitored.cohort.eq(key)]
        for scope, allowed in (("HEPG2", used.cell_id.eq("HEPG2")), ("HUH7", used.cell_id.eq("HUH7")),
                               ("PHH", used.cell_id.eq("PHH")), ("non_hepatic", ~used.cell_id.isin(excluded))):
            subset = used.loc[allowed]
            counts.append(dict(round=name, scope=scope, measured=len(subset), ids=subset.pert_id.nunique(),
                               parents="", lines=subset.cell_id.nunique(), unit="Level5_signature_bank"))

    union = []
    for study in STUDIES:
        sig = signatures(root, study)
        chem_path = args.position/study/"identities.tsv"
        stamp = read_json(chem_path.parent/"receipt.json")
        if sha256(chem_path) != stamp["files"]["identities.tsv"]:
            raise ValueError("Position audit identity catalogue changed")
        chem = table(chem_path)
        sig["parent"] = sig.pert_id.map(chem.set_index("pert_id").parent).fillna(sig.pert_id)
        coords = instances(root, study)
        coords = coords.loc[coords.pert_type.eq("trt_cp") & coords.cell_id.isin(LIVER)]
        tree = sig.loc[sig.pert_type.eq("trt_cp") & sig.cell_id.isin(LIVER)].copy()
        keyed = coords.rename(columns={"cell_id": "cell"})[["inst_id","rna_plate","rna_well","cell","pert_id"]]
        positive_keys = ["rna_plate", "rna_well", "cell", "pert_id"]
        signature_instances = tree[["sig_id", "distil_id"]].assign(inst_id=lambda x: x.distil_id.str.split("|")).explode("inst_id")
        for name, (used, _) in ids_and_cells.items():
            alias = used if study == "GSE92742" else used.rename(
                columns={"rna_plate":"source_rna_plate","rna_well":"source_rna_well","det_plate":"rna_plate","det_well":"rna_well"})
            shared = keyed.merge(alias[positive_keys].drop_duplicates(), on=positive_keys, validate="many_to_one")
            exact = signature_instances.loc[signature_instances.inst_id.isin(set(shared.inst_id)|set(used.sample_id)),"sig_id"]
            tree[f"{name}_exact"] = tree.sig_id.isin(exact)
            tree[f"{name}_hepatic24_parent"] = tree.parent.isin(used.loc[used.cell.isin(LIVER),"parent"])
            tree[f"{name}_nonhepatic24_parent"] = tree.parent.isin(used.loc[~used.cell.isin(excluded),"parent"])
        for name, key in (("P2", "tahoe"), ("P3", "heparg")):
            use = monitored.loc[monitored.study.eq(study) & monitored.cohort.eq(key)]
            map_parent = chem.set_index("pert_id").parent
            tree[f"{name}_exact"] = tree.sig_id.isin(use.sig_id)
            tree[f"{name}_hepatic24_parent"] = tree.parent.isin(use.loc[use.cell_id.isin(LIVER),"pert_id"].map(map_parent))
            tree[f"{name}_nonhepatic24_parent"] = tree.parent.isin(use.loc[~use.cell_id.isin(excluded),"pert_id"].map(map_parent))
        tree["study"] = study
        union.append(tree)
    labels = pd.concat(union, ignore_index=True)
    for (study, cell, hours), group in labels.loc[labels.hours.isin([6,24])].groupby(["study","cell_id","hours"]):
        response = group.loc[group.dose.between(5,20)]
        for name in ("B1", "R8", "R13", "P2", "P3"):
            exact = response.loc[response[f"{name}_exact"]]
            overlaps.append(dict(study=study, cell=cell, hours=hours, round=name, parents=response.parent.nunique(),
                                 exact_signatures=len(exact), exact_parents=exact.parent.nunique(),
                                 hepatic_parent_overlap=response.loc[response[f"{name}_hepatic24_parent"],"parent"].nunique(),
                                 nonhepatic_parent_overlap=response.loc[response[f"{name}_nonhepatic24_parent"],"parent"].nunique()))
    keys = ["study","sig_id","cell_id","hours","dose","pert_id","parent"]
    keys.extend(col for col in labels if col.endswith(("_exact", "_parent")))
    labels[keys].to_csv(out/"signature_exposure.tsv",sep="\t",index=False)
    pd.DataFrame(counts).to_csv(out/"scope.tsv",sep="\t",index=False)
    pd.DataFrame(overlaps).to_csv(out/"overlap.tsv",sep="\t",index=False)
    write_json(out/"receipt.json", dict(code_sha256=sha256(__file__),
        sources={name:dict(path=str(p),sha256=sha256(p)) for name,p in sources.items()},
        files={p.name:sha256(p) for p in sorted(out.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root","base","out","position"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    audit(parser.parse_args())
