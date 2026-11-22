"""Lock exact dose/time pairs and parent-and-plate held LINCS comparisons."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import asset
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import ROOT
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_r18_domains, verify_domain, write_plan
from src.r19_plan.i_stage import parse_landmarks, pinned_bytes
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index
from src.r19_plan.t4_support import eligible_links, matched_pairs, measured_dose
from src.r19_plan.x_access import PHYSICAL, control_registry

STUDY = "GSE92742"
CELL = "HEPG2"
COHORTS = {CELL: (132, 247), "PHH": (67, 104)}
PHH_RANK = 16
PHH_RIDGE = 1.
CONDITION = ["parent", "actual_dose_uM"]
PREDICT = ("source6", "source24", "donor6", "direct24", "fit_controls", "select_controls", "query")
SCORE = ("query", "controls")


def condition_index(rows):
    return pd.MultiIndex.from_frame(rows[CONDITION])


def condition_filter(rows, wanted):
    return rows.loc[condition_index(rows).isin(condition_index(wanted))]


def assigned_pairs(rows):
    paired = matched_pairs(rows)
    paired = paired.loc[paired.cross_time_plate_disjoint & paired.both_times_2plates].copy()
    paired["target_plate"] = [min(plates, key=lambda plate: (crc32(f"T4|{p}|{d}|{plate}".encode()), plate))
                              for p, d, plates in paired[[*CONDITION, "plates_24h"]].itertuples(index=False, name=None)]
    return paired[[*CONDITION, "fold", "target_plate"]]


def scope(rows, pairs, controls, fold, plate, role, *, fixed_operator=False):
    role_index = "ABC".index(role)
    select_role, score_role = "ABC"[(role_index+1)%3], "ABC"[(role_index+2)%3]
    later, earlier = [rows.loc[rows.hours.eq(hour)] for hour in (24, 6)]
    held = pairs.loc[pairs.fold.eq(fold) & pairs.target_plate.eq(plate)]
    wanted = pairs.loc[pairs.fold.ne(fold)]
    source24 = condition_filter(later.loc[later.rna_plate.ne(plate)], wanted).copy()
    source6 = condition_filter(earlier, source24)
    nested = source24.parent.map(lambda p: not fixed_operator and fold_id(p, f"R15-inner-{fold}", 5) == 0)
    source24["phase"] = nested.map({True: "select", False: "inner_fit"})
    query = condition_filter(later.loc[later.rna_plate.eq(plate)], held)
    direct = condition_filter(later.loc[later.rna_plate.ne(plate)], held)
    donors = earlier.loc[earlier.parent.isin(held.parent)]
    inner_count, select_count = source24.loc[~nested].parent.nunique(), source24.loc[nested].parent.nunique()
    sufficient = inner_count > PHH_RANK if fixed_operator else inner_count > 4 and select_count >= 8
    if (not len(held) or len(query[CONDITION].drop_duplicates()) != len(held)
            or len(direct[CONDITION].drop_duplicates()) != len(held)
            or len(condition_filter(donors, held)[CONDITION].drop_duplicates()) != len(held)
            or set(source6.parent) & set(query.parent) or set(source24.parent) & set(query.parent)
            or not sufficient):
        raise ValueError("A T4 query has no independent plate donor or sufficient disjoint parent training support")

    def role_controls(wells, label):
        plates = set(wells.rna_plate)
        selected = controls.loc[controls.role.eq(label) & controls.rna_plate.isin(plates)]
        if set(selected.rna_plate) != plates:
            raise ValueError("T4 plate support does not contain the required independent role")
        return selected

    fit = role_controls(pd.concat((source6, source24, donors, direct), ignore_index=True), role)
    select = role_controls(source24.loc[nested], select_role)
    score = role_controls(query, score_role)
    if pd.concat((source6, source24, donors, direct, query, fit, select, score)).duplicated(PHYSICAL).any():
        raise ValueError("A T4 physical well crosses model, donor or outcome roles")
    return ({"predict": dict(source6=source6, source24=source24, donor6=donors, direct24=direct,
                             fit_controls=fit, select_controls=select, query=query.drop(columns="level3_row_index")),
             "score": dict(query=query, controls=score)},
            dict(fold=fold, plate=plate, role=role, parents=query.parent.nunique(), conditions=len(held),
                 query_wells=len(query), train_parents=source24.parent.nunique(),
                 inner_parents=inner_count, select_parents=select_count,
                 direct_wells=len(direct), dmsos_fit=len(fit), dmsos_select=len(select), dmsos_score=len(score)))


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    cell = args.cell
    support = read_json(args.support_reference)
    if (not support["metadata_only"] or support["expression_read"]
            or support["reservation_access_sha256"] != sha256(args.r18_access / "receipt.json")
            or any(sha256(path) != h for path, h in support["source_metadata_sha256"][STUDY].items())):
        raise ValueError("T4 source metadata no longer match the protected-input support audit")
    protected = read_r18_domains(args)[:4]
    footprint = pd.concat([provenance(table) for table in protected], ignore_index=True)
    inst, sig = geo_metadata(STUDY, args.base / "vcell_liver_r15_lincs/v1")
    inst = prepare(inst)
    candidates = inst.pert_type.eq("trt_cp") & inst.hours.isin((6, 24))
    dose = measured_dose(asset(ROOT, STUDY, "inst_info"), set(inst.loc[candidates, "inst_id"]))
    rows, _ = eligible_links(inst, sig, footprint, dose)
    rows = rows.loc[rows.cell_id.eq(cell)]
    paired = assigned_pairs(rows)
    parents, conditions = COHORTS[cell]
    if (paired.parent.nunique(), len(paired)) != (parents, conditions):
        raise ValueError(f"T4 did not reproduce the audited {cell} time pairs")
    paired_parents = set(paired.parent)
    retained = rows.loc[rows.parent.isin(paired_parents) & rows.hours.eq(6)]
    retained = pd.concat((retained, condition_filter(rows.loc[rows.hours.eq(24)], paired)), ignore_index=True)
    fields = ["inst_id", "sig_id", "parent", "cell_id", "hours", "actual_dose_uM", *PHYSICAL, "det_plate", "det_well"]
    retained = retained[fields].assign(study=STUDY).sort_values(["parent", "actual_dose_uM", "hours", *PHYSICAL])
    controls = control_registry(inst.loc[inst.cell_id.eq(cell)], retained, footprint)
    verify_domain(retained, controls, *protected)
    matrix = read_json(args.matrix_reference)
    if not matrix["metadata_only"] or matrix["expression_read"] or matrix["study"] != STUDY:
        raise ValueError("T4 requires the pinned Level 3 metadata axis")
    genes = parse_landmarks(pinned_bytes(args.landmarks, matrix["landmark_axis_sha256"]))
    expected = matrix["matrix_records"]["combined"]
    combined = pd.concat((retained, controls), ignore_index=True).fillna("")
    indexed, axis, record = metadata_index(Path(expected["path"]), combined, genes.pr_gene_id.to_numpy())
    if record != expected or indexed.level3_row_index.duplicated().any():
        raise ValueError("T4 does not match the registered physical Level 3 matrix")
    treatments = indexed.iloc[:len(retained)].drop(columns="role").copy()
    treatments["hours"] = pd.to_numeric(treatments.hours).astype(int)
    controls = indexed.iloc[len(retained):][["inst_id", "study", "cell_id", *PHYSICAL, "role", "level3_row_index"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".t4-lock-") as tmp:
        stage = Path(tmp)
        files, points = {}, []
        for fold, plate in sorted(set(paired[["fold", "target_plate"]].itertuples(index=False, name=None))):
            for role in "ABC":
                data, point = scope(treatments, paired, controls, fold, plate, role, fixed_operator=cell == "PHH")
                key = f"f{fold}-{role}-{plate}"
                files[key] = {phase: write_plan(stage / (phase+"-plan") / key,
                                               {name+".tsv": rows for name, rows in tables.items()})
                              for phase, tables in data.items()}
                points.append(dict(slice=key, **point))
        pd.DataFrame(points).to_csv(stage / "support.tsv", sep="\t", index=False)
        genes.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        axis.to_csv(stage / "gene_indices.tsv", sep="\t", index=False)
        extension = {"cell": cell, "fixed_rank": PHH_RANK, "fixed_ridge": PHH_RIDGE} if cell == "PHH" else {}
        write_json(stage / "receipt.json", dict(metadata_only=True, expression_read=False,
            parents=parents, conditions=conditions, matrix=record, permitted_datasets=[SAMPLE_IDS, GENE_IDS],
            support_sha256=sha256(args.support_reference), code_sha256=sha256(__file__), slices=files,
            files={f.name: sha256(f) for f in sorted(stage.glob("*.tsv"))}, **extension))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "support-reference",
                 "matrix-reference", "landmarks", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cell", choices=COHORTS, default=CELL)
    print(run(parser.parse_args()))
