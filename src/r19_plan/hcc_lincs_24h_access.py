"""Register independent JHH-5 prediction and score allowlists on existing CMap metadata."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.i_preflight import read_pinned, write_plan
from src.r19_plan.i_stage import parse_landmarks, pinned_bytes
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index
from src.r19_plan.x_inputs import check_unique, control_axis


CELL, VIEW = "JHH5", "nonhepatic"


def load_registry(args):
    if sha256(args.audit / "receipt.json") != sha256(args.audit_pin):
        raise ValueError("CMap target and donor audit receipt changed")
    stamp = read_json(args.audit_pin)
    if not stamp["metadata_only"] or stamp["expression_read"] or not stamp["registered_control_block"]:
        raise ValueError("CMap requires the independently audited metadata provenance")
    scope = {name: read_pinned(args.audit, name, digest) for name, digest in stamp["files"].items()}
    supports = scope["support.tsv"]
    support, = supports.loc[supports.cell.eq(CELL) & supports.donor.eq(VIEW)
                            & supports.scope.eq("donor_covered")].to_dict("records")
    if support["eligible"] != "True":
        raise ValueError("JHH-5 has insufficient independently supported exact-dose data")
    targets = scope["targets.tsv"].loc[lambda rows: rows.cell_id.eq(CELL)]
    source = scope["donors.tsv"].loc[lambda rows: rows.view.eq(VIEW) & rows.parent.isin(targets.parent)]
    donor_support = source.groupby("parent").agg(plates=("rna_plate", "nunique"), cells=("cell_id", "nunique"))
    allowed = donor_support.index[donor_support.plates.ge(2) & donor_support.cells.ge(2)]
    targets = targets.loc[targets.parent.isin(allowed)].copy()
    source = source.loc[source.parent.isin(targets.parent)]
    controls = scope["controls.tsv"].loc[lambda rows: rows.rna_plate.isin(set(source.rna_plate) | set(targets.rna_plate))]
    if (set(targets.rna_plate) & set(source.rna_plate) or targets.parent.nunique() != int(support["parents"])
            or len(targets) != int(support["wells"])):
        raise ValueError("Eligible JHH-5 cohort has a different plate separation or physical support")
    check_unique(pd.concat((targets, source, controls), ignore_index=True))
    return targets, source, controls, scope["splits.tsv"]


def assigned_controls(controls):
    ordered = controls.assign(crc=controls.inst_id.map(lambda s: crc32(s.encode()))).sort_values(["rna_plate", "crc", "inst_id"])
    role = ordered.groupby("rna_plate").cumcount().mod(3).map(dict(enumerate("ABC")))
    return ordered.drop(columns="crc").assign(role=role)


def scoped_splits(target, controls, split_rows):
    slate = split_rows.loc[split_rows.cell.eq(CELL) & split_rows.donor.eq(VIEW)].set_index(["fold", "score_plate"])
    for (fold, plate), query in target.groupby(["fold", "rna_plate"], sort=True):
        held = target.loc[target.fold.eq(fold)]
        fit = target.loc[target.fold.ne(fold) & target.rna_plate.ne(plate) & ~target.scaffold.isin(held.scaffold)]
        evidence = slate.loc[(str(fold), plate)]
        if (fit.parent.nunique() != int(evidence.fit_parents) or len(fit) != int(evidence.fit_wells)
                or len(query) != int(evidence.score_wells) or int(evidence.fit_parents) < 30):
            raise ValueError("Locked fit and query support differs from the metadata audit")
        for i, role in enumerate("ABC"):
            score_role = "ABC"[(i+2)%3]
            fit_controls = controls.loc[controls.role.eq(role) & controls.rna_plate.isin(fit.rna_plate)]
            score_controls = controls.loc[controls.role.eq(score_role) & controls.rna_plate.eq(plate)]
            check_unique(pd.concat((fit, query, fit_controls, score_controls), ignore_index=True))
            control_axis(fit_controls, fit, role)
            control_axis(score_controls, query, score_role)
            yield f"f{fold}-{role}-{plate}", dict(fit=fit, fit_controls=fit_controls,
                query=query.drop(columns="level3_row_index")), dict(query=query, score_controls=score_controls)


def run(args):
    targets, source, controls, splits = load_registry(args)
    matrix_receipt = read_json(args.cmap_pin)
    axis_hash = matrix_receipt["landmark_axis_sha256"]
    landmarks = parse_landmarks(pinned_bytes(args.landmarks, axis_hash))
    genes = landmarks.pr_gene_id.to_numpy()
    matrices = matrix_receipt["matrix_records"]
    treatment_block = pd.concat((targets, source), ignore_index=True)
    treated, trt_axis, trt_matrix = metadata_index(Path(matrices["treated"]["path"]), treatment_block, genes)
    controls, ctl_axis, ctl_matrix = metadata_index(Path(matrices["controls"]["path"]), controls, genes)
    if trt_matrix != matrices["treated"] or ctl_matrix != matrices["controls"]:
        raise ValueError("CMap source matrix metadata differs from the registered axes")
    targets = treated.iloc[:len(targets)].copy()
    source = treated.iloc[len(targets):].copy()
    controls = assigned_controls(controls)
    if args.out.exists():
        raise FileExistsError(args.out)
    with TemporaryDirectory(dir=args.out.parent, prefix=".hcc-24h-access-") as tmp:
        stage = Path(tmp)
        slices, support = {}, []
        for label, predict, score in scoped_splits(targets, controls, splits):
            slices[label] = {phase: write_plan(stage / (phase+"-plan") / label,
                {name+".tsv": rows for name, rows in tables.items()})
                for phase, tables in (("predict", predict), ("score", score))}
            support.append(dict(slice=label, fit_wells=len(predict["fit"]), fit_parents=predict["fit"].parent.nunique(),
                query_wells=len(predict["query"]), fit_controls=len(predict["fit_controls"]), score_controls=len(score["score_controls"])))
        donors = {}
        for role in "ABC":
            source_controls = controls.loc[controls.role.eq(role) & controls.rna_plate.isin(source.rna_plate)]
            control_axis(source_controls, source, role)
            donors[role] = write_plan(stage / "donor-plan" / role, {"source.tsv": source, "controls.tsv": source_controls})
        landmarks.to_csv(stage / "landmarks.tsv", sep="\t", index=False)
        trt_axis.to_csv(stage / "treated_genes.tsv", sep="\t", index=False)
        ctl_axis.to_csv(stage / "controls_genes.tsv", sep="\t", index=False)
        pd.DataFrame(support).to_csv(stage / "support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(study="CMap2020", cell=CELL, metadata_only=True, expression_read=False,
            permitted_HDF5_datasets=[SAMPLE_IDS, GENE_IDS], support_audit_sha256=sha256(args.audit_pin),
            protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__), matrix_reference_sha256=sha256(args.cmap_pin),
            target_parents=targets.parent.nunique(), target_wells=len(targets), target_plates=targets.rna_plate.nunique(),
            donor_cells=source.cell_id.nunique(), donor_plates=source.rna_plate.nunique(), donor_wells=len(source),
            matrices=matrices, slices=slices, donors=donors,
            files={path.name: sha256(path) for path in stage.glob("*.tsv")}))
        stage.rename(args.out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("audit", "audit-pin", "cmap-pin", "landmarks", "protocol", "out"):
        cli.add_argument("--"+field, type=Path, required=True)
    run(cli.parse_args())
