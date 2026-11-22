"""Register matched D1/D2 development control and target access without reading expression."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.i_preflight import write_plan
from src.r19_plan.i_stage import (
    PREDICT, SCORE, check_prediction_scope, check_role, check_wells, checked_query, parse_landmarks,
    parse_tsv, pinned_bytes, plan_receipt, read_tables, slice_roles,
)
from src.r19_plan.r18_matrix_index import GENE_IDS, SAMPLE_IDS, metadata_index
from src.r19_plan.r18_preflight import GEO

CONTROL_KEYS = ["study", "inst_id", "rna_plate", "rna_well", "role"]
SLICES = tuple(f"f{fold}-{role}" for fold in range(3) for role in "ABC")


def equal_roles(rows):
    counts = rows.groupby(["rna_plate", "role"]).size().unstack().reindex(columns=list("ABC"), fill_value=0)
    if counts.empty or counts.isna().any().any() or counts.lt(1).any().any() or not counts.nunique(axis=1).eq(1).all():
        raise ValueError("D1 requires equal independent physical control counts on each plate")
    return counts


def matched(original, balanced):
    aligned = original[CONTROL_KEYS].merge(balanced, on=CONTROL_KEYS, how="inner", validate="one_to_one")
    if set(aligned.rna_plate) != set(original.rna_plate):
        raise ValueError("A previously approved development plate lost its D1 role support")
    return aligned


def register_slice(plan, pin, slice_id, balanced):
    source, _ = plan_receipt(plan, pin, slice_id)
    inputs, _ = read_tables(plan, source, slice_id, "predict", PREDICT)
    score, _ = read_tables(plan, source, slice_id, "score", SCORE)
    check_prediction_scope(inputs, slice_id)
    inputs = {name: matched(rows, balanced) if name.endswith("_controls") else rows for name, rows in inputs.items()}
    score["score_controls"] = matched(score["score_controls"], balanced)
    split = {"predict": inputs, "score": score}
    support = check_split(split, slice_id)
    return split, support


def check_split(split, slice_id):
    inputs, score = split["predict"], split["score"]
    labels, query = inputs["inner_and_outer_labels"], inputs["query_features"]
    held = score["score_labels"]
    fit, select, score_controls = inputs["fit_controls"], inputs["select_controls"], score["score_controls"]
    check_prediction_scope(inputs, slice_id)
    if not checked_query(held.drop(columns="cache_index")).equals(query):
        raise ValueError("D1 changed the registered outer query axis")
    _, _, _, score_role = slice_roles(slice_id)
    check_role(score_controls, held, score_role)
    check_wells(pd.concat((labels, held, fit, select, score_controls), ignore_index=True))
    observed = pd.concat((fit, select, score_controls), ignore_index=True)
    if not observed.level3_row_index.astype(str).str.fullmatch(r"[0-9]+").all() or observed.level3_row_index.duplicated().any():
        raise ValueError("Each D1 control requires a separate nonnegative Level 3 index")
    # Selection only needs the two controls on validation plates, and scoring only its own role on held plates.
    fit_count, select_count, score_count = (frame.groupby("rna_plate").size() for frame in (fit, select, score_controls))
    if not select_count.equals(fit_count.reindex(select_count.index)) or not score_count.equals(fit_count.reindex(score_count.index)):
        raise ValueError("D1 role counts diverge on the validation or score plate axis")
    return dict(slice=slice_id, fit_controls=len(fit), select_controls=len(select), score_controls=len(score_controls),
                inner_fit_parents=labels.loc[labels.inner_phase.eq("inner_fit"), "parent"].nunique(),
                select_parents=labels.loc[labels.inner_phase.eq("select"), "parent"].nunique(),
                score_parents=held.parent.nunique(), score_wells=len(held), score_plates=held.rna_plate.nunique())


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    source, source_hash = plan_receipt(args.i_plan, args.i_reference, SLICES[0])
    if sha256(args.balanced / "receipt.json") != sha256(args.balanced_reference):
        raise ValueError("D1 balanced control receipt changed")
    audit = read_json(args.balanced_reference)
    chosen = parse_tsv(pinned_bytes(args.balanced / "assignments.tsv", audit["files"]["assignments.tsv"]))
    chosen = chosen.assign(study=GEO)[CONTROL_KEYS]
    check_wells(chosen)
    counts = equal_roles(chosen)
    if not audit["metadata_only"] or len(chosen) != audit["kept_wells"] or len(counts) != audit["plates"]:
        raise ValueError("D1 balanced control inventory differs from the original receipt")
    eligible = []
    for role in "ABC":
        slice_id = f"f0-{role}"
        tables, _ = read_tables(args.i_plan, source, slice_id, "predict", PREDICT)
        eligible.append(tables["fit_controls"][CONTROL_KEYS])
    checked = pd.concat(eligible, ignore_index=True)
    check_wells(checked)
    if len(matched(checked, chosen)) != len(chosen):
        raise ValueError("A balanced D1 control lies outside the approved development registry")

    genes = parse_landmarks(pinned_bytes(args.landmarks, source["landmarks_sha256"]))
    matrix_pin = read_json(args.matrix_reference)
    if (not matrix_pin["metadata_only"] or matrix_pin["expression_read"] or matrix_pin["study"] != GEO
            or matrix_pin["landmark_axis_sha256"] != source["landmarks_sha256"]):
        raise ValueError("D1 requires the audited public GSE92742 Level 3 axis")
    expected = matrix_pin["matrix_records"]["combined"]
    indexed, gene_indices, record = metadata_index(Path(expected["path"]), chosen, genes.pr_gene_id.to_numpy())
    if record != expected:
        raise ValueError("The GSE92742 Level 3 axis changed since the independent metadata lock")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".d1-access-") as tmp:
        stage = Path(tmp)
        files, support = {}, []
        for slice_id in SLICES:
            split, row = register_slice(args.i_plan, args.i_reference, slice_id, indexed)
            files[slice_id] = {
                phase: write_plan(stage / (phase + "-plan") / slice_id,
                                  {name + ".tsv": table for name, table in tables.items()})
                for phase, tables in split.items()
            }
            support.append(row)
        pd.DataFrame(support).to_csv(stage / "support.tsv", sep="\t", index=False)
        counts.reset_index().to_csv(stage / "control_counts.tsv", sep="\t", index=False)
        gene_indices.to_csv(stage / "gene_indices.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(
            metadata_only=True, expression_read=False, permitted_hdf5_datasets=[SAMPLE_IDS, GENE_IDS],
            source_access_sha256=source_hash, balanced_sha256=sha256(args.balanced_reference),
            matrix_reference_sha256=sha256(args.matrix_reference), landmarks_sha256=source["landmarks_sha256"],
            matrix=record, physical_controls=len(indexed), plates=len(counts), slice_files=files,
            files={p.name: sha256(p) for p in stage.glob("*.tsv")}, code_sha256=sha256(__file__),
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("i-plan", "i-reference", "balanced", "balanced-reference", "matrix-reference", "landmarks", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
