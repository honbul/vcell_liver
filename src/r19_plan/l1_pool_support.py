"""Audit strictly matched L1 fit, selection, and score pools without expression."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r19_plan.i_stage import PREDICT, check_prediction_scope, check_wells, plan_receipt, read_tables
from src.r19_plan.t4_support import measured_dose

POOL = ["study", "rna_plate", "cell_id", "hours", "actual_dose_uM"]
MINIMA = (2, 3, 5)


def pools(rows):
    checked = rows.loc[rows.actual_dose_uM.ne("")].copy()
    if checked[POOL].isna().any().any():
        raise ValueError("A matched L1 pool has missing experiment metadata")
    group = checked.groupby(POOL, sort=True)
    support = group.parent.transform("nunique")
    inventory = group.agg(wells=("inst_id", "size"), parents=("parent", "nunique")).reset_index()
    summary = []
    for minimum in MINIMA:
        kept = checked.loc[support.ge(minimum)]
        summary.append(dict(min_parents_in_pool=minimum, pool_count=len(inventory),
                            eligible_pools=int(inventory.parents.ge(minimum).sum()),
                            all_wells=len(rows), known_dose_wells=len(checked), eligible_wells=len(kept),
                            eligible_parents=kept.parent.nunique(), plates=kept.rna_plate.nunique()))
    return summary, inventory


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    selections, held = {}, {}
    for fold in range(3):
        sid = f"f{fold}-A"
        reference, _ = plan_receipt(args.plan, args.reference, sid)
        files, _ = read_tables(args.plan, reference, sid, "predict", PREDICT)
        check_prediction_scope(files, sid)
        selections[fold] = files["inner_and_outer_labels"]
        held[fold] = files["query_features"]
    complete = pd.concat(held.values(), ignore_index=True)
    check_wells(complete)
    if set.union(*(set(rows.inst_id) for rows in selections.values())) != set(complete.inst_id):
        raise ValueError("L1 fit and score have inconsistent physical well inventories")
    measured = measured_dose(args.inst, set(complete.inst_id))
    complete = complete.merge(measured, on="inst_id", how="left", validate="one_to_one").fillna({"actual_dose_uM": ""})
    source_dose = pd.to_numeric(complete.dose, errors="coerce")
    actual = pd.to_numeric(complete.actual_dose_uM, errors="coerce")
    disagree = actual.notna() & source_dose.ne(actual)
    dose_by_well = complete.set_index("inst_id").actual_dose_uM
    summaries, inventories = [], []
    for fold in range(3):
        train = selections[fold]
        phases = {name: train.loc[train.inner_phase.eq(name)] for name in ("inner_fit", "select")}
        phases.update(outer_fit=train, score=held[fold])
        for phase, frame in phases.items():
            rows = frame.assign(actual_dose_uM=frame.inst_id.map(dose_by_well))
            points, inventory = pools(rows)
            summaries.extend(dict(fold=fold, phase=phase, **point) for point in points)
            inventories.append(inventory.assign(fold=fold, phase=phase))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".l1-support-") as tmp:
        stage = Path(tmp)
        pd.DataFrame(summaries).to_csv(stage / "support.tsv", sep="\t", index=False)
        pd.concat(inventories).to_csv(stage / "pools.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(
            scope="GSE92742 HEPG2 6h protected-input-excluded DEV-A", metadata_only=True,
            expression_read=False, phases=list(phases), raw_instance_sha256=sha256(args.inst),
            development_split_sha256=sha256(args.reference), code_sha256=sha256(__file__),
            source_and_measured_dose_disagreement_wells=int(disagree.sum()),
            wells_without_exact_dose=int(actual.isna().sum()), minimums=list(MINIMA),
            files={file.name: sha256(file) for file in sorted(stage.iterdir())},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inst", "plan", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
