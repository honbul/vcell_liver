"""Audit the physical metadata for source-only structure-lane calibration."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.design import fold_id
from src.r19_plan.shift_calibration import TARGET_MIN_PLATES, TARGET_MIN_WELLS
from src.r19_plan.x_inputs import check_unique, control_axis, read_plan, reference, slice_axes, validate_predict


def source_folds(rows, fold):
    return rows.parent.map(lambda p: fold_id(p, f"T123A-inner-{fold}", 3))


def select_controls(args, ref, key, tables):
    cell, _, fold, fit_role, select_role, _ = slice_axes(key, ref["cells"])
    rotated = f"{cell}-f{fold}-{select_role}"
    _, _, other = read_plan(args.access, args.pin, rotated, "predict")
    labels, query, fit = (tables[n] for n in ("labels", "query", "fit_controls"))
    select = other["fit_controls"].loc[other["fit_controls"].cell_id.ne(cell)].copy()
    if (set(labels.cell_id) != set(ref["cells"]) - {cell} or set(labels.parent) & set(query.parent)
            or not query.cell_id.eq(cell).all() or not fit.role.eq(fit_role).all()):
        raise ValueError("A scored cell or chemical entered the source-only T.A fit")
    control_axis(select, labels, select_role)
    check_unique(pd.concat((labels, query, fit, select), ignore_index=True))
    return select, rotated


def support(args, key, ref):
    target, _, fold, role, _, _ = slice_axes(key, ref["cells"])
    _, _, plan = read_plan(args.access, args.pin, key, "predict")
    validate_predict(plan, key, ref["cells"])
    select, rotated = select_controls(args, ref, key, plan)
    fit = plan["fit_controls"]
    result = []
    for cell in ref["cells"]:
        rows = plan["query"] if cell == target else plan["labels"].loc[plan["labels"].cell_id.eq(cell)]
        wells, plates = int(fit.cell_id.eq(cell).sum()), fit.loc[fit.cell_id.eq(cell), "rna_plate"].nunique()
        inner = source_folds(rows, fold).groupby(rows.parent).first().value_counts().reindex(range(3), fill_value=0)
        result.append(dict(slice=key, outer_target=target, outer_fold=fold, role=role, context=cell,
            domain="held" if cell==target else "source", parents=rows.parent.nunique(), physical_wells=rows.inst_id.nunique(),
            fit_dmsos=wells, fit_plates=plates, select_dmsos=int(select.cell_id.eq(cell).sum()),
            parent0=inner[0] if cell!=target else None, parent1=inner[1] if cell!=target else None,
            parent2=inner[2] if cell!=target else None,
            eligible=wells >= TARGET_MIN_WELLS and plates >= TARGET_MIN_PLATES and (cell==target or int(inner.min())>=20)))
    return result, dict(predict=ref["slices"][key]["predict"], rotated_key=rotated,
                        rotated_fit_sha256=ref["slices"][rotated]["predict"]["fit_controls.tsv"])


def audit(args):
    ref = reference(args.pin)
    rows, slices = [], {}
    for key in sorted(ref["slices"]):
        local, checks = support(args, key, ref)
        rows.extend(local)
        slices[key] = checks
    args.out.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    if len(table) != 225 or len(slices) != 45 or ref["parents"] != 493 or ref["treatments"] != 14898:
        raise ValueError("The T.A inventory differs from the original fixed X assay support")
    table.to_csv(args.out/"support.tsv", sep="\t", index=False, na_rep="NA")
    write_json(args.out/"receipt.json", dict(expression_read=False, metadata_only=True, eligible=bool(table.eligible.all()),
                  access_sha256=sha256(args.pin), protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                  slices=slices, files={"support.tsv": sha256(args.out/"support.tsv")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("access", "pin", "protocol", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    audit(parser.parse_args())
