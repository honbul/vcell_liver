"""Audit HEPG2-only training support for C3, including independently eligible HEPG2 parents."""

import argparse
from pathlib import Path

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.cache import sort_dose
from src.r15_lincs.data import hepatic_lines, table
from src.r15_lincs.design import fold_id
from src.r19_plan.audit_r18 import DET, PHYSICAL, RNA, combine, geo_metadata, prepare, roles
from src.r17_lincs.inventory import ROOT

STUDY = "GSE92742"


def donors(sig, inst, sizes, wanted, hour, band):
    selected = sig.loc[sig.pert_type.eq("trt_cp") & sig.hours.eq(hour) & sig.dose.between(0, 5)
                       & sig.parent.isin(wanted)]
    linked = combine(selected, inst, exact_dose=False)
    linked = linked.loc[linked.dose_well.between(*band) & linked.plate.map(sizes).fillna(0).ge(2)].copy()
    counts = linked.groupby("cell_id").parent.nunique().sort_index().sort_values(ascending=False, kind="stable")
    linked = linked.loc[linked.cell_id.isin(counts.index[:6])]
    choices = sort_dose(linked).drop_duplicates(["parent", "cell_id"]).sig_id
    return linked.loc[linked.sig_id.isin(choices)].drop_duplicates(PHYSICAL)


def audit(args):
    args.out.mkdir(exist_ok=False)
    raw, sig = geo_metadata(STUDY, args.base / "vcell_liver_r15_lincs/v1")
    inst = prepare(raw)
    _, sizes = roles(inst, STUDY, geo=None)
    exposure = table(args.exposure)
    target = exposure.loc[exposure.cohort.eq("C3") & exposure.A_eligible.eq("True")].copy()
    target["hours"] = target.hours.astype(float)
    target["dose_well"] = target.dose_well.astype(float)
    target = target.drop_duplicates(PHYSICAL)
    nonhep = sig.loc[~sig.cell_id.isin(hepatic_lines(ROOT, STUDY))]
    summaries, blocks, source_registry = [], [], []
    for hour, query in target.groupby("hours"):
        band = (query.dose_well.min(), query.dose_well.max())
        s = sig.loc[sig.pert_type.eq("trt_cp") & sig.hours.eq(hour) & sig.dose.between(0, 5)
                    & sig.cell_id.eq("HEPG2") & sig.qc & sig.parent.ne("")]
        train = combine(s, inst, exact_dose=False).drop_duplicates(PHYSICAL)
        train = train.loc[train.plate.map(sizes).ge(3) & train.dose_well.between(*band)]
        source = donors(nonhep, inst, sizes, set(train.parent) | set(query.parent), hour, band)
        for design, eligible in [("nominal_3uM", train.loc[train.dose.eq(3)]), ("measured_HUH7_range", train)]:
            for scenario in "AB":
                fit = eligible if scenario == "A" else eligible.loc[eligible.parent.isin(source.parent)]
                score = query if scenario == "A" else query.loc[query.parent.isin(source.parent)]
                fit_fold = fit.parent.map(lambda p: fold_id(p, "R15-outer", 3))
                score_fold = score.parent.map(lambda p: fold_id(p, "R15-outer", 3))
                summaries.append(dict(hours=hour, design=design, scenario=scenario, min_uM=band[0], max_uM=band[1],
                                      HEPG2_parents=fit.parent.nunique(), HEPG2_wells=len(fit),
                                      HUH7_parents=score.parent.nunique(), HUH7_wells=len(score)))
                for fold in range(3):
                    outer_train = fit.loc[fit_fold.ne(fold)]
                    selection = outer_train.parent.map(lambda p: fold_id(p, f"R15-inner-{fold}", 5)).eq(0)
                    for phase, rows in [("HEPG2_inner_fit", outer_train.loc[~selection]),
                                         ("HEPG2_select", outer_train.loc[selection]),
                                         ("HUH7_score", score.loc[score_fold.eq(fold)])]:
                        blocks.append(dict(hours=hour, design=design, scenario=scenario, fold=fold,
                                           phase=phase, parents=rows.parent.nunique(), wells=len(rows)))
        fields = ["sig_id", "inst_id", "pert_id", "parent", "cell_id", "hours", "dose", "dose_well", *RNA, *DET, *PHYSICAL]
        source_registry.append(source[fields].assign(cohort="C3", study=STUDY,
            DEV_fit_select_or_pretrain_allowed=False, CONFIRM_R18_B_source_donor_allowed=True))
    for name, formatted in [("c3_hepg2_support", pd.DataFrame(summaries)),
                            ("c3_hepg2_fold_support", pd.DataFrame(blocks)),
                            ("c3_B_hepg2_held_donor_support", pd.concat(source_registry, ignore_index=True))]:
        formatted.to_csv(args.out / f"{name}.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(exposure_sha256=sha256(args.exposure),
        code_sha256=sha256(__file__), files={p.name:sha256(p) for p in args.out.iterdir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "exposure", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    audit(parser.parse_args())
