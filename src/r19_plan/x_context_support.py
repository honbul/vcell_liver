"""Audit same-dose multi-context DEV support in existing GSE92742 metadata."""

import argparse
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import sha256, write_json
from src.r15_lincs.data import asset
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import ROOT
from src.r19_plan.audit_r18 import geo_metadata, prepare
from src.r19_plan.i_preflight import provenance, read_r18_domains
from src.r19_plan.t4_support import eligible_links, measured_dose

STUDY = "GSE92742"
TARGET = "HEPG2"
HOURS = 6
DOSE = "10"
MIN_PARENTS = 100
MIN_FOLD = 20
MIN_PLATES = 3
MAX_CONTEXTS_IN_SEARCH = 12


def support(parents):
    folds = [fold_id(parent, "R15-outer", 3) for parent in parents]
    return len(parents), *(folds.count(fold) for fold in range(3))


def ample(counts):
    n, *folds = counts
    return n >= MIN_PARENTS and min(folds) >= MIN_FOLD


def inventory(rows):
    fields = ["cell_id", "parent", "plate", "well", "inst_id"]
    if rows.empty or rows[fields].isna().any().any() or rows.duplicated(["plate", "well"]).any():
        raise ValueError("X context support requires unique physical RNA wells with resolved lineage")
    shared = set(rows.groupby("plate").cell_id.nunique().loc[lambda n: n > 1].index)
    rows = rows.loc[~rows.plate.isin(shared)]
    parents = {cell: set(frame.parent) for cell, frame in rows.groupby("cell_id")}
    if TARGET not in parents:
        raise ValueError("X context support lacks the HEPG2 anchor context")
    cells = []
    for cell, frame in rows.groupby("cell_id"):
        common = parents[cell] & parents[TARGET]
        matching = frame.loc[frame.parent.isin(common)]
        selected = support(common)
        cells.append(dict(cell=cell, parents=len(parents[cell]), wells=len(frame), plates=frame.plate.nunique(),
                          hepg2_common=selected[0], fold0=selected[1], fold1=selected[2], fold2=selected[3],
                          overlap_plates=matching.plate.nunique(),
                          overlap_eligible=ample(selected) and matching.plate.nunique() >= MIN_PLATES))
    return rows, pd.DataFrame(cells), parents, len(shared)


def choose_loco(rows, cells, parents):
    candidates = cells.loc[cells.overlap_eligible & cells.cell.ne(TARGET)]
    names = candidates.sort_values(["hepg2_common", "cell"], ascending=[False, True]).cell.head(MAX_CONTEXTS_IN_SEARCH)
    scoped = rows.loc[rows.cell_id.isin([TARGET, *names])]
    plate_by_parent = {cell: frame.groupby("parent").plate.agg(frozenset)
                       for cell, frame in scoped.groupby("cell_id")}
    search, best = [], None
    fold = {parent: fold_id(parent, "R15-outer", 3) for parent in parents[TARGET]}
    for others in combinations(names, 4):
        domain = (TARGET, *sorted(others))
        shared = parents[TARGET].intersection(*(parents[cell] for cell in others))
        fold_counts = [0, 0, 0]
        for parent in shared:
            fold_counts[fold[parent]] += 1
        plate_counts = [len(set().union(*plate_by_parent[cell].loc[list(shared)])) for cell in domain]
        eligible = ample((len(shared), *fold_counts)) and min(plate_counts) >= MIN_PLATES
        search.append(dict(contexts="|".join(domain), parents=len(shared), min_context_plates=min(plate_counts),
                           fold0=fold_counts[0], fold1=fold_counts[1], fold2=fold_counts[2], eligible=eligible))
        key = (-len(shared), -min(fold_counts), domain)
        if eligible and (best is None or key < best[0]):
            best = key, domain, shared
    summary = pd.DataFrame(search, columns=["contexts", "parents", "min_context_plates", "fold0", "fold1",
                                           "fold2", "eligible"])
    if best is None:
        return summary, None, None
    _, domain, shared = best
    matched = scoped.loc[scoped.cell_id.isin(domain) & scoped.parent.isin(shared)].copy()
    matched["fold"] = matched.parent.map(fold)
    return summary, domain, matched


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    protected = read_r18_domains(args)[:4]
    footprint = pd.concat([provenance(table) for table in protected], ignore_index=True)
    prior = args.base / "vcell_liver_r15_lincs/v1"
    inst, sig = geo_metadata(STUDY, prior)
    inst = prepare(inst)
    if not {"rna_plate", "rna_well"} <= set(inst):
        raise ValueError("X support cannot substitute detection coordinates for physical RNA wells")
    path = asset(ROOT, STUDY, "inst_info")
    doses = measured_dose(path, set(inst.loc[inst.pert_type.eq("trt_cp") & inst.hours.eq(HOURS), "inst_id"]))
    rows, diagnostics = eligible_links(inst, sig.loc[sig.hours.eq(HOURS)], footprint, doses)
    at_dose = rows.loc[rows.actual_dose_uM.eq(DOSE)]
    if not at_dose.plate.eq(at_dose.rna_plate).all() or not at_dose.well.eq(at_dose.rna_well).all():
        raise ValueError("A candidate X support well is not on the measured RNA axis")
    exact, cells, parents, shared_plates = inventory(at_dose)
    search, selected, matched = choose_loco(exact, cells, parents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-support-") as tmp:
        stage = Path(tmp)
        cells.to_csv(stage / "contexts.tsv", sep="\t", index=False)
        search.to_csv(stage / "loco_candidates.tsv", sep="\t", index=False)
        if selected:
            support_rows = []
            for (cell, fold), frame in matched.groupby(["cell_id", "fold"], sort=True):
                repeated = frame.groupby("parent").plate.nunique().ge(2)
                support_rows.append(dict(cell=cell, fold=fold, parents=frame.parent.nunique(), wells=len(frame),
                                         plates=frame.plate.nunique(), repeat_parents=int(repeated.sum())))
            pd.DataFrame(support_rows).to_csv(stage / "loco_support.tsv", sep="\t", index=False)
        sources = [path, *(asset(ROOT, STUDY, meta) for meta in ("sig_info", "sig_metrics")),
                   prior / "audit" / STUDY / "identities.tsv"]
        write_json(stage / "receipt.json", dict(
            study=STUDY, measured_dose_uM=DOSE, hours=HOURS, minimum_parents=MIN_PARENTS,
            minimum_fold_parents=MIN_FOLD, minimum_context_plates=MIN_PLATES, top_candidates=MAX_CONTEXTS_IN_SEARCH,
            selected_contexts=selected, excluded_shared_plates=shared_plates, diagnostics=diagnostics,
            metadata_only=True, expression_read=False, count_contexts=len(cells),
            reservation_sha256=sha256(args.r18_access / "receipt.json"),
            source_metadata_sha256={str(item): sha256(item) for item in sources},
            code_sha256=sha256(__file__), files={p.name: sha256(p) for p in sorted(stage.iterdir())},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "reports", "r18-registered", "r18-lock", "r18-access", "r18-index", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
