"""Audit previously unscored low-dose adult HCC wells and matched donor support."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.design import fold_id
from src.r16_lincs.audit import geo_exposure
from src.r17_lincs.inventory import CMAP, ROOT, cmap_frames, ids, in_footprint, liver_cells, previous, register_footprint
from src.r19_plan.hcc_lincs_24h_support import SCOPE, TARGETS, evaluate_support, forbidden_signatures, instance_rows, supported_controls
from src.r19_plan.i_audit import structure
from src.r19_plan.i_preflight import provenance, read_pinned, read_r18_domains, verify_domain
from src.r19_plan.t4_scaffold_transfer import scaffold


DOSES = ("3.33333", "1.11111", "1", "0.1")
MIN_DONORS = (1, 2, 4, 8, 16)


def prior_measurements(args):
    ten_pin = args.reports / "r19_plan/data/hcc_lincs24/audit/receipt.json"
    stamp = read_json(ten_pin)
    if sha256(ten_pin) != sha256(args.ten / "receipt.json"):
        raise ValueError("Prior exact 10 µM metadata differ from the pinned audit")
    historic = {name: read_pinned(args.ten, name + ".tsv", stamp["files"][name + ".tsv"])
                for name in ("targets", "donors", "controls")}
    _, old_cmap, _, _ = cmap_frames(args.base)
    footprints, signatures, sources = previous(args.base, old_cmap)
    old_signatures = set().union(*(keys for (study, _), keys in signatures.items() if study == CMAP))
    footprints["ten_development"] = register_footprint(pd.concat([historic["targets"], historic["donors"]]))
    _, geo_ids, geo_positions, geo_sources = geo_exposure(args.base / "vcell_liver_r15_lincs/v1")
    footprints["GEO_releases"] = (geo_ids, geo_positions)
    digests = {name: sha256(path) for name, path in sources.items()}
    digests.update(geo_sources)
    digests["ten_development_receipt"] = sha256(ten_pin)
    return footprints, pd.DataFrame({"sig_id": sorted(old_signatures)}), historic, digests


def summaries(targets, donors):
    records, slices = [], []
    for dose in DOSES:
        source = donors.loc[donors.actual_dose_uM.eq(dose)]
        counts = source.groupby("parent").agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
        for cell in TARGETS:
            target = targets.loc[targets.actual_dose_uM.eq(dose) & targets.cell_id.eq(cell)]
            for minimum in (0, *MIN_DONORS):
                parents = counts.index[counts.cells.ge(minimum) & counts.plates.ge(minimum)]
                selected = target.loc[target.parent.isin(parents)] if minimum else target
                # The prior partition function uses one line and one exact dose at a time.
                _, partitions, _ = evaluate_support(selected, source, cell, "nonhepatic")
                fold = selected.groupby("fold").parent.nunique().reindex(range(3), fill_value=0)
                fit_min = min((r["fit_parents"] for r in partitions), default=0)
                sufficient = (minimum >= 2 and selected.parent.nunique() >= 100 and fold.min() >= 20
                              and selected.rna_plate.nunique() >= 3 and fit_min >= 30)
                if minimum == 8:
                    slices.extend(dict(r, dose=dose, minimum_donor_cells_and_plates=minimum) for r in partitions)
                records.append(dict(cell=cell, dose=dose, minimum_donor_cells_and_plates=minimum,
                    parents=selected.parent.nunique(), wells=len(selected), plates=selected.rna_plate.nunique(),
                    fold0=fold[0], fold1=fold[1], fold2=fold[2], fit_parent_min=fit_min, sufficient=bool(sufficient),
                    previous_10uM_plate_wells=int(selected.previous_10uM_plate.sum())))
    return records, slices


def run(args):
    protected = read_r18_domains(args)
    blocked = pd.concat([provenance(rows) for rows in protected], ignore_index=True)
    registry = ROOT / "LINCS2020"
    paths = dict(samples=args.base / "vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020/instinfo_beta.txt",
                 signatures=registry / "siginfo_beta.txt", identity=registry / "compoundinfo_beta.txt",
                 cells=registry / "cellinfo_beta.txt")
    cells = table(paths["cells"], usecols=["cell_iname", "cell_lineage"])
    hepatic = set(liver_cells(paths["cells"], cmap=True).cell) | set(TARGETS)
    lineage = cells.cell_lineage.str.strip().str.lower()
    nonhepatic = set(cells.loc[~lineage.isin(("", "unknown", "na")), "cell_iname"]) - hepatic
    if not {"A549", "MCF7"} <= nonhepatic or {"HEPG2", "PHH", *TARGETS} & nonhepatic:
        raise ValueError("Unexpected CMap lineages")
    treated, controls, ambiguous = instance_rows(paths["samples"], nonhepatic | set(TARGETS), DOSES)
    footprints, old_signatures, historic, old_digests = prior_measurements(args)
    blocked_members = forbidden_signatures(paths["signatures"], pd.concat((blocked, old_signatures), ignore_index=True))
    foot = register_footprint(blocked)
    before = len(treated), len(controls)
    treated = treated.loc[~in_footprint(treated, foot) & ~ids(treated.inst_id).isin(blocked_members)]
    controls = controls.loc[~in_footprint(controls, foot) & ~ids(controls.inst_id).isin(blocked_members)]
    after = len(treated), len(controls)
    identity = table(paths["identity"], usecols=["pert_id", "canonical_smiles"]).drop_duplicates()
    identity = identity.loc[identity.pert_id.isin(treated.pert_id)].copy()
    identity["parent"] = identity.canonical_smiles.map(structure)
    identity = identity[["pert_id", "parent"]].drop_duplicates()
    identity = identity.loc[identity.parent.ne("") & ~identity.pert_id.duplicated(keep=False)]
    treated, controls = supported_controls(treated.merge(identity, on="pert_id", validate="many_to_one"), controls)
    targets = treated.loc[treated.cell_id.isin(TARGETS)].copy()
    exposure = []
    any_exposed = pd.Series(False, index=targets.index)
    for name, previous_foot in footprints.items():
        seen = in_footprint(targets, previous_foot)
        any_exposed |= seen
        for (cell, dose), rows in targets.loc[seen].groupby(["cell_id", "actual_dose_uM"]):
            exposure.append(dict(source=name, cell=cell, dose=dose, wells=len(rows), parents=rows.parent.nunique()))
    targets = targets.loc[~any_exposed].copy()
    targets["b1_seen"] = False
    targets["previous_10uM_plate"] = targets.rna_plate.isin(historic["targets"].rna_plate)
    targets["fold"] = targets.parent.map(lambda p: fold_id(p, "R15-outer", 3))
    targets["scaffold"] = targets.parent.map({p: scaffold(p) for p in targets.parent.unique()})
    sources = treated.loc[treated.cell_id.isin(nonhepatic) & ~treated.rna_plate.isin(targets.rna_plate)]
    pairs = ["actual_dose_uM", "parent"]
    donors = sources.merge(targets[pairs].drop_duplicates(), on=pairs, validate="many_to_one")
    controls = controls.loc[controls.rna_plate.isin(set(targets.rna_plate) | set(donors.rna_plate))].copy()
    controls["previous_10uM_control"] = in_footprint(controls, register_footprint(historic["controls"]))
    verify_domain(pd.concat((targets, donors), ignore_index=True), controls, *protected[:4])
    stats, slices = summaries(targets, donors)

    with TemporaryDirectory(dir=args.out.parent, prefix=".jhh-low-dose-") as temporary:
        stage = Path(temporary)
        for name, data in (("support", stats), ("splits", slices)):
            pd.DataFrame(data).to_csv(stage / (name + ".tsv"), sep="\t", index=False)
        pd.DataFrame(exposure, columns=["source", "cell", "dose", "wells", "parents"]).to_csv(
            stage / "exposure.tsv", sep="\t", index=False)
        for name, rows, fields in (
            ("targets", targets, ["actual_dose_uM", "parent", "fold", "scaffold", "previous_10uM_plate"]),
            ("donors", donors.assign(view="nonhepatic"), ["actual_dose_uM", "parent", "view"]),
            ("controls", controls, ["previous_10uM_control"]),
        ):
            rows.assign(study=CMAP)[["study", *SCOPE, *fields]].to_csv(stage / (name + ".tsv"), sep="\t", index=False)
        write_json(stage / "receipt.json", dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
            metadata_only=True, matrix_opened=False, expression_read=False,
            source_sha256={name: sha256(path) for name, path in paths.items()}, prior_metadata_sha256=old_digests,
            r18_registry_sha256=sha256(args.r18_registered / "receipt.json"),
            r18_access_sha256=sha256(args.r18_access / "receipt.json"), registered_control_block=True,
            ambiguous_rows=ambiguous, before_signature_and_r18_filter=before, after_signature_and_r18_filter=after,
            prior_target_wells_removed=int(any_exposed.sum()), target_support=len(targets),
            previous_10uM_target_controls=int(controls.loc[controls.rna_plate.isin(targets.rna_plate), "previous_10uM_control"].sum()),
            files={path.name: sha256(path) for path in stage.iterdir()}))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "reports", "ten", "r18-registered", "r18-lock", "r18-access", "r18-index", "protocol", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
