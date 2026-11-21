"""Pin I-track development wells and disjoint fit/select/score access manifests."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r15_lincs.data import table
from src.r15_lincs.design import fold_id
from src.r17_lincs.inventory import in_footprint, register_footprint
from src.r19_plan.contracts import assert_dev_allowed, coord_keys, id_keys
from src.r19_plan.r18_preflight import GEO, PHYSICAL_RNA, ROLES, check_controls, checked_tables

PROVENANCE = ("sig_id", "inst_id", *PHYSICAL_RNA, "det_plate", "det_well")
TARGET_KEYS = ["inst_id", "sig_id", "parent", "cell_id", "hours", "dose", *PHYSICAL_RNA]
STUDIES = ("CMap2020", "GSE70138", GEO)


def read_pin(root, reference):
    receipt = root / "receipt.json"
    if sha256(receipt) != sha256(reference):
        raise ValueError(f"Metadata receipt changed: {root}")
    return read_json(receipt)


def read_pinned(root, name, digest):
    path = root / name
    if sha256(path) != digest:
        raise ValueError(f"Metadata artifact changed: {path}")
    return table(path)


def provenance(rows):
    """Retain source coordinates as well as a supplied canonical physical alias."""
    names = list(PROVENANCE)
    present = [name for name in names if name in rows]
    primary = rows[present].reindex(columns=names, fill_value="")
    if {"plate", "well"} <= set(rows.columns):
        aliases = primary.assign(rna_plate=rows.plate, rna_well=rows.well)
        return pd.concat((primary, aliases), ignore_index=True).drop_duplicates()
    return primary


def exclude(scope, held, name):
    try:
        assert_dev_allowed(scope, provenance(held), scope.iloc[:0])
    except ValueError as exc:
        raise ValueError(f"{name} intersects the I-track candidate inputs: {exc}") from exc


def verify_domain(target, controls, reserved, exceptions, score, donor_controls):
    scope = pd.concat((provenance(target), provenance(controls)), ignore_index=True)
    for name, held in (("R18 treatments and aliases", reserved), ("R18 B donor treatments", exceptions),
                       ("R18 score-only wells", score), ("R18 B donor normalization", donor_controls)):
        exclude(scope, held, name)
    plate_well = coord_keys(target, *PHYSICAL_RNA) & coord_keys(controls, *PHYSICAL_RNA)
    if len(id_keys(target, "inst_id") & id_keys(controls, "inst_id")) or plate_well:
        raise ValueError("A development treatment and control share a physical well")


def register_targets(population, cache):
    if (population.empty or not population.cell_id.eq("HEPG2").all()
        or not pd.to_numeric(population.hours, errors="raise").eq(6).all()
        or not population[TARGET_KEYS].ne("").all().all()
        or len(coord_keys(population, *PHYSICAL_RNA)) != len(population)
        or len(id_keys(population, "inst_id")) != len(population)):
        raise ValueError("Unexpected or duplicate HEPG2 development wells")
    exact = population.merge(
        cache[[*TARGET_KEYS, "index", "quality", "prior_exact"]],
        on=TARGET_KEYS, how="left", validate="one_to_one",
    )
    index = pd.to_numeric(exact["index"], errors="raise")
    if (index.isna().any() or index.duplicated().any() or index.lt(0).any()
        or index.ge(len(cache)).any() or index.mod(1).ne(0).any()
        or not exact.quality.eq("True").all() or not exact.prior_exact.eq("False").all()):
        raise ValueError("An I-track treatment lacks its unique cache index and QC provenance")
    fold = exact.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
    if not fold.eq(pd.to_numeric(exact.outer_fold, errors="raise")).all():
        raise ValueError("Population changed the R17 parent folds")
    return exact.drop(columns=["index", "quality", "prior_exact"]).assign(
        study=GEO, cache_index=index.astype(int), outer_fold=fold,
    ).sort_values("cache_index", ignore_index=True)


def annotate_scaffolds(rows, fit, scaffold):
    keys = rows.parent.map(scaffold)
    known = {scaffold[parent] for parent in set(fit.parent)} - {""}
    status = keys.map(lambda key: "acyclic" if not key else "seen" if key in known else "unseen")
    return rows.assign(scaffold=keys, scaffold_status=status)


def physical_controls(controls, rows, role):
    active = set(rows.rna_plate)
    picked = controls.loc[controls.rna_plate.isin(active) & controls.role.eq(role)]
    if set(picked.rna_plate) != active:
        raise ValueError(f"Missing registered {role} DMSO on a development plate")
    return picked


def register_control_indices(controls, plates):
    if (plates.rna_plate.duplicated().any() or set(plates.rna_plate) != set(controls.rna_plate)):
        raise ValueError("The R17 aggregated control cache has a changed plate axis")
    index = {plate: position for position, plate in enumerate(plates.rna_plate)}
    return controls.assign(cache_plate_index=controls.rna_plate.map(index),
                           cache_role_index=controls.role.map({role: i for i, role in enumerate(ROLES)}))


def split_plans(target, controls):
    check_controls(controls)
    if len(coord_keys(controls, *PHYSICAL_RNA)) != len(controls) or len(id_keys(controls, "inst_id")) != len(controls):
        raise ValueError("Normalized development DMSO aliases overlap")
    scaffold = {parent: MurckoScaffold.MurckoScaffoldSmiles(smiles=parent, includeChirality=False)
                for parent in sorted(set(target.parent))}
    for fold in range(3):
        held = target.loc[target.outer_fold.eq(fold)]
        outer = target.loc[~target.outer_fold.eq(fold)]
        select = outer.parent.map(lambda parent: fold_id(parent, f"R15-inner-{fold}", 5)).eq(0)
        inner = outer.loc[~select]
        validation = outer.loc[select]
        if inner.empty or validation.empty or held.empty or set(outer.parent) & set(held.parent):
            raise ValueError(f"Disjoint I-track partitions are unsupported for outer fold {fold}")
        labels = outer.assign(inner_phase=select.map({True: "select", False: "inner_fit"}))
        score_labels = annotate_scaffolds(held, outer, scaffold)
        query = score_labels.drop(columns="cache_index")
        for i, fit_role in enumerate(ROLES):
            select_role, score_role = ROLES[(i + 1) % 3], ROLES[(i + 2) % 3]
            fit_controls = physical_controls(controls, target, fit_role)
            select_controls = physical_controls(controls, validation, select_role)
            score_controls = physical_controls(controls, held, score_role)
            predictor = {
                "inner_and_outer_labels.tsv": labels,
                "query_features.tsv": query,
                "fit_controls.tsv": fit_controls,
                "select_controls.tsv": select_controls,
            }
            scorer = {"score_labels.tsv": score_labels, "score_controls.tsv": score_controls}
            hidden = pd.concat([provenance(score_labels), provenance(score_controls)], ignore_index=True)
            visible = pd.concat([provenance(labels), provenance(fit_controls), provenance(select_controls)],
                                ignore_index=True)
            exclude(visible, hidden, "I-track score-only wells")
            support = dict(slice=f"f{fold}-{fit_role}", fold=fold, fit_role=fit_role, select_role=select_role,
                           score_role=score_role, fit_DMSO=len(fit_controls),
                           select_DMSO=len(select_controls), score_DMSO=len(score_controls))
            for phase, rows in (("inner_fit", inner), ("select", validation), ("outer_fit", outer),
                                ("score", score_labels)):
                support.update({f"{phase}_wells": len(rows), f"{phase}_parents": rows.parent.nunique(),
                                f"{phase}_plates": rows.rna_plate.nunique()})
            for status in ("seen", "unseen", "acyclic"):
                support[f"score_{status}_scaffold_parents"] = score_labels.loc[
                    score_labels.scaffold_status.eq(status), "parent"
                ].nunique()
            yield support, predictor, scorer


def read_r18_domains(args):
    base = args.reports / "r19_plan" / "data"
    references = base / "preflight" / "r18"
    registry_pin = base / "r18-support-receipt.json"
    registry = checked_tables(args.r18_registered, registry_pin)
    reserved = table(args.r18_registered / "r18_reserved_manifest.tsv")
    exceptions = table(args.r18_registered / "r18_B_donor_exceptions.tsv")
    lock_pin, access_pin = references / "receipt.json", references / "access" / "receipt.json"
    lock = read_pin(args.r18_lock, lock_pin)
    access = read_pin(args.r18_access, access_pin)
    if (not access["metadata_only"] or access["expression_read"]
        or access["registered_sha256"] != sha256(registry_pin) or access["lock_sha256"] != sha256(lock_pin)
        or access["slices"] != 108 or len(access["files"]) != 108):
        raise ValueError("R18 access metadata have an unexpected lineage or slice count")
    registered_controls = read_pinned(args.r18_lock, "control_assignments.tsv", lock["files"]["control_assignments.tsv"])
    scored = []
    for slice_id, roles in sorted(access["files"].items()):
        if set(roles["score"]) != {"score_controls.tsv", "score_labels.tsv"}:
            raise ValueError(f"Unexpected R18 score-only inventory for {slice_id}")
        scored.extend(read_pinned(args.r18_access / "score-plan" / slice_id, name, digest)
                      for name, digest in sorted(roles["score"].items()))
    donors = []
    for study in STUDIES:
        stamp = read_pin(args.r18_index / study, references / "index" / study / "receipt.json")
        if (not stamp["metadata_only"] or stamp["expression_read"] or stamp["lock_sha256"] != sha256(lock_pin)
            or stamp["audited_donor_sha256"] != registry["files"]["r18_B_donor_exceptions.tsv"]):
            raise ValueError(f"R18 B donor control metadata have an unexpected lineage for {study}")
        donors.append(read_pinned(args.r18_index / study, "donor_controls.tsv", stamp["files"]["donor_controls.tsv"]))
    return reserved, exceptions, pd.concat(scored, ignore_index=True).fillna(""), pd.concat(
        donors, ignore_index=True).fillna(""), registered_controls


def write_plan(root, tables):
    root.mkdir(parents=True)
    for name, rows in tables.items():
        rows.to_csv(root / name, sep="\t", index=False)
    return {name: sha256(root / name) for name in sorted(tables)}


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    r19 = args.reports / "r19_plan" / "data"
    pop_stamp = checked_tables(args.population, r19 / "i_views" / "population_receipt.json")
    cache_pin = args.reports / "r15_lincs" / "data" / "cache" / GEO / "receipt.json"
    role_pin = args.reports / "r17_lincs" / "data" / "support" / "receipt.json"
    control_pin = role_pin.parents[1] / "vehicle_stage.json"
    cache_stamp = read_pin(args.r15_cache, cache_pin)
    role_stamp = checked_tables(args.r17_support, role_pin)
    vehicles = args.r17_support.parent / "vehicles"
    vehicle_stamp = read_pin(vehicles, control_pin)
    reserved, exceptions, score, donors, historical = read_r18_domains(args)
    if (cache_stamp["files"]["target_rows.tsv"] != pop_stamp["development_source_sha256"]
        or role_stamp["files"]["assignments.tsv"] != pop_stamp["R17_assignment_sha256"]
        or vehicle_stamp["audit_sha256"] != sha256(role_pin)
        or sha256(args.r18_registered / "r18_reserved_manifest.tsv") != pop_stamp["reserved_sha256"]
        or sha256(args.r18_registered / "r18_B_donor_exceptions.tsv") != pop_stamp["source_exception_sha256"]):
        raise ValueError("The I-track population, cache and protected input metadata have changed")
    target = register_targets(table(args.population / "rows.tsv"),
                              read_pinned(args.r15_cache, "target_rows.tsv", pop_stamp["development_source_sha256"]))
    landmarks = read_pinned(args.r15_cache, "landmarks.tsv", cache_stamp["files"]["landmarks.tsv"])
    if len(landmarks) != 978 or landmarks.pr_gene_id.duplicated().any():
        raise ValueError("Unexpected development gene order")
    if (len(target) != pop_stamp["retained_wells"] or target.parent.nunique() != pop_stamp["retained_parents"]
        or [target.loc[target.outer_fold.eq(i), "parent"].nunique() for i in range(3)] != pop_stamp["expected_counts"]):
        raise ValueError("The locked I-track row set or fold support changed")
    controls = register_control_indices(table(args.r17_support / "assignments.tsv").assign(study=GEO),
                                       read_pinned(vehicles, "vehicles.tsv", vehicle_stamp["files"]["vehicles.tsv"]))
    verify_domain(target, controls, reserved, exceptions, score, donors)
    historical_count = int(in_footprint(controls, register_footprint(historical)).sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".i-preflight-") as tmp:
        stage = Path(tmp)
        files, support = {}, []
        for row, predictor, scorer in split_plans(target, controls):
            identifier = row["slice"]
            files[identifier] = dict(predict=write_plan(stage / "predict-plan" / identifier, predictor),
                                     score=write_plan(stage / "score-plan" / identifier, scorer))
            support.append(row)
        pd.DataFrame(support).to_csv(stage / "support.tsv", sep="\t", index=False)
        write_json(stage / "receipt.json", dict(
            code_sha256=sha256(__file__), metadata_only=True, expression_read=False, fitted=False,
            population_sha256=sha256(args.population / "receipt.json"), cache_metadata_sha256=sha256(cache_pin),
            role_metadata_sha256=sha256(role_pin), control_cache_metadata_sha256=sha256(control_pin),
            r18_access_sha256=sha256(args.r18_access / "receipt.json"),
            historical_R18_fit_select_control_overlap=historical_count, R18_protected_input_overlap=0,
            landmarks_sha256=cache_stamp["files"]["landmarks.tsv"], support_sha256=sha256(stage / "support.tsv"),
            slices=len(support), files=files,
        ))
        stage.rename(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports", "population", "r15-cache", "r17-support", "r18-registered",
                 "r18-lock", "r18-access", "r18-index", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
