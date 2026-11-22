"""Register R18 support and exposure using existing metadata only."""

import argparse
from pathlib import Path
from zlib import crc32

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.lincs_transfer import clean, structural_keys
from src.r15_lincs.cache import sort_dose
from src.r15_lincs.data import DOSE, STUDIES, TIME, asset, hepatic_lines, numbers, table
from src.r15_lincs.design import fold_id
from src.r16_lincs.audit import DET, RNA, INST, SIG, add_units, geo_exposure, token
from src.r17_lincs.inventory import (
    CMAP, ROOT, SOURCES, in_footprint, liver_cells, previous, register_footprint,
)

LOW = (.04, 3.33)
HISTORY = (*SOURCES, "R17")
GEO_COLUMNS = ["inst_id", *RNA, *DET, "cell_id", "pert_id", "pert_iname",
               "pert_type", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]
PHYSICAL = ["physical_axis", "plate", "well"]


def geo_metadata(study, prior):
    inst = table(asset(ROOT, study, "inst_info"), usecols=lambda col: col in GEO_COLUMNS)
    inst = add_units(inst)
    sig_path = asset(ROOT, study, "sig_info")
    source = table(sig_path, usecols=["sig_id", "cell_id", "pert_id", "pert_type", "distil_id",
                                     "pert_idose", "pert_itime"]).rename(columns={"distil_id": "members"})
    source["dose"] = numbers(source.pert_idose, DOSE)
    source["hours"] = numbers(source.pert_itime, TIME)
    metrics_path = asset(ROOT, study, "sig_metrics")
    metrics = table(metrics_path, usecols=["sig_id", "distil_cc_q75", "distil_nsample"])
    source = source.merge(metrics, on="sig_id", how="left", validate="one_to_one")
    source["qc"] = (pd.to_numeric(source.distil_nsample, errors="coerce").ge(2)
                    & pd.to_numeric(source.distil_cc_q75, errors="coerce").ge(.2))
    identity_path = prior / "audit" / study / "identities.tsv"
    stamp = read_json(identity_path.parent / "receipt.json")
    if sha256(identity_path) != stamp["files"][identity_path.name]:
        raise ValueError(f"Changed R15 identity registry: {identity_path}")
    identity = table(identity_path, usecols=["pert_id", "parent"])
    source = attach_parent(source, identity)
    inst["qc_well"] = True
    return inst, source


def attach_parent(source, identity):
    mapping = identity.set_index("pert_id", verify_integrity=True).parent
    return source.assign(parent=source.pert_id.map(mapping).fillna("").replace(r"^UNRESOLVED:.*", "", regex=True))


def cmap_metadata(base, wanted):
    origin = ROOT / "LINCS2020"
    inst_path = base / "vcell_liver_transigen_adaptation_b1/assets/raw/cmap2020/instinfo_beta.txt"
    liver = set(liver_cells(origin / "cellinfo_beta.txt", cmap=True).cell)
    identity = table(origin / "compoundinfo_beta.txt", usecols=["pert_id", "canonical_smiles"]).drop_duplicates("pert_id")
    all_sig = add_units(table(origin / "siginfo_beta.txt", usecols=SIG)).rename(
        columns={"cell_iname": "cell_id", "distil_ids": "members"})
    low = all_sig.hours.eq(24) & all_sig.dose.between(*LOW) & all_sig.pert_type.eq("trt_cp")
    compound_ids = set(all_sig.loc[all_sig.cell_id.isin(liver) & all_sig.pert_type.eq("trt_cp"), "pert_id"])
    identity = identity.loc[identity.pert_id.isin(compound_ids)].copy()
    identity["parent"] = identity.canonical_smiles.map(lambda s: structural_keys(clean(s))[1])
    # All hepatic signatures are retained for alias and exposure checks.
    sig = all_sig.loc[all_sig.cell_id.isin(liver) | (low & all_sig.pert_id.isin(compound_ids))].copy()
    sig["qc"] = (pd.to_numeric(sig.nsample, errors="coerce").ge(2)
                 & pd.to_numeric(sig.cc_q75, errors="coerce").ge(.2) & sig.qc_pass.eq("1"))
    sig = attach_parent(sig, identity).drop_duplicates("sig_id")
    allowed_ids = set(sig.loc[low.reindex(sig.index), "members"].str.split("|").explode())
    chunks = []
    for chunk in pd.read_csv(inst_path, sep="\t", dtype=str, keep_default_na=False, usecols=INST, chunksize=160000):
        hepatic = chunk.cell_iname.isin(liver)
        control = chunk.pert_type.eq("ctl_vehicle") & chunk.qc_pass.eq("1")
        chunks.append(chunk.loc[hepatic | control | chunk.sample_id.isin(allowed_ids)
                                | chunk.sample_id.isin(wanted)])
    inst = add_units(pd.concat(chunks, ignore_index=True)).rename(
        columns={"sample_id": "inst_id", "cell_iname": "cell_id", "cmap_name": "pert_iname"})
    inst["qc_well"] = inst.qc_pass.eq("1")
    return inst.drop_duplicates("inst_id"), sig


def prepare(inst):
    inst = inst.copy()
    is_rna = set(RNA).issubset(inst.columns)
    if not is_rna:
        inst[RNA] = ""
    for col in [*RNA, *DET]:
        if col not in inst:
            inst[col] = ""
        inst[col] = token(inst[col])
    axis = RNA if is_rna else DET
    inst[PHYSICAL] = pd.DataFrame({"physical_axis": axis[0].split("_")[0],
                                  "plate": inst[axis[0]], "well": inst[axis[1]]}, index=inst.index)
    return inst


def membership(sig):
    links = sig.drop(columns="members").assign(inst_id=sig.members.str.split("|")).explode("inst_id")
    return links.loc[links.inst_id.ne("")].drop_duplicates(["sig_id", "inst_id"])


def combine(sig, inst, *, exact_dose=True):
    fields = ["inst_id", "cell_id", "pert_id", *RNA, *DET, *PHYSICAL, "qc_well", "hours", "dose"]
    joined = membership(sig).merge(inst[fields], on=["inst_id", "cell_id", "pert_id"],
                                   suffixes=("", "_well"), validate="many_to_one")
    valid = (joined.plate.ne("") & joined.well.str.fullmatch(r"[A-Z]+\d{2}")
             & joined.hours.eq(joined.hours_well) & joined.dose_well.notna())
    if exact_dose:
        valid &= joined.dose.eq(joined.dose_well)
    return joined.loc[valid].copy()


def flags(study, sig, inst, historical, geo):
    footprints, signatures, _, parent_sets = historical
    links = membership(sig)[["sig_id", "inst_id"]]
    links = links.merge(inst[["inst_id", *RNA, *DET]], on="inst_id", how="left").fillna("")
    for name in HISTORY:
        link_overlap = links.loc[in_footprint(links, footprints[name]), "sig_id"] if name in footprints else []
        sig[name] = sig.sig_id.isin(signatures[study, name]) | sig.sig_id.isin(link_overlap)
        parents = parent_sets[name]
        sig[f"{name}_parent"] = sig.parent.ne("") & sig.parent.isin(parents)
    sig["geo_duplicate"] = False
    if study == CMAP:
        sig_ids, members, physical, _ = geo
        hit = links.loc[in_footprint(links, (members, physical)), "sig_id"]
        sig["geo_duplicate"] = token(sig.sig_id).isin(sig_ids) | sig.sig_id.isin(hit)
    return sig


def parent_history(data, historical):
    parent_sets = {name: set() for name in HISTORY}
    footprints, old_sigs, _ = historical
    charts = {}
    for study, (inst, sig) in data.items():
        if study == CMAP:
            liver = set(liver_cells(ROOT / "LINCS2020/cellinfo_beta.txt", cmap=True).cell)
        else:
            liver = hepatic_lines(ROOT, study)
        registered = sig.loc[sig.cell_id.isin(liver) & sig.parent.ne(""), ["sig_id", "members", "parent"]]
        charts[study] = membership(registered).merge(
            inst[["inst_id", *RNA, *DET]], on="inst_id", how="left").fillna("")
    for name in HISTORY:
        footprint = footprints.get(name)
        for study, chart in charts.items():
            direct = chart.sig_id.isin(old_sigs[study, name])
            parent = chart.loc[direct | in_footprint(chart, footprint) if footprint else direct, "parent"]
            parent_sets[name].update(parent)
    return parent_sets


def roles(inst, study, geo):
    vehicle = inst.pert_type.eq("ctl_vehicle") & inst.qc_well & inst.well.str.fullmatch(r"[A-Z]+\d{2}")
    vehicle &= inst.pert_id.str.upper().eq("DMSO") | inst.pert_iname.str.upper().eq("DMSO")
    controls = inst.loc[vehicle & inst.plate.ne("")].copy()
    if study == CMAP:
        controls = controls.loc[~in_footprint(controls, (geo[1], geo[2]))].copy()
    # A release may list multiple IDs for one coordinate. Count it once before CRC assignment.
    controls = controls.sort_values("inst_id").drop_duplicates(PHYSICAL)
    controls["crc"] = controls.inst_id.map(lambda key: crc32(key.encode()))
    controls = controls.sort_values(["plate", "crc", "inst_id"])
    controls["role"] = controls.groupby("plate").cumcount().mod(3).map(dict(enumerate("ABC")))
    sizes = controls.groupby("plate").size()
    return controls.drop(columns="crc"), sizes


def source_inputs(sig, inst, sizes, wanted, hour, band, *, exact_dose=True):
    selected = sig.loc[sig.pert_type.eq("trt_cp") & sig.hours.eq(hour) & sig.dose.between(*band)
                       & sig.parent.isin(wanted)].copy()
    if selected.empty:
        return combine(selected, inst, exact_dose=exact_dose)
    counts = selected.groupby("cell_id").parent.nunique().sort_index().sort_values(ascending=False, kind="stable")
    selected = selected.loc[selected.cell_id.isin(counts.index[:6])]
    selected = sort_dose(selected).drop_duplicates(["parent", "cell_id"])
    rows = combine(selected, inst, exact_dose=exact_dose)
    return rows.loc[rows.qc_well & rows.plate.map(sizes).fillna(0).ge(2)].drop_duplicates("inst_id")


def support(label, study, sig, rows, controls, sizes, sources):
    lines = []
    levels = [("all", sig, rows)]
    if label == "C2":
        levels.extend((dose, sig.loc[sig.dose.eq(dose)], rows.loc[rows.dose.eq(dose)]) for dose in sorted(sig.dose.unique()))
    for dose, signatures, wells in levels:
        exposed = list(HISTORY)
        history = wells[exposed].any(axis=1)
        physical = wells.drop_duplicates(PHYSICAL)
        qc = wells.loc[wells.qc & wells.qc_well & wells.parent.ne("")]
        kept = qc if label == "C1" else qc.loc[~qc[exposed].any(axis=1) & ~qc.geo_duplicate]
        kept = kept.drop_duplicates(PHYSICAL)
        supported = kept.loc[kept.plate.map(sizes).fillna(0).ge(3)].copy()
        transfer = supported.loc[supported.parent.isin(sources.parent)]
        relevant = controls.loc[controls.plate.isin(supported.plate)]
        lines.append(dict(cohort=label, study=study, hours=int(sig.hours.iloc[0]), dose_uM=dose,
                          signatures=len(signatures), linked_parents=physical.loc[physical.parent.ne(""), "parent"].nunique(),
                          linked_wells=len(physical), qc_parents=qc.parent.nunique(),
                          qc_wells=len(qc.drop_duplicates(PHYSICAL)),
                          prior_exact_signatures=wells.loc[history, "sig_id"].nunique(),
                          geo_duplicate_signatures=signatures.loc[signatures.geo_duplicate, "sig_id"].nunique(),
                          clean_qc_parents=kept.parent.nunique(), clean_qc_wells=len(kept),
                          clean_qc_parent_preexposed=kept.loc[kept[[f"{name}_parent" for name in HISTORY]].any(axis=1),
                                                              "parent"].nunique(),
                          clean_plates=kept.plate.nunique(), A_parents=supported.parent.nunique(),
                          A_wells=len(supported), A_plates=supported.plate.nunique(),
                          B_parents=transfer.parent.nunique(), B_wells=len(transfer), B_plates=transfer.plate.nunique(),
                          DMSO_wells=len(relevant), minimum_DMSO=int(sizes.reindex(supported.plate.unique()).min())
                          if len(supported) else 0, **{
                              f"{name}_wells": len(wells.loc[wells[name]].drop_duplicates(PHYSICAL))
                              for name in exposed}))
    return lines


def donor_output(rows, label, study):
    columns = ["sig_id", "inst_id", "pert_id", "parent", "cell_id", "hours", "dose", *PHYSICAL, *RNA, *DET]
    selected = rows.loc[:, columns]
    if not selected.columns.is_unique:
        raise ValueError(f"{study}: ambiguous source donor metadata columns")
    return selected.assign(cohort=label, study=study)


def read_metadata(base, prior, c1, sources_c1):
    print("Reading GEO release overlap metadata", flush=True)
    geo = geo_exposure(prior)
    geo_data = {study: geo_metadata(study, prior) for study in STUDIES}
    print("Reading CMap metadata", flush=True)
    cmap, sig_cmap = cmap_metadata(base, set(c1.inst_id) | set(sources_c1.inst_id))
    return geo, geo_data, cmap, sig_cmap


def save_tables(args):
    args.out.mkdir(parents=True, exist_ok=False)
    prior = args.base / "vcell_liver_r15_lincs/v1"
    r16 = args.base / "vcell_liver_r16_lincs/v1"
    for sub, names in {"audit": ["target_rows.tsv", "source_wells.tsv"],
                       "confirmation-cache": ["target_rows.tsv", "transfer_rows.tsv"]}.items():
        for name in names:
            stamp = read_json(r16 / sub / "receipt.json")["files"][name]
            if sha256(r16 / sub / name) != stamp:
                raise ValueError("Changed R16 cohort registry")
    c1 = table(r16 / "confirmation-cache/target_rows.tsv")
    c1 = c1.loc[c1.cell_id.eq("HEPG2")].copy()
    for name in ("hours", "dose"):
        c1[name] = pd.to_numeric(c1[name])
    c1["qc"] = c1.quality.eq("True")
    sources_c1 = table(r16 / "audit/source_wells.tsv").drop(columns="cell_id", errors="ignore").rename(
        columns={"sample_id": "inst_id", "cell_iname": "cell_id"})
    for name in ("hours", "dose"):
        sources_c1[name] = pd.to_numeric(sources_c1[name])
    c1_transfer = table(r16 / "confirmation-cache/transfer_rows.tsv")
    if set(sources_c1.parent) != set(c1_transfer.parent):
        raise ValueError("R16 transfer cache source support differs")
    geo, geo_data, cmap, sig_cmap = read_metadata(args.base, prior, c1, sources_c1)
    liver = set(liver_cells(ROOT / "LINCS2020/cellinfo_beta.txt", cmap=True).cell)
    historical = previous(args.base, cmap.loc[cmap.cell_id.isin(liver) & cmap.pert_type.eq("trt_cp")].rename(
        columns={"cell_id": "cell_iname"}))
    r17_target = table(prior / "cache/GSE92742/target_rows.tsv")
    r17_plates = set(table(args.base / "vcell_liver_r17_lincs/v1/support/assignments.tsv").rna_plate)
    r17_target = r17_target.loc[r17_target.cell_id.eq("HEPG2") & pd.to_numeric(r17_target.hours).eq(6)
                                & r17_target.quality.eq("True") & r17_target.prior_exact.eq("False")
                                & r17_target.rna_plate.isin(r17_plates)]
    historical[0]["R17"] = register_footprint(r17_target)
    historical[1]["GSE92742", "R17"].update(r17_target.sig_id)
    historical[2]["R17_role_assignments"] = args.base / "vcell_liver_r17_lincs/v1/support/assignments.tsv"
    data = {study: (prepare(inst), sig) for study, (inst, sig) in {**geo_data, CMAP: (cmap, sig_cmap)}.items()}
    historical = (*historical, parent_history(data, historical))
    control_bank = {study: roles(inst, study, geo) for study, (inst, _) in data.items()}
    exposure_rows, supported_rows, donor_rows, stats, fits, fit_folds = [], [], [], [], [], []

    c1_sig = c1.drop_duplicates("sig_id").merge(sig_cmap[["sig_id", "members"]], on="sig_id", validate="one_to_one")
    c1_sig = flags(CMAP, c1_sig, data[CMAP][0], historical, geo)
    raw_meta = data[CMAP][0][["inst_id", *RNA, *DET, *PHYSICAL, "qc_well"]]
    c1 = c1.drop(columns=RNA).merge(raw_meta, on="inst_id", validate="one_to_one")
    parent_fields = [f"{name}_parent" for name in HISTORY]
    c1 = c1.drop(columns=["B1_parent", "R8_parent"])
    c1 = c1.merge(c1_sig[["sig_id", *HISTORY, *parent_fields, "geo_duplicate"]], on="sig_id", validate="many_to_one")
    c1["dose_well"] = c1.dose
    if c1.geo_duplicate.any() or c1.duplicated(PHYSICAL).any():
        raise ValueError("R16 C1 no longer has the recorded release-clean physical rows")
    sources_c1 = sources_c1.drop(columns=RNA).merge(raw_meta, on="inst_id", validate="one_to_one")
    print("Registering cohort support", flush=True)
    cohorts = [("C1", CMAP, c1_sig, c1, sources_c1)]
    for label, study, cell, hour, band in [
        ("C2", "GSE70138", "HEPG2", 24, LOW), ("C2", CMAP, "HEPG2", 24, LOW),
        ("C3", "GSE92742", "HUH7", 6, (3., 3.)), ("C3", "GSE92742", "HUH7", 24, (3., 3.)),
    ]:
        inst, sig = data[study]
        subset = sig.loc[sig.pert_type.eq("trt_cp") & sig.cell_id.eq(cell) & sig.hours.eq(hour)
                         & sig.dose.between(*band)].copy()
        subset = flags(study, subset, inst, historical, geo)
        linked = combine(subset, inst, exact_dose=label != "C3")
        excluded = hepatic_lines(ROOT, study) if study != CMAP else set(
            liver_cells(ROOT / "LINCS2020/cellinfo_beta.txt", cmap=True).cell
        ).union(*(hepatic_lines(ROOT, release) for release in STUDIES))
        candidates = sig.loc[~sig.cell_id.isin(excluded)].copy()
        if study == CMAP:
            scope = candidates.hours.eq(hour) & candidates.dose.between(*LOW) & candidates.parent.isin(subset.parent)
            candidates = candidates.loc[scope].copy()
            links = membership(candidates[["sig_id", "members"]]).merge(
                inst[["inst_id", *RNA, *DET]], on="inst_id", how="left").fillna("")
            overlap = links.loc[in_footprint(links, (geo[1], geo[2])), "sig_id"]
            candidates = candidates.loc[~token(candidates.sig_id).isin(geo[0]) & ~candidates.sig_id.isin(overlap)]
        wanted = set(subset.loc[subset.qc & ~subset[list(HISTORY)].any(axis=1) & ~subset.geo_duplicate, "parent"])
        donors = source_inputs(candidates, inst, control_bank[study][1], wanted, hour, band,
                               exact_dose=label != "C3")
        if label == "C3":
            # The GEO signature's rounded dose defines the requested cohort, while the
            # constituent metadata gives the physical concentration for transfer matching.
            observed = linked.loc[linked.qc & linked.parent.isin(wanted), "dose_well"]
            measured_band = (float(observed.min()), float(observed.max()))
            donors = donors.loc[donors.dose_well.between(*measured_band)]
            broad = source_inputs(candidates, inst, control_bank[study][1], wanted, hour, (0, 5),
                                  exact_dose=False)
            broad = broad.loc[broad.dose_well.between(*measured_band)]
            _, sizes = control_bank[study]
            train = sig.loc[sig.pert_type.eq("trt_cp") & sig.parent.ne("") & sig.cell_id.eq("HEPG2")
                            & sig.hours.eq(hour) & sig.dose.between(0, 5) & sig.qc]
            train = combine(train, inst, exact_dose=False).drop_duplicates(PHYSICAL)
            train = train.loc[train.plate.map(sizes).fillna(0).ge(3)]
            target = linked.loc[linked.qc & linked.parent.isin(wanted)
                                & linked.plate.map(sizes).fillna(0).ge(3)].drop_duplicates(PHYSICAL)
            for dose_rule, eligible, bsource in [
                ("nominal_3uM", train.loc[train.dose.eq(3) & train.dose_well.between(*measured_band)], donors),
                ("measured_HUH7_range", train.loc[train.dose_well.between(*measured_band)], broad),
                ("same_hour_all_doses_under_5uM", train, broad),
            ]:
                fits.append(dict(cohort=label, hours=hour, dose_rule=dose_rule, dose_min=measured_band[0],
                                 dose_max=measured_band[1], HEPG2_parents=eligible.parent.nunique(),
                                 HEPG2_wells=len(eligible), HEPG2_plates=eligible.plate.nunique(),
                                 B_candidate_parents=bsource.parent.nunique(), B_candidate_wells=bsource.inst_id.nunique(),
                                 HUH7_and_HEPG2_B_parents=eligible.loc[eligible.parent.isin(bsource.parent)
                                                                     & eligible.parent.isin(wanted), "parent"].nunique()))
                for scenario in ("A", "B"):
                    query = target if scenario == "A" else target.loc[target.parent.isin(bsource.parent)]
                    train_source = eligible if scenario == "A" else eligible.loc[eligible.parent.isin(bsource.parent)]
                    source_fold = train_source.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
                    query_fold = query.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
                    for fold in range(3):
                        outer_train = train_source.loc[source_fold.ne(fold)]
                        inner = outer_train.parent.map(lambda parent: fold_id(parent, f"R15-inner-{fold}", 5)).eq(0)
                        for split, group in [("HEPG2_inner_fit", outer_train.loc[~inner]),
                                             ("HEPG2_select", outer_train.loc[inner]),
                                             ("HUH7_score", query.loc[query_fold.eq(fold)])]:
                            fit_folds.append(dict(hours=hour, dose_rule=dose_rule, scenario=scenario,
                                                  fold=fold, split=split, parents=group.parent.nunique(), wells=len(group)))
            donors = broad
        cohorts.append((label, study, subset, linked, donors))

    for label, study, sig, rows, donors in cohorts:
        controls, sizes = control_bank[study]
        rows["cohort"] = label
        rows["study"] = study
        rows["DMSO_count"] = rows.plate.map(sizes).fillna(0).astype(int)
        rows["all_prior_exact"] = rows[[*SOURCES, "R17"]].any(axis=1)
        rows["A_eligible"] = rows.qc & rows.qc_well & rows.parent.ne("") & rows.DMSO_count.ge(3)
        if label != "C1":
            rows["A_eligible"] &= ~rows.all_prior_exact & ~rows.geo_duplicate
        rows["B_eligible"] = rows.A_eligible & rows.parent.isin(donors.parent)
        primary = rows.loc[rows.A_eligible].drop_duplicates(PHYSICAL)
        exposure_rows.append(rows)
        # Reserve entire QC signatures and constituents even when the physical plate fails 3-role support.
        reserved = sig.loc[sig.qc & sig.parent.ne("")]
        if label != "C1":
            reserved = reserved.loc[~reserved[[*SOURCES, "R17"]].any(axis=1) & ~reserved.geo_duplicate]
        linked = membership(reserved)[["sig_id", "inst_id", "parent", "pert_id", "cell_id", "hours", "dose"]].merge(
            data[study][0][["inst_id", *RNA, *DET, *PHYSICAL]], on="inst_id", how="left", validate="many_to_one")
        linked["cohort"] = label
        linked["study"] = study
        supported_rows.append(linked)
        stats.extend(support(label, study, sig, rows, controls, sizes, donors))
        donor_rows.append(donor_output(donors.loc[donors.parent.isin(primary.parent)], label, study))

    columns = ["cohort", "study", "sig_id", "inst_id", "parent", "pert_id", "cell_id", "hours", "dose", "dose_well",
               *PHYSICAL, *RNA, *DET, "qc", "qc_well", *HISTORY, *parent_fields, "geo_duplicate", "DMSO_count",
               "all_prior_exact", "A_eligible", "B_eligible"]
    exposure = pd.concat(exposure_rows, ignore_index=True)
    exposure[columns].to_csv(args.out / "exposure_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(stats).to_csv(args.out / "support.tsv", sep="\t", index=False)
    pd.DataFrame(fits).to_csv(args.out / "c3_transfer_support.tsv", sep="\t", index=False)
    pd.DataFrame(fit_folds).to_csv(args.out / "c3_zero_shot_fold_support.tsv", sep="\t", index=False)
    output_registries(args.out, data, control_bank, exposure, supported_rows, donor_rows)
    write_json(args.out / "receipt.json", {
        "code_sha256": sha256(__file__), "R16_registries": {s: sha256(r16 / s / "receipt.json")
                                                           for s in ("audit", "confirmation-cache")},
        "historical_inputs": {name: sha256(path) for name, path in historical[2].items()},
        "geo_metadata": geo[3], "expression_matrices_read": False,
        "files": {p.name: sha256(p) for p in args.out.iterdir()},
    })


def output_registries(out, data, controls, exposure, rows, donors):
    reserve = pd.concat(rows, ignore_index=True).fillna("")
    is_eligible = exposure.loc[exposure.A_eligible].drop_duplicates(["cohort", "study", *PHYSICAL])
    plates = is_eligible[["study", "plate"]].drop_duplicates()
    dmso = pd.concat([roles.assign(study=study) for study, (roles, _) in controls.items()], ignore_index=True)
    dmso = dmso.merge(plates, on=["study", "plate"], validate="many_to_one")
    count = dmso.groupby(["study", "plate", "role"]).size().unstack().reindex(columns=list("ABC"))
    if (count.isna() | count.lt(1)).any().any():
        raise ValueError("Missing role among retained plates")
    dmso[["study", "inst_id", *PHYSICAL, *RNA, *DET, "role"]].to_csv(
        out / "r18_control_assignments.tsv", sep="\t", index=False)
    count.reset_index().to_csv(out / "r18_control_support.tsv", sep="\t", index=False)

    eligible_cols = [*PHYSICAL, "parent", "hours", "cohort", "study", "B_eligible"]
    eligible = is_eligible[eligible_cols].copy()
    folds, position = [], []
    for group, part in eligible.groupby(["cohort", "study", "hours"]):
        for scenario, support_rows in [("A", part), ("B", part.loc[part.B_eligible])]:
            outer = support_rows.parent.map(lambda parent: fold_id(parent, "R15-outer", 3))
            for fold in range(3):
                fit = support_rows.loc[outer.ne(fold)]
                select = fit.parent.map(lambda parent: fold_id(parent, f"R15-inner-{fold}", 5)).eq(0)
                query = support_rows.loc[outer.eq(fold)]
                position_donors = fit.groupby(["physical_axis", "well"]).plate.agg(set)
                supported = [
                    len(position_donors.get((row.physical_axis, row.well), set()) - {row.plate}) > 0
                    for row in query.itertuples(index=False)
                ]
                position.append((*group, scenario, fold, len(query), sum(supported)))
                for split, assigned in [("inner_fit", fit.loc[~select]), ("select", fit.loc[select]),
                                        ("score", support_rows.loc[outer.eq(fold)])]:
                    folds.append((*group, scenario, fold, split, assigned.parent.nunique(), len(assigned)))
    pd.DataFrame(folds, columns=["cohort", "study", "hours", "scenario", "fold", "split", "parents",
                                 "physical_wells"]).to_csv(out / "r18_fold_support.tsv", sep="\t", index=False)

    pd.DataFrame(position, columns=["cohort", "study", "hours", "scenario", "fold", "query_wells",
                                    "same_coordinate_other_plate_fit_support"]).to_csv(
        out / "r18_position_support.tsv", sep="\t", index=False)

    aliases = []
    for study, (inst, sig) in data.items():
        for cohort, part in reserve.groupby("cohort", sort=True):
            foot = register_footprint(part)
            alias_wells = inst.loc[inst.pert_type.eq("trt_cp") & in_footprint(inst, foot)]
            alias_ids = foot[0] | set(token(alias_wells.inst_id))
            # Reuse the signature member registry so alternate QC/signature definitions cannot evade the hold.
            fields = ["sig_id", "parent", "pert_id", "cell_id", "hours", "dose", "members"]
            possible = membership(sig.loc[sig.pert_type.eq("trt_cp"), fields])
            matched = possible.loc[token(possible.inst_id).isin(alias_ids)]
            held = matched.merge(alias_wells[["inst_id", *PHYSICAL, *RNA, *DET]], on="inst_id", how="left")
            held = held.assign(cohort=cohort, study=study, record_type="release_well_or_signature_alias")
            aliases.append(held)
            unlinked = alias_wells.loc[~alias_wells.inst_id.isin(matched.inst_id)].copy()
            aliases.append(unlinked.assign(cohort=cohort, study=study, record_type="physical_alias", sig_id="", parent=""))
    registry = pd.concat([reserve.assign(record_type="target_signature_member"), *aliases], ignore_index=True)
    registry["DEV_fit_select_or_pretrain_allowed"] = False
    registry["CONFIRM_R18_B_source_donor_allowed"] = False
    rcols = ["cohort", "study", "record_type", "sig_id", "inst_id", "pert_id", "parent", "cell_id", "hours", "dose",
             *PHYSICAL, *RNA, *DET, "DEV_fit_select_or_pretrain_allowed", "CONFIRM_R18_B_source_donor_allowed"]
    registry = registry[rcols].fillna("").drop_duplicates()
    registry.to_csv(out / "r18_reserved_manifest.tsv", sep="\t", index=False)
    donor = pd.concat(donors, ignore_index=True).assign(record_type="B_nonhepatic_input_exception",
                                                       DEV_fit_select_or_pretrain_allowed=False,
                                                       CONFIRM_R18_B_source_donor_allowed=True)
    if any(in_footprint(donor, register_footprint(registry))):
        raise ValueError("A B source donor is also a reserved target physical well")
    donor[rcols].fillna("").drop_duplicates().to_csv(out / "r18_B_donor_exceptions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    save_tables(parser.parse_args())
