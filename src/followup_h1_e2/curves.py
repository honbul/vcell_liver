"""Audit public curve keys and release only matched anchor observations."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nextgen_h1_anchor.inputs import CROSS, json_write, source_registry
from nextgen_h1_v2.data import ROOT as H1, STUDY, partitions, source
from nextgen_h1_v2.predict import anchors_v1, old_receipts
from vc_build.cross_source_data import MODEL_COLS, read_raw, roster_maps, roster_table
from vc_build.gdsc_raw_window import RAW_SHA256, collect_wells, normalize
from vc_build.io import sha256
from vc_build.licob_dose_sources import CTRP_QC

SCREENS = ("CTRPv2", "GDSC2")
ROOT = H1.parents[1] / "followup_20261124/h1_e2"
CTRP = Path("/mnt/S1/sdata/pubdata/DepMap/DrugScreens/CTRP_CTD2")
GDSC = CROSS.parents[1] / "v0plus-public-priors/gdsc2-raw-8.5/GDSC2_public_raw_data_27Oct23.zip"
PAIR = ["cell_id", "compound_id"]
PARENT = "native_compound_id"
DOSE = "dose_uM"
CURVE = [*PAIR, "native_sample", PARENT, "curve_id", "time"]
REG_COLUMNS = ["source_id", *PAIR, PARENT, "dose_min_uM", "dose_max_uM", "source_run"]
CONTRACT = Path(__file__).resolve().parents[2] / "experiments/followup_h1_e2/contract.json"


def read_json(path):
    return json.loads(path.read_text())


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Registered hash mismatch: {path}")
    return path


def registered(screen):
    seal_path = H1.parents[1] / "nextgen_h123/h1_anchor/support.json"
    seal = read_json(seal_path)
    frames = []
    input_hashes = {str(seal_path): sha256(seal_path)}
    for root, rel, group in source_registry():
        path = root / rel
        receipt = checked(path, seal["file_sha256"][str(path)])
        input_hashes[str(path)] = seal["file_sha256"][str(path)]
        block = pd.read_parquet(receipt, columns=[*REG_COLUMNS, group], filters=[("source_id", "==", screen)])
        frames.append(block.rename(columns={group: "lineage"}))
    rows = pd.concat(frames, ignore_index=True)
    rows[PARENT] = rows[PARENT].astype(str)
    if rows.duplicated(PAIR).any() or rows[["lineage", PARENT]].isna().any().any():
        raise ValueError("Registered exact sample–parent rows are not unique")
    if rows.groupby(PARENT).compound_id.nunique().max() != 1:
        raise ValueError("One native drug ID maps to multiple exact parents")
    return rows, input_hashes


def task_roster(task):
    _, dev, meta, held = source(task, query=True)
    freeze = H1 / "frozen-a" / task / "freeze.json"
    checked(freeze, read_json(H1 / "frozen_receipt.json")["packages"][task])
    manifest = read_json(freeze)
    original = anchors_v1(task, old_receipts()[0])
    index = original["index"]
    fit, _ = next(partitions(dev["groups"], "h1v2-anchor-selector"))
    if (set(dev["groups"][fit]) & set(held["groups"]) or
            not np.array_equal(held["groups"][index], original["group"]) or
            manifest["q8"] != held["panel"][original["q"]].tolist()):
        raise ValueError("Curve audit roster differs from the frozen Q8 trial")
    return dev, meta, held, fit, manifest, original


def raw_hashes(screen):
    receipt = CROSS / "receipt.json"
    inputs = read_json(receipt)["inputs"]
    names = ("depmap_models", "ctrp_cells", "ctrp_experiments", "ctrp_compounds") if screen == "CTRPv2" else ("depmap_models",)
    paths = {name: checked(Path(inputs[name]["path"]), inputs[name]["sha256"]) for name in names}
    hashes = {str(paths[name]): inputs[name]["sha256"] for name in names}
    hashes[str(receipt)] = sha256(receipt)
    if screen == "CTRPv2":
        for name in ("README", "COLUMNS", "per_assay_plate", "per_cpd_well", "per_cpd_post_qc"):
            suff = ("v20._" + name) if name.isupper() else (
                "v20.meta." + name if name == "per_assay_plate" else "v20.data." + name)
            path = CTRP / f"{suff}.txt"
            paths[name] = path
            hashes[str(path)] = sha256(path)
        if hashes[str(paths["per_cpd_post_qc"])] != CTRP_QC:
            raise ValueError("CTRP QC-point file differs from vc_build_v1")
    else:
        paths["archive"] = checked(GDSC, RAW_SHA256)
        hashes[str(GDSC)] = RAW_SHA256
    return paths, hashes


def ctrp_extract(reg, anchors, paths):
    cells = read_raw(paths["ctrp_cells"], ["ccl_name", "master_ccl_id"], sep="\t")
    models = read_raw(paths["depmap_models"], MODEL_COLS)
    cells = roster_maps(roster_table(cells, "CTRPv2", "ccl_name", "master_ccl_id"), models)
    cells = cells[["source_key", "cell_id"]].drop_duplicates().dropna()
    runs = read_raw(paths["ctrp_experiments"], ["experiment_id", "master_ccl_id", "snp_fp_status"], sep="\t")
    runs = runs[runs.snp_fp_status.eq("SNP-matched-reference")].merge(
        cells, left_on="master_ccl_id", right_on="source_key", validate="many_to_one")
    runs = runs.rename(columns={"master_ccl_id": "native_sample", "experiment_id": "curve_id"})
    runs = runs[["curve_id", "native_sample", "cell_id"]].drop_duplicates()
    if not runs.curve_id.is_unique:
        raise ValueError("Conflicting CTRP sample identity")

    fields = {"experiment_id": "curve_id", "master_cpd_id": PARENT, "cpd_conc_umol": DOSE}
    keys = [*CURVE[:-1], DOSE]
    by_experiment = ["curve_id", PARENT, DOSE]
    native = reg[[*PAIR, PARENT]]
    native_ids = set(native[PARENT])
    qc_keys = []
    for chunk in pd.read_csv(paths["per_cpd_post_qc"], sep="\t", usecols=list(fields), chunksize=350_000,
                             dtype={"experiment_id": str, "master_cpd_id": str}):
        block = chunk[chunk.master_cpd_id.isin(native_ids)].rename(columns=fields)
        block = block.merge(runs, on="curve_id", validate="many_to_one").merge(
            native, on=["cell_id", PARENT], validate="many_to_one")
        qc_keys.append(block.loc[block[DOSE].gt(0), keys])
    qc_points = pd.concat(qc_keys, ignore_index=True).drop_duplicates()
    anchor_keys = qc_points.merge(anchors, on=PAIR, validate="many_to_one")

    plate = read_raw(paths["per_assay_plate"],
        ["experiment_id", "assay_plate_barcode", "num_dmso_wells", "dmso_plate_avg_log2"], sep="\t")
    plate = plate[pd.to_numeric(plate.num_dmso_wells, errors="coerce").gt(0)
                  & pd.to_numeric(plate.dmso_plate_avg_log2, errors="coerce").notna()]
    plate = plate.rename(columns={"experiment_id": "curve_id", "assay_plate_barcode": "rep_id"})
    if plate.duplicated(["curve_id", "rep_id"]).any():
        raise ValueError("Plate metadata has duplicate experiment and barcode")
    frames, identity_frames = [], []
    columns = fields | {"assay_plate_barcode": "rep_id", "bsub_value_log2": "log_pv"}
    for chunk in pd.read_csv(paths["per_cpd_well"], sep="\t", usecols=list(columns), chunksize=350_000,
                             dtype={"experiment_id": str, "master_cpd_id": str, "assay_plate_barcode": str}):
        block = chunk[chunk.master_cpd_id.isin(native_ids)].rename(columns=columns)
        ok = block.merge(qc_points, on=by_experiment, validate="many_to_one").merge(
            plate[["curve_id", "rep_id"]], on=["curve_id", "rep_id"], validate="many_to_one")
        identity_frames.append(ok[[*keys, "rep_id"]].drop_duplicates())
        measured = ok.merge(anchor_keys[by_experiment], on=by_experiment, validate="many_to_one")
        frames.append(measured.loc[np.isfinite(measured.log_pv), [*keys, "rep_id", "log_pv"]])
    raw_identity = pd.concat(identity_frames, ignore_index=True).drop_duplicates().assign(time="72h")
    data = pd.concat(frames, ignore_index=True)
    data["viability"] = np.exp2(np.minimum(data.pop("log_pv"), 0.0))
    data["time"] = "72h"
    return raw_identity, data, {"published_qc_keys": len(qc_points), "anchor_qc_keys": len(anchor_keys),
                                 "raw_plate_matched_keys": len(raw_identity)}


def gdsc_extract(reg, anchors, paths):
    model = read_raw(paths["depmap_models"], ["ModelID", "SangerModelID"])
    native = reg.merge(model[model.SangerModelID.ne("")], left_on="cell_id", right_on="ModelID",
                       validate="many_to_one").rename(columns={"SangerModelID": "SANGER_MODEL_ID", PARENT: "DRUG_ID"})
    physical = ["SANGER_MODEL_ID", "DRUG_ID"]
    if native.duplicated(physical).any():
        raise ValueError("Conflicting GDSC sample and native drug identity")
    controls, wells, total = collect_wells(paths["archive"], native)
    native_fields = [*PAIR, *physical, "dose_min_uM", "dose_max_uM"]
    wells = wells.merge(native[native_fields], on=physical, validate="many_to_one")
    duration_counts = wells.DURATION.value_counts().sort_index()
    wells = wells.loc[wells.DURATION.eq(4) & wells.CONC.gt(0)].copy()
    raw_identity = gdsc_identity(wells).drop_duplicates()
    picked = wells.merge(anchors, on=PAIR, validate="many_to_one")
    normalized, plate_qc = normalize(controls, picked)
    normalized = pd.concat([gdsc_identity(normalized), normalized[
        ["viability", "dose_min_uM", "dose_max_uM"]]], axis=1)
    return raw_identity, normalized, {"archive_rows": total, "release_duration_distribution": {
        str(k): int(v) for k, v in duration_counts.items()}, **plate_qc}


def gdsc_identity(frame):
    return frame.assign(curve_id=(frame.BARCODE + "|" + frame.SCAN_ID + "|" +
                                 frame.DATE_CREATED.astype(str) + "|" + frame.library),
                        time="DURATION=4").rename(
        columns={"CONC": DOSE, "POSITION": "rep_id", "SANGER_MODEL_ID": "native_sample",
                 "DRUG_ID": PARENT})[[*CURVE, DOSE, "rep_id"]]


def curve_blocks(data, screen):
    result, failures = [], Counter()
    for identity, group in data.groupby(CURVE, sort=True):
        grid = tuple(sorted(group[DOSE].unique()))
        if len(grid) != 16 and screen == "CTRPv2":
            failures["incomplete_published_16_dose_design"] += 1
            continue
        if screen == "GDSC2":
            bounds = group[["dose_min_uM", "dose_max_uM"]].drop_duplicates().to_numpy()
            if len(grid) < 6 or len(bounds) != 1 or not np.allclose([grid[0], grid[-1]], bounds[0], rtol=.02, atol=1e-12):
                failures["incomplete_or_wrong_native_dose_window"] += 1
                continue
        clean = group[~group.duplicated([DOSE, "rep_id"], keep=False)].sort_values([DOSE, "rep_id"])
        at_dose = clean.groupby(DOSE).viability.agg(list)
        if tuple(at_dose.index) != grid:
            failures["ambiguous_replicate_keys"] += 1
            continue
        first = [v[0] for v in at_dose]
        second = [v[1] if len(v) > 1 else np.nan for v in at_dose]
        result.append((*identity, list(grid), first, second, bool(np.isfinite(second).all())))
    return pd.DataFrame(result, columns=[*CURVE, "grid", "first", "second", "two_reps"]), dict(failures)


def extract(screen, out):
    reg, hashes = registered(screen)
    raw_paths, raw = raw_hashes(screen)
    hashes.update(raw)
    all_panels, all_anchors = set(), set()
    for fold in range(3):
        _, _, held, _, manifest, _ = task_roster(f"{screen}-{fold}")
        all_panels.update(held["panel"])
        all_anchors.update(manifest["a8"])
    reg = reg[reg.compound_id.isin(all_panels)]
    anchors = reg.loc[reg.compound_id.isin(all_anchors), PAIR].drop_duplicates()
    load = ctrp_extract if screen == "CTRPv2" else gdsc_extract
    raw_identity, observations, qc = load(reg, anchors, raw_paths)
    curves, failures = curve_blocks(observations, screen)
    out.mkdir()
    tables = {"registered": reg, "raw_keys": raw_identity, "curves": curves}
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    json_write(out / "receipt.json", {"screen": screen, "sources": hashes,
        "raw_well_identity_rows": len(raw_identity), "anchor_measured_rows": len(observations),
        "anchor_resolved_curves": len(curves), "anchor_complete_two_rep_curves": int(curves.two_reps.sum()),
        "anchor_incomplete_curves": failures, "qc": qc, "query_values_exported": False,
        "files": {f"{name}.parquet": sha256(out / f"{name}.parquet") for name in tables},
        "contract_sha256": sha256(CONTRACT), "original_study_receipt_sha256": sha256(STUDY / "receipt.json")})
    print(screen, len(curves), "resolved curves", curves.two_reps.sum(), "two-repeat", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", choices=SCREENS)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    extract(args.screen, args.out)


if __name__ == "__main__":
    main()
