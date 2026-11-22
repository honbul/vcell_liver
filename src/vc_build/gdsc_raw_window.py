"""Reconstruct GDSC2 shared-window viability from independently controlled plates."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from .cross_source_data import read_raw
from .dose_window_diagnostics import TREATMENTS_HASH, WINDOW, compare, measured_points, published_curves
from .io import publish_json, sha256
from .v0plus_cv import training_inputs

RAW_SHA256 = "45dca6c707dcb8b80e73764adc91aeabe9c620af4a3f1f87648c80c35cd74d1b"
GDSC_MEMBER = "GDSC2_public_raw_data_27Oct23/GDSC2_public_raw_data_27Oct23.csv"
PLATE = ["BARCODE", "SCAN_ID"]
WELL = [*PLATE, "POSITION"]
KEY = ["cell_id", "compound_id"]
ENDPOINTS = ["native_auc", "window_auc", "dose_1uM"]
LIBRARY = r"^(L\d+)-D\d+-S$"


def source_rows(train, model_path):
    model = read_raw(model_path, ["ModelID", "SangerModelID"])
    model = model[model.SangerModelID.ne("")]
    core = train[train.source_id.eq("GDSC2")].copy()
    core = core.merge(model, left_on="cell_id", right_on="ModelID", validate="many_to_one")
    key = ["SangerModelID", "native_compound_id"]
    if core.duplicated(key).any():
        raise ValueError("GDSC2 model and drug IDs do not uniquely identify a registered curve")
    return core.rename(columns={"SangerModelID": "SANGER_MODEL_ID", "native_compound_id": "DRUG_ID"})


def collect_wells(archive, core):
    physical_cells = set(core.SANGER_MODEL_ID)
    drugs = set(core.DRUG_ID)
    controls, treatment = [], []
    count_all = 0
    fields = [*WELL, "DATE_CREATED", "SANGER_MODEL_ID", "TAG", "DRUG_ID",
              "CONC", "INTENSITY", "DURATION", "ASSAY"]
    dtype = {key: str for key in [*PLATE, "SANGER_MODEL_ID", "DRUG_ID", "TAG"]}
    with zipfile.ZipFile(archive) as zipped, zipped.open(GDSC_MEMBER) as stream:
        for chunk in pd.read_csv(stream, usecols=fields, dtype=dtype, chunksize=400_000):
            count_all += len(chunk)
            scope = chunk[chunk.SANGER_MODEL_ID.isin(physical_cells) & chunk.ASSAY.eq("Glo")]
            controls.append(scope.loc[scope.TAG.isin(["NC-1", "B"]), [*WELL, "SANGER_MODEL_ID", "TAG", "INTENSITY"]])
            selected = scope.loc[scope.DRUG_ID.isin(drugs) & scope.TAG.str.fullmatch(LIBRARY, na=False)].copy()
            selected["library"] = selected.TAG.str.extract(LIBRARY)[0]
            treatment.append(selected.drop(columns=["ASSAY", "TAG"]))
    return pd.concat(controls, ignore_index=True), pd.concat(treatment, ignore_index=True), count_all


def clean_positions(frame, columns):
    # Repeated tags at a position may be retained only when the requested observation is identical.
    unique = frame.drop_duplicates([*WELL, *columns])
    conflict = unique.duplicated(WELL, keep=False)
    return unique[~conflict], int(unique.loc[conflict, WELL].drop_duplicates().shape[0])


def normalize(controls, treatment):
    controls = controls.dropna(subset=["POSITION", "INTENSITY"])
    treatment = treatment.dropna(subset=["POSITION", "CONC", "INTENSITY", "DATE_CREATED", "DURATION"])
    controls = controls[np.isfinite(controls.INTENSITY)]
    treatment = treatment[np.isfinite(treatment.INTENSITY) & np.isfinite(treatment.CONC)
                          & treatment.CONC.gt(0)]
    all_scans = pd.concat([controls[PLATE], treatment[PLATE]]).drop_duplicates()
    scan_counts = all_scans.groupby("BARCODE").SCAN_ID.nunique()
    single = set(scan_counts[scan_counts.eq(1)].index)
    controls = controls[controls.BARCODE.isin(single)]
    treatment = treatment[treatment.BARCODE.isin(single)]
    controls, ctrl_conflicts = clean_positions(controls, ["TAG", "INTENSITY", "SANGER_MODEL_ID"])
    treatment, treatment_conflicts = clean_positions(treatment, [
        "library", "DRUG_ID", "CONC", "INTENSITY", "SANGER_MODEL_ID", "DATE_CREATED", "DURATION",
    ])
    dual = treatment.merge(controls[WELL], on=WELL, how="left", indicator=True)
    treatment = dual.loc[dual["_merge"].eq("left_only")].drop(columns="_merge")
    control = controls.groupby([*PLATE, "SANGER_MODEL_ID", "TAG"]).INTENSITY.agg(["size", "mean", "std"]).unstack()
    control.columns = ["_".join(pair) for pair in control.columns]
    span = control["mean_NC-1"] - control.mean_B
    control["zprime"] = 1 - 3 * (control["std_NC-1"] + control.std_B) / span.where(span > 0)
    usable = control[control["size_NC-1"].ge(8) & control.size_B.ge(4) & control.zprime.ge(.3)]
    rows = treatment.join(usable[["mean_NC-1", "mean_B", "zprime"]], on=[*PLATE, "SANGER_MODEL_ID"],
                          how="inner", validate="many_to_one")
    ratio = (rows.INTENSITY - rows.mean_B) / (rows["mean_NC-1"] - rows.mean_B)
    rows["viability"] = ratio.clip(0, 1)
    return rows, {
        "eligible_for_plate_control_check": treatment.BARCODE.nunique(),
        "multiple_read_scans_excluded": int(scan_counts.gt(1).sum()),
        "control_well_conflicts": ctrl_conflicts, "treatment_well_conflicts": treatment_conflicts,
        "both_control_and_treatment_wells_excluded": int(dual["_merge"].eq("both").sum()),
        "usable_treatment_plates": rows.BARCODE.nunique(), "normalized_treatment_wells": len(rows),
        "fraction_clipped_viability": float((ratio.lt(0) | ratio.gt(1)).mean()),
        "zprime_quantiles": usable.zprime.quantile([.1, .5, .9]).tolist(),
    }


def summarize_curves(rows, core):
    axes = [*KEY, "BARCODE", "DATE_CREATED", "DURATION", "library"]
    rows = rows.merge(core[[*KEY, "DRUG_ID", "SANGER_MODEL_ID", "dose_min_uM", "dose_max_uM"]],
                      on=["DRUG_ID", "SANGER_MODEL_ID"], validate="many_to_one")
    observed = rows.groupby(axes + ["CONC"]).viability.median().rename("viability").reset_index()
    registered = core.set_index(KEY)[["dose_min_uM", "dose_max_uM"]]
    target = np.linspace(np.log(WINDOW[0]), np.log(WINDOW[1]), 129)
    records = []
    verified_curve_groups = 0
    for identity, group in observed.groupby(axes, sort=False):
        group = group.sort_values("CONC")
        concentrations = group.CONC.to_numpy()
        if len(concentrations) < 5:
            continue
        low, high = registered.loc[identity[:2]]
        if not np.allclose([concentrations.min(), concentrations.max()], [low, high], rtol=.02, atol=1e-12):
            continue
        verified_curve_groups += 1
        if concentrations.min() > WINDOW[0] * (1 + 1e-6) or concentrations.max() < WINDOW[1] * (1 - 1e-6):
            continue
        dose = np.log(concentrations)
        y = group.viability.to_numpy()
        native = trapezoid(np.clip(y, 0, 1), x=dose) / (dose[-1] - dose[0])
        window = np.clip(np.interp(target, dose, y), 0, 1)
        records.append((*identity[:-1], float(native), float(trapezoid(window, dx=1/128)), float(window[-1])))
    plates = pd.DataFrame(records, columns=[*axes[:-1], *ENDPOINTS])
    if plates.empty:
        raise ValueError("No normalized plate curve covers the registered physical range")
    plate_values = plates.groupby(axes[:-1], as_index=False)[ENDPOINTS].median()
    return plate_values, {"matched_native_curve_groups": verified_curve_groups, "window_curve_groups": len(plates)}


def extract(args):
    if sha256(args.raw) != RAW_SHA256:
        raise ValueError("GDSC2 public 8.5 archive did not match published run inputs")
    train, _, receipt = training_inputs(args.bundle)
    model = receipt["inputs"]["depmap_models"]
    if sha256(model["path"]) != model["sha256"]:
        raise ValueError("Core Sanger model mapping changed")
    core = source_rows(train, model["path"])
    controls, treated, total = collect_wells(args.raw, core)
    normalized, qc = normalize(controls, treated)
    plates, curve_qc = summarize_curves(normalized, core)
    collapsed = (plates.groupby(KEY, as_index=False)[ENDPOINTS].median()
                 .rename(columns={"native_auc": "raw_full_auc"}))
    full = core[[*KEY, "lineage_group", "endpoint_value"]].merge(collapsed, on=KEY, validate="one_to_one")
    full = full.rename(columns={"endpoint_value": "native_auc"})
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {"plates": args.out / "plates.parquet", "pairs": args.out / "cells.parquet"}
    plates.to_parquet(paths["plates"], index=False)
    full.to_parquet(paths["pairs"], index=False)
    publish_json(args.out / "receipt.json", {
        "raw_sha256": RAW_SHA256, "bundle_sha256": sha256(args.bundle / "receipt.json"),
        "window_uM": WINDOW, "whole_archive_measurement_rows": total,
        "mapped_core_curve_rows": len(core), "selected_raw_control_rows": len(controls),
        "selected_raw_library_rows_before_pair_join": len(treated), "plate_curve_rows": len(plates),
        "supported_cell_drug_pairs": len(full), "supported_drugs": full.compound_id.nunique(),
        "release_DURATION_days_field": {str(k): int(v) for k, v in plates.DURATION.value_counts().sort_index().items()},
        "qc": qc | curve_qc,
        "artifacts": {name: sha256(path) for name, path in paths.items()},
    })


def summarize_agreement(left, right, lineage_keys):
    same = left.merge(right, on=KEY, validate="one_to_one", suffixes=("_ctrp", "_prism"))
    joined = same.merge(lineage_keys, on="cell_id", validate="many_to_one")
    donor = joined.groupby(["lineage_group", "compound_id"], as_index=False).mean(numeric_only=True)
    return {
        "paired_cell_drug_rows": len(same), "paired_lineage_drug_rows": len(donor),
        "compounds": donor.compound_id.nunique(),
        "within_drug": compare(donor, "compound_id", 30),
        "within_lineage": compare(donor, "lineage_group", 5),
    }


def repeated_plates(plates, identities, *, different_date):
    # Choose repeats by chronology, keeping the recorded assay duration the same.
    ordered = plates.assign(plating_day=plates.DATE_CREATED.str[:10])
    ordered = ordered.sort_values(["plating_day", "BARCODE"]).drop_duplicates([*KEY, "BARCODE"])
    records = []
    for key, group in ordered.groupby(KEY):
        if len(group) < 2:
            continue
        initial = group.iloc[0]
        rest = group.iloc[1:]
        rest = rest[rest.DURATION.eq(initial.DURATION)]
        if different_date:
            rest = rest[rest.plating_day.ne(initial.plating_day)]
        if len(rest):
            records.append((*key, *initial[ENDPOINTS], *rest.iloc[0][ENDPOINTS]))
    columns = [*KEY, *(f"{m}_{s}" for s in ("ctrp", "prism") for m in ENDPOINTS)]
    same = pd.DataFrame(records, columns=columns)
    same = same.merge(identities, on="cell_id", validate="many_to_one")
    donor = same.groupby(["lineage_group", "compound_id"], as_index=False).mean(numeric_only=True)
    return {"paired_cell_drug_rows": len(same), "compounds": donor.compound_id.nunique(),
            "within_drug": compare(donor, "compound_id", 30), "within_lineage": compare(donor, "lineage_group", 5)}


def score(args):
    paths = {"plates": args.gdsc / "plates.parquet", "pairs": args.gdsc / "cells.parquet"}
    receipt = json.loads((args.gdsc / "receipt.json").read_text())
    for name, path in paths.items():
        if sha256(path) != receipt["artifacts"][name]:
            raise ValueError("Normalized GDSC2 measurements differ from their receipt")
    if sha256(args.treatments) != TREATMENTS_HASH or sha256(args.bundle / "receipt.json") != receipt["bundle_sha256"]:
        raise ValueError("Original training sources or public PRISM dose metadata changed")
    train, _, record = training_inputs(args.bundle)
    for name in ("ctrp_cells", "ctrp_experiments", "depmap_models", "prism_curves"):
        source = record["inputs"][name]
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Assay input changed: {name}")
    gdsc = pd.read_parquet(paths["pairs"])
    plates = pd.read_parquet(paths["plates"])
    ctrp, prism = (train[train.source_id.eq(name)] for name in ("CTRPv2", "PRISM20Q2"))
    identity = train[["cell_id", "lineage_group"]].drop_duplicates()
    a = measured_points(record["inputs"], ctrp.merge(gdsc[KEY], on=KEY), args.ctrp_post_qc)
    b, prism_qc = published_curves(record["inputs"], prism.merge(gdsc[KEY], on=KEY), args.treatments)
    prepared = {
        "GDSC2": gdsc[[*KEY, *ENDPOINTS]],
        "CTRPv2": a.merge(ctrp[[*KEY, "endpoint_value"]], on=KEY).rename(columns={"endpoint_value": "native_auc"}),
        "PRISM20Q2": b.merge(prism[[*KEY, "endpoint_value"]], on=KEY).rename(columns={"endpoint_value": "native_auc"}),
    }
    common = prepared["GDSC2"][KEY].merge(a[KEY], on=KEY).merge(b[KEY], on=KEY)
    matched = {}
    for left, right in (("GDSC2", "CTRPv2"), ("GDSC2", "PRISM20Q2"), ("CTRPv2", "PRISM20Q2")):
        matched[f"{left}–{right}"] = {
            "pair_support": summarize_agreement(prepared[left], prepared[right], identity),
            "three_source_support": summarize_agreement(prepared[left].merge(common, on=KEY),
                                                        prepared[right].merge(common, on=KEY), identity),
        }
    raw_native = gdsc.groupby(["lineage_group", "compound_id"], as_index=False)[
        ["native_auc", "raw_full_auc"]
    ].mean()
    observed = [
        (group.native_auc.corr(group.raw_full_auc, method="spearman"),
         group.native_auc.corr(group.raw_full_auc, method="pearson"))
        for _, group in raw_native.groupby("compound_id") if (
            len(group) >= 30 and group.native_auc.nunique() > 1 and group.raw_full_auc.nunique() > 1
        )
    ]
    publish_json(args.out, {
        "gdsc_receipt_sha256": sha256(args.gdsc / "receipt.json"),
        "prism_dose_metadata_sha256": TREATMENTS_HASH, "ctrp_post_qc_sha256": sha256(args.ctrp_post_qc),
        "three_source_cell_drug_pairs": len(common), "prism_curve_qc": prism_qc,
        "gdsc_native_vs_raw_full_window": {
            "within_drug_support": len(observed),
            "spearman_pearson": np.asarray(observed).mean(axis=0).tolist() if observed else [],
        },
        "gdsc_duplicate_plate_association": {
            "separate_plates": repeated_plates(plates, identity, different_date=False),
            "different_plating_dates": repeated_plates(plates, identity, different_date=True),
        }, "pairs": matched,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, fields in (
        ("extract", ("bundle", "raw", "out")), ("score", ("bundle", "gdsc", "ctrp-post-qc", "treatments", "out")),
    ):
        stage = sub.add_parser(name)
        for field in fields:
            stage.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    {"extract": extract, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
