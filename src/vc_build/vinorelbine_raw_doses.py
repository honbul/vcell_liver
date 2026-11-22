"""Audit matched-dose vinorelbine measurements and freeze PDX transfer."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import torch

from .expanded_panel_cv import registered_inputs
from .gdsc_raw_window import RAW_SHA256, collect_wells, normalize, source_rows
from .io import freeze_table, publish_json, sha256, verify_forecast
from .shuen_pdx import agreements, correlation, donor_axis, response_sources
from .structured_kernel import fit_cell_kernels
from .vinorelbine_transfer import DRUG, EXTENSION_SHA, fit_grid, observations, prepare_query
from .v0plus_cv import molecular_inputs, training_inputs
from .v0plus_neural import load_auxiliary


MANIFEST = "b329bf15247a2ba7c00bc9ae4a6424a0439e71d15ad1c35e2de27d7e1f5c2fc8"
WINDOW = (.0025, .05)
ASSAYS = ("GDSC2", "PRISM20Q2")
KINDS = ("fitted", "raw_native", "raw_common")
PRIMARY = "pooled_raw_common"
CONTROLS = ("pooled_fitted", "GDSC2_raw_common", "PRISM20Q2_raw_common")


def load_sources(args):
    extension_file = args.extension / "receipt.json"
    if sha256(extension_file) != EXTENSION_SHA:
        raise ValueError("Unexpected PRISM registration")
    artifact = args.extension / "additional_rows.parquet"
    if sha256(artifact) != json.loads(extension_file.read_text())["artifacts"][artifact.name]:
        raise ValueError("Unexpected PRISM observations")
    _, core, _, _ = registered_inputs(SimpleNamespace(original=args.original, core=args.core))
    extra = pd.read_parquet(artifact)
    vino, hashes = observations(args, core, extra[extra.compound_id.eq(DRUG)])
    return vino, extra, hashes


def integrate(dose, response, native):
    if len(dose) < 5:
        return None
    lower, upper = native
    # Permit only recorded concentration rounding at the native endpoints.
    if not np.allclose([dose[0], dose[-1]], [lower, upper], rtol=.02, atol=1e-12):
        return None
    y = np.clip(response, 0, 1)
    x = np.log(dose)
    native_auc = float(trapezoid(y, x=x) / (x[-1] - x[0]))
    low, high = WINDOW
    if dose[0] > low or dose[-1] < high:
        return native_auc, np.nan
    a, b = np.log([low, high])
    inside = (x > a) & (x < b)
    grid = np.r_[a, x[inside], b]
    return native_auc, float(trapezoid(np.interp(grid, x, y), x=grid) / (b - a))


def gdsc(args):
    if sha256(args.raw) != RAW_SHA256:
        raise ValueError("Unexpected 8.5 GDSC2 raw archive")
    data, _, digest = load_sources(args)
    inputs = json.loads((args.original / "receipt.json").read_text())["inputs"]
    registered = source_rows(data[data.source_id.eq("GDSC2")].assign(compound_id=DRUG),
                             inputs["depmap_models"]["path"])
    if not (registered.dose_min_uM.eq(.0001) & registered.dose_max_uM.eq(.1)).all():
        raise ValueError("GDSC2 has inconsistent registered assay windows")
    controls, treated, total = collect_wells(args.raw, registered)
    normalized, qc = normalize(controls, treated)
    join_keys = ["SANGER_MODEL_ID", "DRUG_ID"]
    normalized = normalized.merge(registered[[*join_keys, "cell_id"]], on=join_keys, validate="many_to_one")
    axes = ["cell_id", "BARCODE", "library", "DATE_CREATED", "DURATION"]
    points = normalized.groupby([*axes, "CONC"], as_index=False).viability.median()
    curves = []
    for (cell, barcode, library, date, duration), curve in points.groupby(axes):
        if (summary := integrate(curve.CONC.to_numpy(), curve.viability.to_numpy(), (.0001, .1))) is not None:
            curves.append((cell, barcode, library, date, duration, *summary))
    by_plate = pd.DataFrame(curves, columns=["cell_id", "plate", "library", "date", "duration", *KINDS[1:]])
    if by_plate.empty:
        raise ValueError("No complete GDSC2 raw vinorelbine curves")
    by_plate = by_plate.groupby(["cell_id", "plate", "date", "duration"], as_index=False)[list(KINDS[1:])].median()
    consensus = by_plate.groupby("cell_id", as_index=False)[list(KINDS[1:])].median()
    output = consensus.merge(registered[["cell_id", "endpoint_value", "lineage_group", "lineage_role"]],
                             on="cell_id", validate="one_to_one").assign(compound_id=DRUG)
    return output, by_plate.assign(compound_id=DRUG), {
        "asset_hashes": {"raw": RAW_SHA256, **digest}, "qc": qc, "total_archive_wells": total,
        "matched_plates": by_plate.plate.nunique(), "duration_field": sorted(by_plate.duration.unique().tolist()),
        "normalized_curves": len(by_plate), "registered_cells": len(registered),
    }


def prism_curves(metrics, manifest):
    wells = pd.read_csv(manifest / "treatments.csv", low_memory=False)
    raw_path = manifest / "lfc.csv"
    header = set(pd.read_csv(raw_path, nrows=0).columns)
    metadata = wells[wells.perturbation_type.eq("experimental_treatment")
                     & wells.column_name.isin(header) & wells.dose.gt(0)].merge(
        metrics[["compound_id", "native_compound_id", "source_run"]].drop_duplicates(),
        left_on=["broad_id", "screen_id"], right_on=["native_compound_id", "source_run"],
        validate="many_to_one",
    )
    plate = metadata.compound_plate.str.split("_", n=1, expand=True)
    metadata["plate_prefix"] = plate[0]
    metadata["cellset"] = plate[1].fillna("PR500") if 1 in plate else "PR500"
    metadata = metadata[metadata.plate_prefix.eq(metadata.groupby("compound_id").plate_prefix.transform("min"))]
    metadata = metadata[["compound_id", "cellset", "detection_plate", "dose", "column_name"]].drop_duplicates()
    if metadata.duplicated("column_name").any():
        raise ValueError("Conflicting PRISM treatment column")
    rna_rows = pd.read_csv(raw_path, usecols=["Unnamed: 0", *metadata.column_name])
    tokens = rna_rows.pop("Unnamed: 0").str.extract(r"^([^_]+)_(ACH-\d+)$")
    rna_rows[["cellset", "cell_id"]] = tokens
    values = rna_rows.dropna(subset=["cell_id"]).melt(id_vars=["cellset", "cell_id"],
                                                  var_name="column_name", value_name="log_fold_change")
    values = values.merge(metadata, on=["cellset", "column_name"], validate="many_to_one")
    values = values[np.isfinite(values.log_fold_change)].merge(
        metrics[["cell_id", "compound_id"]], on=["cell_id", "compound_id"], validate="many_to_one",
    )
    wells = values.groupby(["cell_id", "compound_id", "detection_plate", "dose"], as_index=False).log_fold_change.median()
    native = metadata.groupby("compound_id").dose.agg(["min", "max", "nunique", "size"])
    return wells, native


def prism(args):
    proof = args.raw / "inputs.json"
    if sha256(proof) != MANIFEST:
        raise ValueError("Unexpected PRISM download manifest")
    assets = json.loads(proof.read_text())
    for item in assets:
        if sha256(args.raw / item["file"]) != item["sha256"]:
            raise ValueError(f"PRISM input changed: {item['file']}")
    _, extra, _ = load_sources(args)
    from .licob_prism_restored import RESTORED

    metadata = extra[extra.source_id.eq("PRISM20Q2") & extra.compound_id.isin([DRUG, *RESTORED])]
    wells, support = prism_curves(metadata, args.raw)
    by_dose = wells.groupby(["cell_id", "compound_id", "dose"]).log_fold_change
    values = by_dose.agg(["size", "median"]).reset_index()
    consensus = values[values["size"].ge(2)].rename(columns={"median": "log_fold_change"})
    pairs, plates = [], []
    for groups, output in ((["cell_id", "compound_id"], pairs),
                           (["cell_id", "compound_id", "detection_plate"], plates)):
        input_table = consensus if output is pairs else wells
        for key, curve in input_table.sort_values("dose").groupby(groups, sort=False):
            if (summary := integrate(curve.dose.to_numpy(), np.exp2(curve.log_fold_change.to_numpy()),
                                     support.loc[key[1], ["min", "max"]].to_numpy())) is not None:
                output.append((*key, *summary))
    pairs = pd.DataFrame(pairs, columns=["cell_id", "compound_id", *KINDS[1:]])
    plates = pd.DataFrame(plates, columns=["cell_id", "compound_id", "plate", *KINDS[1:]])
    pairs = pairs.merge(metadata[["cell_id", "compound_id", "endpoint_value", "lineage_group", "lineage_role"]],
                        on=["cell_id", "compound_id"], validate="one_to_one")
    return pairs, plates, {
        "manifest": MANIFEST, "assets": assets, "finite_well_measurements": len(wells),
        "native_concentration": support.to_dict("index"), "matched_compounds": pairs.compound_id.nunique(),
        "supported_cells": pairs.groupby("compound_id").size().to_dict(),
    }


def export_measurements(args, assay, table, plates, notes):
    stage = args.run / assay
    stage.mkdir(exist_ok=False)
    pairs = stage / "cells.parquet"
    repeat = stage / "plates.parquet"
    table.rename(columns={"endpoint_value": "fitted"}).to_parquet(pairs, index=False)
    plates.to_parquet(repeat, index=False)
    publish_json(stage / "receipt.json", {"notes": notes, "cells_sha256": sha256(pairs), "plates_sha256": sha256(repeat)})


def read_measurements(args):
    all_parts = {}
    for assay in ASSAYS:
        stage = args.run / assay
        receipt = json.loads((stage / "receipt.json").read_text())
        inputs = {}
        for name in ("cells", "plates"):
            file = stage / f"{name}.parquet"
            if sha256(file) != receipt[name + "_sha256"]:
                raise ValueError("Raw dose measurements fail hash verification")
            inputs[name] = pd.read_parquet(file)
        all_parts[assay] = inputs | {"receipt": sha256(stage / "receipt.json")}
    return all_parts


def same_cells(parts):
    left, right = (parts[assay]["cells"].query("compound_id == @DRUG").dropna(subset=list(KINDS))
                   for assay in ASSAYS)
    cols = ["cell_id", "lineage_group", "lineage_role", *KINDS]
    paired = left[cols].merge(right[cols], on=cols[:3], suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) < 100:
        raise ValueError("Too few paired native and common-dose curves")
    return paired


def agreement(on):
    result = {}
    size = len(on)
    draws = np.random.default_rng(48817).integers(size, size=(4000, size)) if size >= 5 else None
    for metric in ("spearman", "pearson"):
        points, estimates = {}, {}
        for key in KINDS:
            left, right = (on[f"{key}_{side}"].to_numpy() for side in ("left", "right"))
            points[key] = float(correlation(left, right, metric)) if size >= 5 else None
            if draws is not None:
                estimates[key] = correlation(left[draws], right[draws], metric)
        if estimates:
            for first, second in combinations(KINDS, 2):
                difference = estimates[second] - estimates[first]
                points[f"{second}_minus_{first}"] = {
                    "estimate": points[second] - points[first],
                    "ci95": np.quantile(difference, [.025, .975]).tolist(),
                }
        result[metric] = points
    return result | {"n": size}


def repeats(parts):
    records = {}
    for assay, block in parts.items():
        panels = block["plates"].dropna(subset=list(KINDS[1:]))
        measured = block["cells"][["cell_id", "compound_id", "lineage_group", "lineage_role"]]
        result = {}
        for drug, rows in panels.groupby("compound_id"):
            order = ["date", "plate"] if assay == "GDSC2" else ["plate"]
            ordered = rows.sort_values(order).drop_duplicates(["cell_id", "plate"])
            indexed = ordered.assign(order=ordered.groupby("cell_id").cumcount())
            left, right = (indexed[indexed.order.eq(n)] for n in (0, 1))
            keys = ["cell_id", "compound_id"]
            if "duration" in indexed:
                keys.append("duration")
            cols = [*keys, *KINDS[1:]]
            pairs = left[cols].merge(right[cols], on=keys, suffixes=("_left", "_right"), validate="one_to_one")
            pairs = pairs.merge(measured, on=["cell_id", "compound_id"], validate="one_to_one")
            collapsed = pairs.groupby("lineage_group").mean(numeric_only=True)
            if len(collapsed) >= 5:
                methods = {kind: agreements(collapsed, f"{kind}_left", f"{kind}_right") for kind in KINDS[1:]}
                result[drug] = {"paired_cells": len(pairs), "lineages": len(collapsed), "agreement": methods}
            else:
                result[drug] = {"paired_cells": len(pairs), "lineages": len(collapsed)}
        records[assay] = result
    return records


def source_audit(args, parts, paired):
    lineages = paired.groupby("lineage_group").mean(numeric_only=True)
    adult = paired[paired.lineage_role.eq("adult_hcc")].groupby("lineage_group").mean(numeric_only=True)
    between = agreement(lineages)
    change = [between[name]["raw_common_minus_fitted"] for name in ("spearman", "pearson")]
    gate = change[0]["estimate"] >= .05 and all(item["ci95"][0] > 0 for item in change)
    publish_json(args.run / "raw_audit.json", {
        "between_all": between, "between_adult_hcc": agreement(adult), "repeat_within_assay": repeats(parts),
        "paired_cells": len(paired), "source_hashes": {assay: block["receipt"] for assay, block in parts.items()},
        "raw_matching_gate": gate,
    })


def predict(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen PDX predictions require G1 GPU")
    parts = read_measurements(args)
    paired = same_cells(parts)
    source_audit(args, parts, paired)
    _, rna, _ = training_inputs(args.core)
    roster = pd.Index(sorted(paired.cell_id))
    fit_rna = rna.loc[roster]
    query = pd.concat(prepare_query(args.pdx, fit_rna), names=["track", "row_id"])
    signatures, _, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    aux, aux_receipt = load_auxiliary(args.auxiliary)
    features, matrices = fit_cell_kernels(fit_rna, signatures, aux, device=torch.device("cuda"))
    held = features.cross(query.fillna(fit_rna.median()))["full"]
    fit = matrices["full"]
    ids = query.index.to_frame(index=False)
    predictions = [ids.assign(method="mean", fold=0, predicted_value=0.)]
    data = pd.concat([parts[assay]["cells"].query("compound_id == @DRUG")
                      .set_index("cell_id").loc[roster].reset_index().assign(source_id=assay)
                      for assay in ASSAYS], ignore_index=True)
    for scope in ("pooled", *ASSAYS):
        block = data if scope == "pooled" else data[data.source_id.eq(scope)]
        for key in KINDS:
            output = fit_grid(block.assign(sensitivity=-block[key]), roster, fit, held, (4.,))[4.]
            predictions.append(ids.assign(method=f"{scope}_{key}", fold=0, predicted_value=output))
    freeze_table(pd.concat(predictions, ignore_index=True), args.run / "forecast.parquet", provenance={
        "source_audit_sha256": sha256(args.run / "raw_audit.json"), "pair_cells": len(roster), "window_uM": WINDOW,
        "priors": priors, "auxiliary": aux_receipt["endpoints"], "core_sha256": sha256(args.core / "receipt.json"),
    })


def score(args):
    rows, seal = verify_forecast(args.run / "forecast.parquet")
    if sha256(args.run / "raw_audit.json") != seal["provenance"]["source_audit_sha256"]:
        raise ValueError("The raw 2D audit changed after the 3D prediction freeze")
    studies = response_sources(args.sources)
    labels = {year: studies[year].query("drug == 'vinorelbine'").assign(response=lambda d: -d.tc)
              for year in ("2023", "2024")}
    arms = sorted(rows.method.unique())
    results, gate = {}, {}
    for scale, part in rows.groupby("track"):
        matrix = part.pivot(index="row_id", columns="method", values="predicted_value")
        groups = {year: values.merge(matrix, left_on="model_id", right_index=True, validate="one_to_one")
                  for year, values in labels.items()}
        common = set.intersection(*(set(value.model_id) for value in groups.values()))
        for year, group in groups.items():
            for scope, rows in (("full", group), ("common", group[group.model_id.isin(common)])):
                patient = donor_axis(rows[["model_id", "response", *arms]])
                if len(patient) < 6:
                    raise ValueError("Insufficient PDX scoring support")
                comparisons = {arm: agreements(patient, PRIMARY, "response", arm) for arm in CONTROLS}
                results[f"{scale}/{year}/{scope}"] = {
                    "n": len(patient), "models": len(rows), "score": {
                        arm: agreements(patient, arm, "response") for arm in arms}, "contrasts": comparisons,
                }
                if scale == "cpm" and scope == "common":
                    gate[year] = all(value["ci95"][0] > 0
                                     for comparison in comparisons.values() for value in comparison.values())
    publish_json(args.run / "score.json", {
        "forecast_sha256": seal["sha256"], "comparisons": results, "pass": all(gate.values()), "checks": gate,
    })


def measure(args, stage):
    assay, fn = {"gdsc": ("GDSC2", gdsc), "prism": ("PRISM20Q2", prism)}[stage]
    values, plates, meta = fn(args)
    export_measurements(args, assay, values, plates, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = {
        "gdsc": ("raw", "original", "core", "extension"),
        "prism": ("raw", "original", "core", "extension"),
        "predict": ("core", "pdx", "priors", "hallmark", "targets", "auxiliary"),
        "score": ("sources",),
    }
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, paths in options.items():
        stage = stages.add_parser(name)
        for arg in ("run", *paths):
            stage.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    if args.stage in ("gdsc", "prism"):
        measure(args, args.stage)
    else:
        {"predict": predict, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
