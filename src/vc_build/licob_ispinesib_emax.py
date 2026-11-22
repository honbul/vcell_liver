"""Test matched highest-dose PRISM endpoints in fixed licob expression support."""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from .external import KEY
from .io import freeze_table, publish_json, sha256, verify_forecast
from .licob_cold import checked
from .licob_dose_sources import dose_metadata
from .licob_endpoints import paired_donors, read_outcomes
from .licob_prism_restored import OLD_INPUT
from .shuen_pdx import agreements
from .structured_kernel import fit_cell_kernels
from .vinorelbine_raw_doses import MANIFEST, integrate, prism_curves
from .vinorelbine_transfer import EXTENSION_SHA, fit_grid
from .v0plus_cv import molecular_inputs
from .v0plus_neural import load_auxiliary
from .yang_hcc_external import require_hash


DRUG = "QJZRFPJCWMNVAV-HHHXNRCGSA-N"
VALUES = ("fitted_auc", "raw_auc", "matched_emax")
ARMS = ("drug_mean", *VALUES)


def cohort(args, rna=False):
    lookup = args.licob / "query.parquet"
    if checked(args.licob, ("query", "rna") if rna else ("query",)) != OLD_INPUT:
        raise ValueError("LICOB cohort changed")
    manifest = pd.read_parquet(lookup)
    design = dose_metadata(args.supplement, manifest)
    rows = manifest[manifest.compound_id.eq(DRUG)]
    bounds = design[design.compound_id.eq(DRUG)]
    if len(rows) != 43 or not np.allclose(bounds[["second", "maximum"]], [10 / 3, 10.]):
        raise ValueError("Unexpected primary population or Emax concentrations")
    return rows, bounds


def prepare(args, bounds):
    require_hash(args.raw / "inputs.json", MANIFEST)
    for item in json.loads((args.raw / "inputs.json").read_text()):
        require_hash(args.raw / item["file"], item["sha256"])
    receipt = args.extension / "receipt.json"
    require_hash(receipt, EXTENSION_SHA)
    file = args.extension / "additional_rows.parquet"
    require_hash(file, json.loads(receipt.read_text())["artifacts"][file.name])
    expected = pd.read_parquet(file, filters=[("compound_id", "==", DRUG), ("source_id", "==", "PRISM20Q2")])
    wells, support = prism_curves(expected, args.raw)
    summary = wells.groupby(["cell_id", "dose"]).log_fold_change.agg(["size", "median"]).reset_index()
    summary = summary[summary["size"].ge(2)].rename(columns={"median": "log_fold_change"})
    top = bounds[["second", "maximum"]].iloc[0].to_numpy()
    native = support.loc[DRUG, ["min", "max"]].to_numpy()

    def curve_endpoint(points, keys):
        measured = []
        for group, curve in points.sort_values("dose").groupby(keys):
            dose = curve.dose.to_numpy()
            y = np.clip(np.exp2(curve.log_fold_change.to_numpy()), 0, 1)
            raw = integrate(dose, y, native)
            if raw is None or dose[0] > top[0] or dose[-1] < top[1]:
                continue
            high = np.interp(np.log(top), np.log(dose), y)
            measured.append((*((group,) if isinstance(group, str) else group), raw[0], float(high.min())))
        return pd.DataFrame(measured, columns=[*keys, *VALUES[1:]])

    data = curve_endpoint(summary, ["cell_id"]).merge(
        expected[["cell_id", "lineage_group", "lineage_role", "source_id", "endpoint_value"]],
        on="cell_id", validate="one_to_one",
    ).rename(columns={"endpoint_value": "fitted_auc"})
    repeats = curve_endpoint(wells, ["cell_id", "detection_plate"]).sort_values("detection_plate")
    repeats["index"] = repeats.groupby("cell_id").cumcount()
    matched = repeats[repeats["index"].lt(2)].pivot(index="cell_id", columns="index", values=list(VALUES[1:])).dropna()
    rpt = data[["cell_id", "lineage_group"]].set_index("cell_id").join(
        matched.set_axis([f"{field}_{rep}" for field, rep in matched.columns], axis=1), how="inner",
    ).groupby("lineage_group").mean(numeric_only=True)
    if len(data) < 100 or len(rpt) < 5:
        raise ValueError("Insufficient measured source cells")
    return data, {"fit_support": len(data), "source_adult_hcc_cells": int(data.lineage_role.eq("adult_hcc").sum()),
                  "native_doses": support.to_dict("index"), "paired_detection_lineages": len(rpt),
                  "paired_detection_agreement": {name: agreements(rpt, f"{name}_0", f"{name}_1")
                                                 for name in VALUES[1:]}}


def predict(args):
    query, bounds = cohort(args, rna=True)
    data, notes = prepare(args, bounds)
    rna = pd.read_parquet(args.licob / "rna.parquet")
    ids = pd.Index(sorted(data.cell_id.unique()))
    if set(ids) & set(query.cell_id):
        raise ValueError("Expression training overlaps LICOB")
    edges, _, priors = molecular_inputs(args.priors, args.hallmark, args.targets)
    auxiliary, aux = load_auxiliary(args.auxiliary)
    if not torch.cuda.is_available():
        raise RuntimeError("Ispinesib fitting requires G1 GPU")
    basis, gram = fit_cell_kernels(rna.loc[ids], edges, auxiliary, device=torch.device("cuda"))
    held = basis.cross(rna.loc[query.cell_id])["full"]
    predictions = [query[KEY].assign(method="drug_mean", predicted_value=0.)]
    for name in VALUES:
        estimates = fit_grid(data.assign(sensitivity=-data[name]), ids, gram["full"], held, (4.,))[4.]
        predictions.append(query[KEY].assign(method=name, predicted_value=estimates))
    output = pd.concat(predictions, ignore_index=True).assign(track="LICOB_ISPINESIB_EMAX", fold=0)
    freeze_table(output, args.run / "forecast.parquet", provenance={
        "licob": OLD_INPUT, "prism": MANIFEST, "structural_registry": EXTENSION_SHA,
        "two_high_doses_uM": bounds[["second", "maximum"]].iloc[0].to_list(),
        "source": notes, "priors": priors, "auxiliary": aux["endpoints"],
    })


def score(args):
    query, bounds = cohort(args)
    forecasts, seal = verify_forecast(args.run / "forecast.parquet")
    if len(forecasts) != 4 * len(query):
        raise ValueError("Incomplete ispinesib prediction fixture")
    truth = read_outcomes(args.supplement, query, bounds, selected=("auc", "emax"))
    outcomes = {}
    for endpoint, table in truth.items():
        rows = paired_donors(forecasts, table)
        if rows.lineage_group.nunique() < 30:
            raise ValueError("Insufficient LICOB adult donors")
        wide = rows.pivot(index="lineage_group", columns="method", values="predicted_value")
        wide["response"] = rows.groupby("lineage_group").endpoint_value.first()
        outcomes[endpoint] = {"donors": len(wide),
                              "scores": {name: agreements(wide, name, "response") for name in ARMS},
                              "contrasts": {name: agreements(wide, "matched_emax", "response", name)
                                            for name in ARMS[:-1]}}
    gate = all(row["ci95"][0] > 0 for stats in outcomes["emax"]["contrasts"].values() for row in stats.values())
    publish_json(args.run / "score.json", {"forecast_sha256": seal["sha256"], "outcomes": outcomes, "pass": gate})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    for name, fn, options in (("predict", predict, ("raw", "extension", "priors", "hallmark", "targets", "auxiliary")),
                              ("score", score, ())):
        child = stages.add_parser(name)
        for key in ("run", "licob", "supplement", *options):
            child.add_argument("--" + key, type=Path, required=True)
        child.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
