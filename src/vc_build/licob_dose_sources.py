"""Reconstruct source curve summaries at prespecified LICOB concentrations."""

import argparse
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.integrate import trapezoid
from scipy.special import expit

from .cross_source_data import MODEL_COLS, read_raw, roster_maps, roster_table
from .dose_window_diagnostics import TOLERANCE, TREATMENTS_HASH, integrate_prism
from .expanded_panel_cv import registered_inputs
from .function_data import new_output
from .gdsc_raw_window import RAW_SHA256, collect_wells, normalize, source_rows
from .hcc_panel_support import GDSC_RECEIPT
from .io import publish_json, sha256
from .licob_cold import checked as checked_cold
from .v0plus_licob import ASSET_HASHES
from .yang_hcc_external import require_hash

COLD = "67c11aba490282e30be9ef6d3d4d28ad192a2f5620a9e0b84c978019a0ccdb9e"
CTRP_QC = "033488c593e1a02eedb48aca762d193009ffa767c2793f063f865f10b6a2ac25"
METHODS_PDF = "e809781ccca90499e6b793d849e9749d7aef3a47ee08a8289ef8c349438bfeac"
SCREENS = ("CTRPv2", "GDSC2", "PRISM20Q2")
KEY = ["cell_id", "compound_id"]
VALUES = ["emax", "source_auc", "ic50_bounded"]


def dose_metadata(path, query):
    require_hash(path, ASSET_HASHES["supplement"])
    with closing(load_workbook(path, read_only=True, data_only=True)) as wb:
        raw = list(wb["S5. Drug information"].values)
    table = pd.DataFrame(raw[2:], columns=raw[1]).rename(columns={
        "Catalogue": "catalogue", "Screening max dose (µM)": "maximum",
        "Dilution fold": "dilution", "# Point of doses": "points",
    })
    table = table[["catalogue", "maximum", "dilution", "points"]]
    table = table.assign(second=table.maximum / table.dilution,
                         minimum=table.maximum / table.dilution ** (table.points - 1))
    keys = query[["compound_id", "catalogue"]].drop_duplicates()
    merged = keys.merge(table, on="catalogue", validate="one_to_one")
    if len(merged) != 76 or not merged.maximum.gt(merged.second).all() or not merged.second.gt(merged.minimum).all():
        raise ValueError("Invalid published LICOB concentration design")
    return merged


def prepare(args):
    if checked_cold(args.cold, ("train", "query", "rna")) != COLD:
        raise ValueError("Unexpected exact drug registry")
    _, core, _, extra = registered_inputs(args, include_extension=True)
    require_hash(args.gdsc_extension / "receipt.json", GDSC_RECEIPT)
    path = args.gdsc_extension / "additional_rows.parquet"
    require_hash(path, json.loads((args.gdsc_extension / "receipt.json").read_text())["artifacts"][path.name])
    gdsc = pd.read_parquet(path)
    registered = pd.read_parquet(args.cold / "train.parquet", columns=["row_id"])
    query = pd.read_parquet(args.cold / "query.parquet")
    doses = dose_metadata(args.supplement, query)
    source = pd.concat([core, extra, gdsc], ignore_index=True)
    source = source[source.row_id.isin(registered.row_id) & source.compound_id.isin(doses.compound_id)]
    out = source.merge(doses.drop(columns="catalogue"), on="compound_id", validate="many_to_one")
    if out.duplicated(["source_id", *KEY]).any() or out.compound_id.nunique() != 57:
        raise ValueError("Unstable exact source bridge")
    with new_output(args.out) as area:
        out.to_parquet(area / "sources.parquet", index=False)
        doses.to_parquet(area / "dose.parquet", index=False)
        publish_json(area / "receipt.json", {"cold_sha256": COLD, "methods_pdf_sha256": METHODS_PDF,
            "source_pair_rows": len(out), "source_drugs": int(out.compound_id.nunique()),
            "sources": out.groupby("source_id").size().to_dict(),
            "files": {name: sha256(area / name) for name in ("sources.parquet", "dose.parquet")}})


def registration(path):
    receipt = json.loads((path / "receipt.json").read_text())
    if receipt["cold_sha256"] != COLD:
        raise ValueError("New dose registration")
    for name, digest in receipt["files"].items():
        require_hash(path / name, digest)
    return sha256(path / "receipt.json")


def curve_endpoints(dose_uM, viability, *, minimum, second, maximum):
    """Use actual log-dose support; estimate neither endpoint outside the measured interval."""
    dose = np.log10(dose_uM)
    y = np.clip(viability, 0, 1)
    if len(dose) < 5 or dose[0] > np.log10(second) + 1e-5 or dose[-1] < np.log10(maximum) - 1e-5:
        return None
    bounds = np.log10([minimum, second, maximum])
    emax = min(np.interp(bounds[1:], dose, y))
    auc = trapezoid(y, x=dose) / (dose[-1] - dose[0])
    potency = np.nan
    if dose[0] <= bounds[0] + 1e-5:
        interior = (dose > bounds[0]) & (dose < bounds[2])
        x = np.r_[bounds[0], dose[interior], bounds[2]]
        v = np.interp(x, dose, y)
        # The lowest crossing is defined in the common physical interval.
        found = np.flatnonzero(v <= .5)
        if not len(found):
            potency = bounds[2]
        elif found[0] == 0:
            potency = bounds[0]
        else:
            j = found[0]
            potency = np.interp(.5, v[j - 1:j + 1][::-1], x[j - 1:j + 1][::-1])
    return float(emax), float(auc), float(potency)


def measured(groups, axis, dcol, vcol, expected):
    assay = expected.set_index(KEY)
    rows = []
    for identity, group in groups.groupby(axis, sort=False):
        key = identity[:2]
        meta = assay.loc[key]
        given = group.groupby(dcol)[vcol].median().sort_index()
        if dcol == "CONC" and not np.allclose([given.index[0], given.index[-1]],
                                             [meta.dose_min_uM, meta.dose_max_uM], rtol=.02):
            continue
        value = curve_endpoints(given.index.to_numpy(), given.to_numpy(),
                                minimum=meta.minimum, second=meta.second, maximum=meta.maximum)
        if value:
            rows.append((*key, *value))
    return pd.DataFrame(rows, columns=[*KEY, *VALUES]).groupby(KEY, as_index=False).median()


def ctrp(source, raw, files):
    require_hash(raw, CTRP_QC)
    a = read_raw(files["ctrp_cells"]["path"], ["ccl_name", "master_ccl_id"], sep="\t")
    b = read_raw(files["depmap_models"]["path"], MODEL_COLS)
    cells = roster_maps(roster_table(a, "CTRPv2", "ccl_name", "master_ccl_id"), b)
    cells = cells[["source_key", "cell_id"]].dropna().drop_duplicates()
    run = read_raw(files["ctrp_experiments"]["path"], ["experiment_id", "master_ccl_id", "snp_fp_status"], sep="\t")
    run = run[run.snp_fp_status.eq("SNP-matched-reference")].merge(
        cells, left_on="master_ccl_id", right_on="source_key", validate="many_to_one",
    )[["experiment_id", "cell_id"]].drop_duplicates()
    select = source[[*KEY, "native_compound_id"]].rename(columns={"native_compound_id": "master_cpd_id"})
    frames = []
    for chunk in pd.read_csv(raw, sep="\t", chunksize=400_000, usecols=[
        "experiment_id", "master_cpd_id", "cpd_conc_umol", "cpd_avg_pv"],
        dtype={"experiment_id": str, "master_cpd_id": str}):
        block = chunk[chunk.master_cpd_id.isin(select.master_cpd_id)]
        frames.append(block.merge(run, on="experiment_id", validate="many_to_one")
                      .merge(select, on=["cell_id", "master_cpd_id"], validate="many_to_one"))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[np.isfinite(frame.cpd_avg_pv) & np.isfinite(frame.cpd_conc_umol) & frame.cpd_conc_umol.gt(0)]
    return measured(frame, [*KEY, "experiment_id"], "cpd_conc_umol", "cpd_avg_pv", source), {"selected_points": len(frame)}


def gdsc(source, archive, files):
    require_hash(archive, RAW_SHA256)
    ready = source_rows(source, files["depmap_models"]["path"])
    controls, wells, total = collect_wells(archive, ready)
    points, qc = normalize(controls, wells)
    points = points[points.DURATION.eq(4)].merge(ready[[*KEY, "DRUG_ID", "SANGER_MODEL_ID"]],
                         on=["DRUG_ID", "SANGER_MODEL_ID"], validate="many_to_one")
    frame = measured(points, [*KEY, "BARCODE", "library"], "CONC", "viability", source)
    return frame, {"archive_rows": total, "duration_days_field": 4, **qc}


def prism(source, treatment, curve):
    require_hash(treatment, TREATMENTS_HASH)
    names = ["broad_id", "depmap_id", "screen_id"]
    match = source.rename(columns=dict(zip(["native_compound_id", "cell_id", "source_run"], names)))
    fields = ["upper_limit", "lower_limit", "slope", "ec50", "auc"]
    fit = pd.read_csv(curve, usecols=[*names, *fields, "passed_str_profiling"])
    fit = fit[fit.passed_str_profiling.astype(str).str.lower().eq("true")]
    fit = fit.merge(match[[*names, "compound_id", "minimum", "second", "maximum"]], on=names, validate="many_to_one")
    fit = fit[np.isfinite(fit[fields]).all(axis=1) & fit.ec50.gt(0)]
    dose = pd.read_csv(treatment, usecols=["broad_id", "screen_id", "dose"]).dropna()
    window = dose[dose.dose.gt(0)].groupby(["broad_id", "screen_id"]).dose.agg(["min", "max"])
    fit = fit.join(window, on=["broad_id", "screen_id"]).dropna(subset=["min", "max"])
    fit = fit[fit["min"].lt(fit["max"])]
    auc, _ = integrate_prism(fit, fit["min"].to_numpy(), fit["max"].to_numpy())
    valid = np.abs(auc - fit.auc) <= TOLERANCE
    fit = fit.loc[valid].copy()
    fit = fit[(fit["min"] <= fit.second) & (fit["max"] >= fit.maximum)].copy()
    # Sample the verified continuous fit within the administered dose limits only.
    records = []
    for row in fit.itertuples(index=False):
        x = np.geomspace(row.min, row.max, 129)
        yy = row.lower_limit + (row.upper_limit - row.lower_limit) * expit(
            -row.slope * (np.log(x) - np.log(row.ec50)))
        vals = curve_endpoints(x, yy, minimum=row.minimum, second=row.second, maximum=row.maximum)
        records.append((row.depmap_id, row.compound_id, *vals))
    result = pd.DataFrame(records, columns=[*KEY, *VALUES]).groupby(KEY, as_index=False).median()
    return result, {"verified_native_auc": int(valid.sum()), "dose_supported_curves": len(fit)}


def extract(args):
    digest = registration(args.inputs)
    require_hash(args.original / "receipt.json",
                 "663eb5717b30c6f03ea43c04c4bb076bd233ea0c05405450e17f0c9e85f7fa86")
    original = json.loads((args.original / "receipt.json").read_text())["inputs"]
    for name in ("depmap_models", "ctrp_cells", "ctrp_experiments", "prism_curves"):
        require_hash(Path(original[name]["path"]), original[name]["sha256"])
    rows = pd.read_parquet(args.inputs / "sources.parquet")
    rows = rows[rows.source_id.eq(args.assay)]
    f = {
        "CTRPv2": lambda: ctrp(rows, args.ctrp, original),
        "GDSC2": lambda: gdsc(rows, args.gdsc, original),
        "PRISM20Q2": lambda: prism(rows, args.prism_treatments, original["prism_curves"]["path"]),
    }
    estimates, qc = f[args.assay]()
    result = rows.drop(columns="endpoint_value").merge(estimates, on=KEY, validate="one_to_one")
    with new_output(args.out) as out:
        result.to_parquet(out / "curves.parquet", index=False)
        publish_json(out / "receipt.json", {"input_sha256": digest, "source": args.assay,
            "curve_sha256": sha256(out / "curves.parquet"), "matched_rows": len(result),
            "matched_drugs": int(result.compound_id.nunique()), "ic50_supported_rows": int(result.ic50_bounded.notna().sum()),
            "ic50_supported_drugs": int(result.loc[result.ic50_bounded.notna(), "compound_id"].nunique()), "qc": qc})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    for stage, fields in {
        "register": ("original", "core", "extension", "gdsc-extension", "cold", "supplement"),
        "extract": ("original", "inputs", "ctrp", "gdsc", "prism-treatments"),
    }.items():
        task = sub.add_parser(stage)
        for field in (*fields, "out"):
            task.add_argument("--" + field, type=Path, required=True)
        if stage == "extract":
            task.add_argument("--assay", choices=SCREENS, required=True)
    args = p.parse_args()
    {"register": prepare, "extract": extract}[args.stage](args)


if __name__ == "__main__":
    main()
