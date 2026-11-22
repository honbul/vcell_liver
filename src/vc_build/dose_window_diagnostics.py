"""Compare screen association on identical cell/drug rows at a common concentration window."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.special import expit

from .cross_source_data import MODEL_COLS, read_raw, roster_maps, roster_table
from .io import publish_json, sha256
from .v0plus_cv import training_inputs

TREATMENTS_HASH = "f0ec000acda7c1a01ebae2aa0a2b5dba2821abb674f1fec7a363cdc60d8b379e"
WINDOW = (0.1, 1.0)
TOLERANCE = 0.005


def measured_points(inputs, core, post_qc):
    annotations = read_raw(inputs["ctrp_cells"]["path"], ["ccl_name", "master_ccl_id"], sep="\t")
    models = read_raw(inputs["depmap_models"]["path"], MODEL_COLS)
    names = roster_table(annotations, "CTRPv2", "ccl_name", "master_ccl_id")
    mapped = roster_maps(names, models)[["source_key", "cell_id"]].dropna().drop_duplicates()
    experiments = read_raw(inputs["ctrp_experiments"]["path"], [
        "experiment_id", "master_ccl_id", "snp_fp_status",
    ], sep="\t")
    matched = experiments.loc[experiments.snp_fp_status.eq("SNP-matched-reference")].merge(
        mapped, left_on="master_ccl_id", right_on="source_key", validate="many_to_one",
    )[["experiment_id", "cell_id"]].drop_duplicates()
    keys = core[["cell_id", "compound_id", "native_compound_id"]].rename(
        columns={"native_compound_id": "master_cpd_id"},
    )
    parts = []
    for chunk in pd.read_csv(post_qc, sep="\t", chunksize=400_000, usecols=[
        "experiment_id", "master_cpd_id", "cpd_conc_umol", "cpd_avg_pv",
    ], dtype={"experiment_id": str, "master_cpd_id": str}):
        candidate = chunk[chunk.master_cpd_id.isin(keys.master_cpd_id)]
        parts.append(candidate.merge(matched, on="experiment_id", validate="many_to_one")
                     .merge(keys, on=["cell_id", "master_cpd_id"], validate="many_to_one"))
    points = pd.concat(parts, ignore_index=True).dropna(subset=["cpd_conc_umol", "cpd_avg_pv"])
    points = points[points.cpd_conc_umol.gt(0) & np.isfinite(points.cpd_avg_pv)]
    values = points.groupby(["cell_id", "compound_id", "experiment_id", "cpd_conc_umol"]).cpd_avg_pv.median()
    grid = np.linspace(*np.log(WINDOW), 129)
    estimates = []
    for key, curve in values.groupby(level=[0, 1, 2], sort=False):
        actual = curve.index.get_level_values("cpd_conc_umol").to_numpy()
        if actual.min() > WINDOW[0] or actual.max() < WINDOW[1]:
            continue
        # Published transformed viability is a fraction, and is clipped to match PRISM AUC.
        curve_at_window = np.clip(np.interp(grid, np.log(actual), curve.to_numpy()), 0, 1)
        estimates.append((*key[:2], float(trapezoid(curve_at_window, dx=1 / 128)), float(curve_at_window[-1])))
    result = pd.DataFrame(estimates, columns=["cell_id", "compound_id", "window_auc", "dose_1uM"])
    return result.groupby(["cell_id", "compound_id"], as_index=False).median()


def integrate_prism(rows, low, high):
    u, lo, slope, e = (rows[name].to_numpy()[:, None] for name in ("upper_limit", "lower_limit", "slope", "ec50"))
    t = np.linspace(0, 1, 129)[None, :]
    logdose = np.log(low)[:, None] * (1 - t) + np.log(high)[:, None] * t
    y = np.clip(lo + (u - lo) * expit(-slope * (logdose - np.log(e))), 0, 1)
    return trapezoid(y, dx=1 / 128, axis=1), y[:, -1]


def published_curves(inputs, core, treatments):
    keys = core[["cell_id", "compound_id", "native_compound_id", "source_run"]].rename(
        columns={"cell_id": "depmap_id", "native_compound_id": "broad_id", "source_run": "screen_id"},
    )
    columns = ["broad_id", "depmap_id", "screen_id", "upper_limit", "lower_limit", "slope", "ec50", "auc",
               "passed_str_profiling"]
    parts = []
    for block in pd.read_csv(inputs["prism_curves"]["path"], usecols=columns, chunksize=150_000):
        match = block[block.broad_id.isin(keys.broad_id)
                      & block.passed_str_profiling.astype(str).str.lower().eq("true")]
        parts.append(match.merge(keys, on=["broad_id", "depmap_id", "screen_id"], validate="many_to_one"))
    selected = pd.concat(parts, ignore_index=True)
    dose = pd.read_csv(treatments, usecols=["broad_id", "screen_id", "dose"]).dropna()
    windows = dose[dose.dose.gt(0)].groupby(["broad_id", "screen_id"]).dose.agg(["min", "max"])
    joined = selected.join(windows, on=["broad_id", "screen_id"], validate="many_to_one")
    variables = ["upper_limit", "lower_limit", "slope", "ec50", "auc", "min", "max"]
    joined = joined[np.isfinite(joined[variables]).all(axis=1) & joined.ec50.gt(0)
                    & joined["min"].lt(joined["max"])]
    native, _ = integrate_prism(joined, joined["min"].to_numpy(), joined["max"].to_numpy())
    error = np.abs(native - joined.auc.to_numpy())
    within = joined[error <= TOLERANCE]
    supported = within[within["min"].le(WINDOW[0]) & within["max"].ge(WINDOW[1])].copy()
    supported["window_auc"], supported["dose_1uM"] = integrate_prism(
        supported, np.full(len(supported), WINDOW[0]), np.full(len(supported), WINDOW[1]),
    )
    result = supported.rename(columns={"depmap_id": "cell_id"})
    merged = result.groupby(["cell_id", "compound_id"], as_index=False)[["window_auc", "dose_1uM"]].median()
    return merged, {
        "curve_rows": len(selected), "with_doses_and_finite_fit": len(joined),
        "native_auc_abs_error_quantiles": np.quantile(error, [.5, .9, .99]).tolist(),
        "native_reproduced": len(within), "common_window_rows": len(merged),
    }


def compare(frame, axis, minimum):
    values = ["native_auc", "window_auc", "dose_1uM"]
    rows = []
    for _, group in frame.groupby(axis):
        if len(group) < minimum:
            continue
        row = {}
        for item in values:
            x, y = group[item + "_ctrp"], group[item + "_prism"]
            if x.nunique() > 1 and y.nunique() > 1:
                row[item] = (x.corr(y, method="spearman"), x.corr(y, method="pearson"))
        if len(row) == len(values):
            rows.append(np.asarray([row[key] for key in values]))
    if not rows:
        return {"groups": 0}
    return {"groups": len(rows), **dict(zip(values, np.stack(rows).mean(axis=0).tolist()))}


def run(args):
    if sha256(args.treatments) != TREATMENTS_HASH:
        raise ValueError("PRISM dose metadata is not the registered public Figshare release")
    train, _, receipt = training_inputs(args.bundle)
    inputs = receipt["inputs"]
    for name in ("ctrp_cells", "ctrp_experiments", "depmap_models", "prism_curves"):
        if sha256(inputs[name]["path"]) != inputs[name]["sha256"]:
            raise ValueError(f"Source assay asset changed: {name}")
    digest = sha256(args.ctrp_post_qc)
    ctrp, prism = (train[train.source_id.eq(name)] for name in ("CTRPv2", "PRISM20Q2"))
    keys = ["cell_id", "compound_id"]
    common = ctrp[[*keys, "lineage_group", "endpoint_value"]].merge(
        prism[[*keys, "endpoint_value"]], on=keys, suffixes=("_ctrp", "_prism"), validate="one_to_one",
    ).rename(columns={"endpoint_value_ctrp": "native_auc_ctrp", "endpoint_value_prism": "native_auc_prism"})
    common_keys = common[keys]
    a = measured_points(inputs, ctrp.merge(common_keys, on=keys), args.ctrp_post_qc)
    b, diagnostics = published_curves(inputs, prism.merge(common_keys, on=keys), args.treatments)
    merged = common.merge(a, on=keys, validate="one_to_one").merge(
        b, on=keys, suffixes=("_ctrp", "_prism"), validate="one_to_one",
    )
    paired = merged.groupby(["lineage_group", "compound_id"], as_index=False).mean(numeric_only=True)
    gdsc = train[train.source_id.eq("GDSC2")]
    gdsc = gdsc[gdsc.dose_min_uM.le(WINDOW[0]) & gdsc.dose_max_uM.ge(WINDOW[1])]
    publish_json(args.out, {
        "window_uM": WINDOW, "prism_NATIVE_AUC_tolerance": TOLERANCE,
        "train_receipt_sha256": sha256(args.bundle / "receipt.json"),
        "ctrp_post_qc_sha256": digest, "prism_treatments_sha256": TREATMENTS_HASH,
        "exact_ctrp_prism_cell_drug_pairs": len(common), "matching_window_cells_drugs": len(merged),
        "matching_window_lineage_drugs": len(paired), "matching_window_drugs": paired.compound_id.nunique(),
        "prism_reconstruction": diagnostics,
        "within_drug": compare(paired, "compound_id", minimum=30),
        "within_lineage": compare(paired, "lineage_group", minimum=5),
        "gdsc2_range_metadata": {"rows": len(gdsc), "drugs": gdsc.compound_id.nunique()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bundle", "ctrp-post-qc", "treatments", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
