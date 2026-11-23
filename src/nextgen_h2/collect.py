"""Assemble reporting tables from the locked inner round artifacts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import KEY
from .choose import collected, diff
from .metrics import macro
from .run import read_json


def stress(root, out):
    methods = []
    decisions = read_json(root / "r2_select.json")
    for i in range(3):
        frames = collected(root, "r2", i)
        for axis, df in frames.items():
            metrics = ["spearman", "pearson"] if axis == "drug" else ["ndcg_at_5", "precision_at_5"]
            use = df[df.support.str.count("/").eq(2)]
            for (support, arm), selected in use.groupby(["support", "arm"], sort=True):
                view, fraction, population = support.split("/")
                method = arm.split("/")[2]
                model_weight = float(arm.split("/")[-1]) if method != "C1" else np.nan
                control = None
                if view == "path" and method != "C1":
                    control = f"{view}/{fraction}/C1"
                if view in decisions[str(i)]["views"] and method != "C1":
                    best = decisions[str(i)]["views"][view]["naive"]["spec"]
                    control = f"{view}/{fraction}/{best['method']}/{best['weight']:g}"
                    if arm == control:
                        control = f"{view}/{fraction}/C1"
                for metric in metrics:
                    effect = None if control is None else diff(use, arm, control, support, metric)
                    methods.append({"outer_fit": i, "view": view, "fraction_masked": float(fraction),
                                    "support": population, "arm": method, "weight": model_weight,
                                    "axis": axis, "metric": metric, "point": macro(selected, metric),
                                    "fold_units": len(selected), "control": control, **(effect or {})})
    pd.DataFrame(methods).to_csv(out / "H2_MISSING_STRESS.tsv", sep="\t", index=False)


def mechanism(root, assets, out):
    sig = pd.read_parquet(assets / "lincs_selected_signatures.parquet", columns=[KEY, "dose_um", "time_h", "moa"])
    known_moa = sig.dropna(subset="moa")
    known_moa = known_moa[known_moa.moa.ne("")]
    categories = known_moa.groupby(KEY).moa.agg(lambda v: v.value_counts().sort_index().idxmax())
    profile = sig.groupby(KEY).agg(max_dose=("dose_um", "max"), times=("time_h", "nunique"))
    profile["cmap_mechanism"] = categories
    profile["exposure"] = np.where(profile.max_dose.ge(10.), "≥10uM", "<10uM")
    profile["time_support"] = np.where(profile.times.eq(2), "6h+24h", "one_time")
    results = []
    for outer in range(3):
        unit = collected(root, "r1controls", outer)["drug"]
        for view in ("painting", "lincs"):
            data = unit[unit.support.eq(view + "_observed")].pivot(
                index=["screen", "fold", "unit"], columns="arm", values="spearman").reset_index()
            data = data.join(profile, on="unit")
            data["delta_C2"] = data[view + "/C4"] - data[view + "/C2"]
            data["delta_C3"] = data[view + "/C4"] - data[view + "/C3"]
            for axis in ("screen", "cmap_mechanism", "exposure", "time_support"):
                values = data[axis].fillna("unannotated")
                for label, part in data.groupby(values):
                    if part.unit.nunique() < 10:
                        continue
                    results.append({"outer_fit": outer, "view": view, "axis": axis, "group": label,
                                    "parents": part.unit.nunique(), "screen_drug_units": len(part),
                                    "macro_rho_delta_C2": macro(part, "delta_C2"),
                                    "macro_rho_delta_C3": macro(part, "delta_C3")})
    pd.DataFrame(results).to_csv(out / "H2_INNER_ERROR_STRATA.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    stress(args.root, args.out)
    mechanism(args.root, args.assets, args.out)
