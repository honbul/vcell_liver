"""Build a complete stage, hyperparameter, and learning-curve table from sealed development."""
import argparse
from pathlib import Path

import pandas as pd

from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256

from .search import stage_input


def compile_curves(work, out, sources):
    out.mkdir(exist_ok=False, parents=True)
    config_dir = work.parent / "contracts"
    records, evidence = [], {}
    tables = []
    for path in sources:
        tables.append(pd.read_csv(path, sep="\t", keep_default_na=False))
        evidence[str(path)] = sha256(path)
    for f in range(3):
        _, _, controls = stage_input(work, config_dir, "virtual_controls", f)
        tables.append(controls)
        for i in range(3):
            receipt = work / "virtual_controls" / f"{f}-{i}" / "receipt.json"
            evidence[str(receipt)] = sha256(receipt)
    for candidates in tables:
        for stage, block in candidates.groupby("round_stage", sort=False):
            manifest = config_dir / (stage + ".json")
            registry = read_json(manifest)
            evidence[str(manifest)] = sha256(manifest)
            settings = {}
            for outer in range(3):
                settings[outer] = {item["id"]: item for item in registry[str(outer)]}
            for rec in block.to_dict("records"):
                key = rec["arm"]
                by_fold = settings[int(rec["outer_fit"])]
                tap = key.rsplit("/", 1)[0] if stage == "resid" else key
                meta = by_fold.get(tap, {})
                design = meta.get("design", meta.get("kernel", {}))
                lincs = design.get("lincs", {})
                view = meta.get("view", {})
                generator = meta.get("generator", {})
                residual = meta.get("parameters", {})
                records.append(rec | {"axis_tuned": meta.get("axis", ""), "w_lincs": lincs.get("weight"),
                                      "penalty": design.get("penalty"), "w_mask": lincs.get("mask"),
                                      "LINCS_recipe": lincs.get("recipe"), "LINCS_kernel": lincs.get("kernel"),
                                      "LINCS_transform": lincs.get("transform"), "intercept": design.get("intercept"),
                                      "anchor_penalty": design.get("anchor_penalty"),
                                      "view": view.get("name"), "view_weight": view.get("weight", meta.get("weight")),
                                      "view_transform": view.get("transform"),
                                      "virtual_method": generator.get("method"),
                                      "virtual_regularization": generator.get("regularization"),
                                      "virtual_width": generator.get("width"),
                                      "virtual_control": meta.get("control") if generator else None,
                                      "residual_family": residual.get("family"),
                                      "residual_objective": residual.get("objective"),
                                      "residual_scale": float(key.rsplit("/", 1)[-1]) if residual else None,
                                      **{f"w_{key}": design.get(key) for key in
                                         ("tanimoto", "target", "properties", "identity")}})
    points = pd.DataFrame(records).drop_duplicates(["round_stage", "outer_fit", "axis", "arm", "support"])
    learning = []
    for f in range(3):
        for i in range(3):
            file = work / "resid" / f"{f}-{i}" / "learning.json"
            report = read_json(file)
            evidence[str(file)] = sha256(file)
            if report["inner_query_labels_available_to_estimator"] != 0:
                raise ValueError("H2 nonlinear learning curve exposed held inner query outcomes")
            for curve in report["curve"]:
                dims = {"round_stage": "resid_learning", "outer_fit": f, "inner_fold": i,
                        "axis": "learning", "arm": curve["name"], "support": "OOF inner-fit early stop",
                        "residual_family": curve["family"], "fit_pool_oof_rows": report["fit_pool_oof_rows"],
                        "early_stop_rows": report["early_stop_rows"]}
                if curve["family"] == "gbdt":
                    for section, metrics in curve["data"].items():
                        for metric, history in metrics.items():
                            learning.extend(dims | {"iteration": index, "learning_metric": f"{section}/{metric}",
                                                   "learning_value": v} for index, v in enumerate(history, 1))
                else:
                    for note in curve["data"]:
                        for key in ("fit", "early_stop"):
                            learning.append(dims | {"iteration": note["epoch"], "learning_metric": key,
                                                     "learning_value": note[key]})
    pd.concat([points, pd.DataFrame(learning)], ignore_index=True).replace("", pd.NA).to_csv(
        out / "H2V2_SATURATION_CURVE.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "receipt.json", {"source_files": evidence, "point_rows": len(points),
                                       "curve_rows": len(learning),
                                       "curve_sha256": sha256(out / "H2V2_SATURATION_CURVE.tsv")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("points", nargs="+", type=Path)
    a = parser.parse_args()
    compile_curves(a.work, a.out, a.points)
