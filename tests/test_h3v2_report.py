import json
import os
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

import pandas as pd

from h3v2.data import ROOT
from h3v2.grid import FAMILIES, initial
from h3v2.report import DISPLAY, report_docs
from h3v2.score import AXES


def report_rendering():
    with TemporaryDirectory(dir=ROOT / "tmp") as scratch:
        root = Path(scratch)
        tissues = [f"Tissue {n:02d}" for n in range(28)]
        arms = ("H2", *FAMILIES, "Champion")
        points = pd.DataFrame([{"scope":scope, "support":"all", "screen":"macro", "model":arm, "metric":metric,
                                "seed":"fixed" if arm == "H2" else "mean", "value":.38 + .01 * (arm != "H2"),
                                "seed_sd":None if arm == "H2" else .01}
                               for scope in ("drug", "cell", "held", "GDSC1", "LIMORE", "LICOB")
                               for arm in arms for metric in DISPLAY])
        by_tissue = points[points.scope.eq("drug")].assign(unit_count=20)
        tissue_table = pd.concat([by_tissue.assign(tissue=name) for name in tissues[:-1]], ignore_index=True)
        tissue_table.to_csv(root / "H3V2_PANCANCER_BY_TISSUE.tsv", sep="\t", index=False)
        pd.DataFrame([{"support":"all", "arm":arm, "rows":25, "rank_mse":.07,
                       "additive_mse":.02, "interaction_mse":.05} for arm in arms]).to_csv(
                           root / "H3V2_ERROR_COMPONENTS.tsv", sep="\t", index=False)
        ci = pd.DataFrame([{"scope":"drug", "support":"all", "screen":"macro", "candidate":"Champion", "baseline":"H2",
                            "metric":metric, "delta":.01, "lower":-.01, "upper":.03, "cluster_axis":"scaffold"}
                           for metric in AXES])
        grid, curves = [], []
        for family in FAMILIES:
            config = initial(family)[1]
            capacity = "depth" if config["backend"] == "gbdt" else "width"
            for i, factor in enumerate((.5, 1, 2)):
                variant = config | {capacity:int(config[capacity] * factor)}
                key = {"split":"drug-0", "family":family, "config_id":str(i), "frozen":i == 1,
                       "round":"r1", "backend":config["backend"]}
                device = "cuda" if config["backend"] == "torch" or (family == "gbdt" and i == 2) else "cpu"
                grid.append(key | {"utility":.4 - .02 * abs(1 - i), "budget":120, "configuration":json.dumps(variant)})
                objective = config.get("objective")
                branches = ("l2", "lambdarank") if objective == "l2_lambdarank" else (objective,)
                for branch in branches:
                    name = "ndcg@5" if branch == "lambdarank" else branch
                    for inner in range(3):
                        for iteration in (1, 10, 50):
                            curves.append(key | {"inner":inner, "iteration":iteration, "fit_wall_seconds":200,
                                                 "device":device, "branch":branch, "metric":name, "train_huber":.4, "val_huber":.55,
                                                 "train_objective":.4, "validation_objective":.5})
        grid, curves = pd.DataFrame(grid), pd.DataFrame(curves)
        grid.to_csv(root / "H3V2_GRID_SATURATION.tsv", sep="\t", index=False)
        curves.to_csv(root / "H3V2_FAMILY_LEARNING_CURVES.tsv", sep="\t", index=False)
        statuses = {name:{"verdict":"NO-GO"} for name in (*FAMILIES, "Champion")}
        score = {"overall":"CEILING-CONFIRMED", "verdicts":statuses, "score_passes":1, "tissue_roster":tissues}
        lock = {"external":{"champion":"hybrid", "operating_point":{"id":"simulated"}}}
        report_docs(root, points, ci, curves, grid, score, lock)
        result = (root / "RESULTS.md").read_text()
        assert "+0.01000" in result.splitlines()[2]
        assert all(f"| {tissue} |" in result for tissue in tissues)
        assert "| Tissue 27 | 미추정 | 미추정 | 미추정 | 0 | 0 |" in result
        tuning = (root / "ROUND1_TO_2_FAMILY_TUNING.md").read_text().splitlines()
        tree_times = next(line for line in tuning if line.startswith("| gbdt | r1 |"))
        assert [column.strip() for column in tree_times.split("|")[-3:-1]] == ["0.17", "0.33"]
        script = Path(__file__).resolve().parents[1] / "experiments/nextgen_h3_v2/plot_curves.py"
        env = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"))
        run(["/usr/bin/python3", str(script), str(root)], check=True, env=env)
        for name in ("H3V2_GRID_SATURATION.png", "H3V2_FAMILY_LEARNING_CURVES.png"):
            assert (root / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    print("Rendered numeric summary, 28 tissue rows and both curve figures using simulated results", flush=True)


if __name__ == "__main__":
    report_rendering()
