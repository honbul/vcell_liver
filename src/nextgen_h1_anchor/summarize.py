"""Export aggregate inner results, policy selections and assay budgets."""
import argparse
import json
from pathlib import Path

import pandas as pd

from vc_build.io import sha256

from .inputs import COLD, CROSS, LICOB_LABEL, json_write
from .prepare import TRANSFER
from .tune import config_key


def summarize(data, freezes, out):
    from vc_build.licob_dose_sources import dose_metadata

    out.mkdir()
    receipt = json.loads((data / "receipt.json").read_text())
    limore = pd.read_parquet(CROSS / "test_inputs/rows.parquet", columns=["compound_id", "dose_points"])
    licob = dose_metadata(LICOB_LABEL, pd.read_parquet(COLD / "query.parquet"))
    dose = {
        "LIMORE": limore.groupby("compound_id").dose_points.first().dropna().to_dict(),
        "LICOB": licob.set_index("compound_id").points.to_dict(),
    }
    dose[TRANSFER] = dose["LICOB"]
    policy_rows, inner_rows, work = [], [], {}
    for task in receipt["tasks"]:
        study = task.rsplit("-", 1)[0]
        directory = freezes / task
        path = directory / "freeze.json"
        stamp = json.loads(path.read_text())
        inner_file = directory / "inner.tsv"
        if sha256(inner_file) != stamp["artifacts"]["inner.tsv"]:
            raise ValueError("Tuning log hash mismatch")
        report = pd.read_csv(inner_file, sep="\t")
        if (set(report["round"]) != {1, 2, 3} or set(report.inner_fold) != {0, 1, 2}
                or set(report.k) != {0, 3, 5, 8}):
            raise ValueError("The archived tuning log does not cover the three inner rounds")
        anchors = stamp["anchors"]
        for policy, summary in stamp["round1_by_policy"].items():
            selected = policy == stamp["policy"]
            policy_rows.append({"task": task, "policy": policy, "selected": selected,
                                "r1_direct_ndcg": summary["linear_ndcg"], "r1_bsimple_ndcg": summary["baseline_ndcg"],
                                "r1_paired_gain": summary["gain"], "r1_config": config_key(summary["best_linear"]),
                                "a3": "|".join(anchors[:3]) if selected else "",
                                "a5": "|".join(anchors[:5]) if selected else "",
                                "a8": "|".join(anchors) if selected else "",
                                "q_drugs": len(stamp["query_panel"]) if selected else "",
                                "fit_groups": stamp["fit_groups"], "held_groups": stamp["outer_groups"],
                                "freeze_sha256": sha256(path)})
        policy_rows.append({"task": task, "policy": "random-3-seed-mean", "selected": False,
                            "r1_paired_gain": stamp["random_3seed_gain"], "freeze_sha256": sha256(path)})
        chosen = report[report.policy.eq(stamp["policy"]) & report.k.eq(5)].copy()
        chosen["model_group"] = chosen.recipe.map(lambda r: json.loads(r))
        chosen["family"] = chosen.model_group.map(lambda p: p.get("prior") or p.get("robust") or p["family"])
        grouped = (chosen.groupby(["round", "family", "recipe"], sort=True)
                   .agg(ndcg_at_5=("ndcg_at_5", "mean"), precision_at_5=("precision_at_5", "mean"),
                        inner_n_min=("n", "min"), inner_n_max=("n", "max"), inner_folds=("inner_fold", "nunique"))
                   .reset_index())
        winners = grouped.sort_values(["ndcg_at_5", "precision_at_5"], ascending=False)
        winners = winners.drop_duplicates(["round", "family"]).sort_values(["round", "family"])
        winners.insert(0, "task", task)
        inner_rows.append(winners)
        notes = {k: stamp[k] for k in stamp if k not in {"round1_by_policy", "anchors", "query_panel", "code_sha256"}}
        true_doses = [int(dose.get(study, {}).get(d, 0)) for d in anchors[:5]]
        notes.update(freeze_sha256=sha256(path), r1_selected_policy=stamp["round1_by_policy"][stamp["policy"]],
                     grid_rows=len(report), candidate_counts=report.groupby("round").recipe.nunique().to_dict(),
                     actual_anchor_dose_points=true_doses,
                     cost_known=all(true_doses), query_count=len(stamp["query_panel"]))
        work[task] = notes
    pd.DataFrame(policy_rows).to_csv(out / "H1_ANCHOR_POLICIES.tsv", sep="\t", index=False,
                                     na_rep="NA", float_format="%.8g")
    pd.concat(inner_rows).to_csv(out / "H1_INNER_FAMILY_WINNERS.tsv", sep="\t", index=False, float_format="%.8g")
    json_write(out / "inner_freeze_summary.json", {"tasks": work, "label_receipt_sha256": sha256(data / "receipt.json"),
                                                  "used_held_labels": False})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "freezes", "out"):
        cli.add_argument(arg, type=Path)
    summarize(**vars(cli.parse_args()))
