"""Independent post-score integrity checks that never load held outcomes."""


import numpy as np
import pandas as pd

from vc_build.io import publish_json, sha256

from .data import ROOT, contract, read_json, split_id
from .evaluate import check_forecasts
from .select import STAGES


def audit():
    out = ROOT / "outer-score"
    scored = read_json(out / "score_receipt.json")
    groups, layout, _, checks = check_forecasts()
    conf = contract()
    if scored["predictions"] != checks or scored["score_passes"] != 1:
        raise ValueError("The reported score does not match the outcome-free freezes")
    if scored["screen_score_passes"] != dict.fromkeys(conf["screens"], 1):
        raise ValueError("An outer screen has a different score count")
    if sha256(out / "score-pass.json") != scored["score_claim_sha256"]:
        raise ValueError("The exclusive score claim changed")
    if len({row for _, rows in groups for row in rows.row_id}) != scored["native_observations"]:
        raise ValueError("Native held row count differs across the nine forecasters")
    if len({line for _, rows in groups for line in rows.lineage_group}) != scored["global_lineages"]:
        raise ValueError("Incorrect pooled global-lineage support")
    if any(sha256(out / name) != digest for name, digest in scored["output_sha256"].items()):
        raise ValueError("Score table differs from its sealed hash")

    summary_counts, audit_rows, budgets = {}, 0, {}
    for stage in STAGES:
        book = read_json(ROOT / f"analysis/{stage}-summary.json")
        counts = []
        for index in range(9):
            split = split_id(index)
            group = ROOT / stage / split
            if sha256(group / "forecast.parquet") != book["inputs_sha256"][split]:
                raise ValueError("An inner forecast fails its frozen hash")
            fit = read_json(group / "receipt.json")
            fits = [row for row in fit["runs"] if "seed" in row and not row.get("timing_replay")]
            counts.append(len(fits))
            for row in fits:
                budget = (row["observed_cell_drug_pairs"], row["pair_exposures"], row["updates"])
                if budgets.setdefault((split, row["inner"]), budget) != budget:
                    raise ValueError("H3 methods differ in observed data or update budget")
                if row["observed_cell_drug_pairs"] * conf["epochs"] != row["pair_exposures"]:
                    raise ValueError("An arm used a different observation budget")
                if row["updates"] != conf["epochs"] * conf["updates_per_epoch"] or row["shared_params"] <= 0:
                    raise ValueError("Wrong update budget or frozen gradients")
            if stage not in ("round1", "round2-block"):
                continue
            for fold in range(2):
                path = group / f"gradients-inner{fold}.parquet"
                if sha256(path) != book["inputs_sha256"][f"{split}/{path.stem}"]:
                    raise ValueError("A raw per-batch gradient artifact changed")
                diag = pd.read_parquet(path)
                compute = [row for row in fits if row["inner"] == fold]
                if len(diag) != len(compute) * conf["epochs"] * conf["updates_per_epoch"] * 3:
                    raise ValueError("Missing gradient task pair records")
                ids = ["arm", "update", "loss_a", "loss_b"]
                if diag.duplicated(ids).any():
                    raise ValueError("Duplicated gradient update evidence")
                use = diag.loc[diag.conflict.notna()]
                if (not np.isfinite(use[["gradient_cosine", "norm_ratio", "weighted_norm_ratio"]]).all().all()
                        or not (use.conflict == use.gradient_cosine.lt(0)).all()):
                    raise ValueError("The recorded evidence disagrees with the conflict definition")
                audit_rows += len(diag)
        summary_counts[stage] = sum(counts)
    publish_json(out / "verification.json", {
        "all_forecast_keys_verified": True, "outer_native_rows": scored["native_observations"],
        "outer_forecasts": len(groups), "outer_seed_models": layout,
        "checkpoint_count": sum(len(s["checkpoints"]) for s in checks.values()),
        "round1_and_2_fits": summary_counts, "raw_batch_pair_records": audit_rows,
        "inner_folds_with_same_exposures_across_all_stages": len(budgets),
        "score_passes": scored["score_passes"], "score_receipt_sha256": sha256(out / "score_receipt.json"),
        "verification_loaded_outer_outcomes": False,
    })


if __name__ == "__main__":
    audit()
