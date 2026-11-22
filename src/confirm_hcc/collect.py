"""Verify all nine few-shot runs and collect aggregate 3D tables and claim gates."""

import argparse
from pathlib import Path

import pandas as pd

from vc_build.io import publish_json, sha256, verify_forecast

from .contract import read_json, verified
from .fewshot import A, B, C, D, DRAWS
from .ledger import save
from .pdo_data import BUDGETS, REPEATS, STUDIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _, source = verify_forecast(args.root / "source.parquet")
    reg = args.root / "registration/registration.json"
    registration = read_json(reg)
    if registration["forecast_sha256"] != source["sha256"]:
        raise ValueError("3D registration and forecast do not match")
    results, contrasts, parameters, evidence = [], [], [], {}
    for study in STUDIES:
        for k in BUDGETS:
            p = args.root / study / f"k{k}"
            data, forecast = verify_forecast(p / "predict/forecasts.parquet")
            score = read_json(p / "score/score_receipt.json")
            if (score["study"], score["k"], score["registration_sha256"], score["predictions_sha256"],
                score["repeats"], score["paired_bootstrap_draws"]) != (
                    study, k, sha256(reg), forecast["sha256"], REPEATS, DRAWS):
                raise ValueError("3D score did not use the frozen prediction or repeat contract")
            file_hashes = score["files"]
            for name, digest in file_hashes.items():
                verified(p / "score" / name, digest)
            if set(data.fold) != set(range(REPEATS)):
                raise ValueError("3D split coverage is incomplete")
            fit = verified(p / "predict/calibration_parameters.tsv", forecast["provenance"]["parameters_sha256"])
            parameters.append(pd.read_csv(fit, sep="\t"))
            current = pd.read_csv(p / "score/metrics.tsv", sep="\t")
            for panel, part in current.groupby("track"):
                expected = {B, C, *D} | (set(A) if "/exact68/" in panel else set())
                if set(part.model_id) != expected or len(part) != len(expected):
                    raise ValueError("3D matched comparator missing")
            results.append(current)
            contrasts.append(pd.read_csv(p / "score/paired.tsv", sep="\t"))
            evidence[f"{study}/K{k}"] = {"score_receipt_sha256": sha256(p / "score/score_receipt.json"),
                                         "frozen_predictions_sha256": forecast["sha256"]}
    args.out.mkdir(parents=True, exist_ok=False)
    metrics = pd.concat(results, ignore_index=True)
    paired = pd.concat(contrasts, ignore_index=True)
    params = pd.concat(parameters, ignore_index=True)
    save(args.out / "FEWSHOT_3D_METRICS.tsv", metrics)
    save(args.out / "FEWSHOT_3D_PAIRED_95CI.tsv", paired)
    names = ["local_mean_weight", "gamma_d_eb_structured", "gamma_d_eb_pairwise_rank"]
    params[names] = params[names].apply(pd.to_numeric)
    counts = params.groupby("track")[names].agg(["median", "mean"])
    counts.columns = [a + "_" + b for a, b in counts.columns]
    for key in names[1:]:
        counts[key + "_zero_count"] = params.groupby("track")[key].apply(lambda s: int(s.eq(0).sum()))
    save(args.out / "FEWSHOT_3D_CALIBRATION.tsv", counts.reset_index())
    rows = []
    for (track, model), records in paired.groupby(["track", "model_id"]):
        lcb = records.set_index(["baseline", "metric"]).ci95_low
        pass_joint = bool((lcb.loc[[(b, m) for b in (B, C) for m in ("ndcg_at_5", "residual_spearman")]] > 0).all())
        rows.append({"track": track, "model_id": model, "joint_gate_pass": pass_joint})
    decision = {"stratum_gate": rows, "matched_panels_with_gate": sum(
        row["joint_gate_pass"] for row in rows if "/exact68/" in row["track"]),
        "scope": "Each stratum requires improvement in both NDCG and double-centered interaction over local mean and shrinkage."}
    publish_json(args.out / "fewshot_decision.json", decision)
    transfer = args.root / "transfer"
    stamp = read_json(transfer / "receipt.json")
    _, frozen = verify_forecast(transfer / "transfer_forecasts.parquet")
    if stamp["prediction_sha256"] != frozen["sha256"] or stamp["cross_study_directions"] != 6:
        raise ValueError("Cross-study transfer audit is incomplete")
    name = "TRANSFER_MATCHED_PATIENT_PANELS.tsv"
    save(args.out / name, pd.read_csv(verified(transfer / name, stamp["table_sha256"]), sep="\t"))
    publish_json(args.out / "fewshot_collection_receipt.json", {
        "source_predicted_sha256": source["sha256"], "registration_sha256": sha256(reg),
        "source_support": source["provenance"]["support"], "patient_balance": registration["calibration_balance"],
        "public_workbook_sha256": registration["source_workbook_sha256"], "keyed_evaluation": evidence,
        "transfer_forecast_sha256": frozen["sha256"], "files": {p.name: sha256(p) for p in args.out.iterdir()},
    })


if __name__ == "__main__":
    main()
