"""Collect the goal-section-0 re-runs and report the two aggregates separately.

There are two different numbers that both get called "the validation score of model X", and the
first build printed them in one table without saying which was which:

  seed_mean_of_scores    score each seed's model, then average the scores.  This is the
                         pre-registered primary. It answers "how well does this recipe do".
  score_of_mean_pred     average the three seeds' predictions, then score once.  This is a
                         three-model ensemble, it is always at least as good by Jensen, and it
                         answers a different question. It was never pre-registered.

They are computed and written out under those two names here, never merged.

Validation only. This script does not read the test split.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from . import evaluate as ev
from .data import Dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()

    runs = {}
    for f in sorted(Path(a.runs_dir).glob("*/record.json")):
        r = json.loads(f.read_text())
        runs[(r["kind"], r["seed"])] = (r, f.parent)

    ds = Dataset(a.data_dir, a.manifest_dir, device="cpu")
    va = ds.conditions("val")
    vy, vl = va.y.cpu().numpy(), va.line.cpu().numpy()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- per-run table --------------------------------------------------------------------
    lines = ["kind\tseed\tlr\twd\tbest_epoch\tepochs_run\tval_at_init\tval_macro_delta_rmse\t"
             "train_seconds\tshared_init_sha256\tparams_total"]
    for (kind, seed), (r, _) in sorted(runs.items()):
        lines.append("\t".join([
            kind, str(seed), f"{r['lr']:g}", f"{r['wd']:g}", str(r["best_epoch"]),
            str(r["epochs_run"]), f"{r['val_at_init']:.6f}", f"{r['val_macro_delta_rmse']:.6f}",
            f"{r['train_seconds']:.1f}", r["shared_init_sha256"][:16], str(r["params"]["total"])]))
    (out / "runs.tsv").write_text("\n".join(lines) + "\n")

    # ---- the two aggregates, kept apart ----------------------------------------------------
    agg = {}
    for kind in ["A", "C"]:
        per_seed = {s: runs[(kind, s)][0]["val_macro_delta_rmse"] for s in a.seeds}
        preds = [np.load(runs[(kind, s)][1] / "val_pred.npy") for s in a.seeds]
        ens = ev.macro_delta_rmse(np.mean(preds, 0), vy, vl)
        agg[kind] = dict(
            per_seed_score={str(k): round(v, 6) for k, v in per_seed.items()},
            seed_mean_of_scores=round(st.mean(per_seed.values()), 6),
            seed_sd_of_scores=round(st.stdev(per_seed.values()), 6),
            score_of_mean_prediction=round(float(ens), 6),
            shared_init_sha256={str(s): runs[(kind, s)][0]["shared_init_sha256"] for s in a.seeds},
            val_at_init={str(s): round(runs[(kind, s)][0]["val_at_init"], 6) for s in a.seeds},
            best_epoch={str(s): runs[(kind, s)][0]["best_epoch"] for s in a.seeds},
            params_total=runs[(kind, a.seeds[0])][0]["params"]["total"])

    # the controlled-comparison precondition: same seed, same shared starting weights
    agg["shared_init_matches_across_kinds"] = {
        str(s): bool(runs[("A", s)][0]["shared_init_sha256"]
                     == runs[("C", s)][0]["shared_init_sha256"]) for s in a.seeds}

    rel_per_seed = {str(s): round((runs[("A", s)][0]["val_macro_delta_rmse"]
                                   - runs[("C", s)][0]["val_macro_delta_rmse"])
                                  / runs[("A", s)][0]["val_macro_delta_rmse"] * 100, 4)
                    for s in a.seeds}
    agg["C_over_A_percent"] = dict(
        per_seed=rel_per_seed,
        mean_of_per_seed=round(st.mean(rel_per_seed.values()), 4),
        on_seed_mean_of_scores=round(
            (agg["A"]["seed_mean_of_scores"] - agg["C"]["seed_mean_of_scores"])
            / agg["A"]["seed_mean_of_scores"] * 100, 4),
        on_score_of_mean_prediction=round(
            (agg["A"]["score_of_mean_prediction"] - agg["C"]["score_of_mean_prediction"])
            / agg["A"]["score_of_mean_prediction"] * 100, 4),
        seeds_agreeing_in_sign=int(max(sum(v > 0 for v in rel_per_seed.values()),
                                       sum(v < 0 for v in rel_per_seed.values()))),
        note="positive means C beats A; the three rows are three different statistics, not three "
             "estimates of one")

    (out / "validation_summary.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1), flush=True)


if __name__ == "__main__":
    main()
