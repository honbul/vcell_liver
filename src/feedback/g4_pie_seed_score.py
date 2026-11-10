#!/usr/bin/env python
"""G4 task 2: score PIE seed retrains (HepG2-held-out fold) with PIE's own metrics, next to the
official checkpoint (G3 GPU reproduction) and the G3 simple baselines on the same split.

Runs in the PIE env. For each seed run dir (<runs>/<fold>_seed<S>/eval/<setting>/) it reads
predictions_best_auprc.parquet and scores it with pie.metrics.score (official 6 metrics) plus G3's
target-excluded / delta-scale extras (g3_pie_baselines.extra_metrics). The official checkpoint and
baselines are read from G3's per-pair table (same code, same rows). Checkpoint selection for every
seed is PIE's own (max val AUPRC on source-line val rows; HepG2 never seen).
Outputs (reports/feedback/G4/): pie_seed_summary_<fold>.tsv (per predictor), pie_seed_spread_<fold>.tsv
(mean/min/max/range over seeds), pie_seed_paired_bootstrap_<fold>.tsv, pie_seed_ckpt_<fold>.json;
assets: pred_<fold>_<setting>_seeds.npz (held-out-line settings, for F3 metrics).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_pie_baselines import (BOOT_KEYS, EXTRA_KEYS, SETTINGS, Data, context_mean_of_pairs,  # noqa: E402
                              extra_metrics, paired_boot, scoring_inputs)
from pie.data.splits import load_split  # noqa: E402
from pie.metrics import score  # noqa: E402

BEST_BASE = {"unseen_ctx": "same_target_cal", "unseen_pert": "context_mean_kd", "unseen_ctx_pert": "global_mean_kd"}
RIDGE = {"unseen_ctx": "ridge_stacked_kd", "unseen_pert": "ridge_inline_kd", "unseen_ctx_pert": "ridge_stacked_kd"}
OFFICIAL = ["binary_auprc", "sig_jaccard", "direction_match", "spearman_nsig", "spearman_lfc", "discrimination_score_l1"]


def ckpt_info(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = dict(global_step=ck.get("global_step"), epoch=ck.get("epoch"))
    for k, v in (ck.get("callbacks") or {}).items():
        if "ModelCheckpoint" in k and isinstance(v, dict) and v.get("monitor"):
            bs = v.get("best_model_score")
            info.update(monitor=v.get("monitor"), best_model_score=float(bs) if bs is not None else None)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", default="hepg2")
    ap.add_argument("--runs", default="assets/g4/pie_seeds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--g3-pairs", default="assets/g3_pie/baselines/pie_split_per_pair_hepg2.tsv.gz")
    ap.add_argument("--official-ckpt", default="assets/g3_pie/models/replogle_wdataset/hepg2/best_auprc.ckpt")
    ap.add_argument("--out", default="reports/feedback/G4")
    ap.add_argument("--assets", default="assets/g4/pie_seeds")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--rng", type=int, default=20261110)
    a = ap.parse_args()
    rng = np.random.default_rng(a.rng)
    out, adir = Path(a.out), Path(a.assets)
    out.mkdir(parents=True, exist_ok=True)
    D = Data(a.pre)
    sp = Path(a.splits)
    tests = {s: D.rows(load_split(sp / s / a.fold / "test.json")) for s in SETTINGS}
    g3 = pd.read_csv(a.g3_pairs, sep="\t")
    ck = {"official": ckpt_info(a.official_ckpt)}
    seeds = []
    for S in a.seeds:
        rd = Path(a.runs) / f"{a.fold}_seed{S}"
        if all((rd / "eval" / s / "predictions_best_auprc.parquet").is_file() for s in SETTINGS):
            seeds.append(S)
            ck[f"seed{S}"] = ckpt_info(rd / "best_auprc.ckpt")
            if (rd / "last.ckpt").is_file():
                ck[f"seed{S}"]["last_step"] = ckpt_info(rd / "last.ckpt")["global_step"]
        else:
            print(f"[g4p] seed {S}: eval incomplete in {rd}, skipped")
    print("[g4p] seeds scored:", seeds, json.dumps(ck))
    summary, spread, boots = [], [], []
    for s, rows in tests.items():
        pp_all = {}
        keys = [D.keys[r] for r in rows]
        for S in seeds:
            tbl = pd.read_parquet(Path(a.runs) / f"{a.fold}_seed{S}" / "eval" / s / "predictions_best_auprc.parquet")
            key = {(c, p): i for i, (c, p) in enumerate(zip(tbl["context"], tbl["perturbation"]))}
            oi = [key[k] for k in keys]
            pred = tuple(np.stack(tbl[c].to_numpy()[oi]).astype(np.float32) for c in ("p_de", "lfc_pred", "delta_p_pred"))
            res = score([scoring_inputs(D, rows, pred)])
            pp = res.per_pair.drop(columns=["dataset"])
            ex = extra_metrics(D, rows, pred).assign(context=[k[0] for k in keys], perturbation=[k[1] for k in keys])
            pp = pp.merge(ex, on=["context", "perturbation"])
            name = f"PIE_seed{S}"
            pp_all[name] = pp
            row = dict(setting=s, predictor=name, n_rows=len(rows), **res.context_mean)
            row.update(context_mean_of_pairs(pp, EXTRA_KEYS).to_dict())
            summary.append(row)
            # check against pie eval's own metrics csv
            mt = pd.read_csv(Path(a.runs) / f"{a.fold}_seed{S}" / "eval" / s / "metrics_best_auprc.csv")
            allrow = mt[mt["context"] == "all"].iloc[0]
            row["pie_eval_auprc"] = float(allrow["binary_auprc"])
            if s != "unseen_pert":
                np.savez_compressed(adir / f"pred_{a.fold}_{s}_seed{S}.npz", perts=np.array([k[1] for k in keys]),
                                    genes=np.array(D.genes), delta=pred[2], lfc=pred[1], pde=pred[0])
        for name in ["PIE_repro", BEST_BASE[s], RIDGE[s], "PIE_repro_kd"]:
            sub = g3[(g3["setting"] == s) & (g3["predictor"] == name)]
            if len(sub) == 0:
                continue
            pp_all[name] = sub
            row = dict(setting=s, predictor="PIE_official" if name == "PIE_repro" else name, n_rows=len(sub))
            row.update(context_mean_of_pairs(sub, [k for k in OFFICIAL + EXTRA_KEYS if k in sub.columns]).to_dict())
            summary.append(row)
        # spread over seeds
        sm = pd.DataFrame([r for r in summary if r["setting"] == s and r["predictor"].startswith("PIE_seed")])
        if len(sm):
            for k in OFFICIAL + EXTRA_KEYS:
                if k not in sm:
                    continue
                v = sm[k].astype(float)
                off = [r for r in summary if r["setting"] == s and r["predictor"] == "PIE_official"]
                base = [r for r in summary if r["setting"] == s and r["predictor"] == BEST_BASE[s]]
                spread.append(dict(setting=s, metric=k, n_seeds=len(v), mean=v.mean(), min=v.min(), max=v.max(),
                                   range=v.max() - v.min(), sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                                   official=off[0].get(k) if off else np.nan,
                                   best_baseline=BEST_BASE[s], baseline=base[0].get(k) if base else np.nan))
        # paired bootstrap: each seed vs official ckpt and vs best baseline (official + extra metrics)
        for S in seeds:
            a_pp = pp_all[f"PIE_seed{S}"]
            for ref in ["PIE_repro", BEST_BASE[s], RIDGE[s]]:
                if ref not in pp_all:
                    continue
                for ks in (BOOT_KEYS, EXTRA_KEYS):
                    kk = [k for k in ks if k in pp_all[ref].columns]
                    for b in paired_boot(a_pp[["context", "perturbation", *kk]], pp_all[ref][["context", "perturbation", *kk]],
                                         kk, a.nboot, rng):
                        boots.append(dict(setting=s, a=f"PIE_seed{S}", b="PIE_official" if ref == "PIE_repro" else ref, **b))
    pd.DataFrame(summary).to_csv(out / f"pie_seed_summary_{a.fold}.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(spread).to_csv(out / f"pie_seed_spread_{a.fold}.tsv", sep="\t", index=False, float_format="%.5f")
    pd.DataFrame(boots).to_csv(out / f"pie_seed_paired_bootstrap_{a.fold}.tsv", sep="\t", index=False, float_format="%.5f")
    (out / f"pie_seed_ckpt_{a.fold}.json").write_text(json.dumps(ck, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(spread).to_string())


if __name__ == "__main__":
    main()
