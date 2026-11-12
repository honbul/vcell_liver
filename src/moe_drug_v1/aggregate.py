"""Collect run records, pick one setting per model on validation, and apply the promotion gate.

Selection uses the overall validation macro-ΔRMSE of seed 42 and nothing else. SNU-423 appears in
the tables but is never allowed to choose a setting (plan s7.1, s8.2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs_dir.glob("*/record.json")):
        r = json.loads(p.read_text())
        v = r["val_metrics"]
        rows.append(dict(
            run=p.parent.name, kind=r["kind"], lr=r["lr"], wd=r["wd"], seed=r["seed"],
            val_macro_delta_rmse=r["val_macro_delta_rmse"],
            val_drug_centred_r=v["drug_centred_r"], val_top20_direction=v["top20_direction"],
            val_top20_recall=v["top20_recall"],
            snu423_val_delta_rmse=v.get("snu423_delta_rmse"),
            snu423_val_drug_centred_r=v.get("snu423_drug_centred_r"),
            params_total=r["params"]["total"], params_response=r["params"]["response"],
            params_router=r["params"]["router"], dense_hidden=r.get("dense_hidden"),
            epochs_run=r["epochs_run"], best_epoch=r["best_epoch"],
            train_seconds=r["train_seconds"], infer_ms_per_256=r["infer_ms_per_256"],
            peak_alloc_MB=r.get("peak_alloc_MB"), peak_reserved_MB=r.get("peak_reserved_MB"),
            gpu=r.get("gpu"), budget_limited=r["budget_limited"],
            router_entropy=(r.get("router_usage_val") or {}).get("entropy_mean"),
            router_max_share=(r.get("router_usage_val") or {}).get("max_weight_share"),
            tag=r.get("tag", "")))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--selection-seed", type=int, default=42)
    ap.add_argument("--gate-pct", type=float, default=2.0)
    ap.add_argument("--snu-tolerance-pct", type=float, default=2.0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.runs_dir))
    if df.empty:
        raise SystemExit("no runs found")
    df = df.sort_values(["kind", "seed", "val_macro_delta_rmse"])
    df.to_csv(out / "trials.tsv", sep="\t", index=False, float_format="%.6g")

    cost_cols = ["run", "kind", "lr", "wd", "seed", "params_total", "params_response", "params_router",
                 "dense_hidden", "epochs_run", "train_seconds", "infer_ms_per_256",
                 "peak_alloc_MB", "peak_reserved_MB", "gpu"]
    df[cost_cols].to_csv(out / "resources.tsv", sep="\t", index=False, float_format="%.6g")

    rows = []
    for p in sorted(Path(args.runs_dir).glob("*/record.json")):
        r = json.loads(p.read_text())
        u = r.get("router_usage_val")
        if not u:
            continue
        rows.append(dict(run=p.parent.name, kind=r["kind"], lr=r["lr"], wd=r["wd"], seed=r["seed"],
                         entropy_mean=u["entropy_mean"], entropy_max=u["entropy_max"],
                         max_weight_share=u["max_weight_share"],
                         **{f"mean_weight_e{i}": w for i, w in enumerate(u["mean_weight"])},
                         **{f"argmax_count_e{i}": c for i, c in enumerate(u["argmax_fraction"])}))
    if rows:
        pd.DataFrame(rows).to_csv(out / "routing_usage.tsv", sep="\t", index=False, float_format="%.6g")

    base = json.loads(Path(args.baselines).read_text()) if Path(args.baselines).exists() else {}

    sel = {}
    grid = df[df.seed == args.selection_seed]
    for k, g in grid.groupby("kind"):
        b = g.loc[g.val_macro_delta_rmse.idxmin()]
        sel[k] = dict(lr=float(b.lr), wd=float(b.wd), run=b.run,
                      val_macro_delta_rmse=float(b.val_macro_delta_rmse))

    seedrows = []
    for k, s in sel.items():
        g = df[(df.kind == k) & (df.lr == s["lr"]) & (df.wd == s["wd"])]
        for _, r in g.iterrows():
            seedrows.append(dict(kind=k, seed=int(r.seed), lr=r.lr, wd=r.wd,
                                 val_macro_delta_rmse=r.val_macro_delta_rmse,
                                 snu423_val_delta_rmse=r.snu423_val_delta_rmse,
                                 val_drug_centred_r=r.val_drug_centred_r, run=r.run))
    seeds = pd.DataFrame(seedrows).sort_values(["kind", "seed"])
    seeds.to_csv(out / "seed_results.tsv", sep="\t", index=False, float_format="%.6g")

    gate = {}
    if {"A", "C"} <= set(sel):
        a = seeds[seeds.kind == "A"].set_index("seed")
        c = seeds[seeds.kind == "C"].set_index("seed")
        common = sorted(set(a.index) & set(c.index))
        per_seed = [dict(seed=int(s),
                         A=float(a.val_macro_delta_rmse[s]), C=float(c.val_macro_delta_rmse[s]),
                         rel_pct=float((a.val_macro_delta_rmse[s] - c.val_macro_delta_rmse[s])
                                       / a.val_macro_delta_rmse[s] * 100),
                         snu_rel_pct=(float((a.snu423_val_delta_rmse[s] - c.snu423_val_delta_rmse[s])
                                            / a.snu423_val_delta_rmse[s] * 100)
                                      if pd.notna(a.snu423_val_delta_rmse[s]) else None))
                    for s in common]
        mean_rel = float(np.mean([p["rel_pct"] for p in per_seed])) if per_seed else float("nan")
        mean_snu = float(np.mean([p["snu_rel_pct"] for p in per_seed if p["snu_rel_pct"] is not None])) \
            if per_seed else float("nan")
        cm = float(np.mean([p["C"] for p in per_seed])) if per_seed else float("nan")
        beats_base = {b: (cm < base[b]["macro_delta_rmse"]) for b in base if "macro_delta_rmse" in base[b]}
        gate = dict(
            selection_seed=args.selection_seed, selected=sel, per_seed=per_seed,
            n_seeds=len(per_seed), mean_relative_improvement_pct=mean_rel,
            all_seeds_same_direction=bool(per_seed) and all(p["rel_pct"] > 0 for p in per_seed),
            snu423_mean_relative_change_pct=mean_snu,
            snu423_within_tolerance=bool(mean_snu >= -args.snu_tolerance_pct)
            if np.isfinite(mean_snu) else None,
            C_mean_val_macro=cm, baselines={b: base[b]["macro_delta_rmse"] for b in beats_base},
            beats_baselines=beats_base,
            threshold_pct=args.gate_pct)
        gate["train_B"] = bool(
            gate["n_seeds"] >= 3 and mean_rel >= args.gate_pct and gate["all_seeds_same_direction"]
            and all(beats_base.values()) and gate["snu423_within_tolerance"])
    (out / "selection.json").write_text(json.dumps(gate, indent=1))
    print(df.to_string(index=False), flush=True)
    print("\nselected:", json.dumps(sel, indent=1), flush=True)
    print("\ngate:", json.dumps({k: v for k, v in gate.items() if k != "selected"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
