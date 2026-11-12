"""Open the test split once, for the frozen models only, with one code path for every model.

Nothing here is fitted. The ridge baseline reuses the regularisation strength validation already
chose; no calibrator, panel or threshold is learned from test truth. The plate diagnostic is a
fixed read-out, not a correction (plan s8.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import evaluate as ev
from .data import Dataset
from .models import build, solve_dense_hidden
from .train import predict, router_usage


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_model(run: Path, ds: Dataset, dev: str):
    ck = torch.load(run / "best.pt", map_location=dev, weights_only=False)
    rec = ck["record"]
    hidden = rec.get("dense_hidden")
    if rec["kind"] == "A" and hidden is None:
        hidden = solve_dense_hidden(ds.n_genes, ds.n_bits)[0]
    m = build(rec["kind"], ds.n_genes, ds.n_bits, dense_hidden=hidden).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, rec


def plate_diagnostic(pred, truth, line, drug, plate_of, line_names, min_drugs: int = 5):
    """Within a (cell line, plate) block of >=5 evaluated drugs, remove that block's own mean from
    prediction and truth and correlate what is left. It bounds how much of the agreement survives
    removing a shared per-plate offset; it does not prove the plate effect is gone."""
    rows = []
    for l in np.unique(line):
        for p in np.unique(plate_of[l]):
            if p < 0:
                continue
            m = (line == l) & (plate_of[l][drug] == p)
            if m.sum() < min_drugs:
                continue
            pc = pred[m] - pred[m].mean(0)
            tc = truth[m] - truth[m].mean(0)
            pc = pc - pc.mean(1, keepdims=True)
            tc = tc - tc.mean(1, keepdims=True)
            num = (pc * tc).sum(1)
            den = np.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1))
            r = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
            rows.append(dict(line_index=int(l), cell_line=str(line_names[l]), plate_index=int(p),
                             n_drugs=int(m.sum()), mean_r=float(np.nanmean(r))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--selection", required=True, help="selection.json from aggregate.py")
    ap.add_argument("--baselines-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.data_dir, args.manifest_dir, device=dev)
    te = ds.conditions("test")
    ty, tl, td = te.y.cpu().numpy(), te.line.cpu().numpy(), te.drug.cpu().numpy()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(args.selection).read_text())["selected"]
    seeds = [int(s) for s in args.seeds.split(",")]

    preds, records, freeze = {}, {}, {}
    for kind, s in sel.items():
        per_seed = []
        for sd in seeds:
            # the selected run's own directory name, with only the seed suffix swapped, so the
            # literal lr/wd spelling used at submission time is preserved exactly
            run = Path(args.runs_dir) / re.sub(r"_s\d+$", f"_s{sd}", s["run"])
            if not (run / "best.pt").exists():
                continue
            m, rec = load_model(run, ds, dev)
            p = predict(m, ds, te)
            per_seed.append((sd, p, rec, run))
            freeze[run.name] = dict(sha256_checkpoint=sha256_file(run / "best.pt"),
                                    lr=rec["lr"], wd=rec["wd"], seed=rec["seed"],
                                    best_epoch=rec["best_epoch"], params=rec["params"],
                                    router_usage_test=router_usage(m, ds, te))
        if not per_seed:
            continue
        preds[kind] = per_seed
        records[kind] = [r for _, _, r, _ in per_seed]

    # per-model, per-seed metrics then the seed mean of the primary number
    rows, seed_rmse = [], {}
    for kind, per_seed in preds.items():
        rr = []
        for sd, p, rec, run in per_seed:
            mt, rmse = ev.all_metrics(p, ty, tl, ds.line_names, ds.snu423)
            rr.append(rmse)
            rows.append(dict(model=kind, seed=sd, **{k: v for k, v in mt.items()
                                                     if not isinstance(v, (list, dict))}))
        seed_rmse[kind] = np.mean(rr, 0)          # per-condition RMSE averaged over seeds
    per_seed_df = pd.DataFrame(rows)
    per_seed_df.to_csv(out / "test_per_seed.tsv", sep="\t", index=False, float_format="%.6g")

    # baselines on test, with validation's own lambda; computed by the same metric code
    base = {}
    bdir = Path(args.baselines_dir)
    bjson = bdir / "baselines_test.json"
    if bjson.exists():
        bm = json.loads(bjson.read_text())
        for name in ["no_change", "line_mean", "ridge"]:
            f = bdir / f"pred_{name}_test.npy"
            if f.exists():
                bp = np.load(f)
                base[name] = dict(metrics=bm[name], rmse=ev.per_condition_rmse(bp, ty))

    # seed-mean model predictions for the secondary read-outs and the plate diagnostic
    mean_pred = {k: np.mean([p for _, p, _, _ in v], 0) for k, v in preds.items()}
    summary = {}
    for k, p in mean_pred.items():
        mt, _ = ev.all_metrics(p, ty, tl, ds.line_names, ds.snu423)
        summary[k] = mt
        summary[k]["macro_delta_rmse_seed_mean_of_per_condition_rmse"] = float(
            ev.macro_from_rmse(seed_rmse[k], tl)[0])
    for name, b in base.items():
        summary[name] = b["metrics"]

    # paired bootstrap over drug groups: A vs C, and each against the baselines
    boots = {}
    if {"A", "C"} <= set(seed_rmse):
        boots["A_vs_C"] = ev.paired_bootstrap(seed_rmse["A"], seed_rmse["C"], td, args.n_boot, 0, tl)
        if "B" in seed_rmse:
            boots["B_vs_C"] = ev.paired_bootstrap(seed_rmse["B"], seed_rmse["C"], td, args.n_boot, 0, tl)
            boots["A_vs_B"] = ev.paired_bootstrap(seed_rmse["A"], seed_rmse["B"], td, args.n_boot, 0, tl)
        for name, b in base.items():
            for k in seed_rmse:
                boots[f"{name}_vs_{k}"] = ev.paired_bootstrap(b["rmse"], seed_rmse[k], td,
                                                              args.n_boot, 0, tl)
    # SNU-423, the pre-specified secondary line, same drugs same code
    if ds.snu423 is not None:
        m = tl == ds.snu423
        if m.any() and {"A", "C"} <= set(seed_rmse):
            boots["A_vs_C_snu423"] = ev.paired_bootstrap(seed_rmse["A"][m], seed_rmse["C"][m],
                                                         td[m], args.n_boot, 0, None)
            if "B" in seed_rmse:
                boots["B_vs_C_snu423"] = ev.paired_bootstrap(seed_rmse["B"][m], seed_rmse["C"][m],
                                                             td[m], args.n_boot, 0, None)

    plate_of = ds.plate_np
    diag = {k: plate_diagnostic(p, ty, tl, td, plate_of, ds.line_names) for k, p in mean_pred.items()}
    plate_rows = []
    for k, rs in diag.items():
        for r in rs:
            plate_rows.append(dict(model=k, **r))
    pd.DataFrame(plate_rows).to_csv(out / "plate_diagnostic.tsv", sep="\t", index=False,
                                    float_format="%.6g")

    per_line = {k: ev.line_macro_table(p, ty, tl, ds.line_names) for k, p in mean_pred.items()}
    rows = []
    for k, rs in per_line.items():
        for r in rs:
            rows.append(dict(model=k, **r))
    pd.DataFrame(rows).to_csv(out / "test_per_line.tsv", sep="\t", index=False, float_format="%.6g")

    # per-drug table for the appendix
    drows = []
    for k, p in mean_pred.items():
        rmse = ev.per_condition_rmse(p, ty)
        df = pd.DataFrame(dict(model=k, drug_index=td, line_index=tl, delta_rmse=rmse))
        drows.append(df.groupby(["model", "drug_index"]).delta_rmse.mean().reset_index())
    pd.concat(drows).to_csv(out / "per_drug.tsv", sep="\t", index=False, float_format="%.6g")

    res = dict(summary=summary, bootstrap=boots, freeze=freeze,
               n_test_conditions=int(len(tl)), n_test_drugs=int(len(np.unique(td))),
               n_lines=int(ds.n_lines), seeds=seeds,
               plate_diagnostic_blocks=int(len(plate_rows) / max(len(diag), 1)))
    (out / "test_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                      for k, v in summary.items()}, indent=1), flush=True)
    print("\nbootstrap:", json.dumps(boots, indent=1), flush=True)


if __name__ == "__main__":
    main()
