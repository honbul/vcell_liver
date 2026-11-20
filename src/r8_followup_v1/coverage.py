"""Label-free input coverage of source (out-of-fold episodes) and the HCC targets (plan section 6.4).

Computed from quantities available before any query is measured: n0 of supports / queries, chemical similarity (max Tanimoto of
a query to the supports), source-response similarity (max cosine of the f* representation), global control-state distance,
within-control spread of the supports, support-query plate overlap, unique drugs, rows and plates per support unit, treated wells,
controls. HCC: target metadata (target_rows_meta.tsv) and DMSO control profiles (x0) only; treated HCC profiles are NOT read.
No gate is fitted here; the table only puts the two input ranges side by side.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.data import Features, Episode, overlap
from src.replicate_calibrated_r8.build_store import row_weights
from src.replicate_calibrated_r8 import methods as M
from src.episodic_residual_r2.build_units import rows_to_units
from .common import orig_man, dump, new_run_dir, run_record, pub_man, sha256
from .partitions import Bank
from . import engine as Eng
from . import fmethods as F

torch.set_grad_enabled(False)


def cov_row(inp, fmap, FP, kappa_f, plates_s, plates_q):
    K = len(inp["y_s"]) if inp.get("y_s") is not None else len(inp["g_s"])
    fs, fq = F.fstar(inp, kappa_f)
    ft = fmap.build(inp, FP, fs, fq)
    tq = M.tanimoto(ft["fp_q"], ft["fp_s"])
    a, b = ft["f0r_q"], ft["f0r_s"]
    cs = (a @ b.T) / (a.norm(dim=1, keepdim=True) * b.norm(dim=1)[None, :]).clamp(min=1e-12)
    sp = set()
    for p in plates_s:
        sp |= set(p.split("|"))
    shared = np.mean([bool(set(p.split("|")) & sp) for p in plates_q])
    return dict(n0_query_mean=float(inp["n0_q"].mean()), n0_query_zero_frac=float((inp["n0_q"] == 0).float().mean()), n0_support_mean=float(inp["n0_s"].mean()),
                max_tanimoto_q_to_s=float(tq.max(1).values.mean()), max_cos_f0repr_q_to_s=float(cs.max(1).values.mean()),
                fstar_rms_query=float(fq.pow(2).mean().sqrt()), global_state_norm=float(ft["glob"].norm()), within_ctrl_rms_support=float(ft["ctrl_s"].pow(2).mean().sqrt()),
                within_ctrl_rms_query=float(ft["ctrl_q"].pow(2).mean().sqrt()), query_shares_support_plate_frac=float(shared),
                support_overlap_offdiag_mean=float((inp["o_s"].sum() - torch.diagonal(inp["o_s"]).sum()) / (K * (K - 1))), support_q_mean=float(inp["q_s"].mean()),
                n_support_plates=len(sp), n_query=int(len(inp["g_q"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--constants", nargs=3, required=True, help="inner0 inner1 inner2 constants (kappa_f of each fold)")
    ap.add_argument("--final-constants", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir("coverage", a.run_id)
    bank = Bank()
    rows = []
    for k in range(3):
        c = json.load(open(a.constants[k]))
        ctx = Eng.Ctx(bank, f"outer{k}")
        kf = c["kappa_f"]["16"]
        units = bank.units
        for ep in ctx.episodes("val", [1, 2, 3], None):
            inp = Eng.inputs(ep, 16)
            r = cov_row(inp, ctx.fmap, ctx.st.FP, kf, ep.plates_s[:16], ep.plates_q)
            su = ep.su[:16]
            r.update(side="source_oof", fold=k, target=ep.name, family=ep.family, split_seed=ep.seed, n_unique_support_drugs=int(len(set(ep.g_s[:16]))),
                     rows_per_support_unit=float(units.n_rows.to_numpy()[su].mean()), plates_per_support_unit=float(units.n_plates.to_numpy()[su].mean()),
                     treated_wells_support=int(units.n_trt_wells.to_numpy()[su].sum()))
            rows.append(r)
        print("fold", k, f"{time.time() - t0:.0f}s", flush=True)
    # HCC: global partition (every kept source family behind f0, as in the original HCC evaluation), metadata + control profiles only
    cfin = json.load(open(a.final_constants))
    ctx = Eng.Ctx(bank, "global")
    fz = json.load(open(orig_man("freeze_manifest.json")))
    meta = pd.read_csv(os.path.join(os.environ["R8_PROC"], "target_rows_meta.tsv"), sep="\t")
    x0all = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    st = ctx.st
    gmap = st.groups.set_index("drug_group").group_idx
    for t in ("HUH7", "JHH5", "JHH7"):
        r_ = meta[meta.cell == t].reset_index(drop=True)
        x0 = np.asarray(x0all[r_.row_id.to_numpy()])
        tab, _, ux = rows_to_units(r_, np.zeros_like(x0), x0)
        pos = {g: i for i, g in enumerate(tab.drug_group)}
        w = row_weights(r_)
        by = {}
        for dg, pl, wi, nw in zip(r_.drug_group, r_.det_plate, w, r_.n_trt_wells):
            e = by.setdefault(dg, [[], [], 0])
            e[0].append(pl), e[1].append(wi)
            e[2] += int(nw)
        ctl = r_.groupby("det_plate").n_ctl.first()
        for split, lists in fz["targets"][t].items():
            sup, qry = list(lists["support_order"])[:16], list(lists["query_test"])
            gi = lambda L: np.array([gmap[g] for g in L])
            f0s, n0s = st.f0("hcc", gi(sup), allow_missing=True)
            f0q, n0q = st.f0("hcc", gi(qry), allow_missing=True)
            o, q = overlap([by[g][0] for g in sup], [by[g][1] for g in sup])
            tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
            inp = dict(name=t, g_s=gi(sup), g_q=gi(qry), x0_s=tt(ux[[pos[g] for g in sup]]), x0_q=tt(ux[[pos[g] for g in qry]]), f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q,
                       y_s=None, o_s=tt(o), q_s=tt(q))
            ps = ["|".join(sorted(set(by[g][0]))) for g in sup]
            pq = ["|".join(sorted(set(by[g][0]))) for g in qry]
            r = cov_row(inp, ctx.fmap, st.FP, cfin["kappa_f"]["16"], ps, pq)
            sp = sorted(set(p for g in sup for p in by[g][0]))
            r.update(side="hcc", fold=-1, target=t, family=t, split_seed=int(split), n_unique_support_drugs=len(set(sup)),
                     rows_per_support_unit=float(np.mean([len(by[g][0]) for g in sup])), plates_per_support_unit=float(np.mean([len(set(by[g][0])) for g in sup])),
                     treated_wells_support=int(sum(by[g][2] for g in sup)), control_wells_support_plates=int(ctl[sp].sum()))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "coverage_rows.tsv"), sep="\t", index=False, float_format="%.6g")
    num = [c for c in df.columns if c not in ("side", "fold", "target", "family", "split_seed") and df[c].dtype.kind in "fi"]
    src = df[df.side == "source_oof"].groupby(["family", "target"])[num].mean().groupby(level=0).mean()
    summ = []
    for c in num:
        s = src[c].dropna()
        summ.append(dict(quantity=c, source_family_mean=float(s.mean()), source_q05=float(s.quantile(0.05)), source_q50=float(s.median()), source_q95=float(s.quantile(0.95)),
                         **{f"hcc_{t}": float(df[(df.side == "hcc") & (df.target == t)][c].mean()) for t in ("HUH7", "JHH5", "JHH7")},
                         **{f"hcc_{t}_source_percentile": float((s < df[(df.side == "hcc") & (df.target == t)][c].mean()).mean() * 100) for t in ("HUH7", "JHH5", "JHH7")}))
    summ = pd.DataFrame(summ)
    summ["note"] = "source: out-of-fold outer-test episodes (K=16, seeds 1-3, family-equal); HCC: frozen split lists (K=16, test queries), metadata and control profiles only"
    summ.to_csv(pub_man("source_target_input_coverage.tsv"), sep="\t", index=False, float_format="%.6g")
    dump(run_record(a.run_id, "coverage", t0, n_rows=len(df), hcc_treated_profiles_read=False), os.path.join(out, "result.json"))
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    main()
