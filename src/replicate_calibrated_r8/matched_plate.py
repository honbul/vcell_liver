"""Matched-plate transfer (plan section 10) on source cells only. No HCC value.

For a cell with two detection plates A and B that come from different RNA plates and share enough conditions:
  query   = conditions measured on plate A (truth = their plate-A delta), identical in both arms
  support = the SAME K drug groups, taken either from plate A (in-block) or from plate B (other block)
If a method only helps in-block, what it learns is the plate environment of the support, not the cell.

sets: confirmation      confirmation-family cells, every method (learned models never saw these families)
      train_closed_form source-train cells, closed-form methods only (leave-family-out f0); learned models are not scored here
"""
import argparse, json, os, time
from collections import Counter
import numpy as np
import pandas as pd
import torch

from .common import public, dump, new_run_dir, run_record, proc
from .data import Store, Episode
from .diag import row_delta
from .suite import Suite, current_selection
from .contrasts import family_contrasts, CONTRASTS
from src.episodic_residual_r2.metrics import per_unit

PUB = lambda n: public("manifests", "replicate_calibrated_r8", n)
MIN_SHARED, N_QUERY_MIN, MAX_PAIRS, SEEDS, KS = 36, 20, 6, (1, 2, 3), (8, 16, 32)


def plate_pairs(g):
    """Detection-plate pairs of one cell from different RNA plates, by the number of drug groups measured on both (one condition per group)."""
    by = {}
    for pi, dg, pid, rid, rna in zip(g.plate_idx, g.drug_group, g.pert_id, g.pos, g.rna_plates):
        by.setdefault(pi, {}).setdefault(dg, (pid, rid))
        cur = by[pi][dg]
        if pid < cur[0]:
            by[pi][dg] = (pid, rid)
    rna = dict(zip(g.plate_idx, g.rna_plates))
    ps = sorted(by)
    out = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = ps[i], ps[j]
            if rna[a] == rna[b]:
                continue
            common = sorted(dg for dg in by[a] if dg in by[b] and by[a][dg][0] == by[b][dg][0])
            if len(common) >= MIN_SHARED:
                out.append((len(common), a, b, common))
    out.sort(key=lambda x: (-x[0], x[1], x[2]))
    return out[:MAX_PAIRS], by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir("matched_plate", a.run_id)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = Store(dev)
    rows = pd.read_csv(proc("rows.tsv"), sep="\t")
    rows = rows[rows.role == "source"].reset_index(drop=True)
    rows["pos"] = np.arange(len(rows))
    plates = pd.read_csv(proc("plates.tsv"), sep="\t").set_index("plate_idx")
    RD = row_delta(rows)
    X0 = np.load(os.path.join(os.environ["TB_PROC"], "x0_all.npy"), mmap_mode="r")
    gidx = dict(zip(st.groups.drug_group, st.groups.group_idx))
    suite = Suite(current_selection(with_tsg=False), st, dev, with_tsg=False, shuffle_modes=())
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    recs, elig = [], []
    for ci in range(len(st.cells)):
        role = st.cell_role[ci]
        if role not in ("train", "val"):
            continue
        sset = "confirmation" if role == "val" else "train_closed_form"
        cell, fam = st.cell_name[ci], st.cell_family[ci]
        g = rows[rows.cell == cell]
        pairs, by = plate_pairs(g)
        if sset == "train_closed_form":
            pairs = pairs[:2]
        regime = st.regime_of(ci)
        nb = st.n_behind(regime, fam).cpu().numpy()
        n_used = 0
        for n_common, pa, pb, common in pairs:
            common = [dg for dg in common if nb[gidx[dg]] >= 1]
            if len(common) < MIN_SHARED:
                continue
            n_used += 1
            kind = "x_tag_same_compound_plate" if plates.pert_plate[pa] == plates.pert_plate[pb] else "different_compound_plate"
            for A, Bp in ((pa, pb), (pb, pa)):
                gi = np.array([gidx[dg] for dg in common])
                ra_, rb_ = np.array([by[A][dg][1] for dg in common]), np.array([by[Bp][dg][1] for dg in common])
                rid_a, rid_b = rows.row_id.to_numpy()[ra_], rows.row_id.to_numpy()[rb_]
                dA, dB = RD[ra_], RD[rb_]
                xA, xB = np.asarray(X0[np.sort(rid_a)])[np.argsort(np.argsort(rid_a))], np.asarray(X0[np.sort(rid_b)])[np.argsort(np.argsort(rid_b))]
                for seed in SEEDS if sset == "confirmation" else SEEDS[:1]:
                    p = np.random.RandomState(770000 + 1000 * seed + A * 7 + Bp).permutation(len(common))
                    kmax = max(k for k in KS if len(common) - k >= N_QUERY_MIN)
                    s, q = p[:kmax], p[kmax:][:200]
                    f0s, n0s = st.f0(regime, gi[s], fam)
                    f0q, n0q = st.f0(regime, gi[q], fam)
                    one = torch.ones(kmax, kmax, device=dev)
                    for block, (ys, xs, sp) in (("in_block", (dA[s], xA[s], A)), ("other_block", (dB[s], xB[s], Bp))):
                        ep = Episode(name=cell, family=fam, seed=seed, g_s=gi[s], g_q=gi[q], x0_s=t(xs), x0_q=t(xA[q]), f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q,
                                     y_s=t(ys), y_q=t(dA[q]), o_s=one, q_s=torch.ones(kmax, device=dev),
                                     plates_s=[plates.det_plate[sp]] * kmax, plates_q=[plates.det_plate[A]] * len(q))
                        truth = ep.y_q.cpu().numpy()
                        for K in KS:
                            if K > kmax:
                                continue
                            inp = ep.inputs(K)
                            with torch.no_grad():
                                it = suite.run(ep, K, tsg=False) if sset == "confirmation" else ((n, -1, p_) for n, p_ in suite.closed_form(inp))
                                for method, ms, pred in it:
                                    pu = per_unit(pred.cpu().numpy(), truth)
                                    recs.append(dict(set=sset, cell=cell, family=fam, plate_query=plates.det_plate[A], plate_other=plates.det_plate[Bp], pair_kind=kind,
                                                     support_seed=seed, block=block, K=K, method=method, model_seed=ms, n_query=len(q), n_shared=len(common),
                                                     delta_rmse=float(pu.delta_rmse.mean()), centred_rmse=float(pu.centred_rmse.mean()),
                                                     centred_pearson=float(pu.centred_pearson.mean())))
        elig.append(dict(set=sset, cell=cell, family=fam, n_plate_pairs_used=n_used))
        if n_used:
            print(cell, sset, n_used, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(out, "matched_plate_episodes.tsv.gz"), sep="\t", index=False, float_format="%.6g")
    el = pd.DataFrame(elig)
    # model seeds -> episode; episode -> cell -> family (equal weights)
    ep = df.groupby(["set", "cell", "family", "plate_query", "plate_other", "support_seed", "block", "K", "method"])[["delta_rmse", "centred_rmse", "centred_pearson"]].mean().reset_index()
    out_rows = []
    for v in ("delta_rmse", "centred_rmse"):
        c = ep.groupby(["set", "K", "block", "method", "family", "cell"])[v].mean().reset_index()
        f = c.groupby(["set", "K", "block", "method", "family"])[v].mean().reset_index()
        for (sset, K, method), h in f.groupby(["set", "K", "method"]):
            p = h.pivot(index="family", columns="block", values=v).dropna()
            d = (p["in_block"] - p["other_block"]).to_numpy()
            rs = np.random.RandomState(20261117)
            bs = d[rs.randint(len(d), size=(2000, len(d)))].mean(1)
            out_rows.append(dict(set=sset, K=K, row="method_level", name=method, metric=v, in_block=float(p["in_block"].mean()), other_block=float(p["other_block"].mean()),
                                 in_minus_other=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_families=len(d)))
        # method contrasts inside each block, and their in-block minus other-block difference
        for (sset, block), h in f.groupby(["set", "block"]):
            fc = family_contrasts(h, v)
            for r in fc.itertuples():
                out_rows.append(dict(set=sset, K=r.K, row=f"contrast_{block}", name=r.contrast, metric=v, in_block=r.diff if block == "in_block" else np.nan,
                                     other_block=r.diff if block == "other_block" else np.nan, in_minus_other=np.nan, lo=r.lo, hi=r.hi, n_families=r.n_families))
        for (sset, K), h in f.groupby(["set", "K"]):
            pi = h[h.block == "in_block"].pivot(index="family", columns="method", values=v)
            po = h[h.block == "other_block"].pivot(index="family", columns="method", values=v)
            for name, comb in CONTRASTS.items():
                if not all(m in pi and m in po for m in comb):
                    continue
                d = (sum(c_ * pi[m] for m, c_ in comb.items()) - sum(c_ * po[m] for m, c_ in comb.items())).dropna().to_numpy()
                rs = np.random.RandomState(20261117)
                bs = d[rs.randint(len(d), size=(2000, len(d)))].mean(1)
                out_rows.append(dict(set=sset, K=K, row="contrast_in_minus_other", name=name, metric=v, in_block=np.nan, other_block=np.nan, in_minus_other=float(d.mean()),
                                     lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_families=len(d)))
    tab = pd.DataFrame(out_rows)
    tab.to_csv(PUB("matched_plate_transfer.tsv"), sep="\t", index=False, float_format="%.6g")
    counts = el.groupby("set").agg(cells=("cell", "size"), eligible_cells=("n_plate_pairs_used", lambda x: int((x > 0).sum())), plate_pairs=("n_plate_pairs_used", "sum")).reset_index()
    counts["eligible_families"] = [int(el[(el.set == s) & (el.n_plate_pairs_used > 0)].family.nunique()) for s in counts.set]
    kinds = df.drop_duplicates(["set", "cell", "plate_query", "plate_other"]).groupby(["set", "pair_kind"]).size().rename("ordered_plate_pairs").reset_index()
    counts.to_csv(PUB("matched_plate_eligible.tsv"), sep="\t", index=False)
    kinds.to_csv(PUB("matched_plate_pair_kinds.tsv"), sep="\t", index=False)
    ep.to_csv(PUB("matched_plate_episodes.tsv.gz"), sep="\t", index=False, float_format="%.5g")
    dump(run_record(a.run_id, "matched_plate", t0, n_rows=len(df), eligible=counts.to_dict("records")), os.path.join(out, "result.json"))
    pd.set_option("display.width", 250, "display.max_rows", 600)
    print(counts.to_string())
    print(kinds.to_string())
    print(tab[(tab.metric == "delta_rmse") & (tab.K == 16)].to_string())


if __name__ == "__main__":
    main()
