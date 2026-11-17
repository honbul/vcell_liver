"""D0 / D1 / D2: replicate structure, shared-control covariance and repeatable residual directions. Source values only.

D0  what a 'replicate' is in this release: RNA plate, detection plate, X tag, compound plate, bead batch; control sharing. HCC: metadata only.
D1  plate-pair decomposition of replicate differences of the SAME conditions measured on two plates of one cell:
        d_c = delta(c, p1) - delta(c, p2) = (s_p1 - s_p2) + (e_c,p1 - e_c,p2)
        v_e = mean_gene var_c(d) / 2                      independent per-row component
        v_s = (mean_gene dbar^2 - 2 v_e / n) / 2          component shared by every condition of a plate (control mean error + block effect)
    and the naive control-mean variance ctl_sd^2 / n_ctl of the same plates. No reference f0 is involved.
D2  repeatability of residuals against the same-drug mean with CROSS-FIT references: view A minus the mean over one half of the
    source families, view B minus the mean over the other, disjoint half. The shared-reference version is reported next to it.
"""
import argparse, json, os, time, zlib
from collections import Counter
import numpy as np
import pandas as pd
import torch

from .common import N_GENES, proc, public, dump, new_run_dir, run_record, env
from .data import Store

MIN_COMMON = 20
MAX_PAIRS_PER_CELL = 40


def fam_balanced(df, col, fam="family", cell="cell"):
    return float(df.groupby([fam, cell])[col].mean().groupby(level=0).mean().mean())


def d0(rows, plates, cells, trows):
    cond = rows.groupby(["cell", "pert_id"]).agg(n_rows=("row_id", "size"), n_rna=("rna_plates", "nunique"), n_pert_plate=("pert_plate", "nunique"),
                                                 n_tag=("rep_tag", "nunique"), n_bead=("bead_batch", "nunique")).reset_index()
    cond["kind"] = np.select([cond.n_rows == 1, cond.n_rna == 1, cond.n_pert_plate == 1], ["single_row", "same_rna_plate_only", "x_tag_replicates_one_compound_plate"],
                             "several_compound_plates")
    units = rows.groupby("unit_idx").agg(cell=("cell", "first"), n_rows=("row_id", "size"), n_plates=("plate_idx", "nunique"), n_rna=("rna_plates", "nunique"),
                                         n_pert_plate=("pert_plate", "nunique"), n_trt_wells=("n_trt_wells", "sum"), n_cond=("pert_id", "nunique")).reset_index()
    per_plate = rows.groupby("plate_idx").agg(n_rows=("row_id", "size"), n_units=("unit_idx", "nunique")).reset_index().merge(plates, on="plate_idx")
    rna_of_plate = rows.groupby("plate_idx").rna_plates.agg(lambda s: "|".join(sorted(set("|".join(s).split("|")))))
    per_plate["ctl_same_rna_plate_as_treated"] = (per_plate.plate_idx.map(rna_of_plate) == per_plate.ctl_rna_plates).astype(int)
    per_cell = units.groupby("cell").agg(n_units=("unit_idx", "size"), units_multi_plate=("n_plates", lambda s: int((s > 1).sum())),
                                         units_multi_rna_plate=("n_rna", lambda s: int((s > 1).sum())),
                                         units_multi_compound_plate=("n_pert_plate", lambda s: int((s > 1).sum()))).reset_index()
    per_cell = per_cell.merge(plates.groupby("cell").agg(n_det_plates=("plate_idx", "size"), n_compound_plates=("pert_plate", "nunique"),
                                                         median_n_ctl=("n_ctl", "median")).reset_index(), on="cell")
    per_cell = per_cell.merge(cells[["cell", "family", "meta_role"]], on="cell")
    q = lambda s: {k: float(v) for k, v in s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).items()}
    hcc = {}
    for t, g in trows.groupby("cell"):
        u = g.groupby("drug_group").agg(n_rows=("row_id", "size"), n_plates=("det_plate", "nunique"), n_rna=("rna_plates", "nunique"),
                                        n_pert_plate=("pert_plate", "nunique"), wells=("n_trt_wells", "sum"))
        hcc[t] = dict(units=int(len(u)), det_plates=sorted(g.det_plate.unique()), rna_plates=sorted(set("|".join(g.rna_plates).split("|"))),
                      compound_plates=sorted(g.pert_plate.unique()), rep_tags=sorted(g.rep_tag.unique()), bead_batches=sorted(g.bead_batch.astype(str).unique()),
                      units_by_n_det_plates={str(k): int(v) for k, v in u.n_plates.value_counts().sort_index().items()},
                      units_by_n_rna_plates={str(k): int(v) for k, v in u.n_rna.value_counts().sort_index().items()},
                      units_by_n_compound_plates={str(k): int(v) for k, v in u.n_pert_plate.value_counts().sort_index().items()},
                      treated_wells_total=int(g.n_trt_wells.sum()), control_wells_by_plate={p: int(v) for p, v in g.groupby("det_plate").n_ctl.first().items()},
                      independent_replicate_status="one detection plate: no replicate of any kind" if g.det_plate.nunique() == 1 else
                      ("X-tag replicate plates of one compound plate; biological independence of the cultures is not documented in the release"
                       if g.pert_plate.nunique() == 1 else "several compound plates; each unit's replicate status is listed in units_by_*"))
    summ = dict(source=dict(cells=int(cells.cell.nunique()), plate_rows=int(len(rows)), conditions=int(len(cond)), units=int(len(units)), det_plates=int(len(plates)),
                            conditions_by_kind={k: int(v) for k, v in cond.kind.value_counts().items()},
                            units_by_n_det_plates={str(k): int(v) for k, v in units.n_plates.clip(upper=6).value_counts().sort_index().items()},
                            units_with_two_or_more_rna_plates=int((units.n_rna > 1).sum()), units_with_two_or_more_compound_plates=int((units.n_pert_plate > 1).sum()),
                            cells_with_any_rna_plate_replicate=int((per_cell.units_multi_rna_plate > 0).sum()),
                            cells_with_40_or_more_rna_plate_replicate_units=int((per_cell.units_multi_rna_plate >= 40).sum()),
                            treated_wells_per_row={str(k): int(v) for k, v in rows.n_trt_wells.clip(upper=4).value_counts().sort_index().items()},
                            control_wells_per_plate_quantiles=q(plates.n_ctl), rows_sharing_one_control_pool_quantiles=q(per_plate.n_rows),
                            plates_whose_controls_are_on_the_treated_rna_plate=int(per_plate.ctl_same_rna_plate_as_treated.sum()),
                            control_positional_check=dict(note="between-half squared mean difference over its expectation under independent wells; 1 = no positional structure",
                                                          rows_half_median=float(plates.half_ratio_rows.median()), cols_half_median=float(plates.half_ratio_cols.median()),
                                                          random_half_median=float(plates.half_ratio_random.median()),
                                                          n_plates_rows=int(plates.half_ratio_rows.notna().sum()), n_plates_cols=int(plates.half_ratio_cols.notna().sum()))),
                hcc_metadata_only=hcc,
                not_verifiable_from_release=["whether X1/X2/X3 plates are separately cultured and treated biological replicates or splits of one culture",
                                             "culture passage, treatment date and operator of each plate", "composition of the XC mixed pools (excluded)"])
    return summ, per_cell, cond


def row_delta(rows):
    tb = env("TB_PROC")
    rid = rows.row_id.to_numpy()
    x1 = np.load(os.path.join(tb, "x1_source.npy"), mmap_mode="r")
    x0 = np.load(os.path.join(tb, "x0_all.npy"), mmap_mode="r")
    out = np.empty((len(rid), N_GENES), np.float32)
    for i in range(0, len(rid), 20000):
        j = rid[i:i + 20000]
        out[i:i + 20000] = np.asarray(x1[j]) - np.asarray(x0[j])
    return out


def d1(rows, plates, cells, RD):
    pl = plates.set_index("plate_idx")
    fam = cells.set_index("cell").family
    role = cells.set_index("cell").meta_role
    out = []
    rs = np.random.RandomState(20261117)
    for cell, g in rows.groupby("cell", sort=True):
        pos = g.index.to_numpy()                                  # positions in RD
        by_cond = {}
        for p, pid, pi in zip(pos, g.pert_id.to_numpy(), g.plate_idx.to_numpy()):
            by_cond.setdefault(pid, []).append((pi, p))
        cnt = Counter()
        for lst in by_cond.values():
            ps = sorted(x[0] for x in lst)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    cnt[(ps[i], ps[j])] += 1
        pairs = sorted(k for k, v in cnt.items() if v >= MIN_COMMON)
        if len(pairs) > MAX_PAIRS_PER_CELL:
            pairs = [pairs[i] for i in sorted(rs.choice(len(pairs), MAX_PAIRS_PER_CELL, replace=False))]
        rowpos = {(pi, pid): p for pid, lst in by_cond.items() for pi, p in lst}
        rna = dict(zip(g.plate_idx, g.rna_plates))
        for p1, p2 in pairs:
            common = sorted(pid for pid, lst in by_cond.items() if {p1, p2} <= {x[0] for x in lst})
            a = RD[[rowpos[(p1, c)] for c in common]].astype(np.float64)
            b = RD[[rowpos[(p2, c)] for c in common]].astype(np.float64)
            d = a - b
            n = len(d)
            v_e = d.var(0, ddof=1).mean() / 2
            v_s = ((d.mean(0) ** 2).mean() - 2 * v_e / n) / 2
            naive = 0.5 * (pl.ctl_sd[p1] ** 2 / pl.n_ctl[p1] + pl.ctl_sd[p2] ** 2 / pl.n_ctl[p2])
            kind = "same_rna_plate" if rna[p1] == rna[p2] else ("x_tag_same_compound_plate" if pl.pert_plate[p1] == pl.pert_plate[p2] else "different_compound_plate")
            out.append(dict(cell=cell, family=fam[cell], meta_role=role[cell], plate_1=pl.det_plate[p1], plate_2=pl.det_plate[p2], pair_kind=kind, n_common=n,
                            v_e=v_e, v_s=v_s, naive_ctl_mean_var=naive, n_ctl_1=int(pl.n_ctl[p1]), n_ctl_2=int(pl.n_ctl[p2]),
                            same_bead_batch=int(str(pl.ctl_bead_batch[p1]) == str(pl.ctl_bead_batch[p2])),
                            response_var=float(0.5 * (a.var(0, ddof=1).mean() + b.var(0, ddof=1).mean()))))
    return pd.DataFrame(out)


def unit_residual_variance(st):
    """Within-cell variance of unit residuals against the leave-family-out f0 (source-train cells), with the mean overlap terms."""
    rec = []
    for ci in st.eval_cells("train"):
        ui = st.eligible_units(ci)
        f0, _ = st.f0("lfo", st.u_group[ui], st.cell_family[ci])
        E = st.D[torch.tensor(ui)] - f0
        R = E - E.mean(0, keepdim=True)
        o, q = st.unit_overlap(ui[:400])
        rec.append(dict(cell=st.cell_name[ci], family=st.cell_family[ci], n_units=len(ui), v_tot=float((R ** 2).mean() * len(ui) / (len(ui) - 1)),
                        q_mean=float(q.mean()), o_diag_mean=float(torch.diagonal(o).mean()),
                        o_offdiag_mean=float((o.sum() - torch.diagonal(o).sum()) / max(len(o) * (len(o) - 1), 1))))
    return pd.DataFrame(rec)


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def d2(rows, cells, st, RD):
    """Two views per unit from two different RNA plates; cross-fit references from two disjoint halves of the other source-train families."""
    fams = [f for f, r in zip(st.fam_names, st.fam_role) if r == "train"]
    half = {f: int(zlib.crc32(("half|" + f).encode()) % 2) for f in fams}
    fg = st.FG.numpy().astype(np.float64)
    G = st.G
    Sh = np.zeros((2, G, N_GENES))
    nh = np.zeros((2, G))
    for h in (0, 1):
        m = np.array([st.fam_role[f] == "train" and half[st.fam_names[f]] == h for f in st.fg_fam])
        np.add.at(Sh[h], st.fg_group[m], fg[m])
        nh[h] = np.bincount(st.fg_group[m], minlength=G)
    fold = cells.set_index("cell").cv_fold
    acc = {k: dict(cab=np.zeros((N_GENES, N_GENES)), ctot=np.zeros((N_GENES, N_GENES)), cab_shared=np.zeros((N_GENES, N_GENES)), w=0.0) for k in ("fit", "test")}
    n_cells_fam = cells[cells.meta_role == "train"].groupby("family").size()
    recs = []
    rs = np.random.RandomState(7)
    tr_cells = set(cells.cell[cells.meta_role == "train"])
    for cell, g in rows[rows.cell.isin(tr_cells)].groupby("cell", sort=True):
        f = cells.family[cells.cell == cell].iloc[0]
        fi = int(np.where(st.fam_names == f)[0][0])
        A, B, gi = [], [], []
        for uid, u in g.groupby("unit_idx", sort=True):
            rp = sorted(u.rna_plates.unique())
            if len(rp) < 2:
                continue
            A.append(RD[u.index[u.rna_plates == rp[0]].to_numpy()].mean(0))
            B.append(RD[u.index[u.rna_plates == rp[1]].to_numpy()].mean(0))
            gi.append(st.u_group[uid])
        if len(gi) < MIN_COMMON:
            continue
        gi = np.array(gi)
        j = st.fg_index[fi, gi]
        own = np.where((j >= 0)[:, None], fg[np.clip(j, 0, None)], 0.0)
        hh = half[f]
        S = [Sh[0][gi].copy(), Sh[1][gi].copy()]
        n = [nh[0][gi].copy(), nh[1][gi].copy()]
        S[hh] -= own
        n[hh] -= (j >= 0)
        ok = (n[0] >= 1) & (n[1] >= 1)
        if ok.sum() < MIN_COMMON:
            continue
        A, B = np.stack(A)[ok].astype(np.float64), np.stack(B)[ok].astype(np.float64)
        f1, f2 = S[0][ok] / n[0][ok][:, None], S[1][ok] / n[1][ok][:, None]
        f0 = (S[0][ok] + S[1][ok]) / (n[0][ok] + n[1][ok])[:, None]
        c = lambda x: x - x.mean(0, keepdims=True)
        A1, A2, B1, B2, A0, B0 = c(A - f1), c(A - f2), c(B - f1), c(B - f2), c(A - f0), c(B - f0)
        Ar, Br = c(A), c(B)
        m = len(A)
        perm = np.roll(rs.permutation(m), 1)
        perm = np.where(perm == np.arange(m), (perm + 1) % m, perm)
        tr = lambda x, y: float((x * y).sum())
        recs.append(dict(cell=cell, family=f, cv_fold=int(fold[cell]), n_units=m,
                         corr_same_drug_crossfit=float(0.5 * (rowcorr(A1, B2).mean() + rowcorr(A2, B1).mean())),
                         corr_same_drug_shared_ref=float(rowcorr(A0, B0).mean()),
                         corr_other_drug_crossfit=float(0.5 * (rowcorr(A1, B2[perm]).mean() + rowcorr(A2, B1[perm]).mean())),
                         corr_other_drug_shared_ref=float(rowcorr(A0, B0[perm]).mean()),
                         corr_same_drug_no_reference=float(rowcorr(Ar, Br).mean()), corr_other_drug_no_reference=float(rowcorr(Ar, Br[perm]).mean()),
                         reliability_crossfit=0.5 * (tr(A1, B2) + tr(A2, B1)) / np.sqrt(tr(A0, A0) * tr(B0, B0)),
                         reliability_shared_ref=tr(A0, B0) / np.sqrt(tr(A0, A0) * tr(B0, B0)),
                         residual_var_view=float(0.5 * ((A0 ** 2).mean() + (B0 ** 2).mean()))))
        k = "test" if fold[cell] == 2 else "fit"
        w = 1.0 / n_cells_fam[f]
        cab = (A1.T @ B2 + A2.T @ B1) / (2 * m)
        acc[k]["cab"] += w * 0.5 * (cab + cab.T)
        cs = A0.T @ B0 / m
        acc[k]["cab_shared"] += w * 0.5 * (cs + cs.T)
        acc[k]["ctot"] += w * (A0.T @ A0 + B0.T @ B0) / (2 * m)
        acc[k]["w"] += w
    cellt = pd.DataFrame(recs)
    basis = []
    if acc["fit"]["w"] > 0 and acc["test"]["w"] > 0:
        ev_p, U_p = np.linalg.eigh(acc["fit"]["ctot"])
        ev_r, U_r = np.linalg.eigh(acc["fit"]["cab"])
        U_p, U_r = U_p[:, ::-1], U_r[:, ::-1]
        rnd = np.linalg.qr(np.random.RandomState(0).randn(N_GENES, 64))[0]
        T = acc["test"]
        for r in (16, 32):
            for name, U in (("plain_pca_of_residuals", U_p[:, :r]), ("repeat_cross_covariance", U_r[:, :r]), ("random_subspace", rnd[:, :r])):
                cap = lambda C: float(np.trace(U.T @ C @ U))
                basis.append(dict(rank=r, basis=name, fit_folds="0,1", test_fold=2,
                                  test_repeatable_fraction_captured=cap(T["cab"]) / float(np.trace(T["cab"])),
                                  test_total_variance_fraction_captured=cap(T["ctot"]) / float(np.trace(T["ctot"])),
                                  test_reliability_inside_subspace=cap(T["cab"]) / cap(T["ctot"]),
                                  test_reliability_inside_subspace_shared_ref=cap(T["cab_shared"]) / cap(T["ctot"]),
                                  overlap_with_plain_pca=float((np.linalg.norm(U_p[:, :r].T @ U, "fro") ** 2) / r)))
            basis.append(dict(rank=r, basis="whole_space", fit_folds="", test_fold=2, test_repeatable_fraction_captured=1.0, test_total_variance_fraction_captured=1.0,
                              test_reliability_inside_subspace=float(np.trace(T["cab"]) / np.trace(T["ctot"])),
                              test_reliability_inside_subspace_shared_ref=float(np.trace(T["cab_shared"]) / np.trace(T["ctot"])), overlap_with_plain_pca=np.nan))
    return cellt, pd.DataFrame(basis), half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="diag_v1")
    a = ap.parse_args()
    t0 = time.time()
    out = new_run_dir("diag", a.run_id)
    d = proc()
    rows = pd.read_csv(os.path.join(d, "rows.tsv"), sep="\t")
    plates = pd.read_csv(os.path.join(d, "plates.tsv"), sep="\t")
    cells = pd.read_csv(os.path.join(d, "cells.tsv"), sep="\t")
    trows = pd.read_csv(os.path.join(d, "target_rows_meta.tsv"), sep="\t")
    pub = lambda n: public("manifests", "replicate_calibrated_r8", n)

    s0, per_cell, cond = d0(rows, plates, cells, trows)
    per_cell.to_csv(pub("d0_cells.tsv"), sep="\t", index=False)
    dump(s0, pub("d0_replicate_structure.json"))
    print(json.dumps(s0["source"], indent=1), flush=True)

    RD = row_delta(rows)
    pairs = d1(rows, plates, cells, RD)
    pairs.to_csv(os.path.join(out, "d1_plate_pairs.tsv"), sep="\t", index=False, float_format="%.6g")
    pairs["ratio_shared_over_naive"] = pairs.v_s / pairs.naive_ctl_mean_var
    agg = []
    for (role, kind), g in pairs.groupby(["meta_role", "pair_kind"]):
        agg.append(dict(meta_role=role, pair_kind=kind, n_pairs=len(g), n_cells=g.cell.nunique(), n_families=g.family.nunique(), v_e=fam_balanced(g, "v_e"),
                        v_s=fam_balanced(g, "v_s"), naive_ctl_mean_var=fam_balanced(g, "naive_ctl_mean_var"), response_var=fam_balanced(g, "response_var"),
                        v_s_median=float(g.v_s.median()), frac_pairs_v_s_positive=float((g.v_s > 0).mean()),
                        v_s_same_bead_batch=float(g.v_s[g.same_bead_batch == 1].mean()) if (g.same_bead_batch == 1).any() else np.nan,
                        v_s_other_bead_batch=float(g.v_s[g.same_bead_batch == 0].mean()) if (g.same_bead_batch == 0).any() else np.nan))
    agg = pd.DataFrame(agg)
    agg["v_s_over_naive"] = agg.v_s / agg.naive_ctl_mean_var
    agg["implied_effective_control_wells_ratio"] = agg.naive_ctl_mean_var / agg.v_s
    agg.to_csv(pub("d1_plate_pair_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    by_cell = pairs.groupby(["cell", "family", "meta_role", "pair_kind"]).agg(n_pairs=("v_e", "size"), v_e=("v_e", "mean"), v_s=("v_s", "mean"),
                                                                                naive_ctl_mean_var=("naive_ctl_mean_var", "mean"), n_common=("n_common", "mean")).reset_index()
    by_cell.to_csv(pub("d1_plate_pairs_by_cell.tsv"), sep="\t", index=False, float_format="%.6g")
    print(agg.to_string(), flush=True)

    st = Store("cpu")
    vt = unit_residual_variance(st)
    vt.to_csv(pub("d1_unit_residual_variance.tsv"), sep="\t", index=False, float_format="%.6g")
    indep = pairs[(pairs.meta_role == "train") & (pairs.pair_kind != "same_rna_plate")]
    enough = len(indep) >= 30 and indep.family.nunique() >= 10
    v_e, v_s = fam_balanced(indep, "v_e"), max(fam_balanced(indep, "v_s"), 0.0)
    v_tot, q_mean, o_diag = fam_balanced(vt, "v_tot"), fam_balanced(vt, "q_mean"), fam_balanced(vt, "o_diag_mean")
    v_r = max(v_tot - v_e * q_mean - v_s * o_diag, 0.05 * v_tot)
    model = dict(kind="cov", v_r=float(v_r), v_e=float(v_e), v_s=float(v_s),
                 provenance=dict(run_id=a.run_id, estimated_on="source-train families only; plate pairs from different RNA plates", n_pairs=int(len(indep)),
                                 n_cells=int(indep.cell.nunique()), n_families=int(indep.family.nunique()), replicate_data_sufficient=bool(enough),
                                 v_tot_unit_residual=float(v_tot), mean_q=float(q_mean), mean_o_diag=float(o_diag),
                                 rule="v_r = max(v_tot - v_e * mean q - v_s * mean o_ii, 0.05 v_tot); all three are gene-averaged variances in Level 3 units",
                                 shares=dict(v_r=float(v_r / (v_r + v_e + v_s)), v_e=float(v_e / (v_r + v_e + v_s)), v_s=float(v_s / (v_r + v_e + v_s)))))
    dump(model, pub("covariance_model.json"))
    print(json.dumps(model, indent=1), flush=True)

    cellt, basis, half = d2(rows, cells, st, RD)
    cellt.to_csv(pub("d2_cells.tsv"), sep="\t", index=False, float_format="%.5g")
    basis.to_csv(pub("d2_basis.tsv"), sep="\t", index=False, float_format="%.5g")
    cols = [c for c in cellt.columns if c.startswith(("corr_", "reliab"))]
    d2s = dict(n_cells=int(len(cellt)), n_families=int(cellt.family.nunique()), n_units=int(cellt.n_units.sum()),
               family_balanced={c: fam_balanced(cellt, c) for c in cols},
               reference_halves={str(h): int(sum(v == h for v in half.values())) for h in (0, 1)},
               note="views = the unit's rows on its first and second RNA plate; cross-fit = each view uses a reference built from a disjoint half of the other source-train families")
    dump(d2s, pub("d2_summary.json"))
    print(json.dumps(d2s, indent=1))
    print(basis.to_string())
    dump(run_record(a.run_id, "diag", t0, d1_pairs=int(len(pairs)), covariance_model=model, d2=d2s), os.path.join(out, "result.json"))


if __name__ == "__main__":
    main()
