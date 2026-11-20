"""Matched-plate E1 / E2 episodes (plan sections 3.2, 5.4) for any partition context.

For a cell with detection plates A and B from different RNA plates sharing >= 36 drug groups (original rule and RNG):
  query  = drug groups measured on A, truth = their plate-A delta (identical in E1 and E2)
  E1     = support: the same K drug groups measured on plate A   (query and support share the plate: o_qS = 1)
  E2     = support: the same K drug groups measured on plate B   (other X-tag replicate plate:          o_qS = 0)
Every eligible pair in this release is an X-tag replicate of one compound plate (not a documented independent culture), and
E2 is a noisy measurement of plate A, not the latent biological mean.
Positive control (5.4): the plate-B measurement of the SAME query drug used as the prediction of its plate-A truth, next to the
plate-B measurement of a DIFFERENT query drug of the same pair (fixed derangement).
"""
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.data import Episode
from src.replicate_calibrated_r8.matched_plate import plate_pairs, MIN_SHARED, N_QUERY_MIN, KS
from . import fmethods as F


def load_rows(proc_dir, row_delta):
    rows = pd.read_csv(f"{proc_dir}/rows.tsv", sep="\t")
    rows = rows[rows.role == "source"].reset_index(drop=True)
    rows["pos"] = np.arange(len(rows))
    plates = pd.read_csv(f"{proc_dir}/plates.tsv", sep="\t").set_index("plate_idx")
    return rows, plates, row_delta(rows)


def matched_episodes(st, rows, plates, RD, X0, cell_role="val", max_pairs=2, seeds=(1,)):
    """Yields (meta, ep_in_block, ep_other_block, positive-control dict) for the cells of `cell_role` in the store."""
    gidx = dict(zip(st.groups.drug_group, st.groups.group_idx))
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    for ci in range(len(st.cells)):
        if st.cell_role[ci] != cell_role:
            continue
        cell, fam = st.cell_name[ci], st.cell_family[ci]
        g = rows[rows.cell == cell]
        pairs, by = plate_pairs(g)
        pairs = pairs[:max_pairs]
        regime = st.regime_of(ci)
        nb = st.n_behind(regime, fam).cpu().numpy()
        for n_common, pa, pb, common in pairs:
            common = [dg for dg in common if nb[gidx[dg]] >= 1]
            if len(common) < MIN_SHARED:
                continue
            kind = "x_tag_same_compound_plate" if plates.pert_plate[pa] == plates.pert_plate[pb] else "different_compound_plate"
            for A, Bp in ((pa, pb), (pb, pa)):
                gi = np.array([gidx[dg] for dg in common])
                ra_, rb_ = np.array([by[A][dg][1] for dg in common]), np.array([by[Bp][dg][1] for dg in common])
                rid_a, rid_b = rows.row_id.to_numpy()[ra_], rows.row_id.to_numpy()[rb_]
                dA, dB = RD[ra_], RD[rb_]
                xA = np.asarray(X0[np.sort(rid_a)])[np.argsort(np.argsort(rid_a))]
                xB = np.asarray(X0[np.sort(rid_b)])[np.argsort(np.argsort(rid_b))]
                for seed in seeds:
                    p = np.random.RandomState(770000 + 1000 * seed + A * 7 + Bp).permutation(len(common))
                    kmax = max(k for k in KS if len(common) - k >= N_QUERY_MIN)
                    s, q = p[:kmax], p[kmax:][:200]
                    f0s, n0s = st.f0(regime, gi[s], fam)
                    f0q, n0q = st.f0(regime, gi[q], fam)
                    one = torch.ones(kmax, kmax)
                    eps = {}
                    for block, (ys, xs, sp, oq) in (("in_block", (dA[s], xA[s], A, 1.0)), ("other_block", (dB[s], xB[s], Bp, 0.0))):
                        ep = Episode(name=cell, family=fam, seed=seed, g_s=gi[s], g_q=gi[q], x0_s=t(xs), x0_q=t(xA[q]), f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q,
                                     y_s=t(ys), y_q=t(dA[q]), o_s=one, q_s=torch.ones(kmax), plates_s=[plates.det_plate[sp]] * kmax,
                                     plates_q=[plates.det_plate[A]] * len(q))
                        ep.o_qs = torch.full((len(q), kmax), oq)
                        eps[block] = ep
                    der = np.roll(np.arange(len(q)), 1)
                    pc = dict(truth=dA[q], same_drug_other_plate=dB[q], other_drug_other_plate=dB[q][der],
                              ctrl_shift_rms=float(np.sqrt(((xA[s] - xB[s]) ** 2).mean())), ctrl_rms=float(np.sqrt(((xA[s] - xA[s].mean(0)) ** 2).mean())))
                    meta = dict(cell=cell, family=fam, plate_query=plates.det_plate[A], plate_other=plates.det_plate[Bp], pair_kind=kind, support_seed=seed,
                                kmax=kmax, n_query=len(q), n_shared=len(common))
                    yield meta, eps["in_block"], eps["other_block"], pc
