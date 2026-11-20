"""Branch section 10: where to spend the same plate-row budget (retrospective budget simulation on existing replicate plates).

Primary design ('third_plate'): a cell with three detection plates A, B, C from three different RNA plates that share >= 36 drug
groups (same condition). The query truth is fixed on plate C (never a support plate), so neither arm shares the query plate.
  D16   16 support drugs x 1 plate row: drugs 1-8 measured on A, drugs 9-16 measured on B          (16 plate rows)
  REP8  the first 8 of those drugs (pre-fixed nested subset), each the equal mean of its A and B rows (16 plate rows, 8 drugs)
Each of the three plates serves once as the query plate (rotation); support seeds 1-3 fix the drug order.
Auxiliary design ('ab_mean', only if fewer than 10 source-train families are eligible for the primary design): query truth = the
equal mean of plates A and B; D16 = 8 drugs on A + 8 on B; REP8 = the A/B mean of 8 drugs (same mean query-support overlap 0.5).
Plate rows are not wet-lab cost: treated wells, control wells and plates of every arm are recorded separately. The three plates
are X-tag replicates of one compound plate whenever the release says so (recorded); they are not documented independent cultures.
"""
from itertools import combinations
import numpy as np
import torch

from src.replicate_calibrated_r8.data import Episode

MIN_COMMON, N_QUERY_MIN, MAX_TRIPLES = 36, 20, 2


def plate_triples(g):
    by = {}
    for pi, dg, pid, pos, rna in zip(g.plate_idx, g.drug_group, g.pert_id, g.pos, g.rna_plates):
        cur = by.setdefault(pi, {}).get(dg)
        if cur is None or pid < cur[0]:
            by[pi][dg] = (pid, pos)
    rna = dict(zip(g.plate_idx, g.rna_plates))
    out = []
    for a, b, c in combinations(sorted(by), 3):
        if len({rna[a], rna[b], rna[c]}) < 3:
            continue
        common = sorted(dg for dg in by[a] if dg in by[b] and dg in by[c] and by[a][dg][0] == by[b][dg][0] == by[c][dg][0])
        if len(common) >= MIN_COMMON:
            out.append((len(common), a, b, c, common))
    out.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    return out[:MAX_TRIPLES], by


def budget_episodes(st, rows, plates, RD, X0, cell_role="val", seeds=(1, 2, 3), design="third_plate"):
    gidx = dict(zip(st.groups.drug_group, st.groups.group_idx))
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    rid = rows.row_id.to_numpy()
    ntw = rows.n_trt_wells.to_numpy()
    for ci in range(len(st.cells)):
        if st.cell_role[ci] != cell_role:
            continue
        cell, fam = st.cell_name[ci], st.cell_family[ci]
        trip, by = plate_triples(rows[rows.cell == cell])
        regime = st.regime_of(ci)
        nb = st.n_behind(regime, fam).cpu().numpy()
        for n_common, a, b, c, common in trip:
            common = [dg for dg in common if nb[gidx[dg]] >= 1]
            if len(common) < MIN_COMMON:
                continue
            rots = ((a, b, c), (b, c, a), (c, a, b)) if design == "third_plate" else ((a, b, None), (b, a, None))
            for X, Y, Q in rots:
                gi = np.array([gidx[dg] for dg in common])
                pos = lambda P: np.array([by[P][dg][1] for dg in common])
                pX, pY = pos(X), pos(Y)
                x0 = lambda p: np.asarray(X0[np.sort(rid[p])])[np.argsort(np.argsort(rid[p]))]
                xX, xY = x0(pX), x0(pY)
                if Q is not None:
                    pQ = pos(Q)
                    truth_all, xq_all = RD[pQ], x0(pQ)
                else:
                    truth_all, xq_all = 0.5 * (RD[pX] + RD[pY]), 0.5 * (xX + xY)
                for seed in seeds:
                    p = np.random.RandomState(660000 + 1000 * seed + 7 * int(X) + 3 * int(Y) + (int(Q) if Q is not None else 0)).permutation(len(common))
                    s16, q = p[:16], p[16:][:200]
                    if len(q) < N_QUERY_MIN:
                        continue
                    f0q, n0q = st.f0(regime, gi[q], fam)
                    arms = {}
                    # D16: 8 drugs on X, 8 drugs on Y
                    onX = np.arange(16) < 8
                    ys = np.where(onX[:, None], RD[pX[s16]], RD[pY[s16]])
                    xs = np.where(onX[:, None], xX[s16], xY[s16])
                    o = (onX[:, None] == onX[None, :]).astype(np.float32)
                    oq = np.zeros((len(q), 16), np.float32) if Q is not None else np.full((len(q), 16), 0.5, np.float32)
                    arms["D16"] = (s16, ys, xs, o, np.ones(16, np.float32), oq,
                                   dict(plate_rows=16, unique_drugs=16, treated_wells=int(ntw[pX[s16[onX]]].sum() + ntw[pY[s16[~onX]]].sum()), support_plates=2))
                    s8 = s16[:8]
                    ys8 = 0.5 * (RD[pX[s8]] + RD[pY[s8]])
                    xs8 = 0.5 * (xX[s8] + xY[s8])
                    o8 = np.full((8, 8), 0.5, np.float32)
                    oq8 = np.zeros((len(q), 8), np.float32) if Q is not None else np.full((len(q), 8), 0.5, np.float32)
                    arms["REP8"] = (s8, ys8, xs8, o8, np.full(8, 0.5, np.float32), oq8,
                                    dict(plate_rows=16, unique_drugs=8, treated_wells=int(ntw[pX[s8]].sum() + ntw[pY[s8]].sum()), support_plates=2))
                    ctl = int(plates.n_ctl[X] + plates.n_ctl[Y])
                    meta = dict(cell=cell, family=fam, design=design, plate_x=plates.det_plate[X], plate_y=plates.det_plate[Y],
                                plate_query=plates.det_plate[Q] if Q is not None else "mean(X,Y)", support_seed=seed, n_query=len(q), n_shared=len(common),
                                same_compound_plate=bool(len({plates.pert_plate[X], plates.pert_plate[Y]} | ({plates.pert_plate[Q]} if Q is not None else set())) == 1),
                                control_wells_support_plates=ctl)
                    eps = {}
                    for arm, (ss, y, xsup, oo, qq, oqs, cost) in arms.items():
                        f0s, n0s = st.f0(regime, gi[ss], fam)
                        ep = Episode(name=cell, family=fam, seed=seed, g_s=gi[ss], g_q=gi[q], x0_s=t(xsup), x0_q=t(xq_all[q]), f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q,
                                     y_s=t(y), y_q=t(truth_all[q]), o_s=t(oo), q_s=t(qq), plates_s=[], plates_q=[])
                        ep.o_qs = t(oqs)
                        eps[arm] = (ep, cost)
                    yield meta, eps


def count_eligible(st, rows, cell_role):
    fams, cells = set(), 0
    for ci in range(len(st.cells)):
        if st.cell_role[ci] != cell_role:
            continue
        trip, _ = plate_triples(rows[rows.cell == st.cell_name[ci]])
        if trip:
            fams.add(st.cell_family[ci])
            cells += 1
    return len(fams), cells
