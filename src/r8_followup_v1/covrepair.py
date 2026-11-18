"""Covariance repair V0 -> V1 -> V2 -> V3 (plan section 4.3, Appendix A).

Working covariance of one gene's residuals of the n units of a cell:  Sigma = v_r I + v_e diag(q) + v_s O,  O = W W'.
Within-cell centring (H = I - 11'/n) gives  E[E'HE/(n-1)] = mu'H mu/(n-1) + v_r + v_e mean(q) + v_s (mean(O_diag) - mean(O_off)).
  V0  v_r = v_tot - v_e q400 - v_s Od400                         (original diag.py: first-400-unit summary, f0, no off-diagonal term)
  V1  v_r = v_tot - v_e q400 - v_s (Od400 - Ooff400)             (formula only)
  V2  v_r = v_tot - v_e q_all - v_s (Od_all - Ooff_all)          (every eligible unit; exact sums, no N x N matrix)
  V3  as V2 with residuals against f* = f0 n/(n + kappa_f)       (the predictor's anchor)
v_e, v_s come from the plate-pair decomposition (no reference involved) restricted to the partition's training families.
Floor: v_r_used = max(v_r_raw, 0.05 v_tot) as in the original; raw values, floor use and per-cell negative fractions are kept.
"""
import numpy as np
import pandas as pd
import torch

from src.replicate_calibrated_r8.diag import fam_balanced, unit_residual_variance
from .common import sha_str

FLOOR_FRAC = 0.05


def plate_weights(st, u):
    """Plate-aggregated weights of unit u (dict plate -> sum of row weights) and q_u = sum of squared ROW weights."""
    a, b = st._ur_ptr[u], st._ur_ptr[u + 1]
    pl, w = st._ur_plate[a:b], st._ur_w[a:b]
    d = {}
    for p, x in zip(pl, w):
        d[p] = d.get(p, 0.0) + float(x)
    return d, float((w.astype(np.float64) ** 2).sum())


def overlap_means_exact(st, units):
    """mean q, mean O_diag, mean O_off over the units, by the sum formulas of Appendix A."""
    return overlap_means_dicts([plate_weights(st, u) for u in units])


def overlap_means_rows(plate_lists, weight_lists):
    """Same from raw (plate ids, row weights) per unit (test helper; rows on the same plate are summed as in data.overlap)."""
    out = []
    for pl, wl in zip(plate_lists, weight_lists):
        d = {}
        for p, x in zip(pl, wl):
            d[p] = d.get(p, 0.0) + float(x)
        out.append((d, float((np.asarray(wl, np.float64) ** 2).sum())))
    return overlap_means_dicts(out)


def overlap_means_dicts(items):
    n = len(items)
    tot = {}
    od = np.zeros(n)
    q = np.zeros(n)
    for i, (d, qi) in enumerate(items):
        q[i] = qi
        od[i] = sum(x * x for x in d.values())
        for p, x in d.items():
            tot[p] = tot.get(p, 0.0) + x
    s2 = sum(x * x for x in tot.values())
    return float(q.mean()), float(od.mean()), float((s2 - od.sum()) / (n * (n - 1)))


def overlap_means_matrix(st, units):
    o, q = st.unit_overlap(units)
    o, q = o.double(), q.double()
    n = len(units)
    return float(q.mean()), float(torch.diagonal(o).mean()), float((o.sum() - torch.diagonal(o).sum()) / (n * (n - 1)))


def pair_constants(pairs, train_families):
    """v_e, v_s (family-balanced, different-RNA-plate pairs of the partition's training families) and the sufficiency rule."""
    indep = pairs[pairs.family.isin(train_families) & (pairs.pair_kind != "same_rna_plate")]
    v_s_raw = fam_balanced(indep, "v_s")
    return dict(v_e=fam_balanced(indep, "v_e"), v_s=max(v_s_raw, 0.0), v_s_raw=v_s_raw, n_pairs=int(len(indep)), n_pair_cells=int(indep.cell.nunique()),
                n_pair_families=int(indep.family.nunique()), sufficient=bool(len(indep) >= 30 and indep.family.nunique() >= 10))


def unit_table(st, kappa_f=None):
    """Per source-train cell of the partition: v_tot against f0 (V0-V2) and f* (V3), first-400 and all-unit overlap means."""
    base = unit_residual_variance(st)                  # original function: v_tot (f0), q / O_diag / O_off over the first 400 units
    rec = []
    for r, ci in zip(base.itertuples(), st.eval_cells("train")):
        assert r.cell == st.cell_name[ci]
        ui = st.eligible_units(ci)
        q_all, od_all, oo_all = overlap_means_exact(st, ui)
        f0, n = st.f0("lfo", st.u_group[ui], st.cell_family[ci])
        if kappa_f is not None:
            sh = (n / (n + kappa_f))[:, None] if kappa_f > 0 else (n > 0).float()[:, None]
            E = st.D[torch.tensor(ui)].double() - (f0 * sh).double()
            R = E - E.mean(0, keepdim=True)
            v3 = float((R ** 2).mean() * len(ui) / (len(ui) - 1))
        else:
            v3 = np.nan
        rec.append(dict(cell=r.cell, family=r.family, n_units=int(len(ui)), n_units_overlap400=int(min(400, len(ui))), v_tot_f0=r.v_tot, v_tot_fstar=v3,
                        q400=r.q_mean, od400=r.o_diag_mean, oo400=r.o_offdiag_mean, q_all=q_all, od_all=od_all, oo_all=oo_all,
                        units_sha256=sha_str(",".join(map(str, ui))), units400_sha256=sha_str(",".join(map(str, ui[:400])))))
    return pd.DataFrame(rec)


def versions(ut, pc, kappa_f=None):
    """The four covariance versions from one unit table and one pair-constant dict."""
    ve, vs = pc["v_e"], pc["v_s"]
    spec = dict(V0=("v_tot_f0", "q400", "od400", None), V1=("v_tot_f0", "q400", "od400", "oo400"), V2=("v_tot_f0", "q_all", "od_all", "oo_all"),
                V3=("v_tot_fstar", "q_all", "od_all", "oo_all"))
    out = {}
    hall = sha_str("|".join(ut.units_sha256))
    h400 = sha_str("|".join(ut.units400_sha256))
    for v, (vt, q, od, oo) in spec.items():
        if v == "V3" and kappa_f is None:
            continue
        cell_raw = ut[vt] - ve * ut[q] - vs * (ut[od] - (ut[oo] if oo else 0.0))
        t = ut.assign(_raw=cell_raw)
        v_tot = fam_balanced(t, vt)
        raw = v_tot - ve * fam_balanced(t, q) - vs * (fam_balanced(t, od) - (fam_balanced(t, oo) if oo else 0.0))
        used = max(raw, FLOOR_FRAC * v_tot)
        out[v] = dict(version=v, kind="cov", v_r=float(used), v_e=float(ve), v_s=float(vs), v_r_raw=float(raw), v_r_floor=float(FLOOR_FRAC * v_tot),
                      floor_applied=bool(raw < FLOOR_FRAC * v_tot), v_tot=float(v_tot), q_mean=fam_balanced(t, q), o_diag_mean=fam_balanced(t, od),
                      o_off_mean=fam_balanced(t, oo) if oo else 0.0, o_off_mean_unused=fam_balanced(t, "oo400") if not oo else np.nan,
                      frac_cells_raw_negative=float((cell_raw < 0).mean()), frac_cells_below_floor=float((cell_raw < FLOOR_FRAC * ut[vt]).mean()),
                      frac_families_raw_negative=float((t.groupby("family")._raw.mean() < 0).mean()),
                      n_cells=int(len(ut)), n_families=int(ut.family.nunique()), n_units=int(ut.n_units.sum()),
                      n_units_overlap_summary=int(ut.n_units_overlap400.sum() if v in ("V0", "V1") else ut.n_units.sum()),
                      unit_list_sha256=h400 if v in ("V0", "V1") else hall, residual_reference="f0 (unshrunk, leave-family-out)" if v != "V3" else f"f* (kappa_f={kappa_f})",
                      kappa_f=kappa_f if v == "V3" else np.nan, **{k: pc[k] for k in ("v_s_raw", "n_pairs", "n_pair_cells", "n_pair_families", "sufficient")})
    return out


def cov_dict(v):
    return dict(kind="cov", v_r=v["v_r"], v_e=v["v_e"], v_s=v["v_s"])
