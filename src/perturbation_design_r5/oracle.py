"""Sealed access to treated values. Oracle: one representative row per revealed pool group, counted in a cost ledger.
Truth: query-unit deltas, used only by the scorer after a whole selection sequence has finished."""
import json, os, subprocess
import numpy as np
import pandas as pd

from .common import REPO, MAN

FREEZE = os.path.join(MAN, "freeze_manifest.json")


def _hcc_allowed():
    if os.environ.get("R5_FINAL_EVAL") != "1":
        raise PermissionError("HCC treated values are sealed: only experiments/perturbation_design_r5/hcc_eval.sbatch may open them")
    if not os.path.exists(FREEZE):
        raise PermissionError("freeze manifest missing")
    rel = os.path.relpath(FREEZE, REPO)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise PermissionError("freeze manifest is not committed at HEAD")


_TARGET_CACHE = {}


def _target_arrays(st):
    if "w" not in _TARGET_CACHE:
        _hcc_allowed()
        _TARGET_CACHE["w"] = np.load(os.path.join(st.path, "well_delta_target.npy"))
        _TARGET_CACHE["u"] = np.load(os.path.join(st.path, "unit_delta_target.npy"))
    return _TARGET_CACHE["w"], _TARGET_CACHE["u"]


class Oracle:
    def __init__(self, st, ep, override=None):
        self.ep = ep
        self.pool_set = set(int(i) for i in ep.pool)
        self.revealed = {}
        self.ledger = []
        self.override = override          # fixtures only: {group index: vector}
        if ep.role == "hcc":
            self._w = _target_arrays(st)[0]
        else:
            self._w = st.well_delta

    def reveal(self, gi):
        gi = int(gi)
        if gi not in self.pool_set:
            raise PermissionError("only acquisition-pool groups can be revealed")
        if gi in self.revealed:
            raise ValueError("group already revealed (no repeat measurement)")
        r = self.ep.rep.iloc[gi]
        v = self.override[gi] if self.override is not None else np.asarray(self._w[int(r.store_idx)], dtype=np.float32)
        self.revealed[gi] = v
        self.ledger.append(dict(group=self.ep.groups[gi], pert_id=r.pert_id, det_plate=r.det_plate, trt_wells=int(r.n_trt_wells), ctl_wells=int(r.n_ctl)))
        return v

    def cost(self):
        led = pd.DataFrame(self.ledger)
        if not len(led):
            return dict(trt_wells=0, plates=0, ctl_wells_overhead=0)
        pl = led.drop_duplicates("det_plate")
        return dict(trt_wells=int(led.trt_wells.sum()), plates=int(len(pl)), ctl_wells_overhead=int(pl.ctl_wells.sum()))


class Truth:
    """Query truth per query group. stress=True: truth only from rows on replicate tags other than the group's first tag."""

    def __init__(self, st, ep):
        self.ep = ep
        if ep.role == "hcc":
            _, u = _target_arrays(st)
            tu = st.target_units[st.target_units.cell == ep.cell].set_index("drug_group")
            self.vals = np.stack([u[int(tu.loc[ep.groups[q], "unit_idx"])] for q in ep.query])
            self.mask = np.ones(len(ep.query), bool)
            return
        if not ep.stress:
            su = st.units[st.units.cell == ep.cell].set_index("drug_group")
            self.vals = np.stack([np.asarray(st.unit_delta[int(su.loc[ep.groups[q], "unit_idx"])]) for q in ep.query])
            self.mask = np.ones(len(ep.query), bool)
            return
        rows = st.rows[st.rows.cell == ep.cell]
        vals, mask = [], []
        for q in ep.query:
            g = ep.groups[q]
            s = rows[rows.drug_group == g]
            first = s.rep_tag.min()
            s = s[s.rep_tag != first]
            if len(s) == 0:
                vals.append(np.zeros(st.well_delta.shape[1], np.float32))
                mask.append(False)
                continue
            cond = [np.asarray(st.well_delta[s.store_idx.to_numpy()[j]]).mean(0) for j in s.groupby("pert_id").indices.values()]
            vals.append(np.mean(cond, 0))
            mask.append(True)
        self.vals = np.stack(vals)
        self.mask = np.array(mask)


def score(pred, truth):
    """Per query unit: delta RMSE, Pearson, centred Pearson (own means over the query set)."""
    t = truth.vals[truth.mask]
    p = pred[truth.mask]
    rmse = np.sqrt(((p - t) ** 2).mean(1))
    pc = _rowcorr(p, t)
    cpc = _rowcorr(p - p.mean(0), t - t.mean(0))
    return dict(rmse=float(rmse.mean()), pearson=float(np.nanmean(pc)), centred_pearson=float(np.nanmean(cpc)), n=int(len(t))), rmse


def _rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / den
