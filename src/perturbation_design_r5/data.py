"""Store access, f0 tables and episode construction. Treated values of a target are never held by an Episode:
they are reachable only through oracle.Oracle (one representative row per revealed group) and evaluate.Truth (query units)."""
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .common import proc, protocol, sha_hex, TARGETS


class Store:
    def __init__(self, path=None, allow_target=False):
        self.path = path or proc()
        p = self.path
        self.rows = pd.read_csv(os.path.join(p, "rows.tsv"), sep="\t")
        self.units = pd.read_csv(os.path.join(p, "units_source.tsv"), sep="\t")
        self.unit_delta = np.load(os.path.join(p, "unit_delta_source.npy"), mmap_mode="r")
        self.well_delta = np.load(os.path.join(p, "well_delta_source.npy"), mmap_mode="r")
        self.fg = pd.read_csv(os.path.join(p, "famgroup.tsv"), sep="\t")
        self.fgv = np.load(os.path.join(p, "famgroup.npy"), mmap_mode="r")
        self.groups = pd.read_csv(os.path.join(p, "groups.tsv"), sep="\t")
        self.gpos = pd.Series(np.arange(len(self.groups)), index=self.groups.drug_group)
        self.fp = np.load(os.path.join(p, "group_fp.npy"))
        bp = os.path.join(p, "bases.npz")
        self.bases = dict(np.load(bp)) if os.path.exists(bp) else None
        self.target_units = pd.read_csv(os.path.join(p, "units_target.tsv"), sep="\t")   # metadata only
        self.allow_target = allow_target

    def families(self, role):
        u = self.units
        return sorted(set(u.family[u.meta_role == role]))


def f0_table(st, allowed_families):
    """Equal mean over allowed families of the family-level unit mean, per drug group."""
    fg = st.fg[st.fg.family.isin(allowed_families)]
    idx = fg.groupby("drug_group", sort=True).indices
    names = np.array(sorted(idx))
    vals = np.empty((len(names), st.fgv.shape[1]), np.float32)
    ncell = np.empty(len(names), np.float32)
    fgi = fg.fg_idx.to_numpy()
    nc = fg.n_cells.to_numpy()
    for i, g in enumerate(names):
        j = idx[g]
        vals[i] = np.asarray(st.fgv[fgi[j]]).mean(0)
        ncell[i] = nc[j].sum()
    return names, vals, ncell


def tanimoto(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter = a @ b.T
    na = a.sum(1)[:, None]
    nb = b.sum(1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


@dataclass
class Episode:
    role: str
    cell: str
    family: str
    seed: int
    groups: np.ndarray            # eligible groups in the episode (sorted by name)
    query: np.ndarray             # indices into groups
    pool: np.ndarray              # indices into groups, permuted order
    initial: np.ndarray           # first 4 pool entries
    f0: np.ndarray                # (n, 978)
    n_f0: np.ndarray              # cells behind f0
    fp: np.ndarray                # (n, 2048) bool
    feat: np.ndarray              # standardised [f0_repr(32), log1p n_f0]
    rep: pd.DataFrame             # representative row per group (metadata + control features), aligned with groups
    stress: bool = False
    extra: dict = field(default_factory=dict)


def target_cells(st, role, pr=None):
    """(cell, family) list for a role, following the protocol."""
    pr = pr or protocol()
    ep = pr["episodes"]
    if role == "hcc":
        return [(t, t) for t in TARGETS]
    u = st.units[st.units.meta_role == ("train" if role == "dev" else "val")]
    out = []
    for f, idx in u.groupby("family").indices.items():
        sub = u.iloc[idx]
        allowed = allowed_families(st, role, f)
        okg = set(st.fg.drug_group[st.fg.family.isin(allowed)])
        cnt = sub[sub.drug_group.isin(okg)].groupby("cell").size()
        cnt = cnt[cnt >= ep["virtual_target_min_groups"]]
        if len(cnt):
            best = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            out.append((best, f))
    if role == "dev":
        out = sorted(out, key=lambda cf: sha_hex(cf[1]))[:16]
    return sorted(out, key=lambda cf: cf[1])


def allowed_families(st, role, target_family):
    fams_tr = set(st.fg.family[st.fg.meta_role == "train"])
    if role == "dev":
        return fams_tr - {target_family}
    if role == "meta_val":
        return fams_tr
    if role == "hcc":
        return set(st.fg.family)
    raise ValueError(role)


_F0_CACHE = {}


def make_episode(st, role, cell, family, seed, stress=False, pr=None):
    pr = pr or protocol()
    ep = pr["episodes"]
    allowed = allowed_families(st, role, family)
    key = (role, family)
    if key not in _F0_CACHE:
        if len(_F0_CACHE) >= 2:
            _F0_CACHE.pop(next(iter(_F0_CACHE)))
        _F0_CACHE[key] = f0_table(st, allowed)
    names, F0, ncell = _F0_CACHE[key]
    pos = pd.Series(np.arange(len(names)), index=names)
    if role == "hcc":
        meas = st.target_units[st.target_units.cell == cell].drug_group
    else:
        meas = st.units[st.units.cell == cell].drug_group
    rows = st.rows[st.rows.cell == cell]
    flag = "is_rep_stress" if stress else "is_rep"
    rep = rows[rows[flag] == 1].set_index("drug_group")
    elig = sorted(set(meas) & set(pos.index) & set(rep.index))
    if role != "hcc" and len(elig) > ep["pool_cap"]:
        elig = sorted(sorted(elig, key=lambda g: sha_hex(f"{cell}|{g}"))[:ep["pool_cap"]])
    groups = np.array(elig)
    n = len(groups)
    rs = np.random.RandomState(20261117 + 1000 * seed)
    perm = rs.permutation(n)
    nq = int(round(0.3 * n))
    query = np.sort(perm[:nq])
    pool = perm[nq:]
    f0 = F0[pos[groups].to_numpy()]
    nf = ncell[pos[groups].to_numpy()]
    b = st.bases
    f0r = (f0 - b["f0_mu"]) @ b["f0_P"]
    feat = np.concatenate([f0r, np.log1p(nf)[:, None]], 1)
    feat = (feat - b["feat_mean"]) / b["feat_sd"]
    fp = st.fp[st.gpos[groups].to_numpy()]
    r = rep.loc[groups, ["pert_id", "det_plate", "rep_tag", "n_trt_wells", "n_ctl", "ctl_sd", "store_idx", "role"]].reset_index()
    return Episode(role=role, cell=cell, family=family, seed=seed, groups=groups, query=query, pool=pool,
                   initial=pool[:4].copy(), f0=f0.astype(np.float32), n_f0=nf, fp=fp, feat=feat.astype(np.float32), rep=r, stress=stress)
