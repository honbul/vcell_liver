"""Source store with family-balanced same-drug means, plate-overlap structure of support sets and the two state feature blocks.

Differences from the predecessor (R2 data.py):
  * XC mixed pools are absent; f0 is the equal mean over FAMILIES of the family's mean unit delta (not over cells).
  * roles can be overridden for the inner family cross-validation (fold families become virtual targets).
  * every episode carries the plate-overlap matrix of its support units (metadata only) for the observation model.
  * the control state is split into a global block (support-mean control profile vs. source) and the within block.
"""
import json, os, zlib
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch

from .common import MAN, N_GENES, proc, sha256, man_file


@dataclass
class Episode:
    """One support/query problem of one target cell. `y_q` is the only field holding query labels."""
    name: str
    family: str
    seed: int
    g_s: np.ndarray
    g_q: np.ndarray
    x0_s: torch.Tensor
    x0_q: torch.Tensor
    f0_s: torch.Tensor
    f0_q: torch.Tensor
    n0_s: torch.Tensor          # number of source FAMILIES behind f0 (0 -> f0 is the zero vector)
    n0_q: torch.Tensor
    y_s: torch.Tensor           # measured support deltas (the K labels a method may use)
    y_q: torch.Tensor           # query truth: loss / scoring only
    o_s: torch.Tensor           # (Kmax, Kmax) plate overlap  o_ij = sum_p w_ip w_jp  (metadata)
    q_s: torch.Tensor           # (Kmax,) sum over the unit's plate rows of w_row^2   (metadata)
    plates_s: list = field(default_factory=list)
    plates_q: list = field(default_factory=list)

    def inputs(self, K):
        """Everything a method may see at support size K. No query label inside."""
        assert K <= len(self.g_s)
        return dict(name=self.name, g_s=self.g_s[:K], g_q=self.g_q, x0_s=self.x0_s[:K], x0_q=self.x0_q, f0_s=self.f0_s[:K], f0_q=self.f0_q,
                    n0_s=self.n0_s[:K], n0_q=self.n0_q, y_s=self.y_s[:K], o_s=self.o_s[:K, :K], q_s=self.q_s[:K])


def overlap(plate_lists, weight_lists):
    """o_ij = sum over shared plates of w_ip w_jp and q_i = sum over plate rows of w^2, from per-unit (plate ids, row weights)."""
    ids = sorted(set(p for pl in plate_lists for p in pl))
    pos = {p: i for i, p in enumerate(ids)}
    W = np.zeros((len(plate_lists), len(ids)))
    q = np.zeros(len(plate_lists))
    for i, (pl, wl) in enumerate(zip(plate_lists, weight_lists)):
        for p, w in zip(pl, wl):
            W[i, pos[p]] += w
            q[i] += w * w
    return W @ W.T, q


class Store:
    def __init__(self, device="cpu", fold=None, verify=True):
        d = proc()
        if verify:
            man = json.load(open(man_file("store_manifest.json")))
            for f, e in man["files"].items():
                assert sha256(os.path.join(d, f)) == e["sha256"], f"{f} differs from the store manifest"
        cells = pd.read_csv(os.path.join(d, "cells.tsv"), sep="\t")
        units = pd.read_csv(os.path.join(d, "units.tsv"), sep="\t")
        groups = pd.read_csv(os.path.join(d, "groups.tsv"), sep="\t")
        ur = pd.read_csv(os.path.join(d, "unit_rows.tsv"), sep="\t")
        r2 = os.environ["R2_PROC"]
        idx = units.r2_unit_idx.to_numpy()
        delta = np.load(os.path.join(r2, "unit_delta.npy"), mmap_mode="r")[idx]
        x0 = np.load(os.path.join(r2, "unit_x0.npy"), mmap_mode="r")[idx]
        fp = np.load(os.path.join(os.environ["TB_PROC"], "ecfp4.npy"))[groups.rep_drug_idx.to_numpy()]
        role = cells.meta_role.to_numpy().copy()
        if fold is not None:                       # inner cross-validation: fold families are the virtual targets, meta-validation is unused
            role = np.where(cells.meta_role == "val", "off", np.where(cells.cv_fold == fold, "val", "train"))
        self.fold = fold
        self._init(units, groups, cells, role, delta, x0, fp, ur, device)

    @classmethod
    def from_arrays(cls, units, groups, cells, delta, x0, fp, unit_rows=None, device="cpu"):
        """Fixture constructor (synthetic data). cells needs cell_idx, cell, family, meta_role."""
        self = cls.__new__(cls)
        self.fold = None
        if unit_rows is None:
            unit_rows = pd.DataFrame(dict(unit_idx=units.unit_idx, plate_idx=pd.factorize(units.plates)[0], w=1.0))
        self._init(units, groups, cells, cells.meta_role.to_numpy().copy(), delta, x0, fp, unit_rows, device)
        return self

    def _init(self, units, groups, cells, role, delta, x0, fp, ur, device):
        self.units, self.groups, self.cells = units, groups, cells
        self.dev = torch.device(device)
        u, c = units, cells
        assert (u.unit_idx.to_numpy() == np.arange(len(u))).all() and (c.cell_idx.to_numpy() == np.arange(len(c))).all()
        assert (groups.group_idx.to_numpy() == np.arange(len(groups))).all()
        self.G = len(groups)
        self.D = torch.tensor(np.asarray(delta), dtype=torch.float32, device=self.dev)
        self.X0 = torch.tensor(np.asarray(x0), dtype=torch.float32, device=self.dev)
        self.FP = torch.tensor(np.asarray(fp), dtype=torch.float32, device=self.dev)
        self.u_cell, self.u_group = u.cell_idx.to_numpy(), u.group_idx.to_numpy()
        self.cell_name, self.cell_family, self.cell_role = c.cell.to_numpy(), c.family.to_numpy(), np.asarray(role)
        fams = sorted(set(self.cell_family))
        self.fam_names = np.array(fams)
        self.cell_fam = np.array([fams.index(f) for f in self.cell_family])
        self.fam_role = np.array([self.cell_role[self.cell_fam == i][0] for i in range(len(fams))])
        for i in range(len(fams)):
            assert len(set(self.cell_role[self.cell_fam == i])) == 1, "a family straddles roles"
        self.cell_units = {i: np.where(self.u_cell == i)[0] for i in range(len(c))}
        self.plates = u.plates.to_numpy() if "plates" in u else np.array([""] * len(u))
        # unit -> plate rows (for the overlap structure)
        ur = ur.sort_values("unit_idx", kind="stable")
        self._ur_plate, self._ur_w = ur.plate_idx.to_numpy(), ur.w.to_numpy()
        self._ur_ptr = np.searchsorted(ur.unit_idx.to_numpy(), np.arange(len(u) + 1))
        # family x group means (float64 accumulation on CPU, fixed order)
        d64 = np.asarray(delta, np.float64)
        key = self.cell_fam[self.u_cell].astype(np.int64) * self.G + self.u_group
        codes, uniq = pd.factorize(key, sort=True)
        cnt = np.bincount(codes).astype(np.float64)
        fg = np.zeros((len(uniq), N_GENES))
        np.add.at(fg, codes, d64)
        fg /= cnt[:, None]
        self.fg_fam, self.fg_group = (uniq // self.G).astype(int), (uniq % self.G).astype(int)
        self.fg_index = np.full((len(fams), self.G), -1, dtype=np.int64)
        self.fg_index[self.fg_fam, self.fg_group] = np.arange(len(uniq))
        self._fg64 = fg
        self.FG = torch.tensor(fg, dtype=torch.float32, device=self.dev)
        self.S_train, self.n_train = self._table(self.fam_role == "train")
        self.S_all, self.n_all = self._table(self.fam_role != "__none__")
        self.fg_index_t = torch.tensor(self.fg_index, device=self.dev)
        del self._fg64

    def _table(self, fam_mask):
        m = fam_mask[self.fg_fam]
        S = np.zeros((self.G, N_GENES))
        np.add.at(S, self.fg_group[m], self._fg64[m])
        n = np.bincount(self.fg_group[m], minlength=self.G).astype(np.float32)
        return torch.tensor(S, dtype=torch.float32, device=self.dev), torch.tensor(n, device=self.dev)

    # ---- f0 (family-balanced)
    def regime_of(self, cell_idx):
        r = self.cell_role[cell_idx]
        assert r in ("train", "val"), "cell is outside this store's roles"
        return "lfo" if r == "train" else "train"

    def f0(self, regime, g, family=None, allow_missing=False):
        """Family-balanced same-drug mean of drug groups g and the number of families behind it.

        regime 'lfo'   : source-train families minus `family` (the virtual target's own family)
               'train' : source-train families (virtual target is outside source-train)
               'hcc'   : every kept source family (frozen reference list)
        A group without any allowed family gets the zero vector and n = 0 (only reachable with allow_missing).
        """
        g = torch.as_tensor(np.asarray(g), device=self.dev)
        if regime == "hcc":
            S, n = self.S_all[g], self.n_all[g]
        else:
            S, n = self.S_train[g], self.n_train[g]
            if regime == "lfo":
                fi = int(np.where(self.fam_names == family)[0][0])
                assert self.fam_role[fi] == "train"
                j = self.fg_index_t[fi, g]
                has = j >= 0
                S = S - torch.where(has[:, None], self.FG[j.clamp(min=0)], torch.zeros_like(S))
                n = n - has.float()
            else:
                assert regime == "train"
        assert allow_missing or (n >= 1).all(), "f0 requested for a drug group without any allowed source family"
        return S / n.clamp(min=1)[:, None], n

    def n_behind(self, regime, family=None):
        if regime == "hcc":
            return self.n_all
        if regime == "train":
            return self.n_train
        fi = int(np.where(self.fam_names == family)[0][0])
        return self.n_train - (self.fg_index_t[fi] >= 0).float()

    def eligible_units(self, cell_idx):
        ui = self.cell_units[cell_idx]
        n = self.n_behind(self.regime_of(cell_idx), self.cell_family[cell_idx])
        return ui[(n[torch.tensor(self.u_group[ui], device=self.dev)] >= 1).cpu().numpy()]

    def unit_overlap(self, units):
        pl = [self._ur_plate[self._ur_ptr[i]:self._ur_ptr[i + 1]] for i in units]
        wl = [self._ur_w[self._ur_ptr[i]:self._ur_ptr[i + 1]] for i in units]
        o, q = overlap(pl, wl)
        return torch.tensor(o, dtype=torch.float32, device=self.dev), torch.tensor(q, dtype=torch.float32, device=self.dev)

    def episode_from_units(self, cell_idx, sup_units, qry_units, seed=-1):
        su, qu = np.asarray(sup_units), np.asarray(qry_units)
        assert not (set(su) & set(qu)) and (self.u_cell[su] == cell_idx).all() and (self.u_cell[qu] == cell_idx).all()
        regime, fam = self.regime_of(cell_idx), self.cell_family[cell_idx]
        gs, gq = self.u_group[su], self.u_group[qu]
        f0s, n0s = self.f0(regime, gs, fam)
        f0q, n0q = self.f0(regime, gq, fam)
        st, qt = torch.tensor(su, device=self.dev), torch.tensor(qu, device=self.dev)
        o, q = self.unit_overlap(su)
        return Episode(name=self.cell_name[cell_idx], family=fam, seed=seed, g_s=gs, g_q=gq, x0_s=self.X0[st], x0_q=self.X0[qt], f0_s=f0s, f0_q=f0q,
                       n0_s=n0s, n0_q=n0q, y_s=self.D[st], y_q=self.D[qt], o_s=o, q_s=q, plates_s=list(self.plates[su]), plates_q=list(self.plates[qu]))

    # ---- fixed evaluation episodes (same rule as R2: permute eligible units; first 20% = query; rest = nested support order)
    def eval_episode(self, cell_idx, seed, query_cap=None, min_groups=40, max_k=32):
        ui = self.eligible_units(cell_idx)
        if len(ui) < min_groups:
            return None
        p = np.random.RandomState(880000 + 1000 * seed + int(zlib.crc32(self.cell_name[cell_idx].encode()) % 100000)).permutation(len(ui))
        nq = int(round(0.2 * len(ui)))
        q, s = ui[p[:nq]], ui[p[nq:]]
        if query_cap:
            q = q[:query_cap]
        return self.episode_from_units(cell_idx, s[:max_k], q, seed=seed)

    def eval_cells(self, role, min_groups=40):
        return [int(ci) for ci in np.where(self.cell_role == role)[0] if len(self.eligible_units(ci)) >= min_groups]

    def episodes(self, role, seeds, cap=None):
        return [self.eval_episode(ci, s, query_cap=cap) for ci in self.eval_cells(role) for s in seeds]

    # ---- training sampler (source-train families only)
    def make_sampler(self, k_values=(8, 16, 32), q_max=64, min_query=8):
        fams = {}
        for ci in np.where(self.cell_role == "train")[0]:
            ui = self.eligible_units(ci)
            if len(ui) >= min(k_values) + min_query:
                fams.setdefault(self.cell_family[ci], {})[int(ci)] = ui
        names = sorted(fams)

        def sample(rs):
            f = names[rs.randint(len(names))]
            cs = sorted(fams[f])
            c = cs[rs.randint(len(cs))]
            ui = fams[f][c]
            ks = [k for k in k_values if len(ui) >= k + min_query]
            K = ks[rs.randint(len(ks))]
            p = rs.permutation(len(ui))
            Q = min(q_max, len(ui) - K)
            return K, self.episode_from_units(c, ui[p[:K]], ui[p[K:K + Q]])

        sample.families = names
        return sample


def dummy_context(name, dim):
    """Deterministic pseudo-random vector per cell name: same size as the global state, no state information."""
    return np.random.RandomState(zlib.crc32(("dummy|" + str(name)).encode()) % (2 ** 31)).randn(dim).astype(np.float32)


class Features:
    """Label-free feature maps fitted on source-train cells only (of the store's current roles)."""
    KEYS = ("P_ctrl", "s_ctrl", "mu_f0", "P_f0", "s_f0", "mu_g", "P_g", "s_g")

    def __init__(self, *a):
        for k, v in zip(self.KEYS, a):
            setattr(self, k, v)

    @classmethod
    def fit(cls, st, rank=32, rank_g=16):
        tm = (st.cell_role == "train")[st.u_cell]
        ui = np.where(tm)[0]
        X = st.X0[torch.tensor(ui, device=st.dev)].double()
        ci = torch.tensor(st.u_cell[ui], device=st.dev)
        nc = len(st.cells)
        cm = torch.zeros(nc, N_GENES, device=st.dev, dtype=torch.float64).index_add_(0, ci, X)
        cn = torch.zeros(nc, device=st.dev, dtype=torch.float64).index_add_(0, ci, torch.ones(len(ci), device=st.dev, dtype=torch.float64))
        cmean = cm / cn.clamp(min=1)[:, None]
        W = X - cmean[ci]                                               # within-cell deviations of control profiles
        evals, evecs = torch.linalg.eigh(W.T @ W / len(W))
        P_ctrl, s_ctrl = evecs[:, -rank:].flip(1), torch.sqrt(evals[-rank:].flip(0).clamp(min=1e-12))
        ok = st.n_train >= 1
        F = (st.S_train[ok] / st.n_train[ok][:, None]).double()
        mu = F.mean(0)
        e2, v2 = torch.linalg.eigh((F - mu).T @ (F - mu) / len(F))
        P_f0, s_f0 = v2[:, -rank:].flip(1), torch.sqrt(e2[-rank:].flip(0).clamp(min=1e-12))
        # global block: family-balanced cell-mean control profiles of source-train cells
        tr_cells = np.where((st.cell_role == "train") & (cn.cpu().numpy() > 0))[0]
        fam_of = st.cell_fam[tr_cells]
        fm = torch.stack([cmean[torch.tensor(tr_cells[fam_of == f], device=st.dev)].mean(0) for f in sorted(set(fam_of))])
        mu_g = fm.mean(0)
        e3, v3 = torch.linalg.eigh((fm - mu_g).T @ (fm - mu_g) / len(fm))
        P_g, s_g = v3[:, -rank_g:].flip(1), torch.sqrt(e3[-rank_g:].flip(0).clamp(min=1e-12))
        return cls(*[t.float() for t in (P_ctrl, s_ctrl, mu, P_f0, s_f0, mu_g, P_g, s_g)])

    def save(self, path):
        np.savez(path, **{k: getattr(self, k).cpu().numpy() for k in self.KEYS})

    @classmethod
    def load(cls, path, device):
        z = np.load(path)
        return cls(*[torch.tensor(z[k], device=device) for k in cls.KEYS])

    def ctrl(self, x0, x0_support):
        """Within block: deviation from the support mean control profile."""
        return ((x0 - x0_support.mean(0, keepdim=True)) @ self.P_ctrl) / self.s_ctrl

    def glob(self, x0_support):
        """Global block: support-mean control profile relative to the source-train mean (one vector per episode)."""
        return ((x0_support.mean(0, keepdim=True) - self.mu_g) @ self.P_g) / self.s_g

    def f0repr(self, f, n0):
        return torch.cat([((f - self.mu_f0) @ self.P_f0) / self.s_f0, torch.log1p(n0)[:, None] / 3.0], 1)

    def build(self, inp, fp_table, fstar_s=None, fstar_q=None, dummy=False):
        """Feature blocks from a method-visible input dict. f0 representation uses the shrunk anchor f* when given."""
        out = {}
        for side, fs in (("s", fstar_s), ("q", fstar_q)):
            out["fp_" + side] = fp_table[torch.as_tensor(inp["g_" + side], device=fp_table.device)]
            out["ctrl_" + side] = self.ctrl(inp["x0_" + side], inp["x0_s"])
            out["f0r_" + side] = self.f0repr(inp["f0_" + side] if fs is None else fs, inp["n0_" + side])
        g = self.glob(inp["x0_s"])
        out["glob"] = torch.tensor(dummy_context(inp["name"], g.shape[1]), device=g.device)[None] if dummy else g
        return out
