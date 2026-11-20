"""Source unit store, leave-family-out same-drug mean (f0), label-free features and episode construction."""
import json, os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch

from .common import MAN, proc, sha256

N_GENES = 978


@dataclass
class Episode:
    """One support/query problem of one target cell. `y_q` is the only field holding query labels."""
    name: str                   # cell (or HCC target) name
    family: str
    seed: int
    g_s: np.ndarray             # support drug-group indices in the fixed support order (length >= any K used)
    g_q: np.ndarray             # query drug-group indices
    x0_s: torch.Tensor
    x0_q: torch.Tensor
    f0_s: torch.Tensor
    f0_q: torch.Tensor
    n0_s: torch.Tensor          # number of source cells behind f0
    n0_q: torch.Tensor
    y_s: torch.Tensor           # measured support deltas (the K labels a method may use)
    y_q: torch.Tensor           # query truth: loss / scoring only
    plates_s: list = field(default_factory=list)
    plates_q: list = field(default_factory=list)

    def inputs(self, K):
        """Everything a method may see at support size K. No query label inside."""
        assert K <= len(self.g_s)
        return dict(g_s=self.g_s[:K], g_q=self.g_q, x0_s=self.x0_s[:K], x0_q=self.x0_q, f0_s=self.f0_s[:K], f0_q=self.f0_q,
                    n0_s=self.n0_s[:K], n0_q=self.n0_q, y_s=self.y_s[:K])


class Source:
    """All source units on one device plus the tables needed for f0 under the three exclusion regimes."""

    def __init__(self, device="cpu", proc_dir=None, verify=True, fp_path=None):
        d = proc_dir or proc()
        if verify:
            man = json.load(open(os.path.join(MAN, "data_contract.json")))
            for f, e in man["files"].items():
                assert sha256(os.path.join(d, f)) == e["sha256"], f"{f} differs from the committed data contract"
        self.units = pd.read_csv(os.path.join(d, "units.tsv"), sep="\t")
        self.groups = pd.read_csv(os.path.join(d, "groups.tsv"), sep="\t")
        self.cells = pd.read_csv(os.path.join(d, "cells.tsv"), sep="\t")
        fp_path = fp_path or os.path.join(os.environ["TB_PROC"], "cmap2020_v1", "ecfp4.npy")
        fp = np.load(fp_path)
        self._init(np.load(os.path.join(d, "unit_delta.npy")), np.load(os.path.join(d, "unit_x0.npy")), fp[self.groups.rep_drug_idx.to_numpy()], device)

    @classmethod
    def from_arrays(cls, units, groups, cells, delta, x0, fp, device="cpu"):
        """Fixture constructor (synthetic data)."""
        self = cls.__new__(cls)
        self.units, self.groups, self.cells = units, groups, cells
        self._init(delta, x0, fp, device)
        return self

    def _init(self, delta, x0, fp, device):
        self.dev = torch.device(device)
        u, c = self.units, self.cells
        assert (u.unit_idx.to_numpy() == np.arange(len(u))).all() and (c.cell_idx.to_numpy() == np.arange(len(c))).all()
        assert (self.groups.group_idx.to_numpy() == np.arange(len(self.groups))).all()
        assert c.groupby("family").meta_role.nunique().max() == 1, "a family straddles meta-train and meta-validation"
        self.G = len(self.groups)
        self.D = torch.tensor(delta, dtype=torch.float32, device=self.dev)
        self.X0 = torch.tensor(x0, dtype=torch.float32, device=self.dev)
        self.FP = torch.tensor(fp, dtype=torch.float32, device=self.dev)
        self.u_cell = u.cell_idx.to_numpy()
        self.u_group = u.group_idx.to_numpy()
        self.cell_name = c.cell.to_numpy()
        self.cell_family = c.family.to_numpy()
        self.cell_role = c.meta_role.to_numpy()
        self.cell_units = {i: np.where(self.u_cell == i)[0] for i in range(len(c))}
        self.family_cells = {f: g.cell_idx.to_numpy() for f, g in c.groupby("family")}
        self.train_cell_mask = self.cell_role == "train"
        self._delta64 = np.asarray(delta, np.float64)
        self.S_train, self.n_train = self._table(self.train_cell_mask)
        self.S_all, self.n_all = self._table(np.ones(len(c), bool))
        del self._delta64

    def _table(self, cell_mask):
        """Per drug group: sum over allowed cells of unit deltas, and the number of allowed cells."""
        um = cell_mask[self.u_cell]
        gi = self.u_group[um]
        S = np.zeros((self.G, N_GENES))
        np.add.at(S, gi, self._delta64[um])             # CPU float64, fixed order: bit-reproducible tables
        n = np.bincount(gi, minlength=self.G).astype(np.float32)
        return torch.tensor(S, dtype=torch.float32, device=self.dev), torch.tensor(n, device=self.dev)

    # ---- f0
    def f0_table(self, regime, family=None):
        """(sum table, count table) of the cells allowed for an episode.

        regime 'meta_train': meta-train cells minus every cell of `family` (the virtual target's family).
        regime 'meta_val'  : meta-train cells only. regime 'hcc': all source cells.
        """
        if regime == "hcc":
            assert family is None
            return self.S_all, self.n_all
        if regime == "meta_val":
            assert family is None or not self.train_cell_mask[self.family_cells[family]].any(), "meta_val regime used for a meta-train family"
            return self.S_train, self.n_train
        assert regime == "meta_train" and family is not None
        cells = self.family_cells[family]
        assert self.train_cell_mask[cells].all(), "meta_train regime used for a meta-validation family"
        ui = np.concatenate([self.cell_units[c] for c in cells])
        ut = torch.tensor(ui, device=self.dev)
        gi = torch.tensor(self.u_group[ui], device=self.dev)
        S = self.S_train.clone().index_add_(0, gi, -self.D[ut])
        n = self.n_train.clone().index_add_(0, gi, -torch.ones(len(gi), device=self.dev))
        return S, n

    @staticmethod
    def f0_from(S, n, g):
        g = torch.as_tensor(g, device=S.device)
        nn = n[g]
        assert (nn >= 1).all(), "f0 requested for a drug group without any allowed source cell"
        return S[g] / nn[:, None], nn

    def regime_of(self, cell_idx):
        return "meta_train" if self.cell_role[cell_idx] == "train" else "meta_val"

    def eligible_units(self, cell_idx, n_table):
        """Units of a cell whose drug group has at least one allowed source cell behind f0."""
        ui = self.cell_units[cell_idx]
        ok = (n_table[torch.tensor(self.u_group[ui], device=self.dev)] >= 1).cpu().numpy()
        return ui[ok]

    def episode_from_units(self, cell_idx, sup_units, qry_units, S, n, seed=-1):
        su, qu = np.asarray(sup_units), np.asarray(qry_units)
        assert not (set(su) & set(qu)) and (self.u_cell[su] == cell_idx).all() and (self.u_cell[qu] == cell_idx).all()
        gs, gq = self.u_group[su], self.u_group[qu]
        f0s, n0s = self.f0_from(S, n, gs)
        f0q, n0q = self.f0_from(S, n, gq)
        st, qt = torch.tensor(su, device=self.dev), torch.tensor(qu, device=self.dev)
        pl = self.units.plates.to_numpy()
        return Episode(name=self.cell_name[cell_idx], family=self.cell_family[cell_idx], seed=seed, g_s=gs, g_q=gq, x0_s=self.X0[st], x0_q=self.X0[qt],
                       f0_s=f0s, f0_q=f0q, n0_s=n0s, n0_q=n0q, y_s=self.D[st], y_q=self.D[qt], plates_s=list(pl[su]), plates_q=list(pl[qu]))

    # ---- fixed evaluation episodes
    def eval_episode(self, cell_idx, seed, query_cap=None, min_groups=40, max_k=32):
        """Deterministic episode: permute eligible units; first 20% = query, rest = support order (nested K)."""
        regime = self.regime_of(cell_idx)
        S, n = self.f0_table(regime, self.cell_family[cell_idx] if regime == "meta_train" else None)
        ui = self.eligible_units(cell_idx, n)
        if len(ui) < min_groups:
            return None
        p = np.random.RandomState(770000 + 1000 * seed + cell_idx).permutation(len(ui))
        nq = int(round(0.2 * len(ui)))
        q, s = ui[p[:nq]], ui[p[nq:]]
        if query_cap:
            q = q[:query_cap]
        return self.episode_from_units(cell_idx, s[:max_k], q, S, n, seed=seed)

    def eval_cells(self, role, min_groups=40):
        out = []
        for ci in np.where(self.cell_role == role)[0]:
            regime = self.regime_of(ci)
            _, n = self.f0_table(regime, self.cell_family[ci] if regime == "meta_train" else None)
            if len(self.eligible_units(ci, n)) >= min_groups:
                out.append(int(ci))
        return out

    # ---- training sampler (meta-train families only)
    def make_sampler(self, k_values=(8, 16, 32), q_max=64, min_query=8):
        fams = {}
        for f, cells in self.family_cells.items():
            if not self.train_cell_mask[cells].all():
                continue
            S, n = self.f0_table("meta_train", f)
            el = {int(c): self.eligible_units(c, n) for c in cells}
            el = {c: u for c, u in el.items() if len(u) >= min(k_values) + min_query}
            if el:
                fams[f] = el
        names = sorted(fams)

        def sample(rs):
            f = names[rs.randint(len(names))]
            S, n = self.f0_table("meta_train", f)
            cs = sorted(fams[f])
            c = cs[rs.randint(len(cs))]
            ui = fams[f][c]
            ks = [k for k in k_values if len(ui) >= k + min_query]
            K = ks[rs.randint(len(ks))]
            p = rs.permutation(len(ui))
            Q = min(q_max, len(ui) - K)
            return K, self.episode_from_units(c, ui[p[:K]], ui[p[K:K + Q]], S, n)

        sample.families = names
        return sample


class Features:
    """Label-free feature maps fitted on meta-train units only."""

    def __init__(self, P_ctrl, s_ctrl, mu_f0, P_f0, s_f0):
        self.P_ctrl, self.s_ctrl, self.mu_f0, self.P_f0, self.s_f0 = P_ctrl, s_ctrl, mu_f0, P_f0, s_f0

    @classmethod
    def fit(cls, src, rank=32):
        tm = src.train_cell_mask[src.u_cell]
        ui = np.where(tm)[0]
        X = src.X0[torch.tensor(ui, device=src.dev)].double()
        ci = torch.tensor(src.u_cell[ui], device=src.dev)
        nc = len(src.cells)
        cm = torch.zeros(nc, N_GENES, device=src.dev, dtype=torch.float64).index_add_(0, ci, X)
        cn = torch.zeros(nc, device=src.dev, dtype=torch.float64).index_add_(0, ci, torch.ones(len(ci), device=src.dev, dtype=torch.float64))
        W = X - (cm / cn.clamp(min=1)[:, None])[ci]                     # within-cell (plate) deviations of control profiles
        evals, evecs = torch.linalg.eigh(W.T @ W / len(W))
        P_ctrl = evecs[:, -rank:].flip(1)
        s_ctrl = torch.sqrt(evals[-rank:].flip(0).clamp(min=1e-12))
        ok = src.n_train >= 1
        F = (src.S_train[ok] / src.n_train[ok][:, None]).double()
        mu = F.mean(0)
        e2, v2 = torch.linalg.eigh((F - mu).T @ (F - mu) / len(F))
        P_f0 = v2[:, -rank:].flip(1)
        s_f0 = torch.sqrt(e2[-rank:].flip(0).clamp(min=1e-12))
        return cls(P_ctrl.float(), s_ctrl.float(), mu.float(), P_f0.float(), s_f0.float())

    def save(self, path):
        np.savez(path, **{k: getattr(self, k).cpu().numpy() for k in ("P_ctrl", "s_ctrl", "mu_f0", "P_f0", "s_f0")})

    @classmethod
    def load(cls, path, device):
        z = np.load(path)
        return cls(*[torch.tensor(z[k], device=device) for k in ("P_ctrl", "s_ctrl", "mu_f0", "P_f0", "s_f0")])

    def ctrl(self, x0, x0_support):
        """Control-state feature: deviation from the support mean control profile (no label involved)."""
        return ((x0 - x0_support.mean(0, keepdim=True)) @ self.P_ctrl) / self.s_ctrl

    def f0repr(self, f0, n0):
        return torch.cat([((f0 - self.mu_f0) @ self.P_f0) / self.s_f0, torch.log1p(n0)[:, None] / 3.0], 1)

    def build(self, inp, fp_table):
        """Feature blocks of an episode's support and query units from a method-visible input dict."""
        out = {}
        for side in ("s", "q"):
            out["fp_" + side] = fp_table[torch.as_tensor(inp["g_" + side], device=fp_table.device)]
            out["ctrl_" + side] = self.ctrl(inp["x0_" + side], inp["x0_s"])
            out["f0r_" + side] = self.f0repr(inp["f0_" + side], inp["n0_" + side])
        return out


def residual_basis(src, rank=32):
    """Top principal axes of offset-removed residuals of meta-train cells (leave-family-out f0; each cell weighted equally).

    Source-only: no meta-validation or HCC unit contributes.
    """
    C = torch.zeros(N_GENES, N_GENES, device=src.dev, dtype=torch.float64)
    n_cells = 0
    for fam, cells in src.family_cells.items():
        if not src.train_cell_mask[cells].all():
            continue
        S, n = src.f0_table("meta_train", fam)
        for c in cells:
            ui = src.eligible_units(c, n)
            if len(ui) < 16:
                continue
            f0, _ = src.f0_from(S, n, src.u_group[ui])
            E = (src.D[torch.tensor(ui, device=src.dev)] - f0).double()
            R = E - E.mean(0, keepdim=True)
            C += R.T @ R / len(R)
            n_cells += 1
    evals, evecs = torch.linalg.eigh(C / n_cells)
    return evecs[:, -rank:].flip(1).float(), (evals[-rank:].flip(0) / evals.sum()).float()
