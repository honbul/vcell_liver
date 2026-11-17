"""R4 unit store (read-only) with the R7 test seal, training-only statistics, and the knowledge tensors."""
import os, subprocess
import numpy as np
import pandas as pd

from .common import MAN, r4proc, kg, tb, HCC

FREEZE = os.path.join(MAN, "freeze_manifest.json")


class Store:
    def __init__(self):
        self.units = pd.read_csv(r4proc("units.tsv"), sep="\t")
        self.groups = pd.read_csv(r4proc("groups.tsv"), sep="\t")
        fp = np.load(tb("ecfp4.npy"))
        self.group_fp = fp[self.groups.rep_drug_idx.to_numpy()].astype(np.float32)
        self.x0 = np.load(r4proc("unit_x0.npy"))
        self._open = np.load(r4proc("unit_delta_open.npy"))
        u = self.units
        self.role = u.role.to_numpy()
        self.cell_idx = u.cell_idx.to_numpy()
        self.group_idx = u.group_idx.to_numpy()
        self.n_cells = int(self.cell_idx.max()) + 1
        self.train = np.where(self.role == "train")[0]
        self.val = np.where(self.role == "validation")[0]
        self.test = np.where(self.role == "test")[0]
        for a, b in [(self.train, self.val), (self.train, self.test), (self.val, self.test)]:
            assert not set(u.drug_group[a]) & set(u.drug_group[b])

    def fp(self, idx):
        return self.group_fp[self.group_idx[idx]]

    def delta(self, idx):
        """Measured deltas of train / validation units. Test units are refused here."""
        idx = np.asarray(idx)
        assert not (self.role[idx] == "test").any(), "test deltas are sealed (R7 final evaluation only)"
        return self._open[self.units.open_idx.to_numpy()[idx]]

    def test_delta(self, idx):
        assert os.environ.get("R7_FINAL_EVAL") == "1", "test deltas only inside the R7 final evaluation job"
        assert subprocess.run(["git", "ls-files", "--error-unmatch", FREEZE], capture_output=True).returncode == 0, "R7 freeze manifest not committed"
        assert subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() == "", "worktree not clean"
        idx = np.asarray(idx)
        assert (self.role[idx] == "test").all()
        t = np.load(r4proc("unit_delta_test.npy"), mmap_mode="r")
        return np.asarray(t[self.units.test_idx.to_numpy()[idx]])

    def source(self, idx):
        return self.units.is_hcc.to_numpy()[idx] == 0


class TrainStats:
    """Fitted on training-drug units only."""

    def __init__(self, S):
        tr = S.train
        y = S.delta(tr).astype(np.float64)
        c = S.cell_idx[tr]
        m = np.zeros((S.n_cells, 978)); n = np.bincount(c, minlength=S.n_cells)
        np.add.at(m, c, y)
        assert (n > 0).all(), "a cell without training units"
        self.cell_mean = (m / n[:, None]).astype(np.float32)
        self.x0_mu = S.x0[tr].mean(0).astype(np.float32)
        self.x0_sd = (S.x0[tr].std(0) + 1e-3).astype(np.float32)

    def x0s(self, S, idx):
        return (S.x0[idx] - self.x0_mu) / self.x0_sd


class Knowledge:
    """Graph + per-group seeds (built by kg_build, metadata only)."""
    SEED = ["mech", "actB", "actF", "sign_pos", "sign_neg", "action_unknown", "map_exact", "map_connect_only", "multi_gene_target"]

    def __init__(self, n_groups):
        self.edge_index = np.load(kg("edge_index.npy"))
        self.edge_rel = np.load(kg("edge_rel.npy"))
        self.edge_feat = np.load(kg("edge_feat.npy"))
        self.measured = np.load(kg("node_measured.npy")).astype(np.float32)
        self.lm_idx = np.load(kg("landmark_node_idx.npy"))
        self.n_nodes = len(self.measured)
        s = pd.read_csv(kg("drug_seeds.tsv"), sep="\t")
        self.seeds = s
        st = pd.read_csv(kg("group_status.tsv"), sep="\t").sort_values("group_idx")
        assert len(st) == n_groups and (st.group_idx.to_numpy() == np.arange(n_groups)).all()
        self.status = st
        self.m_available = st.m_available.to_numpy().astype(np.float32)
        # per group: rows of seeds; tid blocks for the augmentation
        self.by_group = {g: d for g, d in s.groupby("group_idx")}
        self.blocks = {g: sorted({t for x in d.tids for t in str(x).split("|")}) for g, d in self.by_group.items()}

    def group_status_vec(self, gi, rows):
        """4 drug-level evidence status features of the (possibly augmented) seed rows."""
        if rows is None or len(rows) == 0:
            return np.zeros(4, np.float32)
        return np.array([rows.mech.max(), max(rows.actB.max(), rows.actF.max()), min(np.log1p(len(rows)) / np.log(201), 1.0),
                         1.0 - rows.action_unknown.mean()], np.float32)
