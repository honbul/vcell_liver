"""Guarded access to the built dataset. Training and evaluation share these tables and id maps."""
import json, os
import numpy as np
import pandas as pd

from .tsg_adapter import sha256

MAN_DIR = "manifests/transigen_adaptation_b1"
FREEZE = os.path.join(MAN_DIR, "freeze_manifest.json")
EXCLUDED = ("HUH7", "HUH751", "JHH5", "JHH7")


def norm_name(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


class Data:
    def __init__(self, proc_dir=None, verify_hashes=True):
        proc_dir = proc_dir or os.path.join(os.environ["TB_PROC"], "cmap2020_v1")
        real = os.path.realpath(proc_dir)
        # A study run must never fall back to the upstream example files.
        assert "data_example" not in real and "TranSiGen" not in real, f"refusing upstream example path: {real}"
        man = json.load(open(os.path.join(MAN_DIR, "build_manifest.json")))
        self.dir, self.man = proc_dir, man
        if verify_hashes:
            for f in ["rows.tsv", "drugs.tsv", "ecfp4.npy", "x0_all.npy", "x1_source.npy"]:
                got = sha256(os.path.join(proc_dir, f))
                assert got == man["files"][f]["sha256"], f"{f} differs from the committed build manifest"
        self.rows = pd.read_csv(os.path.join(proc_dir, "rows.tsv"), sep="\t")
        self.drugs = pd.read_csv(os.path.join(proc_dir, "drugs.tsv"), sep="\t")
        self.genes = pd.read_csv(os.path.join(proc_dir, "genes.tsv"), sep="\t", dtype=str)
        assert len(self.rows) == man["n_rows"] and (self.rows.row_id.to_numpy() == np.arange(len(self.rows))).all()
        self.n_src = man["n_source_rows"]
        assert (self.rows.role.iloc[: self.n_src] == "source").all() and (self.rows.role.iloc[self.n_src:] == "target").all()
        self.x0 = np.load(os.path.join(proc_dir, "x0_all.npy"), mmap_mode="r")
        self.fp = np.load(os.path.join(proc_dir, "ecfp4.npy"))
        assert self.x0.shape == (len(self.rows), 978) and self.fp.shape == (len(self.drugs), 2048)
        # id round-trips shared by every consumer
        assert (self.drugs.drug_idx.to_numpy() == np.arange(len(self.drugs))).all()
        assert (self.drugs.pert_id.to_numpy()[self.rows.drug_idx.to_numpy()] == self.rows.pert_id.to_numpy()).all()
        g = self.drugs.set_index("pert_id").drug_group
        assert (g.reindex(self.rows.pert_id).to_numpy() == self.rows.drug_group.to_numpy()).all()
        src_cells = set(self.rows.cell[: self.n_src])
        bad = [c for c in src_cells if norm_name(c) in EXCLUDED or "HUH" in norm_name(c) or "JHH" in norm_name(c)]
        assert not bad, f"target HCC family in source: {bad}"

    # ---- source
    def source_split(self):
        s = pd.read_csv(os.path.join(MAN_DIR, "source_split.tsv"), sep="\t")
        role = self.rows.cell.iloc[: self.n_src].map(s.set_index("cell").role)
        assert role.notna().all()
        role = role.to_numpy().copy()
        # protocol amendment 1: degenerate source rows (a plate whose values are ~0.001) are used nowhere
        ex = pd.read_csv(os.path.join(MAN_DIR, "source_rows_excluded_degenerate.tsv"), sep="\t")
        assert (ex.row_id < self.n_src).all()
        role[ex.row_id.to_numpy()] = "excluded"
        return np.where(role == "train")[0], np.where(role == "val")[0]

    def source_x1(self):
        x = np.load(os.path.join(self.dir, "x1_source.npy"), mmap_mode="r")
        assert x.shape == (self.n_src, 978)
        return x

    # ---- target
    def split_table(self, target, split_seed):
        t = pd.read_csv(os.path.join(MAN_DIR, "splits", f"target_{target}_split{split_seed}.tsv"), sep="\t")
        assert t.drug_group.is_unique and set(t.role) <= {"adapt_pool", "val", "test"}
        return t

    def target_rows(self, target, split_seed, role, k=None):
        """Row ids of a target for a split role. role='adapt' needs k (nested subset of the pool)."""
        t = self.split_table(target, split_seed)
        if role == "adapt":
            pool = t[t.role == "adapt_pool"].sort_values("adapt_rank")
            assert k is not None and k <= len(pool), "k larger than the adapt pool"
            groups = set(pool.drug_group.iloc[:k])
        else:
            assert role in ("val", "test")
            groups = set(t.drug_group[t.role == role])
        r = self.rows
        sel = r[(r.role == "target") & (r.cell == target) & r.drug_group.isin(groups)]
        return sel.row_id.to_numpy()

    def target_x1(self, row_ids, target, split_seed, purpose):
        """Measured treated profiles of target rows. Test rows are refused unless this is the final test."""
        row_ids = np.asarray(row_ids)
        assert (row_ids >= self.n_src).all(), "not target rows"
        t = self.split_table(target, split_seed)
        role = self.rows.drug_group.iloc[row_ids].map(t.set_index("drug_group").role)
        assert role.notna().all() and (self.rows.cell.iloc[row_ids] == target).all()
        if purpose == "final_test":
            assert os.environ.get("TB_FINAL_TEST") == "1", "test X1 is only for final_test.py"
            assert os.path.exists(FREEZE), "freeze manifest missing: target test stays closed"
        else:
            assert purpose in ("adapt", "val")
            assert not (role == "test").any(), "target test X1 requested outside the final test"
        x = np.load(os.path.join(self.dir, "x1_target.npy"), mmap_mode="r")
        assert x.shape == (len(self.rows) - self.n_src, 978)
        return np.asarray(x[row_ids - self.n_src])


def condition_delta(rows, delta_rows):
    """Equal mean over plates of plate-level deltas -> one vector per (cell, pert_id).

    Returns (table with cell, pert_id, drug_group, n_plates; array) in a deterministic order.
    """
    key = rows[["cell", "pert_id", "drug_group"]].reset_index(drop=True)
    grp = key.groupby(["cell", "pert_id"], sort=True)
    idx = grp.indices
    names = sorted(idx)
    out = np.stack([np.asarray(delta_rows)[idx[n]].mean(0) for n in names])
    first = grp.head(1).set_index(["cell", "pert_id"]).drug_group
    tab = pd.DataFrame(dict(cell=[n[0] for n in names], pert_id=[n[1] for n in names],
                            drug_group=[first[n] for n in names], n_plates=[len(idx[n]) for n in names]))
    return tab, out
