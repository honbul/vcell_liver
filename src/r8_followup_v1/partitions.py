"""Family partitions of the follow-up and one in-memory bank of the original R8 store (read-only).

A partition is a role per cell: 'train' (families whose data may fit nuisance quantities and constants),
'val' (families that are scored: virtual targets), 'off' (unused). Store(role) reuses the original Store code so
f0 / f* follow exactly the original rules: a 'val' cell gets f0 from the partition's 'train' families only.
"""
import json, os, zlib
import numpy as np
import pandas as pd

from src.replicate_calibrated_r8.data import Store
from src.replicate_calibrated_r8.common import sha256 as _sha
from .common import orig_man, sha_str


def inner_val_family(fam):
    return zlib.crc32(("r8f_inner|" + str(fam)).encode()) % 3 == 0


class Bank:
    """Arrays of the original store, loaded and hash-checked once; Store objects for any partition."""

    def __init__(self, verify=True):
        d = os.environ["R8_PROC"]
        if verify:
            man = json.load(open(orig_man("store_manifest.json")))
            for f, e in man["files"].items():
                assert _sha(os.path.join(d, f)) == e["sha256"], f"{f} differs from the store manifest"
        self.cells = pd.read_csv(os.path.join(d, "cells.tsv"), sep="\t")
        self.units = pd.read_csv(os.path.join(d, "units.tsv"), sep="\t")
        self.groups = pd.read_csv(os.path.join(d, "groups.tsv"), sep="\t")
        self.ur = pd.read_csv(os.path.join(d, "unit_rows.tsv"), sep="\t")
        r2 = os.environ["R2_PROC"]
        idx = self.units.r2_unit_idx.to_numpy()
        self.delta = np.asarray(np.load(os.path.join(r2, "unit_delta.npy"), mmap_mode="r")[idx], np.float32)
        self.x0 = np.asarray(np.load(os.path.join(r2, "unit_x0.npy"), mmap_mode="r")[idx], np.float32)
        self.fp = np.load(os.path.join(os.environ["TB_PROC"], "ecfp4.npy"))[self.groups.rep_drug_idx.to_numpy()]
        self.fam = self.cells.family.to_numpy()
        self.meta = self.cells.meta_role.to_numpy()
        self.fold = self.cells.cv_fold.to_numpy()

    # ---------------------------------------------------------------- roles
    def roles(self, name):
        """name: 'global' | 'outer{k}' | 'inner{k}' | 'final_inner'."""
        m, f, fam = self.meta, self.fold, self.fam
        if name == "global":
            return m.copy()
        if name.startswith("outer"):
            k = int(name[5:])
            return np.where(m == "val", "off", np.where(f == k, "val", "train")).astype(object)
        if name.startswith("inner") or name == "final_inner":
            base = (m == "train") if name == "final_inner" else ((m == "train") & (f != int(name[5:])))
            iv = np.array([inner_val_family(x) for x in fam])
            return np.where(~base, "off", np.where(iv, "val", "train")).astype(object)
        raise ValueError(name)

    def store(self, name, delta=None):
        role = self.roles(name)
        st = Store.__new__(Store)
        st.fold = None
        st._init(self.units, self.groups, self.cells, role, self.delta if delta is None else delta, self.x0, self.fp, self.ur, "cpu")
        st.partition = name
        return st

    def describe(self, name):
        role = self.roles(name)
        out = {}
        for r in ("train", "val", "off"):
            fams = sorted(set(self.fam[role == r]))
            out[r] = dict(n_cells=int((role == r).sum()), n_families=len(fams), families_sha256=sha_str("|".join(fams)))
        return out


PARTITIONS = ["global", "outer0", "outer1", "outer2", "inner0", "inner1", "inner2", "final_inner"]
INNER_OF = {"outer0": "inner0", "outer1": "inner1", "outer2": "inner2", "global": "final_inner"}
