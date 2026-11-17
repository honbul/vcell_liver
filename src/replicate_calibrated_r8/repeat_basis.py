"""Repeat-informed output basis (revision R-1; plan section 8, optional modification).

Top eigenvectors of the symmetrised cross-covariance between two views of the same unit measured on two different RNA plates,
each view taken against a reference built from a DISJOINT half of the other source-train families (cross-fit, so a shared
noisy reference cannot create the repeatability). Source-train cells of the store's current roles only: with --fold k the
fold-k families contribute nothing, so the basis used inside a cross-validation fold never saw the fold's virtual targets.
"""
import argparse, os, time, zlib
import numpy as np
import pandas as pd

from .common import N_GENES, proc, public, sha256, dump
from .data import Store
from .diag import row_delta, MIN_COMMON

RANK = 16


def repeat_basis(st, rows, RD, rank=RANK):
    fams = [f for f, r in zip(st.fam_names, st.fam_role) if r == "train"]
    half = {f: int(zlib.crc32(("half|" + f).encode()) % 2) for f in fams}
    fg = st.FG.cpu().numpy().astype(np.float64)
    G = st.G
    Sh, nh = np.zeros((2, G, N_GENES)), np.zeros((2, G))
    for h in (0, 1):
        m = np.array([st.fam_role[f] == "train" and half[st.fam_names[f]] == h for f in st.fg_fam])
        np.add.at(Sh[h], st.fg_group[m], fg[m])
        nh[h] = np.bincount(st.fg_group[m], minlength=G)
    tr_cells = [st.cell_name[i] for i in range(len(st.cell_name)) if st.cell_role[i] == "train"]
    n_cells_fam = pd.Series(st.cell_family[st.cell_role == "train"]).value_counts()
    cab, wsum, n_cells, n_units = np.zeros((N_GENES, N_GENES)), 0.0, 0, 0
    fam_of = dict(zip(st.cell_name, st.cell_family))
    for cell, g in rows[rows.cell.isin(set(tr_cells))].groupby("cell", sort=True):
        f = fam_of[cell]
        fi = int(np.where(st.fam_names == f)[0][0])
        A, B, gi = [], [], []
        for uid, u in g.groupby("unit_idx", sort=True):
            rp = sorted(u.rna_plates.unique())
            if len(rp) < 2:
                continue
            A.append(RD[u.index[u.rna_plates == rp[0]].to_numpy()].mean(0))
            B.append(RD[u.index[u.rna_plates == rp[1]].to_numpy()].mean(0))
            gi.append(st.u_group[uid])
        if len(gi) < MIN_COMMON:
            continue
        gi = np.array(gi)
        j = st.fg_index[fi, gi]
        own = np.where((j >= 0)[:, None], fg[np.clip(j, 0, None)], 0.0)
        hh = half[f]
        S, n = [Sh[0][gi].copy(), Sh[1][gi].copy()], [nh[0][gi].copy(), nh[1][gi].copy()]
        S[hh] -= own
        n[hh] -= (j >= 0)
        ok = (n[0] >= 1) & (n[1] >= 1)
        if ok.sum() < MIN_COMMON:
            continue
        A, B = np.stack(A)[ok].astype(np.float64), np.stack(B)[ok].astype(np.float64)
        f1, f2 = S[0][ok] / n[0][ok][:, None], S[1][ok] / n[1][ok][:, None]
        c = lambda x: x - x.mean(0, keepdims=True)
        A1, A2, B1, B2 = c(A - f1), c(A - f2), c(B - f1), c(B - f2)
        m = len(A)
        cc = (A1.T @ B2 + A2.T @ B1) / (2 * m)
        w = 1.0 / n_cells_fam[f]
        cab += w * 0.5 * (cc + cc.T)
        wsum += w
        n_cells += 1
        n_units += m
    ev, U = np.linalg.eigh(cab / wsum)
    U = U[:, ::-1][:, :rank]
    return U.astype(np.float32), dict(n_cells=n_cells, n_units=n_units, families_weight=float(wsum), top_eigenvalues=[float(x) for x in ev[::-1][:rank]],
                                      n_negative_eigenvalues=int((ev < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    t0 = time.time()
    rows = pd.read_csv(proc("rows.tsv"), sep="\t")
    rows = rows[rows.role == "source"].reset_index(drop=True)
    RD = row_delta(rows)
    info = {}
    for fold in (None, 0, 1, 2):
        name = f"repeat_basis_{'all' if fold is None else 'fold' + str(fold)}_{a.tag}.npy"
        path = proc(name)
        assert not os.path.exists(path), "basis files are immutable"
        U, rec = repeat_basis(Store("cpu", fold=fold), rows, RD)
        np.save(path, U)
        rec.update(path=path, sha256=sha256(path), rank=RANK, orthonormal_error=float(np.abs(U.T @ U - np.eye(RANK)).max()))
        info["all" if fold is None else f"fold{fold}"] = rec
        print(name, rec["n_cells"], rec["n_units"], f"{time.time() - t0:.0f}s", flush=True)
    A = np.load(info["all"]["path"])
    for k in ("fold0", "fold1", "fold2"):
        info[k]["subspace_overlap_with_all"] = float(np.linalg.norm(A.T @ np.load(info[k]["path"]), "fro") ** 2 / RANK)
    dump(dict(tag=a.tag, rank=RANK, rule="top eigenvectors of the symmetrised cross-fit cross-covariance of two RNA-plate views; source-train cells of the roles in force",
              bases=info, commit=os.environ.get("R8_COMMIT", "")), public("manifests", "replicate_calibrated_r8", "repeat_basis.json"))


if __name__ == "__main__":
    main()
