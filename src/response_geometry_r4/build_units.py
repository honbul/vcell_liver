"""Collapse B plate rows to (cell, drug_group) units for the global drug split.

Outputs (git-ignored, R4_PROC):
  units.tsv            every unit with role (train / validation / test), family, plates, n_plates
  unit_x0.npy          unit X0 (input; all roles)
  unit_delta_open.npy  unit deltas of train + validation units (row k = units with open_idx k)
  unit_delta_test.npy  unit deltas of test units, sealed: read only by test_eval.py after the freeze
  half_delta_a/b.npy   split-half deltas of training units with >= 2 distinct detection plates (row = half_idx)
  train_rows_*.npy     training-drug plate rows for the upstream (row-level) WAVE recipe
  groups.tsv           representative fingerprint row per group
"""
import json, os
import numpy as np
import pandas as pd

from .common import MAN, B_MAN, proc, public, sha256, dump, HCC
from src.transigen_adaptation_b1.data import Data


def main():
    out = proc()
    assert not os.path.exists(proc("units.tsv")), "units already built; refusing to overwrite"
    D = Data()
    split = pd.read_csv(os.path.join(MAN, "drug_split.tsv"), sep="\t").set_index("drug_group").role
    fam = pd.read_csv(os.path.join(MAN, "cell_families.tsv"), sep="\t").set_index("cell").family
    rows = D.rows.copy()
    ex = pd.read_csv(os.path.join(B_MAN, "source_rows_excluded_degenerate.tsv"), sep="\t").row_id.to_numpy()
    keep = np.ones(len(rows), bool)
    keep[ex] = False
    rows = rows[keep & rows.cell.isin(fam.index).to_numpy()].copy()
    assert set(rows.cell) == set(fam.index), "cell set differs from the family table"
    rows["role"] = rows.drug_group.map(split)
    assert rows.role.notna().all()
    rid = rows.row_id.to_numpy()
    x0 = np.asarray(D.x0, np.float32)[rid]
    x1 = np.empty_like(x0)
    src = rid < D.n_src
    x1[src] = np.asarray(D.source_x1(), np.float32)[rid[src]]
    x1t = np.load(os.path.join(D.dir, "x1_target.npy"), mmap_mode="r")      # target rows: role from the R4 drug split
    x1[~src] = np.asarray(x1t, np.float32)[rid[~src] - D.n_src]
    delta = x1 - x0
    assert np.isfinite(delta).all()

    # rows -> condition (cell, pert_id): equal mean over plates -> unit (cell, group): equal mean over conditions
    rows = rows.reset_index(drop=True)
    cond_codes, cond_names = pd.factorize(rows.cell + "\t" + rows.pert_id, sort=True)
    cc = np.bincount(cond_codes).astype(np.float64)
    cd = np.zeros((len(cond_names), 978)); cx = np.zeros_like(cd)
    np.add.at(cd, cond_codes, delta.astype(np.float64)); np.add.at(cx, cond_codes, x0.astype(np.float64))
    cd /= cc[:, None]; cx /= cc[:, None]
    cond_cell = np.array([n.split("\t")[0] for n in cond_names])
    cond_group = rows.groupby(cond_codes).drug_group.first().reindex(range(len(cond_names))).to_numpy()
    u_codes, u_names = pd.factorize(pd.Series(cond_cell) + "\t" + pd.Series(cond_group), sort=True)
    uc = np.bincount(u_codes).astype(np.float64)
    ud = np.zeros((len(u_names), 978)); ux = np.zeros_like(ud)
    np.add.at(ud, u_codes, cd); np.add.at(ux, u_codes, cx)
    ud /= uc[:, None]; ux /= uc[:, None]
    rows["u"] = u_codes[cond_codes]
    g = rows.groupby("u")
    tab = pd.DataFrame(dict(cell=[n.split("\t")[0] for n in u_names], drug_group=[n.split("\t")[1] for n in u_names],
                            n_conditions=uc.astype(int), n_rows=g.size().reindex(range(len(u_names))).to_numpy(),
                            plates=g.det_plate.agg(lambda s: "|".join(sorted(set(s)))).reindex(range(len(u_names))).to_numpy()))
    tab.insert(0, "unit_idx", np.arange(len(tab)))
    tab["n_plates"] = tab.plates.str.count(r"\|") + 1
    tab["family"] = tab.cell.map(fam)
    tab["is_hcc"] = tab.cell.isin(HCC).astype(int)
    tab["role"] = tab.drug_group.map(split)
    drugs = D.drugs.sort_values(["drug_group", "pert_id"])
    groups = drugs.groupby("drug_group", sort=True).agg(rep_pert_id=("pert_id", "first"), rep_drug_idx=("drug_idx", "first"), n_pert_ids=("pert_id", "size")).reset_index()
    groups.insert(0, "group_idx", np.arange(len(groups)))
    tab["group_idx"] = tab.drug_group.map(groups.set_index("drug_group").group_idx)
    cells = sorted(tab.cell.unique())
    tab["cell_idx"] = tab.cell.map({c: i for i, c in enumerate(cells)})
    is_test = (tab.role == "test").to_numpy()
    tab["open_idx"] = -1; tab.loc[~is_test, "open_idx"] = np.arange((~is_test).sum())
    tab["test_idx"] = -1; tab.loc[is_test, "test_idx"] = np.arange(is_test.sum())

    # split halves over distinct detection plates (training units only), alternate assignment in sorted plate order
    tab["half_idx"] = -1
    ha, hb, k = [], [], 0
    for u in np.where((tab.role == "train").to_numpy() & (tab.n_plates >= 2).to_numpy())[0]:
        r = rows.iloc[g.indices[u]]
        pl = sorted(set(r.det_plate))
        side = r.det_plate.map({p: i % 2 for i, p in enumerate(pl)}).to_numpy()
        ha.append(delta[r.index.to_numpy()[side == 0]].mean(0)); hb.append(delta[r.index.to_numpy()[side == 1]].mean(0))
        tab.loc[u, "half_idx"] = k; k += 1

    tr = rows.role.to_numpy() == "train"
    rows_tr = rows[tr]
    files = dict(unit_x0=ux.astype(np.float32), unit_delta_open=ud[~is_test].astype(np.float32), unit_delta_test=ud[is_test].astype(np.float32),
                 half_delta_a=np.array(ha, np.float32), half_delta_b=np.array(hb, np.float32),
                 train_rows_x0=x0[tr], train_rows_x1=x1[tr])
    for n, a in files.items():
        np.save(proc(n + ".npy"), a)
    rows_tr[["row_id", "cell", "pert_id", "drug_group", "det_plate", "drug_idx"]].to_csv(proc("train_rows.tsv"), sep="\t", index=False)
    tab.to_csv(proc("units.tsv"), sep="\t", index=False)
    groups.to_csv(proc("groups.tsv"), sep="\t", index=False)
    names = [n + ".npy" for n in files] + ["units.tsv", "groups.tsv", "train_rows.tsv"]
    by = lambda m: {r: int(v) for r, v in tab[m].role.value_counts().items()}
    man = dict(track="response_geometry_r4", prior_build_manifest_sha256=sha256(os.path.join(B_MAN, "build_manifest.json")), prior_proc_dir=D.dir,
               rows_used=int(len(rows)), degenerate_rows_dropped=int((~keep).sum()), units=int(len(tab)), cells=len(cells), families=int(tab.family.nunique()),
               units_by_role=by(tab.role.notna()), hcc_units_by_role={h: by(tab.cell == h) for h in HCC},
               training_rows=int(tr.sum()), training_units_with_halves=k,
               test_deltas_sealed_in="unit_delta_test.npy (only test_eval.py opens it)",
               files={f: dict(sha256=sha256(proc(f)), bytes=os.path.getsize(proc(f))) for f in names}, out_dir=out)
    dump(man, public("manifests", "response_geometry_r4", "data_contract.json"))
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
