"""Collapse the prior study's source plate rows to (cell, drug_group) units. Source rows only; no target value is read."""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import MAN, proc, public, sha256, dump
from src.transigen_adaptation_b1.data import Data


def rows_to_units(rows, delta_rows, x0_rows):
    """rows: frame with cell, pert_id, drug_group, det_plate (any index). Arrays aligned with rows.

    plate rows -> condition (cell, pert_id): equal mean over plates -> unit (cell, drug_group): equal mean over conditions.
    Returns (unit table sorted by cell, drug_group; unit delta; unit x0).
    """
    r = rows[["cell", "pert_id", "drug_group", "det_plate"]].reset_index(drop=True)
    cond_key = r.cell + "\t" + r.pert_id
    cond_codes, cond_names = pd.factorize(cond_key, sort=True)
    n_c = len(cond_names)
    cnt = np.bincount(cond_codes, minlength=n_c).astype(np.float64)
    cd = np.zeros((n_c, delta_rows.shape[1]))
    cx = np.zeros((n_c, delta_rows.shape[1]))
    np.add.at(cd, cond_codes, np.asarray(delta_rows, np.float64))
    np.add.at(cx, cond_codes, np.asarray(x0_rows, np.float64))
    cd /= cnt[:, None]
    cx /= cnt[:, None]
    first = r.groupby(cond_codes).head(1)
    cond = pd.DataFrame(dict(cell=first.cell.to_numpy(), drug_group=first.drug_group.to_numpy()), index=cond_codes[first.index])
    cond = cond.sort_index()
    assert r.groupby(cond_codes).drug_group.nunique().max() == 1
    unit_key = cond.cell + "\t" + cond.drug_group
    u_codes, u_names = pd.factorize(unit_key, sort=True)
    n_u = len(u_names)
    ucnt = np.bincount(u_codes, minlength=n_u).astype(np.float64)
    ud = np.zeros((n_u, cd.shape[1]))
    ux = np.zeros((n_u, cd.shape[1]))
    np.add.at(ud, u_codes, cd)
    np.add.at(ux, u_codes, cx)
    ud /= ucnt[:, None]
    ux /= ucnt[:, None]
    r["u"] = u_codes[cond_codes]
    plates = r.groupby("u").det_plate.agg(lambda s: "|".join(sorted(set(s))))
    tab = pd.DataFrame(dict(cell=[n.split("\t")[0] for n in u_names], drug_group=[n.split("\t")[1] for n in u_names],
                            n_conditions=ucnt.astype(int), n_rows=np.bincount(r.u, minlength=n_u), plates=plates.reindex(range(n_u)).to_numpy()))
    return tab, ud.astype(np.float32), ux.astype(np.float32)


def group_table(drugs):
    """One representative pert_id (lexicographically first) per drug group; its fingerprint is the group's fingerprint."""
    d = drugs.sort_values(["drug_group", "pert_id"])
    g = d.groupby("drug_group", sort=True).agg(rep_pert_id=("pert_id", "first"), rep_drug_idx=("drug_idx", "first"), n_pert_ids=("pert_id", "size")).reset_index()
    g.insert(0, "group_idx", np.arange(len(g)))
    return g


def main():
    out = proc()
    assert not os.path.exists(os.path.join(out, "units.tsv")), "units already built; refusing to overwrite"
    D = Data()
    fam = pd.read_csv(os.path.join(MAN, "source_families.tsv"), sep="\t")
    n = D.n_src
    rows = D.rows.iloc[:n]
    ex = pd.read_csv("manifests/transigen_adaptation_b1/source_rows_excluded_degenerate.tsv", sep="\t").row_id.to_numpy()
    keep = np.ones(n, bool)
    keep[ex] = False
    x0 = np.asarray(D.x0[:n], np.float32)
    delta = np.asarray(D.source_x1(), np.float32) - x0
    tab, ud, ux = rows_to_units(rows[keep], delta[keep], x0[keep])
    groups = group_table(D.drugs)
    cells = fam[["cell", "family", "meta_role"]].sort_values("cell").reset_index(drop=True)
    cells.insert(0, "cell_idx", np.arange(len(cells)))
    assert set(tab.cell) <= set(cells.cell)
    tab["cell_idx"] = tab.cell.map(cells.set_index("cell").cell_idx)
    tab["group_idx"] = tab.drug_group.map(groups.set_index("drug_group").group_idx)
    assert tab.group_idx.notna().all() and np.isfinite(ud).all() and np.isfinite(ux).all()
    tab.insert(0, "unit_idx", np.arange(len(tab)))
    cells["n_units"] = cells.cell_idx.map(tab.groupby("cell_idx").size()).fillna(0).astype(int)
    tab.to_csv(os.path.join(out, "units.tsv"), sep="\t", index=False)
    groups.to_csv(os.path.join(out, "groups.tsv"), sep="\t", index=False)
    cells.to_csv(os.path.join(out, "cells.tsv"), sep="\t", index=False)
    np.save(os.path.join(out, "unit_delta.npy"), ud)
    np.save(os.path.join(out, "unit_x0.npy"), ux)
    files = ["units.tsv", "groups.tsv", "cells.tsv", "unit_delta.npy", "unit_x0.npy"]
    man = dict(track="episodic_residual_r2", built_from=dict(prior_build_manifest_sha256=sha256("manifests/transigen_adaptation_b1/build_manifest.json"),
                                                               prior_files=D.man["files"], prior_proc_dir=D.dir),
               n_source_rows_used=int(keep.sum()), n_degenerate_rows_dropped=int((~keep).sum()), n_units=int(len(tab)), n_cells=int(tab.cell_idx.nunique()),
               n_drug_groups_total=int(len(groups)), n_drug_groups_in_source=int(tab.group_idx.nunique()),
               units_with_multiple_conditions=int((tab.n_conditions > 1).sum()), units_with_multiple_plates=int(tab.plates.str.contains("|", regex=False).sum()),
               unit_delta_rms=float(np.sqrt((ud.astype(np.float64) ** 2).mean())), target_values_read=False,
               files={f: dict(sha256=sha256(os.path.join(out, f)), bytes=os.path.getsize(os.path.join(out, f))) for f in files}, out_dir=out)
    dump(man, public("manifests", "episodic_residual_r2", "data_contract.json"))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
