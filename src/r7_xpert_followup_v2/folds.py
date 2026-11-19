"""Drug-group folds of the R4 training drugs and the trajectory manifest (private plan 2026-11-19 sections 4.2-4.4).

Outer: the R4 training drug groups are split into 3 folds (all cell-line units of a group move together). For outer
fold k, T_k = units of fold-k groups (evaluation), A_k = the other training units (fitting set).
Inner: inside every A_k the groups are split again into 3 folds; inner model (k, j) trains on A_k minus inner fold j
and predicts inner fold j -> genuine OOF predictions on A_k from models that never saw T_k.
Checkpoint selection of every model uses R4 validation units with is_hcc == 0 (disjoint from every A_k and T_k;
never an evaluation set of this track). HCC validation/test units are never a selection or fitting input.

Output (server mirror -> repo): configs/r7_xpert_followup_v2/folds.tsv, trajectories.tsv, folds_summary.json."""
import json, os
import numpy as np
import pandas as pd

from .common import CFG, SEEDS, public, dump, sha_list, stamp

FOLD_SEED = 20261119
K = 3


def assign(S):
    tg = np.unique(S.group_idx[S.train])
    rng = np.random.RandomState(FOLD_SEED)
    outer = np.empty(len(tg), int); outer[rng.permutation(len(tg))] = np.arange(len(tg)) % K
    inner = np.full((K, len(tg)), -1)
    for k in range(K):
        a = np.where(outer != k)[0]
        r = np.random.RandomState(FOLD_SEED + 1 + k)
        x = np.empty(len(a), int); x[r.permutation(len(a))] = np.arange(len(a)) % K
        inner[k, a] = x
    gtab = S.groups.set_index("group_idx").drug_group
    return pd.DataFrame(dict(group_idx=tg, drug_group=gtab.loc[tg].to_numpy(), outer=outer, inner_k0=inner[0], inner_k1=inner[1], inner_k2=inner[2]))


def load_folds():
    return pd.read_csv(os.path.join(CFG, "folds.tsv"), sep="\t")


def unit_sets(S, folds, scope, outer=-1, inner=-1):
    """Unit index arrays of a trajectory. Training units are R4 'train' units only."""
    u = S.units
    val_src = S.val[u.is_hcc.to_numpy()[S.val] == 0]
    out = dict(select_src=val_src, select_all=S.val)
    gi = S.group_idx[S.train]
    fo = folds.set_index("group_idx")
    o = fo.outer.loc[gi].to_numpy()
    if scope == "full":
        out.update(train=S.train, val=S.val, test=S.test)
    elif scope == "outer":
        out.update(train=S.train[o != outer], T=S.train[o == outer])
    elif scope == "inner":
        i = fo[f"inner_k{outer}"].loc[gi].to_numpy()
        a = o != outer
        out.update(train=S.train[a & (i != inner)], Ain=S.train[a & (i == inner)])
    else:
        raise ValueError(scope)
    return out


def trajectories(epochs=100):
    rows = []
    for s in SEEDS:
        for var in ["official", "ecfp"]:
            rows.append(dict(traj_id=f"full_{var}_L0_s{s}", stage="A_correction", scope="full", recipe="L0", variant=var, outer=-1, inner=-1, seed=s,
                             epochs=epochs, selectors="U,U_all", predict="val,test"))
    for s in SEEDS:
        for k in range(K):
            for rec in ["L0", "L1"]:
                rows.append(dict(traj_id=f"outer_{rec}_k{k}_s{s}", stage="B_outer", scope="outer", recipe=rec, variant="official", outer=k, inner=-1, seed=s,
                                 epochs=epochs, selectors="U,R,P", predict="T"))
            for j in range(K):
                rows.append(dict(traj_id=f"inner_L0_k{k}_j{j}_s{s}", stage="C_inner", scope="inner", recipe="L0", variant="official", outer=k, inner=j, seed=s,
                                 epochs=epochs, selectors="U,R,P", predict="Ain"))
    return pd.DataFrame(rows)


def main():
    from src.evidence_gated_r7.data import Store
    S = Store()
    f = assign(S)
    os.makedirs(os.path.dirname(public(CFG, "x")), exist_ok=True)
    f.to_csv(public(CFG, "folds.tsv"), sep="\t", index=False)
    T = trajectories()
    T.to_csv(public(CFG, "trajectories.tsv"), sep="\t", index=False)
    u = S.units
    tr = u.iloc[S.train].copy()
    tr["outer"] = f.set_index("group_idx").outer.loc[tr.group_idx].to_numpy()
    summ = dict(n_train_groups=int(len(f)), n_train_units=int(len(tr)), groups_per_outer=f.outer.value_counts().sort_index().tolist(),
                units_per_outer=tr.outer.value_counts().sort_index().tolist(),
                hcc_units_per_outer={h: tr[tr.cell == h].outer.value_counts().reindex(range(K), fill_value=0).tolist() for h in ["HUH7", "JHH5", "JHH7"]},
                min_units_per_cell_per_outer=int(tr.groupby(["cell", "outer"]).size().unstack(fill_value=0).min().min()),
                cells_with_zero_units_in_some_outer=tr.groupby(["cell", "outer"]).size().unstack(fill_value=0).pipe(lambda d: d[(d == 0).any(axis=1)].index.tolist()),
                n_cells=int(tr.cell.nunique()), n_source_families=int(tr[tr.is_hcc == 0].family.nunique()),
                selection_units_src=int((u.is_hcc.to_numpy()[S.val] == 0).sum()),
                folds_sha=sha_list(f.to_csv(sep="\t", index=False).splitlines()), n_trajectories=int(len(T)),
                trajectories_by_stage=T.stage.value_counts().to_dict(), fold_seed=FOLD_SEED, **stamp())
    for k in range(K):
        for j in range(K):
            s = unit_sets(S, f, "inner", k, j)
            summ[f"inner_k{k}_j{j}"] = dict(train=int(len(s["train"])), held=int(len(s["Ain"])))
    dump(summ, public(CFG, "folds_summary.json"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
