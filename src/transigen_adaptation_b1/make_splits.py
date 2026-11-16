"""Fix the source train/validation cell lines and the per-target drug-group splits. Reads row metadata only."""
import argparse, json, os
import numpy as np
import pandas as pd
import yaml

from .tsg_adapter import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--config", default="configs/transigen_adaptation_b1/protocol.yaml")
    ap.add_argument("--manifest-dir", default="manifests/transigen_adaptation_b1")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sdir = os.path.join(a.manifest_dir, "splits")
    assert not os.path.exists(os.path.join(a.manifest_dir, "splits_summary.json")), "splits already fixed; refusing to overwrite"
    os.makedirs(sdir, exist_ok=True)
    rows = pd.read_csv(os.path.join(a.proc, "rows.tsv"), sep="\t")
    src = rows[rows.role == "source"]

    # ---- source: held-out cell lines for validation
    sv = cfg["splits"]["source_validation"]
    cells = np.array(sorted(src.cell.unique()))
    perm = np.random.RandomState(sv["seed"]).permutation(len(cells))
    n_val = int(round(sv["fraction_of_source_cell_lines"] * len(cells)))
    val_cells = set(cells[perm[:n_val]])
    st = src.groupby("cell").agg(rows=("row_id", "size"), conditions=("pert_id", "nunique")).reset_index()
    st["role"] = np.where(st.cell.isin(val_cells), "val", "train")
    st["liver_non_hcc"] = st.cell.isin(cfg["source_liver_non_hcc_flagged"]).astype(int)
    st.to_csv(os.path.join(a.manifest_dir, "source_split.tsv"), sep="\t", index=False)
    train_groups = set(src.drug_group[~src.cell.isin(val_cells)])
    train_group_cells = src[~src.cell.isin(val_cells)].groupby("drug_group").cell.nunique()

    summary = dict(source=dict(cells=len(cells), train_cells=int(len(cells) - n_val), val_cells=sorted(val_cells),
                               train_rows=int(st.rows[st.role == "train"].sum()), val_rows=int(st.rows[st.role == "val"].sum()),
                               train_conditions=int(st.conditions[st.role == "train"].sum()),
                               val_conditions=int(st.conditions[st.role == "val"].sum())), targets={})
    el = cfg["eligibility"]
    fr = cfg["splits"]["target"]["fractions"]
    for tgt in sorted(cfg["targets"]):
        r = rows[(rows.role == "target") & (rows.cell == tgt)]
        g_all = sorted(r.drug_group.unique())
        g = [x for x in g_all if x in train_groups]
        n = len(g)
        status = "full" if n >= el["full"]["min_drug_groups"] else ("pilot" if n >= el["pilot"]["min_drug_groups"] else "excluded")
        info = dict(drug_groups_measured=len(g_all), drug_groups_eligible=n, dropped_not_in_source_train=len(g_all) - n,
                    conditions_eligible=int(r[r.drug_group.isin(g)].pert_id.nunique()),
                    rows_eligible=int(r.drug_group.isin(g).sum()), plates=int(r.det_plate.nunique()),
                    plates_per_condition_median=float(r[r.drug_group.isin(g)].groupby("pert_id").det_plate.nunique().median()),
                    status=status, splits={})
        if status != "excluded":
            n_test = int(round(fr[2] * n))
            n_val_t = int(round(fr[1] * n))
            for seed in cfg["splits"]["target"]["split_seeds"]:
                rs = np.random.RandomState(100000 + seed)   # same stream layout for every target
                p = [g[i] for i in rs.permutation(n)]
                test, val, pool = p[:n_test], p[n_test:n_test + n_val_t], p[n_test + n_val_t:]
                pool = [pool[i] for i in rs.permutation(len(pool))]     # nested order: first k = adapt set
                tab = pd.DataFrame(dict(target=tgt, split_seed=seed, drug_group=test + val + pool,
                                        role=["test"] * len(test) + ["val"] * len(val) + ["adapt_pool"] * len(pool),
                                        adapt_rank=[-1] * (len(test) + len(val)) + list(range(len(pool)))))
                tab["n_conditions"] = tab.drug_group.map(r.groupby("drug_group").pert_id.nunique())
                tab["n_rows"] = tab.drug_group.map(r.groupby("drug_group").size())
                tab["n_source_train_cells"] = tab.drug_group.map(train_group_cells)
                assert tab.drug_group.is_unique and len(tab) == n
                assert not (set(test) & set(val)) and not (set(test) & set(pool)) and not (set(val) & set(pool))
                path = os.path.join(sdir, f"target_{tgt}_split{seed}.tsv")
                tab.to_csv(path, sep="\t", index=False)
                k_main = min(cfg["splits"]["k_main"], len(pool))
                info["splits"][str(seed)] = dict(adapt_pool=len(pool), val=len(val), test=len(test), k_main=k_main,
                                                 reduced_k=bool(k_main < cfg["splits"]["k_main"]),
                                                 k_curve=[k for k in cfg["splits"]["k_curve"] if k <= len(pool)],
                                                 enough_val_test=bool(len(val) >= el["full"]["min_val"] and len(test) >= el["full"]["min_test"]),
                                                 sha256=sha256(path))
        summary["targets"][tgt] = info
    json.dump(summary, open(os.path.join(a.manifest_dir, "splits_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
