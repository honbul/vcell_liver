"""Global drug-group split 70/15/15. Reads drugs.tsv / rows.tsv metadata only (no expression)."""
import numpy as np, pandas as pd

from .common import tb, public, dump, sha_list, SPLIT_SEED, HCC


def main():
    drugs = pd.read_csv(tb("drugs.tsv"), sep="\t")
    rows = pd.read_csv(tb("rows.tsv"), sep="\t", usecols=["role", "cell", "drug_group"])
    groups = np.array(sorted(drugs.drug_group.unique()))
    assert len(groups) == 8288
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(groups))
    n_tr, n_va = int(round(0.70 * len(groups))), int(round(0.15 * len(groups)))
    role = np.empty(len(groups), dtype=object)
    role[perm[:n_tr]] = "train"; role[perm[n_tr:n_tr + n_va]] = "validation"; role[perm[n_tr + n_va:]] = "test"
    tab = pd.DataFrame(dict(drug_group=groups, role=role))
    tab.to_csv(public("manifests", "response_geometry_r4", "drug_split.tsv"), sep="\t", index=False)
    r = rows.merge(tab, on="drug_group")
    cnt = r.drop_duplicates(["cell", "drug_group"]).groupby(["role"]).size().to_dict()
    hcc = {h: r[r.cell == h].drop_duplicates("drug_group").role.value_counts().to_dict() for h in HCC}
    summ = dict(seed=SPLIT_SEED, groups=tab.role.value_counts().to_dict(), units_by_role_before_qc=cnt, hcc_groups_by_role=hcc,
                sha_train=sha_list(tab.drug_group[tab.role == "train"]), sha_validation=sha_list(tab.drug_group[tab.role == "validation"]),
                sha_test=sha_list(tab.drug_group[tab.role == "test"]))
    dump(summ, public("manifests", "response_geometry_r4", "drug_split_summary.json"))
    print(summ)


if __name__ == "__main__":
    main()
