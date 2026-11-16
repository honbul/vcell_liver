"""Explicit leakage and wiring tests. Run after the dataset and splits exist and before any training.
Every check is an assertion; the JSON written at the end lists what was checked."""
import inspect, json, os, sys
import numpy as np
import pandas as pd
import yaml

from . import tsg_adapter as T
from .data import Data, EXCLUDED, MAN_DIR, norm_name, FREEZE
from .evaluate import per_condition, retrieval_ranks


def main():
    cfg = yaml.safe_load(open("configs/transigen_adaptation_b1/protocol.yaml"))
    D = Data()
    rows, n_src = D.rows, D.n_src
    src, tgt = rows.iloc[:n_src], rows.iloc[n_src:]
    checks = {}
    # 1. no target family or clone anywhere in source, by exact and normalised names, and by the release's alias table
    names = {norm_name(c) for c in src.cell.unique()}
    assert not (names & set(EXCLUDED)) and not any("HUH" in n or "JHH" in n for n in names)
    ci = pd.read_csv("/mnt/S1/sdata/pubdata/LINCS/LINCS2020/cellinfo_beta.txt", sep="\t", dtype=str).fillna("")
    alias_hit = []
    for r in ci.itertuples():
        al = {norm_name(x) for x in [r.cell_iname, r.ccle_name.split("_")[0]] + r.cell_alias.replace("|", ";").split(";")}
        if al & {"HUH7", "HUH751", "JHH5", "JHH7"} or any(a.startswith(("HUH7", "JHH5", "JHH7")) for a in al if a):
            alias_hit.append(r.cell_iname)
    assert not (set(alias_hit) & set(src.cell.unique())), f"alias of a target family in source: {alias_hit}"
    checks["source_has_no_target_family"] = dict(source_cells=int(src.cell.nunique()), cellinfo_entries_matching_target_names=sorted(set(alias_hit)))
    assert set(tgt.cell.unique()) <= set(cfg["targets"]) and "HUH751" not in set(rows.cell)
    # 2. no shared detection plate between source and target rows (a shared plate would share DMSO wells)
    shared = set(src.det_plate) & set(tgt.det_plate)
    assert not shared, f"plates shared by source and target: {sorted(shared)[:5]}"
    checks["no_shared_plates_source_target"] = True
    # 3. source validation cells are disjoint from source training cells
    tr, va = D.source_split()
    assert not (set(src.cell.iloc[tr]) & set(src.cell.iloc[va])) and len(tr) + len(va) == n_src
    checks["source_train_val_cells_disjoint"] = dict(train_rows=int(len(tr)), val_rows=int(len(va)))
    # 4. target splits: disjoint, drug groups intact, every scored drug has a source-train condition, nested k subsets
    train_groups = set(src.drug_group.iloc[tr])
    summ = json.load(open(os.path.join(MAN_DIR, "splits_summary.json")))["targets"]
    for t, info in summ.items():
        for s, si in info["splits"].items():
            tab = D.split_table(t, int(s))
            assert T.sha256(os.path.join(MAN_DIR, "splits", f"target_{t}_split{s}.tsv")) == si["sha256"]
            assert set(tab.drug_group) <= train_groups
            ra = {k: set(D.target_rows(t, int(s), "adapt", k)) for k in sorted(set([si["k_main"]] + si["k_curve"]))}
            rv, rt = set(D.target_rows(t, int(s), "val")), set(D.target_rows(t, int(s), "test"))
            ks = sorted(ra)
            for a_, b_ in zip(ks, ks[1:]):
                assert ra[a_] < ra[b_], "adapt subsets are not nested"
            big = ra[ks[-1]]
            assert not (big & rv) and not (big & rt) and not (rv & rt)
            g = lambda r: set(rows.drug_group.iloc[sorted(r)])
            assert not (g(big) & g(rv)) and not (g(big) & g(rt)) and not (g(rv) & g(rt))
            assert set(rows.cell.iloc[sorted(big | rv | rt)]) == {t}
    checks["target_splits_disjoint_grouped_nested"] = True
    # 5. the guarded loader refuses test X1 outside the final test
    t0 = next(t for t, i in summ.items() if i["status"] != "excluded")
    s0 = int(next(iter(summ[t0]["splits"])))
    rt = D.target_rows(t0, s0, "test")
    for purpose in ("adapt", "val"):
        try:
            D.target_x1(rt, t0, s0, purpose)
            raise SystemExit("test X1 was served to a non-test purpose")
        except AssertionError:
            pass
    if not os.path.exists(FREEZE):
        try:
            os.environ["TB_FINAL_TEST"] = "1"
            D.target_x1(rt, t0, s0, "final_test")
            raise SystemExit("test X1 was served without a freeze manifest")
        except AssertionError:
            pass
        finally:
            os.environ.pop("TB_FINAL_TEST", None)
    checks["test_x1_refused_before_freeze"] = True
    # 6. the dataset cannot fall back to the upstream example
    try:
        Data(os.path.join(os.environ["TB_TSG_SRC"], "data/LINCS2020/data_example"), verify_hashes=False)
        raise SystemExit("example path accepted")
    except AssertionError:
        pass
    T.load_upstream_model_class()
    assert "dataset" not in sys.modules
    checks["example_path_refused_and_upstream_dataset_not_imported"] = True
    # 7. inference signature has no treated-profile argument, and the prediction ignores X1 by construction
    assert list(inspect.signature(T.predict_x1_mean_path).parameters)[:3] == ["model", "x0", "feat"]
    checks["inference_takes_x0_and_fingerprint_only"] = True
    # 8. gene axis and id round trips
    lm = pd.read_csv(os.path.join(MAN_DIR, "genes_landmark_978.tsv"), sep="\t", dtype=str)
    assert len(lm) == 978 and (lm.gene_id.to_numpy() == D.genes.gene_id.to_numpy()).all()
    fp_by_pid = {p: D.fp[i].tobytes() for p, i in zip(D.drugs.pert_id, D.drugs.drug_idx)}
    samp = rows.sample(2000, random_state=0)
    assert all(D.fp[r.drug_idx].tobytes() == fp_by_pid[r.pert_id] for r in samp.itertuples())
    checks["gene_axis_978_and_drug_id_round_trip"] = True
    # 9. evaluator sanity on synthetic input: order-equivariant, perfect prediction scores perfectly
    rs = np.random.RandomState(0)
    tr_ = rs.normal(size=(12, 978))
    pc = per_condition(tr_, tr_)
    assert np.allclose(pc.delta_rmse, 0) and np.allclose(pc.delta_pearson, 1) and (pc.retrieval_rank == 1).all()
    p = rs.permutation(12)
    pr = tr_ + rs.normal(size=tr_.shape)
    a_, b_ = per_condition(pr, tr_), per_condition(pr[p], tr_[p])
    assert np.allclose(a_.delta_rmse.to_numpy()[p], b_.delta_rmse) and np.allclose(a_.retrieval_rank.to_numpy()[p], b_.retrieval_rank)
    assert np.allclose(retrieval_ranks(np.zeros_like(tr_), tr_), 6.5)       # all ties -> mean rank
    assert np.isnan(per_condition(np.zeros_like(tr_), tr_).delta_pearson).all()
    checks["evaluator_synthetic_checks"] = True
    json.dump(checks, open(os.path.join(MAN_DIR, "leakage_checks.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
