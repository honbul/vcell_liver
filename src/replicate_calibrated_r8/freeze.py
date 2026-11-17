"""Write the freeze manifest: configuration, code commit, every source-side selection, checkpoint hashes, the fixed support /
query lists per (target, split_seed) and the label cost of every support set. Reads split files and plate METADATA only;
no HCC expression value."""
import argparse, json, os, subprocess
import numpy as np
import pandas as pd
import yaml

from .common import CFG, MAN, B_MAN, TARGETS, public, sha256, dump, proc
from .suite import current_selection
from .contrasts import CONTRASTS

REQUIRED = ("anchor_selection.json", "model_selection.json", "tsg_selection.json", "store_manifest.json", "reference_families.tsv", "covariance_model.json", "cv_selection.json", "cov2_selection.json", "cov_strength_decision.json")


def support_and_query(target, split):
    path = os.path.join(B_MAN, "splits", f"target_{target}_split{split}.tsv")
    t = pd.read_csv(path, sep="\t")
    pool = t[t.role == "adapt_pool"].sort_values("adapt_rank")
    return dict(split_file=path, split_file_sha256=sha256(path), support_order=list(pool.drug_group.iloc[:32]), adapt_pool_size=int(len(pool)),
                query_test=list(t.drug_group[t.role == "test"]), query_val_secondary=list(t.drug_group[t.role == "val"]))


def support_cost(targets, k_values):
    """Label cost of each support set from plate metadata: drug groups, plate rows, treated wells, plates, control wells."""
    meta = pd.read_csv(proc("target_rows_meta.tsv"), sep="\t")
    out = []
    for t, splits in targets.items():
        m = meta[meta.cell == t]
        for s, e in splits.items():
            for K in k_values:
                if K > len(e["support_order"]):
                    continue
                r = m[m.drug_group.isin(e["support_order"][:K])]
                assert r.drug_group.nunique() == K
                pl = r.drop_duplicates("det_plate")
                out.append(dict(target=t, split_seed=int(s), K=K, drug_groups=K, conditions=int(r.pert_id.nunique()), plate_rows=int(len(r)), treated_wells=int(r.n_trt_wells.sum()),
                                detection_plates=int(r.det_plate.nunique()), rna_plates=int(r.rna_plates.nunique()), compound_plates=int(r.pert_plate.nunique()),
                                independent_plates="not documented" if r.pert_plate.nunique() == 1 else int(r.pert_plate.nunique()),
                                control_wells=int(pl.n_ctl.sum()), drug_groups_on_two_or_more_plates=int((r.groupby("drug_group").det_plate.nunique() > 1).sum()),
                                target_validation_labels=0))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG))
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    assert dirty == "", "working tree has uncommitted changes"
    for f in REQUIRED:
        assert os.path.exists(os.path.join(MAN, f)), f"{f} is not committed"
        pub = os.path.join(os.environ["R8_PUBLIC"], "manifests", "replicate_calibrated_r8", f)
        assert not os.path.exists(pub) or sha256(pub) == sha256(os.path.join(MAN, f)), f"{f}: server copy differs from the committed file"
    sel = current_selection(committed_only=True)
    ev = cfg["evaluation"]["hcc"]
    for n, seeds in sel["learned"].items():
        assert sorted(seeds) == [str(s) for s in ev["model_seeds"]], (n, sorted(seeds))
        for e in seeds.values():
            assert sha256(e["checkpoint"]) == e["checkpoint_sha256"]
    for e in sel["tsg"]["checkpoints"].values():
        assert sha256(e["checkpoint"]) == e["checkpoint_sha256"]
    fb = proc("feature_basis.npz")
    targets = {t: {str(s): support_and_query(t, s) for s in ev["split_seeds"]} for t in TARGETS}
    for t in TARGETS:
        for s, e in targets[t].items():
            assert not (set(e["support_order"]) & set(e["query_test"])) and not (set(e["support_order"]) & set(e["query_val_secondary"]))
    cost = support_cost(targets, cfg["episodes"]["k_values"])
    cost.to_csv(public("manifests", "replicate_calibrated_r8", "support_cost.tsv"), sep="\t", index=False)
    fz = dict(track="replicate_calibrated_r8", frozen_on=subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip(),
              code_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              config=dict(path=CFG, sha256=sha256(CFG)), note=a.note,
              primary=dict(K=cfg["episodes"]["k_main"], metric="delta_rmse", query_set="test",
                           contrasts=["C1_iid-C0_iid", "B00-C0_iid", "B00-C1_iid", "B01-B00", "B10-B00", "B11-B00", "interaction(B11-B10-B01+B00)"]),
              all_contrasts=sorted(CONTRASTS), selection=sel,
              feature_basis=dict(path=fb, sha256=sha256(fb)), files={f: sha256(os.path.join(MAN, f)) for f in REQUIRED},
              k_values=cfg["episodes"]["k_values"], k_main=cfg["episodes"]["k_main"], model_seeds=ev["model_seeds"], split_seeds=ev["split_seeds"],
              target_validation_labels=0, bootstrap=ev["bootstrap"], targets=targets, hcc_expression_values_read_before_freeze_in_this_study=False,
              exposure="Pre-specified follow-up on an already-seen public benchmark: the HCC data and these test splits were scored in studies B and R2.")
    out = public("manifests", "replicate_calibrated_r8", "freeze_manifest.json")
    assert not os.path.exists(out) and not os.path.exists(os.path.join(MAN, "freeze_manifest.json")), "already frozen"
    dump(fz, out)
    print(json.dumps({k: v for k, v in fz.items() if k not in ("targets", "selection")}, indent=1)[:3000])
    print(cost.to_string())


if __name__ == "__main__":
    main()
