"""Section 4.1 selection-population audit, part 1 (CPU, XPert env; reads files only, trains nothing).

Connects the executed original XPert runs to (a) the upstream working copy that ran them, (b) the h5ad 'valid' split they
selected on, (c) their run logs and saved checkpoints, and asks: did HCC validation units enter the automatic checkpoint
selection (upstream EarlyStopping on validation loss4 = 1 - unit-mean delta Pearson over every 'valid' row)?
Also records whether per-epoch snapshots exist (they decide between re-selection and retraining).
Output: manifests/r7_xpert_followup_v2/selection_population_audit_part1.json (+ per-epoch curves TSV)."""
import glob, json, os, re, subprocess
import numpy as np
import pandas as pd

from .common import ORIG_XPERT_RUNS, UPSTREAM_REF, HCC, orig_xpert, xp_work, pman, dump, sha256, sha_list, stamp
from src.evidence_gated_r7.data import Store

PAT = re.compile(r"Epoch (\d+), Valid Total Loss: [\d.]+, Loss1: [\d.]+, Loss2: [\d.-]+, Loss3: [\d.]+, Loss4: ([\d.]+)")
FILES = ["train_xpert.py", "utils.py", "datasets/MyDataset.py", "configs/config_l1000.yaml", "models/model_XPert.py", "models/model_utils.py"]


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def blob(path):
    return subprocess.run(["git", "hash-object", path], capture_output=True, text=True).stdout.strip()


def main():
    import anndata as ad, torch
    up = os.environ["XP_UPSTREAM"]
    head = git(up, "rev-parse", "HEAD")
    upstream = dict(path=up, head=head, matches_protocol_ref=head == UPSTREAM_REF,
                    porcelain_lines=len([l for l in git(up, "status", "--porcelain").splitlines() if l.strip()]),
                    files={f: dict(worktree_blob=blob(os.path.join(up, f)), head_blob=git(up, "rev-parse", f"HEAD:{f}")) for f in FILES})
    for f in upstream["files"].values():
        f["identical"] = f["worktree_blob"] == f["head_blob"]

    S = Store()
    u = S.units
    h5 = xp_work("processed_data", "l1000_r7sdst.h5ad")
    A = ad.read_h5ad(h5, backed="r")
    obs = A.obs.copy()
    split_counts = obs.r7_split.value_counts().to_dict()
    va = obs[obs.r7_split == "valid"]
    vidx = va.unit_idx.to_numpy().astype(int)
    same_as_store_val = bool(len(vidx) == len(S.val) and (np.sort(vidx) == np.sort(S.val)).all())
    tr_idx = obs[obs.r7_split == "train"].unit_idx.to_numpy().astype(int)
    roles_in_valid = u.role.iloc[vidx].value_counts().to_dict()
    hcc_mask = u.is_hcc.to_numpy()[vidx] == 1
    hcc_by_cell = u.cell.iloc[vidx[hcc_mask]].value_counts().to_dict()
    src_idx = np.sort(vidx[~hcc_mask])
    prior_idx = np.sort(vidx)
    # cross-check h5ad X: treated profile = X0 + delta of the store for a sample of valid rows
    rows = np.where(obs.r7_split.to_numpy() == "valid")[0][:256]
    x = np.asarray(A.X[rows[0]:rows[-1] + 1])[rows - rows[0]]
    chk = float(np.abs(x - (S.x0[vidx[:256]].astype(np.float32) + S.delta(vidx[:256]).astype(np.float32))).max())

    runs = {}
    curves = []
    for var, names in ORIG_XPERT_RUNS.items():
        for r in names:
            work = os.path.join(os.environ["R7_ORIG_ASSETS"], "xpert_runs_work", r)
            logs = glob.glob(os.path.join(work, "experiment", "l1000", "*", "*.log"))
            assert len(logs) == 1, (r, logs)
            txt = open(logs[0]).read()
            args = dict(re.findall(r"^(nfold|dataset|seed|drug_feat|include_cell_idx|use_gradscaler|config|mode): (.*)$", txt, re.M))
            nval = re.findall(r"Valid data:(\d+)", txt)
            ep = [(int(a), float(b)) for a, b in PAT.findall(txt)]
            v = np.array([b for _, b in ep])
            for e, b in ep:
                curves.append(dict(run=r, variant=var, epoch=e, logged_loss4_all_valid=b))
            pth = glob.glob(os.path.join(work, "experiment", "l1000", "*", "*.pth"))
            snaps = sorted(os.path.basename(p) for p in glob.glob(os.path.join(work, "experiment", "l1000", "*", "*")) if p.endswith((".pth", ".pt", ".ckpt")))
            ck = torch.load(orig_xpert(r, "expt", "r7_split_fold_early_stop.pth"), map_location="cpu")
            ck_epoch = int(ck["epoch"])
            tied = np.where(v == v.min())[0].tolist()
            runs[r] = dict(variant=var, log=logs[0], args=args, logged_valid_rows=[int(n) for n in nval], epochs_logged=len(v),
                           early_stop_triggered="Early stopping" in txt or len(v) < 100,
                           checkpoint=orig_xpert(r, "expt", "r7_split_fold_early_stop.pth"),
                           checkpoint_sha256=sha256(orig_xpert(r, "expt", "r7_split_fold_early_stop.pth")),
                           checkpoint_epoch=ck_epoch, checkpoint_logged_loss4=float(v[ck_epoch]),
                           log_min_loss4_3dp=float(v.min()), log_epochs_at_min_3dp=tied,
                           checkpoint_files_in_experiment_folder=snaps, per_epoch_snapshots=len(snaps) > 1,
                           run_workcopy_files_identical_to_upstream_head={f: blob(os.path.join(work, f)) == upstream["files"][f]["head_blob"] for f in FILES if f != "configs/config_l1000.yaml"},
                           h5ad_link=os.path.realpath(os.path.join(work, "processed_data", "l1000_r7sdst.h5ad")))
    pd.DataFrame(curves).to_csv(pman("orig_xpert_logged_curves.tsv"), sep="\t", index=False)

    fam = u.family.to_numpy()
    rec = dict(
        question="did HCC validation units enter the automatic checkpoint selection of the original R7 XPert runs?",
        upstream_working_copy=upstream,
        code_path=dict(prep="src/evidence_gated_r7/xpert_prep.py: tr, va = S.train, S.val (no HCC filter) -> h5ad r7_split='valid'",
                       loader="upstream utils.load_dataloader: val_data = data[data.obs[nfold] == 'valid'] (all rows)",
                       dataset="upstream datasets/MyDataset.py: every row, no cell filter",
                       selector="upstream train_xpert.py: stopper.step(val_loss4, ...), val_loss4 = sum(1 - Pearson(delta_hat, delta)) / n_valid_rows",
                       checkpoint="upstream utils.EarlyStopping.save_checkpoint overwrites one {fold}_fold_early_stop.pth on each strict improvement"),
        h5ad=dict(path=h5, sha256=sha256(h5), split_counts={k: int(v) for k, v in split_counts.items()},
                  valid_rows_equal_store_validation_units=same_as_store_val, train_rows_equal_store_train_units=bool((np.sort(tr_idx) == np.sort(S.train)).all()),
                  roles_in_valid={k: int(v) for k, v in roles_in_valid.items()}, treated_profile_crosscheck_max_abs=chk),
        prior_selection_population=dict(rule="all h5ad 'valid' rows (= every R4 validation unit, HCC included), unit-weighted",
                                        n_units=int(len(prior_idx)), n_hcc_units=int(hcc_mask.sum()), hcc_units_by_cell={k: int(v) for k, v in hcc_by_cell.items()},
                                        hcc_share_of_unit_weight=float(hcc_mask.mean()), n_drug_groups=int(u.drug_group.iloc[prior_idx].nunique()),
                                        unit_idx_sha256=sha_list(prior_idx.tolist())),
        corrected_selection_population=dict(rule="R4 validation units with is_hcc == 0 (source only), same unit-weighted selector",
                                            n_units=int(len(src_idx)), n_drug_groups=int(u.drug_group.iloc[src_idx].nunique()),
                                            n_families=int(len(set(fam[src_idx]))), empty=bool(len(src_idx) == 0),
                                            unit_idx_sha256=sha_list(src_idx.tolist())),
        runs=runs,
        finding=dict(hcc_validation_in_selection=bool(hcc_mask.sum() > 0 and same_as_store_val),
                     per_epoch_snapshots_available=any(r["per_epoch_snapshots"] for r in runs.values()),
                     consequence="no per-epoch snapshots: source-only re-selection from saved checkpoints is impossible; "
                                 "the correction retrains XPert with a source-only selection mask (plan 4.1 last paragraph)"),
        hcc_cells=list(HCC), **stamp())
    dump(rec, pman("selection_population_audit_part1.json"))
    print(json.dumps({k: rec[k] for k in ["prior_selection_population", "corrected_selection_population", "finding"]}, indent=1))
    print({r: (v["checkpoint_epoch"], v["checkpoint_logged_loss4"], v["log_epochs_at_min_3dp"]) for r, v in runs.items()})


if __name__ == "__main__":
    main()
