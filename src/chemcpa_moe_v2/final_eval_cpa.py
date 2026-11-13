"""The single test pass over the held-out drug groups, for every frozen arm and every baseline.

This is the only script in the study that reads chemCPA's "ood" split, which is this study's test
set. It loads the checkpoints selection already chose, scores them once, and writes the checkpoint
hashes alongside the numbers so the freeze is checkable.

Nothing is selected, calibrated or re-fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .train_cpa import ConditionEvaluator, _import_chemcpa, build, macro_delta_rmse, make_module


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# The baselines' drug-centred correlation is the mean of per-condition correlations
# (src/moe_drug_v1/evaluate.py). This file used to pool every condition and gene into one vector
# and correlate once, which is a different statistic -- a pooled correlation is dominated by the
# conditions with the largest response, so the two cannot be compared and the arms were being read
# against the baselines on unequal terms. The baselines' own function is imported here so there is
# one definition in the study; the pooled number is kept under a name that says what it is.
from src.moe_drug_v1.evaluate import drug_centred_correlation  # noqa: E402


def drug_centred_r(P: np.ndarray, T: np.ndarray, line: np.ndarray) -> float:
    """Mean per-condition correlation after removing each line's mean response over the drugs."""
    return float(drug_centred_correlation(P, T, line)[0])


def drug_centred_r_pooled(P: np.ndarray, T: np.ndarray, line: np.ndarray) -> float:
    """The same centring, but all conditions and genes correlated as one vector. Not comparable
    with `drug_centred_r`; reported only so the earlier figure stays traceable."""
    p, t = P.copy(), T.copy()
    for l in np.unique(line):
        m = line == l
        p[m] -= p[m].mean(0)
        t[m] -= t[m].mean(0)
    p, t = p.ravel(), t.ravel()
    if p.std() < 1e-12 or t.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def top20(P: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    direction, recall = [], []
    for p, t in zip(P, T):
        k = np.argsort(-np.abs(t))[:20]
        direction.append(float((np.sign(p[k]) == np.sign(t[k])).mean()))
        kp = set(np.argsort(-np.abs(p))[:20].tolist())
        recall.append(len(kp & set(k.tolist())) / 20)
    return float(np.mean(direction)), float(np.mean(recall))


def plate_diagnostic(P, T, line, plate, min_drugs: int = 5) -> dict:
    """Remove each (cell line, plate) block's own mean from both sides, then re-correlate.

    A fixed read-out, not a corrector: nothing is fitted and no test truth is used to build it. It
    bounds how much of the agreement survives taking out a per-block offset, which is the part a
    shared measurement environment could produce on its own.
    """
    p2, t2, keep = P.copy(), T.copy(), np.zeros(len(P), bool)
    for l in np.unique(line):
        for q in np.unique(plate[line == l]):
            m = (line == l) & (plate == q)
            if m.sum() < min_drugs:
                continue
            p2[m] -= p2[m].mean(0)
            t2[m] -= t2[m].mean(0)
            keep |= m
    if not keep.any():
        return dict(n_blocks=0)
    a, b = p2[keep].ravel(), t2[keep].ravel()
    r = 0.0 if a.std() < 1e-12 or b.std() < 1e-12 else float(np.corrcoef(a, b)[0, 1])
    return dict(n_blocks=int(sum(1 for l in np.unique(line)
                                 for q in np.unique(plate[line == l])
                                 if ((line == l) & (plate == q)).sum() >= min_drugs)),
                n_conditions=int(keep.sum()),
                plate_centred_r=round(r, 6),
                macro_delta_rmse_plate_centred=round(
                    macro_delta_rmse(p2[keep], t2[keep], line[keep]), 6))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, drug: np.ndarray, line: np.ndarray,
                     n: int = 2000, seed: int = 0) -> dict:
    """Resample drug groups, each carrying all of its cell lines, and compare two macro scores."""
    rng = np.random.default_rng(seed)
    drugs = np.unique(drug)
    diffs = []
    for _ in range(n):
        pick = rng.choice(drugs, len(drugs), replace=True)
        m = np.concatenate([np.where(drug == d)[0] for d in pick])
        diffs.append(macro_delta_rmse(a[m], b[m], line[m]))
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemcpa-src", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--freeze-manifest", required=True)
    ap.add_argument("--pred-dir", required=True,
                    help="where the test predictions go; they are arrays and stay on the server, "
                         "never in Git")
    ap.add_argument("--baselines-test", default=None,
                    help="baselines_test.json, so the arms and the minimum baselines appear in "
                         "one table scored on exactly the same conditions")
    ap.add_argument("--baseline-preds-dir", default=None,
                    help="directory holding pred_{no_change,line_mean,ridge}_test.npy; the "
                         "baselines are then scored on the arms' rows with the arms' functions")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--expert-rank", type=int, default=32)
    ap.add_argument("--max-control-cells", type=int, default=64)
    ap.add_argument("--patience", type=int, default=3,
                    help="chemCPA's own patience field; it only fills a config slot here, "
                         "because nothing in this script trains")
    ap.add_argument("--seed", type=int, default=42,
                    help="the seed the base model is built with before a checkpoint is "
                         "loaded over it; the loaded weights decide everything that is scored")
    args = ap.parse_args()

    import anndata as ad
    ChemCPA, load_dataset_splits, PerturbationDataModule = _import_chemcpa(args.chemcpa_src)
    sel = json.loads(Path(args.selection).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    cfg, datasets, dataset, dm, dcfg = build(args, ChemCPA, load_dataset_splits,
                                             PerturbationDataModule)
    adata = ad.read_h5ad(args.adata)
    drug_to_idx = {d: i for i, d in enumerate(dataset.drugs_names_unique_sorted)}
    ev = ConditionEvaluator(adata, Path(args.v2_dir), Path(args.manifest_dir), "test",
                            drug_to_idx, dataset.atomic_covars_dict["cell_type"],
                            max_control_cells=args.max_control_cells, seed=0)
    print(f"test conditions: {len(ev.rows)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results, preds_by_arm, hashes = {}, {}, {}
    truth = line = drug = None

    for arm, s in sel["selected"].items():
        if arm not in sel["aggregates"]:
            continue
        lrs = sel["aggregates"][arm]["lr_scale"]
        per_seed, preds = {}, []
        for sd in args.seeds:
            run = Path(args.runs_dir) / f"{arm}_lr{lrs}_s{sd}"
            ck = run / "best.pt"
            if not ck.exists():
                # a silently skipped seed changes the primary metric's definition, so it stops
                raise SystemExit(f"{ck} is missing; the seed set this pass was declared with "
                                 f"({args.seeds}) is not complete for {arm}")
            hashes[str(run.name)] = sha256(ck)
            lm, _, _ = make_module(ChemCPA, cfg, dcfg, arm, args)
            lm.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
            lm = lm.to(dev)
            score, P, T, L = ev.evaluate(lm.model, device=dev)
            per_seed[str(sd)] = round(score, 6)
            preds.append(P)
            truth, line = T, L
            del lm
            torch.cuda.empty_cache()
        if not preds:
            raise SystemExit(f"{arm} produced no scored seed")
        preds_by_arm[arm] = preds
        import statistics as st
        results[arm] = dict(
            lr_scale=lrs, per_seed_score=per_seed,
            seed_mean_of_scores=round(st.mean(per_seed.values()), 6),
            score_of_mean_prediction=round(macro_delta_rmse(np.mean(preds, 0), truth, line), 6))

    drug = np.array([r[1] for r in ev.rows])
    line = np.array([r[0] for r in ev.rows])

    # SNU-423, the pre-specified HCC secondary. The identifier is the one the rest of the study
    # uses (src/moe_drug_v1/data.py, src/moe_drug_v1/prepare.py); a wrong one silently produced no
    # SNU-423 row in the first test pass, so a missing line is an error here rather than a skip.
    SNU423 = "CVCL_0366"
    lines_tsv = pd.read_csv(Path(args.manifest_dir) / "lines.tsv", sep="\t").sort_values("line_index")
    snu = None
    for i, cid in enumerate(lines_tsv.cell_line_id):
        if str(cid) == SNU423:
            snu = i
    if snu is None:
        raise SystemExit(f"{SNU423} (SNU-423) is not in lines.tsv; the pre-specified secondary "
                         "cannot be computed")
    plate_of = np.load(Path(args.v2_dir) / "plate.npy")
    plate = np.array([plate_of[r[0], r[1]] for r in ev.rows])
    # Every secondary is computed the way the primary is -- once per seed, then averaged -- and the
    # ensembled figure is kept beside it under its own name. Mixing the two is what let a 0.50 %
    # ensemble margin be quoted next to a seed-mean table whose margin is 0.95 %.
    def secondaries(P: np.ndarray) -> dict:
        d_, r_ = top20(P, truth)
        out_ = dict(drug_centred_r=round(drug_centred_r(P, truth, line), 6),
                    drug_centred_r_pooled=round(drug_centred_r_pooled(P, truth, line), 6),
                    top20_direction=round(d_, 6), top20_recall=round(r_, 6),
                    plate_diagnostic=plate_diagnostic(P, truth, line, plate))
        m = line == snu
        if m.any():
            out_["snu423_delta_rmse"] = round(
                float(np.sqrt(np.mean((P[m] - truth[m]) ** 2, 1)).mean()), 6)
            out_["snu423_n_drugs"] = int(m.sum())
            out_["snu423_drug_centred_r"] = round(drug_centred_r(P[m], truth[m], line[m]), 6)
        return out_

    for arm, preds in preds_by_arm.items():
        per_seed_sec = {str(sd): secondaries(P_) for sd, P_ in zip(args.seeds, preds)}
        keys = [k for k in per_seed_sec[str(args.seeds[0])] if k != "plate_diagnostic"]
        results[arm]["secondaries_per_seed"] = per_seed_sec
        results[arm]["secondaries_seed_mean"] = {
            k: round(float(np.mean([per_seed_sec[s][k] for s in per_seed_sec])), 6)
            for k in keys}
        P = np.mean(preds, 0)
        results[arm]["secondaries_of_mean_prediction"] = secondaries(P)
        # keep the predictions, so any further read-out of this frozen pass is arithmetic on
        # what was already computed and never a second opening of the test split
        np.save(pred_dir / f"test_pred_{arm}.npy", P.astype(np.float32))
    np.save(pred_dir / "test_truth.npy", truth.astype(np.float32))
    np.save(pred_dir / "test_line.npy", line)
    np.save(pred_dir / "test_drug.npy", drug)

    # Paired bootstrap over drug groups. The pre-registered primary is the seed mean of the
    # per-seed scores, so that is what the interval is built on; the ensemble interval is kept
    # beside it, labelled, instead of standing in for it. The resample only re-indexes
    # per-condition RMSEs that are computed once, so running both is cheap.
    def rmse_rows(P_):
        return np.sqrt(np.mean((P_ - truth) ** 2, axis=1))

    rmse_by_arm = {a_: [rmse_rows(P_) for P_ in preds_by_arm[a_]] for a_ in preds_by_arm}
    rmse_ens = {a_: rmse_rows(np.mean(preds_by_arm[a_], 0)) for a_ in preds_by_arm}
    uniq_lines = np.unique(line)

    def macro_from_rows(r_, idx):
        li = line[idx]
        vals = [r_[idx][li == l_].mean() for l_ in uniq_lines if (li == l_).any()]
        return float(np.mean(vals))

    def seed_mean_from_rows(rs, idx):
        return float(np.mean([macro_from_rows(r_, idx) for r_ in rs]))

    groups = np.unique(drug)
    where_g = {g: np.where(drug == g)[0] for g in groups}
    rng = np.random.default_rng(0)
    resamples = [np.concatenate([where_g[g] for g in rng.choice(groups, len(groups), replace=True)])
                 for _ in range(2000)]
    all_idx = np.arange(len(truth))

    boots: dict = {"primary_seed_mean_of_scores": {}, "ensemble_of_mean_prediction": {}}
    arms = [a_ for a_ in ["P0", "P1", "P2", "P3"] if a_ in preds_by_arm]
    for i, a_ in enumerate(arms):
        for b_ in arms[i + 1:]:
            pair = {
                "primary_seed_mean_of_scores":
                    lambda idx, a_=a_, b_=b_: (seed_mean_from_rows(rmse_by_arm[a_], idx),
                                               seed_mean_from_rows(rmse_by_arm[b_], idx)),
                "ensemble_of_mean_prediction":
                    lambda idx, a_=a_, b_=b_: (macro_from_rows(rmse_ens[a_], idx),
                                               macro_from_rows(rmse_ens[b_], idx))}
            for tag, point in pair.items():
                sa, sb = point(all_idx)
                d = np.empty(len(resamples))
                for bi, m in enumerate(resamples):
                    xa, xb = point(m)
                    d[bi] = xa - xb
                boots[tag][f"{a_}_vs_{b_}"] = dict(
                    a=round(sa, 6), b=round(sb, 6), diff=round(sa - sb, 6),
                    diff_ci=[round(float(np.percentile(d, 2.5)), 6),
                             round(float(np.percentile(d, 97.5)), 6)],
                    rel_improvement_pct=round(float((sa - sb) / sa * 100), 4))

    # Two reference points computed on *these* rows, so the arms are not compared with numbers
    # scored on a different condition set. Neither reads a test response: the zero prediction uses
    # none, and the per-line mean is the mean over that line's **training** drugs.
    if truth is None:
        raise SystemExit("no arm checkpoint was loaded; nothing to score")
    Yall = np.load(Path(args.v2_dir) / "Y.npy")
    valid = np.load(Path(args.v2_dir) / "valid.npy")
    splits_all = pd.read_csv(Path(args.manifest_dir) / "splits.tsv",
                             sep="\t").sort_values("group_index")
    train_cols = np.where(splits_all.split.to_numpy() == "train")[0]
    line_mean = np.zeros((Yall.shape[0], Yall.shape[2]), dtype=np.float64)
    for l_ in range(Yall.shape[0]):
        m_ = valid[l_, train_cols]
        if m_.any():
            line_mean[l_] = Yall[l_, train_cols[m_]].mean(0)
    ref = dict(
        no_change=round(macro_delta_rmse(np.zeros_like(truth), truth, line), 6),
        line_mean_train_only=round(
            macro_delta_rmse(np.array([line_mean[l_] for l_ in line]), truth, line), 6))
    if args.baselines_test and Path(args.baselines_test).exists():
        bt = json.loads(Path(args.baselines_test).read_text())
        ref["full_test_set_baselines"] = {
            k: v.get("macro_delta_rmse") for k, v in bt.items() if isinstance(v, dict)}
        ref["full_test_set_note"] = ("scored on every valid test condition, not only the rows the "
                                     "single-cell evaluator can form; use the two numbers above "
                                     "for a like-for-like comparison with the arms")

    # The baselines on exactly the arms' rows, through exactly the arms' functions. Their test
    # predictions were written by `moe_drug_v1.baselines --split test`; nothing is refitted here.
    # Rows are matched by (line index, drug column), which both sides index from the same
    # lines.tsv and splits.tsv, and every arm row must be found or the script stops.
    baselines_same_rows = {}
    if args.baseline_preds_dir:
        bdir = Path(args.baseline_preds_dir)
        cols = np.where(splits_all.split.to_numpy() == "test")[0]
        li_, dj_ = np.where(valid[:, cols])
        pos = {(int(a_), int(cols[b_])): k for k, (a_, b_) in enumerate(zip(li_, dj_))}
        take = [pos.get((int(r[0]), int(r[1]))) for r in ev.rows]
        if any(t is None for t in take):
            raise SystemExit("a scored arm row has no baseline prediction")
        take = np.array(take)
        for name in ["no_change", "line_mean", "ridge"]:
            f = bdir / f"pred_{name}_test.npy"
            if not f.exists():
                raise SystemExit(f"{f} is missing")
            Pb = np.load(f)
            if len(Pb) != len(li_):
                raise SystemExit(f"{f} has {len(Pb)} rows, the test condition table has {len(li_)}")
            Pb = Pb[take].astype(np.float64)
            b = dict(macro_delta_rmse=round(macro_delta_rmse(Pb, truth, line), 6))
            b.update(secondaries(Pb))
            baselines_same_rows[name] = b

    rep = dict(n_test_conditions=len(ev.rows), results=results, reference=ref,
               baselines_same_rows=baselines_same_rows, bootstrap=boots,
               checkpoint_sha256=hashes,
               note="one pass, no selection, no calibration; percentages are relative to the "
                    "first-named arm")
    (out / "test_results.json").write_text(json.dumps(rep, indent=1))
    Path(args.freeze_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.freeze_manifest).write_text(json.dumps(
        dict(checkpoint_sha256=hashes, selection=sel["selected"],
             n_test_conditions=len(ev.rows)), indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
