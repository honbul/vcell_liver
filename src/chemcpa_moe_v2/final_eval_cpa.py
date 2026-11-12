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


def drug_centred_r(P: np.ndarray, T: np.ndarray, line: np.ndarray) -> float:
    """Correlation after removing each cell line's mean response, so a per-line constant scores 0."""
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
    ap.add_argument("--baselines-test", default=None,
                    help="baselines_test.json, so the arms and the minimum baselines appear in "
                         "one table scored on exactly the same conditions")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--expert-rank", type=int, default=32)
    ap.add_argument("--max-control-cells", type=int, default=64)
    args = ap.parse_args()

    import anndata as ad
    ChemCPA, load_dataset_splits, PerturbationDataModule = _import_chemcpa(args.chemcpa_src)
    sel = json.loads(Path(args.selection).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg, datasets, dataset, dm, dcfg = build(args, ChemCPA, load_dataset_splits,
                                             PerturbationDataModule)
    adata = ad.read_h5ad(args.adata)
    drug_to_idx = {d: i for i, d in enumerate(dataset.drugs_names_unique_sorted)}
    ev = ConditionEvaluator(adata, Path(args.v2_dir), Path(args.manifest_dir), "test",
                            drug_to_idx, max_control_cells=args.max_control_cells, seed=0)
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
                continue
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
            continue
        preds_by_arm[arm] = preds
        import statistics as st
        results[arm] = dict(
            lr_scale=lrs, per_seed_score=per_seed,
            seed_mean_of_scores=round(st.mean(per_seed.values()), 6),
            score_of_mean_prediction=round(macro_delta_rmse(np.mean(preds, 0), truth, line), 6))

    drug = np.array([r[1] for r in ev.rows])
    line = np.array([r[0] for r in ev.rows])

    # SNU-423, the pre-specified secondary
    lines_tsv = pd.read_csv(Path(args.manifest_dir) / "lines.tsv", sep="\t").sort_values("line_index")
    snu = None
    for i, cid in enumerate(lines_tsv.cell_line_id):
        if str(cid) == "CVCL_0564":          # SNU-423
            snu = i
    plate_of = np.load(Path(args.v2_dir) / "plate.npy")
    plate = np.array([plate_of[r[0], r[1]] for r in ev.rows])
    for arm, preds in preds_by_arm.items():
        P = np.mean(preds, 0)
        results[arm]["plate_diagnostic"] = plate_diagnostic(P, truth, line, plate)
        results[arm]["drug_centred_r"] = round(drug_centred_r(P, truth, line), 6)
        d, r_ = top20(P, truth)
        results[arm]["top20_direction"] = round(d, 6)
        results[arm]["top20_recall"] = round(r_, 6)
        if snu is not None and (line == snu).any():
            m = line == snu
            results[arm]["snu423_delta_rmse"] = round(
                float(np.sqrt(np.mean((P[m] - truth[m]) ** 2, 1)).mean()), 6)

    boots = {}
    arms = [a_ for a_ in ["P0", "P1", "P2", "P3"] if a_ in preds_by_arm]
    for i, a_ in enumerate(arms):
        for b_ in arms[i + 1:]:
            A = np.mean(preds_by_arm[a_], 0)
            B = np.mean(preds_by_arm[b_], 0)
            rng = np.random.default_rng(0)
            groups = np.unique(drug)
            d = []
            for _ in range(2000):
                pick = rng.choice(groups, len(groups), replace=True)
                m = np.concatenate([np.where(drug == g)[0] for g in pick])
                d.append(macro_delta_rmse(A[m], truth[m], line[m])
                         - macro_delta_rmse(B[m], truth[m], line[m]))
            d = np.array(d)
            boots[f"{a_}_vs_{b_}"] = dict(
                diff=round(float(macro_delta_rmse(A, truth, line)
                                 - macro_delta_rmse(B, truth, line)), 6),
                diff_ci=[round(float(np.percentile(d, 2.5)), 6),
                         round(float(np.percentile(d, 97.5)), 6)],
                rel_improvement_pct=round(float(
                    (macro_delta_rmse(A, truth, line) - macro_delta_rmse(B, truth, line))
                    / macro_delta_rmse(A, truth, line) * 100), 4))

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

    rep = dict(n_test_conditions=len(ev.rows), results=results, reference=ref, bootstrap=boots,
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
