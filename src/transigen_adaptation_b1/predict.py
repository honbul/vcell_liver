"""One code path that produces every model's condition-level predicted delta for a target split,
on validation (pre-freeze, also selects the baseline hyper-parameters) or on test (final, selections given)."""
import json, os
import numpy as np
import torch

from . import tsg_adapter as T
from .adapt_target import source_checkpoint, run_id, ARMS
from .baselines import ALPHAS, fit_target_ridge
from .data import condition_delta, MAN_DIR
from .evaluate import per_condition

LAMBDAS = [0, 0.25, 0.5, 1]
SEEDS = [42, 43, 44]


def load_run(rid):
    p = os.path.join(os.environ["TB_RUNS"], "adapt", rid, "result.json")
    assert os.path.exists(p), f"expected run is missing: {rid}"          # never skip silently
    r = json.load(open(p))
    assert T.sha256(r["checkpoint"]) == r["checkpoint_sha256"], f"checkpoint of {rid} changed"
    return r


def predictions(D, S, target, split, k, role, dev, lr_by_arm, max_updates, selection=None, arms=ARMS, extra_runs=None):
    """Returns (condition table, truth delta, {(model, seed): pred delta}, selection, run records).

    role = 'val': baseline hyper-parameters are chosen here on target_val and returned.
    role = 'test': `selection` must be given; test truth is requested with purpose 'final_test'.
    """
    assert role in ("val", "test")
    purpose = "val" if role == "val" else "final_test"
    if role == "test":
        assert selection is not None, "test predictions need frozen selections"
    di = D.rows.drug_idx.to_numpy()
    rows_a = np.sort(D.target_rows(target, split, "adapt", k))
    rows_e = np.sort(D.target_rows(target, split, role))
    x0_a, x0_e = np.asarray(D.x0[rows_a]), np.asarray(D.x0[rows_e])
    fp_a, fp_e = D.fp[di[rows_a]], D.fp[di[rows_e]]
    meta_a, meta_e = D.rows.iloc[rows_a], D.rows.iloc[rows_e]
    d_a = D.target_x1(rows_a, target, split, "adapt") - x0_a
    cond_a, truth_a = condition_delta(meta_a, d_a)
    cond_e, truth_e = condition_delta(meta_e, D.target_x1(rows_e, target, split, purpose) - x0_e)
    cd = lambda rows_delta, meta=meta_e: condition_delta(meta, rows_delta)[1]
    rmse = lambda p: float(per_condition(p, truth_e).delta_rmse.mean())
    sel = dict(selection or {})
    choose = selection is None            # hyper-parameters are chosen only when none are supplied (validation stage)
    assert not (choose and role == "test")
    P = {}
    n = len(cond_e)
    P[("no_change", "-")] = np.zeros((n, 978))
    P[("target_adapt_mean", "-")] = np.repeat(truth_a.mean(0, keepdims=True), n, 0)
    P[("source_same_drug_mean", "-")] = S.same_drug(cond_e.drug_group)
    P[("source_ridge", "-")] = cd(S.ridge_rows(x0_e, fp_e))
    F_a, F_e = S.features(x0_a, fp_a), S.features(x0_e, fp_e)
    if choose:
        sc = {al: rmse(cd(fit_target_ridge(F_a, d_a, al)(F_e))) for al in ALPHAS}
        sel["target_ridge_alpha"] = min(ALPHAS, key=lambda a_: (sc[a_], a_))
        sel["target_ridge_val"] = {str(a_): v for a_, v in sc.items()}
    P[("target_ridge", "-")] = cd(fit_target_ridge(F_a, d_a, sel["target_ridge_alpha"])(F_e))

    tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    def nn_delta(model, x0, fp, meta):
        p = T.predict_x1_mean_path(model, tt(x0), tt(fp)).cpu().numpy()      # inputs: X0 and fingerprint only
        return condition_delta(meta, p - x0)[1]
    runs = []
    for seed in SEEDS:
        e = source_checkpoint(seed)
        m, _ = T.load_checkpoint(e["checkpoint"], dev, seed=seed)
        b0_e, b0_a = nn_delta(m, x0_e, fp_e, meta_e), nn_delta(m, x0_a, fp_a, meta_a)
        P[("B0", seed)] = b0_e
        off = (truth_a - b0_a).mean(0, keepdims=True)
        P[("b0_mean_offset", seed)] = b0_e + off
        if choose:
            sc = {lam: rmse(b0_e + lam * off) for lam in LAMBDAS}
            sel[f"offset_lambda_seed{seed}"] = min(LAMBDAS, key=lambda l_: (sc[l_], l_))
            sel[f"offset_lambda_val_seed{seed}"] = {str(l_): v for l_, v in sc.items()}
        P[("b0_mean_offset_shrunk", seed)] = b0_e + sel[f"offset_lambda_seed{seed}"] * off
        for arm in arms:
            r = load_run(run_id(target, split, k, arm, lr_by_arm[arm], seed, max_updates))
            assert r["adapt_rows_hash"] == __import__("hashlib").sha256(rows_a.astype(np.int64).tobytes()).hexdigest()[:16]
            m, _ = T.load_checkpoint(r["checkpoint"], dev, seed=seed)
            P[(arm, seed)] = nn_delta(m, x0_e, fp_e, meta_e)
            if role == "val":      # the recomputed validation prediction must reproduce what the run stored
                stored = np.load(os.path.join(os.path.dirname(r["checkpoint"]), "pred_val_delta.npy"))
                assert np.allclose(stored, P[(arm, seed)], atol=1e-4), f"validation prediction of {r['run_id']} not reproduced"
            runs.append(r)
    for name, rid in (extra_runs or {}).items():
        r = load_run(rid)
        m, _ = T.load_checkpoint(r["checkpoint"], dev, seed=r["seed"])
        P[(name, r["seed"])] = nn_delta(m, x0_e, fp_e, meta_e)
        runs.append(r)
    for arm in ("B0",) + tuple(arms):                     # ensembles are separate entries, never mixed with seed means
        P[(arm + "_ensemble3", "ens")] = np.mean([P[(arm, s)] for s in SEEDS], 0)
    for key, p in P.items():
        assert p.shape == truth_e.shape and np.isfinite(p).all(), key
    info = dict(n_adapt_conditions=int(len(cond_a)), n_adapt_rows=int(len(rows_a)), n_eval_conditions=int(n))
    return cond_e, truth_e, P, sel, runs, info
