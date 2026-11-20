"""Tables for the report, from the committed per-condition test scores (no model or expression file needed).

Seed means and ensembles are kept in separate tables. Intervals are drug-group block bootstraps within a
target, conditional on the drugs measured in that target; they are not intervals over HCC lines.
"""
import argparse, json, os
import numpy as np
import pandas as pd

REP = os.path.join(os.environ.get("TB_REP_DIR", "reports/transigen_adaptation_b1"), "tables")
METRICS = ["delta_rmse", "delta_pearson", "centred_pearson", "top20_direction", "top20_recall", "retrieval_mrr", "retrieval_top5"]
SEEDED = ["B0", "B1_scratch", "B2_finetune", "b0_mean_offset", "b0_mean_offset_shrunk"]
BASE = ["no_change", "target_adapt_mean", "source_same_drug_mean", "source_ridge", "target_ridge"]
ORDER = ["B0", "B1_scratch", "B2_finetune"] + BASE + ["b0_mean_offset", "b0_mean_offset_shrunk"]


def load():
    pc = pd.read_csv(os.path.join(REP, "test_per_condition.tsv.gz"), sep="\t", dtype={"seed": str})
    pc["retrieval_mrr"] = 1 / pc.retrieval_rank
    pc["retrieval_top5"] = (pc.retrieval_rank <= 5).astype(float)
    return pc


def boot_weights(groups_all, n_boot, seed):
    rs = np.random.RandomState(seed)
    g = np.array(sorted(groups_all))
    return g, rs.multinomial(len(g), np.ones(len(g)) / len(g), size=n_boot)      # (B, n_groups) multiplicities


def model_scores(pc_t, models, metric, W=None, g_index=None):
    """Score per model for one target: weighted condition mean within (split, seed) -> mean over seeds -> mean over splits.

    With W (B x n_groups) returns a (B, n_models) array; without, a vector. The same group draw is applied to
    every split, seed and model, so all contrasts are paired.
    """
    out = []
    for m in models:
        d = pc_t[pc_t.model == m]
        per_split = []
        for s, ds in d.groupby("split_seed"):
            per_seed = []
            for sd, x in ds.groupby("seed"):
                v = x[metric].to_numpy(float)
                ok = np.isfinite(v)
                if W is None:
                    per_seed.append(v[ok].mean() if ok.any() else np.nan)
                else:
                    w = W[:, [g_index[g] for g in x.drug_group]].astype(float)
                    den = (w * ok).sum(1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        per_seed.append(np.where(den > 0, (w * np.where(ok, v, 0)).sum(1) / den, np.nan))
            per_split.append(np.mean(per_seed, 0))
        out.append(np.nanmean(np.stack(per_split), 0) if W is not None else np.nanmean(per_split))
    return np.stack(out, -1) if W is not None else np.array(out)


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    pc = load()
    main_pc = pc[pc.main == 1]
    targets = sorted(main_pc.target.unique())
    models = [m for m in ORDER if m in set(main_pc.model)] + sorted(m for m in set(main_pc.model) if m.startswith("B2_label_permuted"))
    prim, sec, con, var = [], [], [], []
    boot_store = {}
    for t in targets:
        d = main_pc[main_pc.target == t]
        g, W = boot_weights(set(d.drug_group), a.n_boot, a.seed)
        gi = {x: i for i, x in enumerate(g)}
        for metric in METRICS:
            ms = [m for m in models if m in set(d.model)]
            pt = model_scores(d, ms, metric)
            bt = model_scores(d, ms, metric, W, gi)
            if metric == "delta_rmse":
                boot_store[t] = (ms, pt, bt)
            for i, m in enumerate(ms):
                lo, hi = ci(bt[:, i])
                row = dict(target=t, model=m, metric=metric, score=pt[i], ci_lo=lo, ci_hi=hi)
                (prim if metric == "delta_rmse" else sec).append(row)
            for ref in ["B0", "B1_scratch"] + BASE + ["b0_mean_offset", "b0_mean_offset_shrunk"] + [m for m in ms if m.startswith("B2_label")]:
                if ref in ms and "B2_finetune" in ms and not ref.startswith("B2_label"):
                    i, j = ms.index("B2_finetune"), ms.index(ref)
                    lo, hi = ci(bt[:, i] - bt[:, j])
                    con.append(dict(target=t, contrast=f"B2_finetune - {ref}", metric=metric, primary=ref in ("B0", "B1_scratch"),
                                    diff=pt[i] - pt[j], ci_lo=lo, ci_hi=hi, lower_is_better=metric == "delta_rmse"))
            if "b0_mean_offset" in ms and "B0" in ms:
                i, j = ms.index("b0_mean_offset"), ms.index("B0")
                lo, hi = ci(bt[:, i] - bt[:, j])
                con.append(dict(target=t, contrast="b0_mean_offset - B0", metric=metric, primary=False, diff=pt[i] - pt[j], ci_lo=lo, ci_hi=hi,
                                lower_is_better=metric == "delta_rmse"))
        # seed / split variation of the primary metric, separately from the drug bootstrap
        r = d.groupby(["model", "split_seed", "seed"]).delta_rmse.mean().reset_index()
        for m, x in r.groupby("model"):
            var.append(dict(target=t, model=m, n_runs=len(x), mean=x.delta_rmse.mean(), sd_over_split_seed_runs=x.delta_rmse.std(ddof=1) if len(x) > 1 else np.nan,
                            min=x.delta_rmse.min(), max=x.delta_rmse.max(),
                            sd_over_seeds_within_split=x.groupby("split_seed").delta_rmse.std(ddof=1).mean() if x.seed.nunique() > 1 else np.nan,
                            sd_over_splits_of_seed_mean=x.groupby("split_seed").delta_rmse.mean().std(ddof=1)))
    # equal mean over targets (descriptive; bootstraps are independent per target)
    eq = []
    common = [m for m in models if all(m in boot_store[t][0] for t in targets)]
    for m in common:
        pt = np.mean([boot_store[t][1][boot_store[t][0].index(m)] for t in targets])
        bt = np.mean([boot_store[t][2][:, boot_store[t][0].index(m)] for t in targets], 0)
        lo, hi = ci(bt)
        eq.append(dict(model=m, metric="delta_rmse", score=pt, ci_lo=lo, ci_hi=hi, n_targets=len(targets)))
    for ref in common:
        if ref != "B2_finetune" and "B2_finetune" in common:
            f = lambda k: np.mean([boot_store[t][k][..., boot_store[t][0].index("B2_finetune")] - boot_store[t][k][..., boot_store[t][0].index(ref)] for t in targets], 0)
            lo, hi = ci(f(2))
            eq.append(dict(model=f"B2_finetune - {ref}", metric="delta_rmse", score=float(f(1)), ci_lo=lo, ci_hi=hi, n_targets=len(targets)))
    # ensembles and other per-run summaries from the by-run table
    runs = pd.read_csv(os.path.join(REP, "test_scores_by_run.tsv"), sep="\t", dtype={"seed": str})
    ens = runs[(runs.main == 1) & runs.model.str.endswith("_ensemble3")].groupby(["target", "model"])[["delta_rmse", "delta_pearson", "centred_pearson"]].mean().reset_index()
    mag = runs[(runs.main == 1) & ~runs.model.str.endswith("_ensemble3")].groupby(["target", "model"])[
        ["magnitude_ratio_gene_median", "magnitude_ratio_global", "retrieval_mrr", "shuffled_mrr", "retrieval_top5", "shuffled_top5", "n_conditions"]].mean().reset_index()
    first = runs.split_seed.min()
    cur = runs[(runs.split_seed == first) & ~runs.model.str.endswith("_ensemble3") & ~runs.model.str.startswith("B2_label")]
    curve = cur.groupby(["target", "k", "model"]).agg(delta_rmse=("delta_rmse", "mean"), sd_over_seeds=("delta_rmse", "std"), n_seeds=("delta_rmse", "size"),
                                                      delta_pearson=("delta_pearson", "mean"), centred_pearson=("centred_pearson", "mean"),
                                                      n_adapt_conditions=("n_adapt_conditions", "first"), n_test_conditions=("n_conditions", "first")).reset_index()
    for name, rows in [("primary_delta_rmse", prim), ("secondary_metrics", sec), ("contrasts", con), ("seed_split_variation", var), ("equal_target_mean", eq)]:
        pd.DataFrame(rows).to_csv(os.path.join(REP, f"{name}.tsv"), sep="\t", index=False, float_format="%.5f")
    ens.to_csv(os.path.join(REP, "ensemble_scores_separate.tsv"), sep="\t", index=False, float_format="%.5f")
    mag.to_csv(os.path.join(REP, "magnitude_and_retrieval.tsv"), sep="\t", index=False, float_format="%.5f")
    curve.to_csv(os.path.join(REP, "learning_curve_k.tsv"), sep="\t", index=False, float_format="%.5f")
    print(pd.DataFrame(prim).pivot(index="model", columns="target", values="score").reindex(models).round(4).to_string())
    print(pd.DataFrame(con).query("metric == 'delta_rmse'").round(4).to_string())


if __name__ == "__main__":
    main()
