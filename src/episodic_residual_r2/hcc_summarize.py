"""Tables from the per-unit HCC scores: per target and K, every method, drug-group bootstrap intervals, paired
contrasts against the references, and the equal-weight mean over the three targets. No prediction is recomputed."""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import runs, public, dump, sha256

METRICS = ["delta_rmse", "centred_rmse", "centred_pearson", "delta_pearson", "top20_direction", "mrr", "mrr_centred"]
REFS = ["mean_offset", "source_same_drug_mean",
        # added after the frozen evaluation, summary only (no prediction recomputed): references needed to read plan section 12
        "allfeat_krr", "shrunk_mean_offset", "residual_adapter__zero_residual"]


def boot_weights(groups_by_split, n_boot, seed):
    """Cluster bootstrap over the union of query drug groups: a drawn group carries all its split appearances."""
    union = sorted(set().union(*[set(g) for g in groups_by_split.values()]))
    pos = {g: i for i, g in enumerate(union)}
    rs = np.random.RandomState(seed)
    W = rs.multinomial(len(union), np.ones(len(union)) / len(union), size=n_boot).astype(np.float64)       # (n_boot, n_union)
    return {s: W[:, [pos[g] for g in gs]] for s, gs in groups_by_split.items()}


def target_table(d, n_boot, seed):
    """d: per-unit rows of one target, one K, one query set, model seeds already averaged. Returns point estimates and bootstrap draws per method."""
    splits = sorted(d.split_seed.unique())
    piv = {s: d[d.split_seed == s].pivot(index="drug_group", columns="method", values=METRICS) for s in splits}
    gbs = {s: list(piv[s].index) for s in splits}
    W = boot_weights(gbs, n_boot, seed)
    methods = sorted(d.method.unique())
    point, draws = {}, {}
    for m in methods:
        for v in METRICS:
            pts, bs = [], []
            for s in splits:
                x = piv[s][(v, m)].to_numpy(float)
                ok = np.isfinite(x)
                pts.append(x[ok].mean())
                w = W[s][:, ok]
                den = w.sum(1)
                bs.append(np.where(den > 0, (w * x[ok]).sum(1) / np.maximum(den, 1e-12), np.nan))
            point[(m, v)] = float(np.mean(pts))
            draws[(m, v)] = np.nanmean(np.stack(bs), 0)
    n_groups = len(set().union(*[set(g) for g in gbs.values()]))
    return methods, point, draws, n_groups, {int(s): len(g) for s, g in gbs.items()}


def ci(x):
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--kind", default="hcc")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    src_dir = runs(a.kind, a.run_id)
    res = json.load(open(os.path.join(src_dir, "result.json")))
    assert sha256(res["per_unit_table"]) == res["per_unit_sha256"]
    df = pd.read_csv(res["per_unit_table"], sep="\t")
    df["mrr"] = 1 / df.retrieval_rank
    df["mrr_centred"] = 1 / df.retrieval_rank_centred
    by_seed = df.groupby(["target", "query_set", "K", "method", "model_seed"])[METRICS].mean().reset_index()
    # model seeds averaged per unit (seeded methods), closed-form methods have a single row
    u = df.groupby(["target", "query_set", "split_seed", "K", "method", "drug_group", "is_test"])[METRICS + ["truth_rms", "pred_rms"]].mean().reset_index()
    rows, contrasts, eq_draws, eq_point = [], [], {}, {}
    for (target, qset, K), d in u.groupby(["target", "query_set", "K"]):
        methods, point, draws, n_groups, n_by_split = target_table(d, a.n_boot, a.seed)
        for m in methods:
            r = dict(target=target, query_set=qset, K=K, method=m, n_query_groups_union=n_groups, n_query_per_split=json.dumps(n_by_split))
            for v in METRICS:
                lo, hi = ci(draws[(m, v)])
                r.update({v: point[(m, v)], v + "_lo": lo, v + "_hi": hi})
                eq_draws.setdefault((qset, K, m, v), {})[target] = draws[(m, v)]
                eq_point.setdefault((qset, K, m, v), {})[target] = point[(m, v)]
            rows.append(r)
            for ref in REFS:
                if m == ref or ref not in methods:
                    continue
                for v in METRICS:
                    dd = draws[(m, v)] - draws[(ref, v)]
                    lo, hi = ci(dd)
                    contrasts.append(dict(target=target, query_set=qset, K=K, method=m, reference=ref, metric=v, diff=point[(m, v)] - point[(ref, v)], lo=lo, hi=hi))
                    eq_draws.setdefault((qset, K, m, v, ref), {})[target] = dd
                    eq_point.setdefault((qset, K, m, v, ref), {})[target] = point[(m, v)] - point[(ref, v)]
    targets = sorted(u.target.unique())
    for key, per_t in eq_draws.items():
        if set(per_t) != set(targets):
            continue                                              # a K that one target cannot supply is not averaged
        dr = np.mean([per_t[t] for t in targets], 0)              # independent resampling inside each target
        pt = float(np.mean([eq_point[key][t] for t in targets]))
        lo, hi = ci(dr)
        if len(key) == 4:
            qset, K, m, v = key
            rows.append(dict(target="EQUAL_MEAN_3", query_set=qset, K=K, method=m, **{v: pt, v + "_lo": lo, v + "_hi": hi}))
        else:
            qset, K, m, v, ref = key
            contrasts.append(dict(target="EQUAL_MEAN_3", query_set=qset, K=K, method=m, reference=ref, metric=v, diff=pt, lo=lo, hi=hi))
    tab = pd.DataFrame(rows)
    tab = tab.groupby(["target", "query_set", "K", "method"], sort=False).first().reset_index()            # merges the per-metric rows of EQUAL_MEAN_3
    con = pd.DataFrame(contrasts)
    pub = lambda n: public("predictions" if n.startswith("per_unit") else "manifests", "episodic_residual_r2", n)
    tag = "hcc" if a.kind == "hcc" else a.kind
    tab.to_csv(pub(f"{tag}_results.tsv"), sep="\t", index=False, float_format="%.6g")
    con.to_csv(pub(f"{tag}_contrasts.tsv"), sep="\t", index=False, float_format="%.6g")
    by_seed.to_csv(pub(f"{tag}_by_model_seed.tsv"), sep="\t", index=False, float_format="%.6g")
    if a.kind == "hcc":
        for t, d in u.groupby("target"):
            d.to_csv(pub(f"per_unit_metrics_{t}.tsv"), sep="\t", index=False, float_format="%.5g")
        dump(dict(run_id=a.run_id, note="Prediction arrays stay on the server; only paths and SHA-256 are published.", prediction_files=res["prediction_files"],
                  per_unit_all_seeds=dict(path=res["per_unit_table"], sha256=res["per_unit_sha256"]), label_budget=res["label_budget"], failures=res["failures"],
                  frozen_code_commit=res["frozen_code_commit"], eval_commit=res["commit"], eval_fix=res["eval_fix"], slurm_job=res["slurm_job"]),
             public("predictions", "episodic_residual_r2", "prediction_files.json"))
    pd.set_option("display.width", 250, "display.max_rows", 500)
    show = tab[(tab.query_set == "test")][["target", "K", "method", "delta_rmse", "delta_rmse_lo", "delta_rmse_hi", "centred_pearson", "mrr_centred"]]
    print(show.to_string())
    print(con[(con.query_set == "test") & (con.metric == "delta_rmse") & (con.reference == "mean_offset")].to_string())


if __name__ == "__main__":
    main()
