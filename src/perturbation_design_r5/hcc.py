"""Frozen evaluation on HUH7 / JHH5 / JHH7 (once), or the same code path on meta-validation cells with --mock.
HCC treated values open only with R5_FINAL_EVAL=1 and a committed freeze manifest (oracle._hcc_allowed)."""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import public, dump_json, runs, sha_file, MAN, commit
from .oracle import FREEZE, _hcc_allowed
from .replay import replay, HYPER
from .evaluate import episode_table, summary

BOOT = dict(n=2000, seed=20261117)


def group_bootstrap(pu, refs, policies):
    """Per line: resample the line's query groups (union over seeds) with replacement; each episode's K-error is the
    multiplicity-weighted mean over its query groups; AUC over K; mean over seeds; paired difference vs reference."""
    rs = np.random.RandomState(BOOT["seed"])
    out, draws = [], {}
    for cell, g in pu.groupby("cell"):
        groups = np.array(sorted(g.group.unique()))
        gi = pd.Series(np.arange(len(groups)), index=groups)
        W = np.zeros((BOOT["n"] + 1, len(groups)))
        W[0] = 1
        W[1:] = np.stack([np.bincount(rs.randint(0, len(groups), len(groups)), minlength=len(groups)) for _ in range(BOOT["n"])])
        aucs = {}
        for pol, gp in g.groupby("policy"):
            per_seed = []
            for seed, gs in gp.groupby("seed"):
                ks = sorted(gs.K.unique())
                errs = []
                for K in ks:
                    gk = gs[gs.K == K]
                    m = np.zeros(len(groups))
                    m[gi[gk.group].to_numpy()] = gk.rmse.to_numpy()
                    mask = np.zeros(len(groups))
                    mask[gi[gk.group].to_numpy()] = 1
                    w = W * mask
                    errs.append((w * m).sum(1) / np.maximum(w.sum(1), 1e-12))
                errs = np.stack(errs, 1)
                per_seed.append(np.trapz(errs, ks, axis=1) / (ks[-1] - ks[0]))
            aucs[pol] = np.mean(per_seed, 0)
        draws[cell] = aucs
        for ref in refs:
            for pol in policies:
                if pol == ref or pol not in aucs or ref not in aucs:
                    continue
                d = aucs[pol] - aucs[ref]
                out.append(dict(cell=cell, policy=pol, reference=ref, diff=d[0], lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
    cells = sorted(draws)
    for ref in refs:
        for pol in policies:
            if pol == ref or not all(pol in draws[c] and ref in draws[c] for c in cells):
                continue
            d = np.mean([draws[c][pol] - draws[c][ref] for c in cells], 0)
            out.append(dict(cell="HCC3_equal_mean" if len(cells) == 3 else "equal_mean", policy=pol, reference=ref, diff=d[0],
                            lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="same code path on meta-validation cells; no HCC value opened")
    a = ap.parse_args()
    fr = json.load(open(FREEZE))
    assert sha_file(HYPER) == fr["hyper_sha256"], "hyper.json differs from the frozen one"
    pols = fr["policies"]
    seeds = fr["seeds"]
    if a.mock:
        role, tag = "meta_val", "hcc_mock_v1"
        cells = fr.get("mock_cells")
    else:
        _hcc_allowed()
        role, tag = "hcc", fr["hcc_run_tag"]
        cells = None
        assert not os.path.exists(os.path.join(runs(), tag)), "the HCC evaluation has already been run"
    rows, pu, seqs, info = replay(role, pols, seeds, tag, stress=False, cells=cells)
    ept = episode_table(rows)
    d = public("manifests", "perturbation_design_r5", "runs", tag)
    os.makedirs(d, exist_ok=True)
    ept.to_csv(os.path.join(d, "episodes.tsv"), sep="\t", index=False, float_format="%.5f")
    per_line = ept.groupby(["cell", "readout", "policy"])[["auc", "rmse_K4", "rmse_K8", "rmse_K16", "rmse_K32", "pearson_K16", "cpearson_K16"]].mean().reset_index()
    per_line.to_csv(os.path.join(d, "per_line.tsv"), sep="\t", index=False, float_format="%.5f")
    eq = per_line.groupby(["readout", "policy"]).mean(numeric_only=True).reset_index()
    eq.to_csv(os.path.join(d, "equal_mean.tsv"), sep="\t", index=False, float_format="%.5f")
    con = group_bootstrap(pu, ["random", "diversity", "fixed_vr"], pols)
    con.to_csv(os.path.join(d, "contrasts_group_bootstrap.tsv"), sep="\t", index=False, float_format="%.5f")
    sq = pd.DataFrame(seqs)
    sq[["role", "cell", "seed", "policy", "sequence"]].to_json(os.path.join(d, "sequences.jsonl"), orient="records", lines=True)
    sq.groupby(["cell", "policy"])[["trt_wells", "plates", "ctl_wells_overhead", "wall_s"]].mean().to_csv(os.path.join(d, "cost.tsv"), sep="\t", float_format="%.3f")
    dump_json(dict(info, freeze_commit=fr.get("code_commit"), run_commit=commit()), os.path.join(d, "info.json"))
    with pd.option_context("display.width", 200):
        print(eq[eq.readout == "main"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(con.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
