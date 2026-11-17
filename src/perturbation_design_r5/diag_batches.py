"""Dev-only diagnostic of selected batches: within-batch similarity, plate concentration, predicted noise of picks."""
import argparse, json, os
import numpy as np
import pandas as pd

from .common import runs, public, protocol
from .data import Store, make_episode
from .models import EpisodeKernels, nu_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hyper", required=True)
    a = ap.parse_args()
    st = Store()
    pr = protocol()
    hp = json.load(open(a.hyper))
    sq = pd.read_json(os.path.join(runs(), a.tag, "sequences.jsonl"), lines=True)
    out = []
    cache = {}
    for _, r in sq.iterrows():
        key = (r.cell, r.seed)
        if key not in cache:
            ep = make_episode(st, r.role, r.cell, r.family, int(r.seed), pr=pr)
            ek = EpisodeKernels(ep, st.bases)
            cache = {key: (ep, ek, nu_vector(ep, ek, hp), pd.Series(np.arange(len(ep.groups)), index=ep.groups))}
        ep, ek, nu, pos = cache[key]
        idx = pos[r.sequence].to_numpy()
        for b in range(1, len(idx) // 4):
            bi = idx[4 * b:4 * b + 4]
            iu = np.triu_indices(len(bi), 1)
            out.append(dict(family=r.family, seed=r.seed, policy=r.policy, batch=b,
                            tan_within=ek.tan[np.ix_(bi, bi)][iu].mean(), cos_within=ek.cos[np.ix_(bi, bi)][iu].mean(),
                            tan_to_prev=ek.tan[np.ix_(bi, idx[:4 * b])].max(1).mean(),
                            plates=ep.rep.det_plate.iloc[bi].nunique(), nu_pick=nu[bi].mean(), nu_pool=nu[ep.pool].mean(),
                            ctl_sd_pick=ep.rep.ctl_sd.iloc[bi].mean()))
    df = pd.DataFrame(out)
    tab = df.groupby(["family", "policy"]).mean(numeric_only=True).groupby("policy").mean().drop(columns=["seed", "batch"])
    tab.to_csv(public("manifests", "perturbation_design_r5", "develop", f"batch_diag_{a.tag}.tsv"), sep="\t", float_format="%.4f")
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
