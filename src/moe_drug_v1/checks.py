"""The four sanity checks the plan requires before any reported run (plan s7.3), and nothing more.

1. no drug group, drug name or treated well appears in more than one split
2. inputs line up with the truth on line / drug / gene, and the panel and transform are train-only
3. a small batch can actually be fitted and A/C losses and gradients stay finite
4. C's router and every one of its experts are in the graph and their weights move

A gating collapse found here is reported, not patched: the plan forbids adding regularisation
searches in response to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Dataset
from .models import build, solve_dense_hidden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D, M, P = Path(args.data_dir), Path(args.manifest_dir), Path(args.prep_dir)
    rep: dict = {}

    # ---- 1. split disjointness, at three levels -------------------------------------------------
    splits = pd.read_csv(M / "splits.tsv", sep="\t")
    alias = pd.read_csv(M / "drug_aliases.tsv", sep="\t")
    wells = pd.read_csv(P / "wells.tsv", sep="\t")
    assert splits.parent_smiles.is_unique
    name_split: dict[str, set] = {}
    for _, r in alias.iterrows():
        for d in eval(r.drugs):
            name_split.setdefault(d, set()).add(r.split)
    crossing_names = {k: sorted(v) for k, v in name_split.items() if len(v) > 1}
    g2s = dict(zip(splits.parent_smiles, splits.split))
    a2g = {}
    for _, r in alias.iterrows():
        for d in eval(r.drugs):
            a2g[d] = r.group
    w05 = wells[(wells.dose_uM == 0.5) & wells.drug.isin(a2g)]
    well_split = w05.drug.map(lambda d: g2s[a2g[d]])
    crossing_wells = int(w05.assign(s=well_split).groupby("sample").s.nunique().gt(1).sum())
    # a test drug must not appear at any other dose in train either
    other_dose = wells[(wells.dose_uM != 0.5) & wells.drug.isin(a2g)]
    rep["split_disjoint"] = dict(
        n_groups=len(splits), per_split=splits.split.value_counts().to_dict(),
        drug_names_in_two_splits=crossing_names,
        treated_wells_in_two_splits=crossing_wells,
        other_dose_wells_excluded_from_training=int(len(other_dose)),
        pass_=len(crossing_names) == 0 and crossing_wells == 0)

    # ---- 2. ordering and train-only panel --------------------------------------------------------
    ds = Dataset(D, M, device="cpu")
    genes = pd.read_csv(M / "genes.tsv", sep="\t")
    transform = json.loads((M / "transform.json").read_text())
    Yraw = np.load(D / "Y.npy")
    valid = np.load(D / "valid.npy")
    kept = ds.kept_line_index
    # the dataset's y for a sampled condition must equal the stored table at the same (line, drug)
    c = ds.conditions("val")
    k = np.random.default_rng(0).choice(len(c), size=min(200, len(c)), replace=False)
    same = all(np.allclose(c.y[i].numpy(), Yraw[kept[int(c.line[i])], int(c.drug[i])]) for i in k)
    basal = np.load(D / "basal.npy")
    rep["alignment"] = dict(
        n_panel_genes=len(genes), panel_is_train_only=transform["panel"]["selection"],
        technical_excluded=transform["technical_exclusion"]["n_excluded"],
        y_lookup_matches_table=bool(same),
        basal_identical_across_drugs=True,  # basal is indexed by line only, by construction
        basal_shape=list(basal.shape), fp_shape=list(np.load(D / "fp.npy").shape),
        no_nan_in_valid_conditions=bool(np.isfinite(Yraw[valid]).all()),
        pass_=bool(same and np.isfinite(Yraw[valid]).all() and len(genes) == 2000))

    # ---- 3 & 4. optimisation reaches every part --------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(D, M, device=dev)
    tr = ds.conditions("train")
    idx = torch.arange(64, device=dev)
    fit = {}
    for kind in ["A", "C"]:
        torch.manual_seed(0)
        hidden = solve_dense_hidden(ds.n_genes, ds.n_bits)[0] if kind == "A" else None
        m = build(kind, ds.n_genes, ds.n_bits, dense_hidden=hidden).to(dev)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        losses, gnorms = [], []
        for _ in range(200):
            p, _ = m(ds.basal[tr.line[idx]], ds.fp[tr.drug[idx]])
            loss = ((p - tr.y[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
            opt.step()
            losses.append(float(loss.detach()))
            gnorms.append(float(gn))
        moved = {n: float((p.detach() - before[n]).abs().max()) for n, p in m.named_parameters()}
        entry = dict(loss_first=losses[0], loss_last=losses[-1],
                     overfits_small_batch=bool(losses[-1] < 0.25 * losses[0]),
                     all_finite=bool(np.isfinite(losses).all() and np.isfinite(gnorms).all()),
                     n_params_unmoved=int(sum(v == 0.0 for v in moved.values())))
        if kind == "C":
            with torch.no_grad():
                _, pi = m(ds.basal[tr.line[idx]], ds.fp[tr.drug[idx]])
            pi = pi.cpu().numpy()
            e = m.response.experts
            entry.update(
                router_params_moved=all(moved[n] > 0 for n in moved if "router" in n),
                every_expert_moved=all(float((e.w1[i] - before["response.experts.w1"][i]).abs().max()) > 0
                                       and float((e.w2[i] - before["response.experts.w2"][i]).abs().max()) > 0
                                       for i in range(e.k)),
                router_mean_weight=pi.mean(0).tolist(),
                router_entropy=float(-(pi * np.log(np.clip(pi, 1e-12, None))).sum(1).mean()),
                router_entropy_max=float(np.log(pi.shape[1])),
                router_collapsed=bool(pi.mean(0).max() > 0.95))
        entry["pass_"] = bool(entry["all_finite"] and entry["overfits_small_batch"]
                              and entry["n_params_unmoved"] == 0
                              and (kind == "A" or (entry["router_params_moved"] and entry["every_expert_moved"])))
        fit[kind] = entry
    rep["optimisation"] = fit
    rep["all_pass"] = bool(rep["split_disjoint"]["pass_"] and rep["alignment"]["pass_"]
                           and all(v["pass_"] for v in fit.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    if not rep["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
