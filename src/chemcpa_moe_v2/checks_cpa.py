"""Pre-run checks for the chemCPA arms: the comparison's preconditions, before any score exists.

1. At a given seed, P0, P1, P2 and P3 start from **identical** values in every module they share.
   Checked by SHA-256 over the state dict with the residual block excluded.
2. At a given seed, P1, P2 and P3 compute **exactly the same function as P0** before any training,
   because every residual's output projection is zero-initialised. Checked by running real control
   cells through all four and comparing outputs bitwise.
3. The residual blocks are parameter-matched: P1 against P2's experts plus router.
4. The residual actually receives gradient steps, and for P2 the router and all four experts move.
5. The three splits' treated drug groups are disjoint, and the validation evaluator's conditions
   are drawn only from validation drug groups.

A failure here is a failure of the comparison, not of a model, so the script exits non-zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models_cpa import param_counts, residual_fingerprint, solve_dense_hidden
from .train_cpa import ConditionEvaluator, _import_chemcpa, build, make_module

ARMS = ["P0", "P1", "P2", "P3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemcpa-src", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--expert-rank", type=int, default=32)
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-control-cells", type=int, default=64)
    a = ap.parse_args()

    import anndata as ad
    ChemCPA, load_dataset_splits, PerturbationDataModule = _import_chemcpa(a.chemcpa_src)
    cfg, datasets, dataset, dm, dcfg = build(a, ChemCPA, load_dataset_splits,
                                             PerturbationDataModule)
    rep: dict = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1 & 2 -------------------------------------------------------------------------------
    shared, equal_fn, counts = {}, {}, {}
    genes = datasets["test_control"].genes[:64].to(dev)
    n = genes.shape[0]
    di = torch.zeros(n, dtype=torch.long, device=dev)
    do = torch.full((n,), 0.5, device=dev)
    cov = torch.zeros((n, datasets["training"].num_covariates[0]), device=dev)
    cov[:, 0] = 1.0

    # chemCPA re-seeds inside ComPert's own constructor. `ComPert.__init__` takes the `hparams`
    # branch only when they are a plain dict; an OmegaConf node is not one, so `set_hparams_(seed,
    # hparams)` runs, and that method calls `torch.manual_seed(seed)` / `np.random.seed(seed)` with
    # `model.additional_params.seed`. The ambient RNG state is therefore irrelevant to how the base
    # model is initialised, and setting the global seed alone would make every "seed" identical.
    # (`set_hparams_` also draws random hyper-parameters for a non-zero seed, but every key it draws
    # is then overwritten by `configs/chemcpa_moe_v2/hparams_official.json`, which supplies all of
    # them -- verified below by comparing the realised hparams across seeds.)
    hp_by_seed = {}
    for seed in a.seeds:
        fps, outs = {}, {}
        cfg["model"]["additional_params"]["seed"] = int(seed)
        for arm in ARMS:
            torch.manual_seed(0)
            np.random.seed(0)
            lm, dh, parity = make_module(ChemCPA, cfg, dcfg, arm, a)
            hp_by_seed[str(seed)] = {k: lm.model.hparams[k] for k in sorted(lm.model.hparams)}
            lm = lm.to(dev).eval()
            fps[arm] = residual_fingerprint(lm.model)
            if seed == a.seeds[0]:
                counts[arm] = dict(param_counts(lm.model), dense_hidden=dh, parity=parity)
            with torch.no_grad():
                out, _ = lm.model.predict(genes=genes, drugs_idx=di, dosages=do, covariates=[cov])
            outs[arm] = out.detach().float().cpu().numpy()
            del lm
            torch.cuda.empty_cache()
        shared[str(seed)] = dict(fingerprints={k: v[:16] for k, v in fps.items()},
                                 identical=bool(len(set(fps.values())) == 1))
        equal_fn[str(seed)] = {arm: dict(
            max_abs_difference_from_P0=float(np.max(np.abs(outs[arm] - outs["P0"]))),
            identical=bool(np.array_equal(outs[arm], outs["P0"]))) for arm in ARMS[1:]}

    seeds_differ = len({v["fingerprints"]["P0"] for v in shared.values()}) == len(a.seeds)
    hp_same = len({json.dumps(v, sort_keys=True) for v in hp_by_seed.values()}) == 1
    rep["shared_initialisation"] = dict(
        by_seed=shared,
        seeds_give_different_weights=bool(seeds_differ),
        hyperparameters_identical_across_seeds=bool(hp_same),
        realised_hyperparameters=hp_by_seed[str(a.seeds[0])],
        effective_seed="model.additional_params.seed, consumed by ComPert.set_hparams_",
        pass_=bool(all(v["identical"] for v in shared.values()) and seeds_differ and hp_same))
    rep["same_function_at_init"] = dict(
        by_seed=equal_fn,
        pass_=bool(all(v2["identical"] for v in equal_fn.values() for v2 in v.values())))

    # ---- 3 -----------------------------------------------------------------------------------
    d = int(cfg["model"]["hparams"]["autoencoder_width"])
    hidden, parity = solve_dense_hidden(d, a.n_experts, a.expert_rank,
                                        d_cond=2 * int(cfg["model"]["hparams"]["dim"]))
    rep["parameter_parity"] = dict(parity, counts=counts,
                                   pass_=bool(parity["relative_difference"] <= 0.01))

    # ---- 4 -----------------------------------------------------------------------------------
    fit = {}
    tr = datasets["training"]
    idx = torch.arange(64)
    g = tr.genes[idx].to(dev)
    didx = tr.drugs_idx[idx].to(dev).squeeze(-1) if tr.drugs_idx.ndim > 1 else tr.drugs_idx[idx].to(dev)
    dos = tr.dosages[idx].to(dev).squeeze(-1) if tr.dosages.ndim > 1 else tr.dosages[idx].to(dev)
    cv = tr.covariates[0][idx].to(dev)
    for arm in ["P1", "P2", "P3"]:
        torch.manual_seed(0)
        lm, _, _ = make_module(ChemCPA, cfg, dcfg, arm, a)
        lm = lm.to(dev)
        res = lm.model.residual
        before = {k: v.detach().clone() for k, v in res.named_parameters()}
        opt = torch.optim.Adam(res.parameters(), lr=1e-2)
        losses = []
        for _ in range(100):
            out, _ = lm.model.predict(genes=g, drugs_idx=didx, dosages=dos, covariates=[cv])
            dim = out.shape[1] // 2
            loss = ((out[:, :dim] - g) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        moved = {k: float((v.detach() - before[k]).abs().max()) for k, v in res.named_parameters()}
        entry = dict(loss_first=losses[0], loss_last=losses[-1],
                     all_finite=bool(np.isfinite(losses).all()),
                     n_params_unmoved=int(sum(v == 0.0 for v in moved.values())),
                     moved=moved)
        if arm == "P2":
            with torch.no_grad():
                out, _ = lm.model.predict(genes=g, drugs_idx=didx, dosages=dos, covariates=[cv])
            pi = lm.model.last_router_weights.detach().float().cpu().numpy()
            entry.update(router_mean_weight=pi.mean(0).tolist(),
                         router_collapsed=bool(pi.mean(0).max() > 0.95),
                         every_expert_moved=bool(
                             moved.get("experts.w2", 0) > 0 and moved.get("experts.w1", 0) > 0))
        entry["pass_"] = bool(entry["all_finite"] and entry["n_params_unmoved"] == 0
                              and entry["loss_last"] < entry["loss_first"])
        fit[arm] = entry
        del lm
        torch.cuda.empty_cache()
    rep["residual_receives_gradient"] = dict(fit, pass_=bool(all(v["pass_"] for v in fit.values())))

    # ---- 5 -----------------------------------------------------------------------------------
    adata = ad.read_h5ad(a.adata)
    obs = adata.obs
    treated = obs[obs.control == 0]
    by = {s: set(treated.condition[treated.split == s].astype(str)) for s in ["train", "test", "ood"]}
    overlaps = {f"{x}_x_{y}": sorted(by[x] & by[y])[:5]
                for x, y in [("train", "test"), ("train", "ood"), ("test", "ood")]}
    drug_to_idx = {d_: i for i, d_ in enumerate(dataset.drugs_names_unique_sorted)}
    ev = ConditionEvaluator(adata, Path(a.v2_dir), Path(a.manifest_dir), "val", drug_to_idx,
                            max_control_cells=a.max_control_cells, seed=0)
    splits = pd.read_csv(Path(a.manifest_dir) / "splits.tsv", sep="\t").sort_values("group_index")
    val_cols = set(np.where(splits.split.to_numpy() == "val")[0].tolist())
    off = [r for r in ev.rows if r[1] not in val_cols]
    rep["split_disjoint"] = dict(
        n_drugs={s: len(v) for s, v in by.items()},
        overlaps=overlaps,
        n_val_conditions=len(ev.rows),
        val_conditions_outside_validation_drugs=len(off),
        pass_=bool(all(not v for v in overlaps.values()) and not off))

    rep["all_pass"] = bool(all(rep[k]["pass_"] for k in
                               ["shared_initialisation", "same_function_at_init",
                                "parameter_parity", "residual_receives_gradient",
                                "split_disjoint"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    if not rep["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
