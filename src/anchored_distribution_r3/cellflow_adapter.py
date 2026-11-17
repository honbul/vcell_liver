"""CellFlow (external comparator) retrained on the same split, genes and cells as every other method.

Two stages, because CellFlow lives in its own JAX environment and this package's environment has
no JAX:

    export   (study python)     python -m src.anchored_distribution_r3.cellflow_adapter export --setting dev
             writes $R3_ASSETS/cellflow/<setting>/ : the real cells of the setting's fitting wells
             (the same rows the DDPMs train on), the control cells of every (plate, line) involved,
             the ECFP4 table and the evaluation-well table (ids and cell counts only).
    train    (CellFlow python)  python src/anchored_distribution_r3/cellflow_adapter.py train --setting dev --iters N
             trains CellFlow once, then writes its NATIVE output for every evaluation well:
             $R3_ASSETS/gen/cellflow/<setting>/<run_id>__ss{seed}.npz, key str(gid) -> float32
             [n_eval, 2000] in log1p(CP10k) units (no clipping, no mean anchoring).
             `evaluate.py --methods "cells:<that path with {seed}>"` scores it like any other method.

Adapter choices (fixed before any evaluation score was seen; one configuration, not tuned):
    * sample representation: the 2,000 genes directly (`sample_rep="X"`), not the tutorial's PCA
      latent, so that CellFlow is scored on the same gene panel without a decoder of ours.
    * drug condition: the study's 2048-bit ECFP4 vector passed as the perturbation representation
      (`perturbation_covariate_reps`); the control "drug" is the all-zero vector.
    * cell line: sample covariate, one-hot over the cell-line index.
    * source distributions: `split_covariates=["plate", "line"]`, i.e. the same-plate, same-line
      DMSO cells; targets are that plate/line's training-drug treated cells.
    * prediction: CellFlow's ODE maps each source cell deterministically. For an evaluation well
      the source cells are n_eval control cells of its (plate, line), drawn with the same seeded
      rule as `baselines.control_cells`. No treated statistic of an evaluation well is used.

The `train` stage imports nothing from this package (only numpy/pandas/anndata/jax/cellflow).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CELLFLOW_COMMIT = "03bff12e71326742ad53a0909a4a8fee36958794"   # v0.0.9

CONFIG = dict(
    sample_rep="X", drug_rep="ecfp4_2048", line_rep="onehot", split_covariates=["plate", "line"],
    solver="otfm", condition_mode="deterministic", regularization=0.0, pooling="mean",
    layers_before_pool=dict(drug=[256, 256], line=[64]), layers_after_pool=[256, 256],
    condition_embedding_dim=256, cond_output_dropout=0.9,
    time_encoder_dims=[1024, 1024], hidden_dims=[1024, 1024, 1024], decoder_dims=[2048, 2048],
    conditioning="concatenation", probability_path=None, match_fn="match_linear(defaults)",
    optimizer="adam(1e-4)", batch_size=1024, ode="library default (Tsit5, PID rtol=atol=1e-5)",
)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def cf_dir(setting) -> Path:
    return Path(os.environ["R3_ASSETS"]) / "cellflow" / setting


# ------------------------------------------------------------------ stage 1: export (study env)
def export(setting: str) -> None:
    from .setting import build   # noqa: PLC0415  (study environment only)

    out = cf_dir(setting)
    if (out / "export.json").exists():
        raise SystemExit(f"{out}/export.json exists; exports are written once")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st = build(setting, "cpu")
    d, fw, ew = st.d, st.fit_wells, st.eval_wells
    X = np.concatenate([np.asarray(d.X[r.start:r.end]) for r in fw.itertuples()]).astype(np.float32)
    n = (fw.end - fw.start).to_numpy()
    obs = pd.DataFrame(dict(gid=np.repeat(fw.gid.to_numpy(), n), plate=np.repeat(fw.plate_index.to_numpy(), n),
                            line=np.repeat(fw.line.to_numpy(), n), drug=np.repeat(fw.drug.to_numpy(), n)))
    keys = sorted({(int(p), int(l)) for p, l in zip(fw.plate_index, fw.line)}
                  | {(int(p), int(l)) for p, l in zip(ew.plate_index, ew.line)})
    C, cobs = [], []
    for p, l in keys:
        c = d.ctrl_cells(p, l).astype(np.float32)        # same order as baselines.control_cells sees
        C.append(c); cobs += [(p, l, i) for i in range(len(c))]
    C = np.concatenate(C)
    np.save(out / "train_X.npy", X); obs.to_csv(out / "train_obs.tsv", sep="\t", index=False)
    np.save(out / "ctrl_X.npy", C)
    pd.DataFrame(cobs, columns=["plate", "line", "pos"]).to_csv(out / "ctrl_obs.tsv", sep="\t", index=False)
    np.save(out / "fp.npy", d.fp.astype(np.float32))
    ew[["gid", "plate_index", "line", "drug", "n_eval"]].rename(columns={"plate_index": "plate"}).to_csv(
        out / "eval_wells.tsv", sep="\t", index=False)
    rec = dict(setting=setting, n_train_cells=int(len(X)), n_train_wells=int(len(fw)),
               n_train_conditions=int(fw.groupby(["line", "drug"]).ngroups), n_train_drugs=int(fw.drug.nunique()),
               n_ctrl_cells=int(len(C)), n_ctrl_pops=len(keys), n_eval_wells=int(len(ew)),
               n_eval_drugs=int(ew.drug.nunique()), n_lines_total=int(d.valid.shape[0]), n_genes=int(X.shape[1]),
               eval_drugs_disjoint_from_train=not (set(fw.drug) & set(ew.drug)),
               sha256={f: sha256_file(out / f) for f in ["train_X.npy", "ctrl_X.npy", "fp.npy", "eval_wells.tsv"]},
               commit=os.popen("git rev-parse HEAD").read().strip(), wall_s=time.time() - t0)
    (out / "export.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


# ------------------------------------------------------------------ stage 2: train + sample (CellFlow env)
def train(setting: str, iters: int, seed: int, sample_seeds, smoke: bool) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import anndata as ad   # noqa: PLC0415
    import jax   # noqa: PLC0415
    import optax   # noqa: PLC0415
    from cellflow.model import CellFlow   # noqa: PLC0415

    src = cf_dir(setting)
    exp = json.loads((src / "export.json").read_text())
    cfg = dict(CONFIG, setting=setting, iters=int(iters), seed=int(seed), cellflow_commit=CELLFLOW_COMMIT)
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    rid = f"cf009-{setting}-x{exp['n_genes']}-it{iters}-bs{cfg['batch_size']}-s{seed}-{h}" + ("-smoke" if smoke else "")
    gen = Path(os.environ["R3_ASSETS"]) / "gen" / "cellflow" / setting
    run = Path(os.environ["R3_ASSETS"]) / "runs" / "cellflow" / rid
    if (run / "record.json").exists():
        raise SystemExit(f"run id {rid} already exists; run ids are immutable")
    gen.mkdir(parents=True, exist_ok=True); run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"run={rid} jax={jax.__version__} devices={jax.devices()}", flush=True)

    Xt = np.load(src / "train_X.npy"); ot = pd.read_csv(src / "train_obs.tsv", sep="\t")
    Xc = np.load(src / "ctrl_X.npy"); oc = pd.read_csv(src / "ctrl_obs.tsv", sep="\t")
    fp = np.load(src / "fp.npy"); W = pd.read_csv(src / "eval_wells.tsv", sep="\t")
    L = exp["n_lines_total"]
    assert not set(ot.drug) & set(W.drug), "evaluation drugs must not be training drugs"
    fp_rep = {f"d{i}": fp[i] for i in sorted(set(ot.drug) | set(W.drug))}
    fp_rep["control"] = np.zeros(fp.shape[1], np.float32)
    line_rep = {f"l{i}": np.eye(L, dtype=np.float32)[i] for i in range(L)}

    def obs_frame(o, control):
        f = pd.DataFrame(dict(drug="control" if control else ("d" + o.drug.astype(str)).to_numpy(),
                              line=("l" + o.line.astype(str)).to_numpy(), plate=("p" + o.plate.astype(str)).to_numpy(),
                              is_control=control))
        f.index = [("c" if control else "t") + str(i) for i in range(len(f))]
        return f

    obs = pd.concat([obs_frame(oc, True), obs_frame(ot, False)])
    for c in ["drug", "line", "plate"]:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(np.concatenate([Xc, Xt]), obs=obs)
    adata.uns["fp"] = fp_rep; adata.uns["line_emb"] = line_rep
    del Xt
    print(f"adata {adata.shape} controls={int(obs.is_control.sum())} load={time.time()-t0:.0f}s", flush=True)

    mlp = lambda dims: {"layer_type": "mlp", "dims": list(dims), "dropout_rate": 0.0}   # noqa: E731
    cf = CellFlow(adata, solver=cfg["solver"])
    cf.prepare_data(sample_rep="X", control_key="is_control",
                    perturbation_covariates={"drug": ("drug",)}, perturbation_covariate_reps={"drug": "fp"},
                    sample_covariates=["line"], sample_covariate_reps={"line": "line_emb"},
                    split_covariates=cfg["split_covariates"], max_combination_length=1, null_value=0.0)
    n_src, n_tgt = int(cf.train_data.n_controls), int(cf.train_data.n_perturbations)
    print(f"source dists={n_src} target dists={n_tgt} prepare={time.time()-t0:.0f}s", flush=True)
    cf.prepare_model(
        condition_mode=cfg["condition_mode"], regularization=cfg["regularization"], pooling=cfg["pooling"],
        layers_before_pool={k: mlp(v) for k, v in cfg["layers_before_pool"].items()},
        layers_after_pool=mlp(cfg["layers_after_pool"]),
        condition_embedding_dim=cfg["condition_embedding_dim"], cond_output_dropout=cfg["cond_output_dropout"],
        time_encoder_dims=cfg["time_encoder_dims"], hidden_dims=cfg["hidden_dims"], decoder_dims=cfg["decoder_dims"],
        conditioning=cfg["conditioning"], probability_path=cfg["probability_path"],
        optimizer=optax.adam(1e-4), seed=seed)
    t1 = time.time()
    cf.train(num_iterations=iters, batch_size=cfg["batch_size"], valid_freq=10 ** 9)
    train_s = time.time() - t1
    loss = np.asarray(cf.trainer.training_logs["loss"], dtype=np.float64)
    k = max(1, len(loss) // 20)
    curve = [float(loss[i:i + k].mean()) for i in range(0, len(loss), k)]
    print(f"train {iters} it: {train_s:.0f}s ({train_s / iters * 1000:.1f} ms/it) loss curve {np.round(curve, 4).tolist()}", flush=True)
    assert np.isfinite(loss).all(), "non-finite training loss"

    # ---- native prediction for every evaluation well
    t2 = time.time()
    groups = {k: g.pos.to_numpy() for k, g in oc.groupby(["plate", "line"])}
    start = {}
    o = 0
    for (p, l), g in oc.groupby(["plate", "line"], sort=False):
        start[(p, l)] = o; o += len(g)
    M = max(len(v) for v in groups.values())            # one source size -> one compiled ODE solve
    if smoke:
        W = W.iloc[:: max(1, len(W) // 24)].reset_index(drop=True)
    out = {s: {} for s in sample_seeds}
    for (p, l), g in W.groupby(["plate", "line"]):
        C = Xc[start[(p, l)]: start[(p, l)] + len(groups[(p, l)])]
        pad = C[np.arange(M) % len(C)]                   # cyclic padding; only rows < len(C) are used
        ca = ad.AnnData(pad.copy(), obs=pd.DataFrame(dict(drug="control", line=f"l{l}", plate=f"p{p}", is_control=True),
                                                     index=[f"s{i}" for i in range(M)]))
        ca.uns["fp"] = fp_rep; ca.uns["line_emb"] = line_rep
        drugs = sorted(set(g.drug))
        cov = pd.DataFrame(dict(drug=[f"d{x}" for x in drugs], line=f"l{l}", plate=f"p{p}", is_control=False,
                                condition=[f"d{x}" for x in drugs]))
        pred = cf.predict(adata=ca, covariate_data=cov, sample_rep="X", condition_id_key="condition")
        pred = {k: np.asarray(v)[: len(C)] for k, v in pred.items()}
        for r in g.itertuples():
            for s in sample_seeds:
                rng = np.random.default_rng([s, int(r.gid)])                  # same rule as baselines.control_cells
                idx = np.sort(rng.permutation(len(C))[: int(r.n_eval)])
                out[s][str(r.gid)] = pred[f"d{r.drug}"][idx].astype(np.float32)
    sample_s = time.time() - t2
    files = {}
    for s in sample_seeds:
        f = gen / f"{rid}__ss{s}.npz"
        np.savez(f, **out[s]); files[str(f)] = sha256_file(f)
    allv = np.concatenate([v for v in out[sample_seeds[0]].values()])
    try:
        ms = jax.devices()[0].memory_stats(); peak = ms.get("peak_bytes_in_use", 0) / 2 ** 30
    except Exception:   # noqa: BLE001
        peak = None
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    rec = dict(run_id=rid, kind="cellflow_train", status="ok", cfg=cfg, cellflow_commit=CELLFLOW_COMMIT,
               iterations=int(iters), batch_size=cfg["batch_size"], n_train_cells=exp["n_train_cells"],
               n_ctrl_cells=exp["n_ctrl_cells"], n_source_dists=n_src, n_target_dists=n_tgt,
               n_train_conditions=exp["n_train_conditions"], n_eval_wells=int(len(W)), source_pad_M=int(M),
               train_wall_s=train_s, sample_wall_s=sample_s, total_wall_s=time.time() - t0,
               peak_jax_gpu_gb=peak, nvidia_smi_at_end=smi, loss_curve_20=curve, final_loss=curve[-1],
               out_finite=bool(np.isfinite(allv).all()), out_min=float(allv.min()), out_neg_frac=float((allv < 0).mean()),
               out_mean=float(allv.mean()), sample_seeds=list(sample_seeds), npz_sha256=files,
               npz_pattern=str(gen / f"{rid}__ss{{seed}}.npz"), job=os.environ.get("SLURM_JOB_ID"),
               commit=os.popen("git rev-parse HEAD").read().strip(), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (run / "record.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export", "train"])
    ap.add_argument("--setting", default="dev")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true", help="short run: scores nothing, predicts ~24 wells")
    a = ap.parse_args()
    if a.stage == "export":
        export(a.setting)
    else:
        if a.iters <= 0:
            sys.exit("--iters is required for the train stage")
        train(a.setting, a.iters, a.seed, [int(s) for s in a.sample_seeds.split(",")], a.smoke)


if __name__ == "__main__":
    main()
