"""Train one arm (P0 / P1 / P2 / P3) of the chemCPA comparison on Tahoe at 0.5 uM.

The training step is chemCPA's own. This module subclasses `chemCPA.lightning_module.ChemCPA` and
overrides exactly three things:

  * `configure_optimizers` -- calls the published one and then adds the residual block's parameters
    to the autoencoder group. The published method enumerates encoder, decoder, embedding encoder
    and covariate embeddings by name, so a new module would otherwise never receive a gradient
    step. Learning rate, weight decay, the StepLR(step_size_lr, gamma=0.9) schedule and the
    adversary alternation are left exactly as they are.
  * `on_validation_epoch_end` -- replaces chemCPA's R2 read-out with this study's primary metric,
    macro-DeltaRMSE over (cell line, drug) conditions, computed on the **validation** drug groups
    only (chemCPA's "test" split; see make_anndata.py for the naming).
  * `on_train_end` -- removed, because the published one scores the ood split, and the ood split
    here is this study's held-out test set.

`ComPert.predict` is re-bound on the instance so the decoder's shared hidden state passes through
the residual. Everything else about the model -- encoder, drug embedding, doser, covariate
embeddings, both adversaries, the Gaussian head and the GaussianNLL objective -- is the published
code, untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .models_cpa import (attach_residual, decoder_with_residual, param_counts,
                         residual_fingerprint, solve_dense_hidden)


def _import_chemcpa(src: str):
    if src not in sys.path:
        sys.path.insert(0, src)
    from chemCPA.data.data import PerturbationDataModule, load_dataset_splits  # noqa: E402
    from chemCPA.lightning_module import ChemCPA  # noqa: E402
    return ChemCPA, load_dataset_splits, PerturbationDataModule


def patched_predict(self, genes, drugs=None, drugs_idx=None, dosages=None, covariates=None,
                    return_latent_basal=False):
    """`ComPert.predict`, with the residual applied inside the decoder.

    Line for line the published method until the decoder call; the only change is that the decoder
    is run through `decoder_with_residual`, which adds R(h) to the hidden state the final linear
    map reads.
    """
    from chemCPA.model import _move_inputs
    genes, drugs, drugs_idx, dosages, covariates = _move_inputs(
        genes, drugs, drugs_idx, dosages, covariates, device=self.device)
    latent_basal = self.encoder(genes)
    latent_treated = latent_basal

    drug_embedding = None
    if self.num_drugs > 0:
        drug_embedding = self.compute_drug_embeddings_(drugs=drugs, drugs_idx=drugs_idx,
                                                       dosages=dosages)
        latent_treated = latent_treated + drug_embedding

    emb_cov = None
    if self.num_covariates[0] > 0:
        for cov_type, e in enumerate(self.covariates_embeddings):
            e = e.to(self.device)
            cov_idx = covariates[cov_type].argmax(1)
            latent_treated = latent_treated + e(cov_idx)
            emb_cov = e(cov_idx)

    if self.num_covariates[0] > 0 and self.num_drugs > 0:
        cell_drug_embedding = torch.cat([emb_cov, drug_embedding], dim=1)
    elif self.num_drugs > 0:
        cell_drug_embedding = drug_embedding
    else:
        cell_drug_embedding = torch.zeros_like(latent_basal)

    gene_reconstructions = decoder_with_residual(self, latent_treated, cell_drug_embedding)
    dim = gene_reconstructions.size(1) // 2
    mean = gene_reconstructions[:, :dim]
    var = F.softplus(gene_reconstructions[:, dim:])
    out = torch.cat([mean, var], dim=1)
    if return_latent_basal:
        return out, cell_drug_embedding, (latent_basal, drug_embedding, latent_treated)
    return out, cell_drug_embedding


def macro_delta_rmse(pred: np.ndarray, truth: np.ndarray, line: np.ndarray) -> float:
    """Per-condition RMSE -> mean over that line's drugs -> unweighted mean over lines."""
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
    per_line = [rmse[line == l].mean() for l in np.unique(line) if (line == l).any()]
    return float(np.mean(per_line))


class ConditionEvaluator:
    """Turns single-cell predictions into the (cell line, drug) response this study scores.

    For a condition, the model is given that cell line's own DMSO cells from the plate the
    condition's wells sit on, together with the drug index and the 0.5 uM dose. The predicted
    response is the mean of those predicted treated profiles minus the observed DMSO mean -- the
    same subtraction the truth uses, so the comparison is between predicted and observed treated
    means and nothing else.
    """

    def __init__(self, adata, v2_dir: Path, manifest_dir: Path, split: str,
                 drug_to_idx: dict, cov_onehot: dict, max_control_cells: int = 64, seed: int = 0):
        import anndata  # noqa: F401
        self.Y = np.load(v2_dir / "Y.npy")
        self.valid = np.load(v2_dir / "valid.npy")
        self.plate_of = np.load(v2_dir / "plate.npy")
        # The truth is the unweighted mean over a condition's wells, each referenced to its own
        # plate's DMSO pool, and 9.2 % of conditions have wells on more than one plate. Scoring a
        # prediction built on the majority plate alone against that truth compares two different
        # control terms, so the same well weighting is applied to the prediction.
        pw = v2_dir / "plate_weights.npy"
        if not pw.exists():
            raise SystemExit(f"{pw} is missing; rebuild the v2 data with prepare_v2 so the "
                             "prediction can use the same plate weighting as the truth")
        self.plate_weights = np.load(pw)
        # the control term the truth subtracts: the whole plate DMSO pool for that cell line, not
        # the mean of whichever control cells happen to be fed to the model
        self.dmso_plate_mean = np.load(v2_dir / "dmso_plate_mean.npy")
        splits = pd.read_csv(manifest_dir / "splits.tsv", sep="\t").sort_values("group_index")
        lines = pd.read_csv(manifest_dir / "lines.tsv", sep="\t").sort_values("line_index")
        aliases = pd.read_csv(manifest_dir / "drug_aliases.tsv", sep="\t")
        import ast
        label_of = {str(r.group): sorted(str(x) for x in ast.literal_eval(r.drugs))[0]
                    .replace("+", "-").replace("_", "-").strip() for _, r in aliases.iterrows()}

        want = splits.split.to_numpy() == split
        self.drug_cols = np.where(want)[0]
        self.drug_idx = np.array([drug_to_idx.get(label_of[g], -1)
                                  for g in splits.group_smiles.astype(str).to_numpy()[want]])

        obs = adata.obs
        self.line_name = dict(zip(lines.cell_line_id, range(len(lines))))

        # The covariate one-hot a cell line must receive is the column the *training loader's own*
        # encoder gave that line's `cell_type` name. chemCPA fits `OneHotEncoder` on
        # `np.unique(obs.cell_type)`, so its column order is the sorted cell NAME; `lines.tsv` is
        # ordered by CVCL identifier. Writing `cov[:, line_index] = 1` -- which this evaluator did
        # until 2026-11-13 -- therefore handed the model a different cell line's identity for 45 of
        # the 50 lines (SNU-423's expression arrived labelled HEC-1-A). The vector is now looked up
        # by name in the loader's own `atomic_covars_dict`, and a line that has no entry is a hard
        # error rather than a silently wrong row.
        pair = (obs[["cell_line_id", "cell_type"]].astype(str).drop_duplicates())
        if pair.cell_line_id.duplicated().any():
            bad = sorted(pair.cell_line_id[pair.cell_line_id.duplicated()].unique())[:5]
            raise SystemExit(f"a cell_line_id maps to more than one cell_type name: {bad}")
        name_of_id = dict(zip(pair.cell_line_id, pair.cell_type))
        self.cov_onehot_of_line, self.cov_audit = {}, []
        for l_, cid in enumerate(lines.cell_line_id.astype(str)):
            nm = name_of_id.get(cid)
            vec = cov_onehot.get(nm) if nm is not None else None
            if vec is not None:
                v = np.asarray(vec, dtype=np.float32)
                self.cov_onehot_of_line[l_] = v
                self.cov_audit.append(dict(line_index=l_, cell_line_id=cid, cell_name=nm,
                                           onehot_column=int(np.argmax(v))))

        ctrl = obs.index[(obs.control == 1).to_numpy()]
        ctrl_pos = np.where((obs.control == 1).to_numpy())[0]
        cl = obs.cell_line_id.astype(str).to_numpy()[ctrl_pos]
        pl = obs.plate.astype(str).to_numpy()[ctrl_pos]
        plates = sorted(pd.unique(obs.plate.astype(str)))
        self.plates = plates
        rng = np.random.default_rng(seed)
        self.control_rows = {}
        for (c, p), idx in pd.DataFrame(dict(c=cl, p=pl, i=ctrl_pos)).groupby(["c", "p"]):
            v = idx.i.to_numpy()
            if len(v) > max_control_cells:
                v = rng.choice(v, max_control_cells, replace=False)
            self.control_rows[(c, plates.index(p))] = np.sort(v)
        self.X = adata.X
        self.cell_line_ids = list(lines.cell_line_id)
        self.n_lines = len(lines)
        self.dmso_idx = drug_to_idx.get("DMSO", 0)

        # the conditions that can actually be scored: valid truth, a known drug, control cells
        rows, self.plate_mix, dropped_plate_weight = [], {}, []
        for j, d in enumerate(self.drug_cols):
            if self.drug_idx[j] < 0:
                continue
            for l in range(self.n_lines):
                if not self.valid[l, d]:
                    continue
                p = int(self.plate_of[l, d])
                key = (self.cell_line_ids[l], p)
                if p < 0 or key not in self.control_rows:
                    continue
                if l not in self.cov_onehot_of_line:
                    raise SystemExit(
                        f"{self.cell_line_ids[l]} has scorable conditions but no covariate "
                        "one-hot in the training loader's encoder; the model cannot be told "
                        "which cell line this is")
                # the plates the truth's wells came from, kept only where control cells exist,
                # renormalised so the prediction is still a weighted mean
                w = self.plate_weights[l, d]
                mix = [(q, float(w[q])) for q in np.where(w > 0)[0]
                       if (self.cell_line_ids[l], int(q)) in self.control_rows]
                if not mix:
                    continue
                tot = sum(x for _, x in mix)
                if tot < 1.0 - 1e-6:
                    dropped_plate_weight.append((l, d, round(1.0 - tot, 4)))
                mix = [(int(q), x / tot) for q, x in mix]
                self.plate_mix[(l, d)] = mix
                rows.append((l, d, j, p))
        self.rows = rows
        self.scored_lines = sorted({r[0] for r in rows})
        self.n_multi_plate_conditions = sum(1 for v in self.plate_mix.values() if len(v) > 1)
        self.conditions_with_dropped_plate_weight = dropped_plate_weight

    def evaluate(self, model, device="cuda", control_mode: str = "observed"
                 ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Score the arms' conditions.

        `control_mode="observed"` is the pre-registered primary: the predicted response is the
        predicted treated mean minus the **observed** plate DMSO pool, the same term the truth
        subtracts, so predicted-minus-true is exactly predicted-treated-minus-observed-treated.

        `control_mode="model"` is a diagnostic: the same control cells are passed through the model
        with the DMSO drug index at dose 0, and that prediction is subtracted instead. It cancels
        whatever constant error the decoder makes in reproducing a cell line's absolute profile,
        and so separates "cannot reconstruct the profile" from "cannot predict the response".
        It is never used to select anything.
        """
        assert control_mode in ("observed", "model")
        model.eval()
        preds, truths, lines = [], [], []
        n_cov = model.num_covariates[0]
        with torch.no_grad():
            for (l, d, j, p) in self.rows:
                v = self.cov_onehot_of_line[l]
                assert len(v) == n_cov, (len(v), n_cov)
                delta = None
                # one term per plate the condition's wells sit on, combined with the same weights
                # the truth used; for the 90.8 % of conditions on a single plate this is one term
                for q, wq in self.plate_mix[(l, d)]:
                    idx = self.control_rows[(self.cell_line_ids[l], q)]
                    g = torch.as_tensor(np.asarray(self.X[idx]), dtype=torch.float32, device=device)
                    n = g.shape[0]
                    di = torch.full((n,), int(self.drug_idx[j]), dtype=torch.long, device=device)
                    do = torch.full((n,), 0.5, dtype=torch.float32, device=device)
                    cov = torch.as_tensor(v, dtype=torch.float32, device=device
                                          ).unsqueeze(0).expand(n, -1).contiguous()
                    out, _ = model.predict(genes=g, drugs_idx=di, dosages=do, covariates=[cov])
                    dim = out.shape[1] // 2
                    pred_treated = out[:, :dim].mean(0).float().cpu().numpy()
                    if control_mode == "observed":
                        base = self.dmso_plate_mean[q, l]
                    else:
                        dz = torch.full((n,), int(self.dmso_idx), dtype=torch.long, device=device)
                        z = torch.zeros((n,), dtype=torch.float32, device=device)
                        oc, _ = model.predict(genes=g, drugs_idx=dz, dosages=z, covariates=[cov])
                        base = oc[:, :dim].mean(0).float().cpu().numpy()
                    term = wq * (pred_treated - base)
                    delta = term if delta is None else delta + term
                preds.append(delta)
                truths.append(self.Y[l, d])
                lines.append(l)
        P, T, L = np.array(preds), np.array(truths), np.array(lines)
        return macro_delta_rmse(P, T, L), P, T, L


def build(args, ChemCPA, load_dataset_splits, PerturbationDataModule):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(dict(
        dataset=dict(dataset_path=args.adata, perturbation_key="condition", dose_key="dose",
                     covariate_keys="cell_type", smiles_key="SMILES", degs_key="all_DEGs",
                     pert_category="cov_drug_dose_name", split_key="split", use_drugs_idx=True),
        model=dict(hparams=json.loads(Path(args.hparams).read_text()),
                   additional_params=dict(patience=args.patience, decoder_activation="linear",
                                          doser_type="amortized", seed=args.seed),
                   embedding=dict(model="rdkit", datapath=args.embedding),
                   append_ae_layer=False, enable_cpa_mode=False, load_pretrained=False),
        training=dict(run_eval_r2=False),
        profiling=dict(run_profiler=False, outdir="./")))
    datasets, dataset = load_dataset_splits(**cfg["dataset"], return_dataset=True)
    dm = PerturbationDataModule(datasplits=datasets,
                                train_bs=cfg["model"]["hparams"]["batch_size"])
    dcfg = dict(num_genes=datasets["training"].num_genes,
                num_drugs=datasets["training"].num_drugs,
                num_covariates=datasets["training"].num_covariates,
                use_drugs_idx=dataset.use_drugs_idx,
                canon_smiles_unique_sorted=dataset.canon_smiles_unique_sorted)
    return cfg, datasets, dataset, dm, dcfg


def make_module(ChemCPA, cfg, dcfg, arm: str, args):
    import lightning as L

    class Arm(ChemCPA):
        """chemCPA's LightningModule with a residual on the decoder's shared hidden state."""

        def configure_optimizers(self):
            groups = super().configure_optimizers()
            res = getattr(self.model, "residual", None)
            if res is not None:
                # the published method enumerates the autoencoder's parts by name, so a module it
                # has never heard of gets no gradient step unless it is added here
                groups[0]["optimizer"].add_param_group(
                    dict(params=list(res.parameters()),
                         lr=self.model.hparams["autoencoder_lr"],
                         weight_decay=self.model.hparams["autoencoder_wd"]))
            return groups

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            return None

        def on_validation_epoch_end(self):
            ev = getattr(self, "condition_evaluator", None)
            if ev is None:
                return
            self.eval()
            score, *_ = ev.evaluate(self.model, device=self.device)
            self.log("val_macro_delta_rmse", score, on_step=False, on_epoch=True)
            self.history.append(dict(epoch=int(self.current_epoch), val_macro_delta_rmse=score))
            if score < self.best["val"] - 1e-9:
                self.best = dict(val=score, epoch=int(self.current_epoch))
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.state_dict().items()}
            print(json.dumps(dict(epoch=int(self.current_epoch), val=round(score, 6),
                                  best=round(self.best["val"], 6))), flush=True)
            self.train()

        def on_train_end(self):
            # the published hook scores the ood split; here ood is the held-out test set
            return None

    lm = Arm(cfg, dcfg)
    lm.history, lm.best, lm.best_state = [], dict(val=float("inf"), epoch=-1), None
    lm.condition_evaluator = None

    d_hidden = lm.model.hparams["autoencoder_width"]
    dense_hidden = None
    parity = None
    if arm == "P1":
        dense_hidden, parity = solve_dense_hidden(
            d_hidden, args.n_experts, args.expert_rank,
            d_cond=2 * lm.model.hparams["dim"])
    attach_residual(lm.model, arm, n_experts=args.n_experts, expert_rank=args.expert_rank,
                    dense_hidden=dense_hidden)
    lm.model.predict = types.MethodType(patched_predict, lm.model)
    return lm, dense_hidden, parity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemcpa-src", required=True)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--hparams", required=True, help="json of chemCPA's config/model/hparams")
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", required=True, choices=["P0", "P1", "P2", "P3"])
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--check-every", type=int, default=1)
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--expert-rank", type=int, default=32)
    ap.add_argument("--max-control-cells", type=int, default=64)
    args = ap.parse_args()

    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    import anndata as ad

    ChemCPA, load_dataset_splits, PerturbationDataModule = _import_chemcpa(args.chemcpa_src)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg, datasets, dataset, dm, dcfg = build(args, ChemCPA, load_dataset_splits,
                                             PerturbationDataModule)
    # the official learning rates, scaled by the one allowed factor
    for k in ["autoencoder_lr", "adversary_lr", "dosers_lr"]:
        cfg["model"]["hparams"][k] = float(cfg["model"]["hparams"][k]) * args.lr_scale

    lm, dense_hidden, parity = make_module(ChemCPA, cfg, dcfg, args.arm, args)
    counts = param_counts(lm.model)
    shared_sha = residual_fingerprint(lm.model)

    adata = ad.read_h5ad(args.adata)
    drug_to_idx = {d: i for i, d in enumerate(dataset.drugs_names_unique_sorted)}
    lm.condition_evaluator = ConditionEvaluator(
        adata, Path(args.v2_dir), Path(args.manifest_dir), "val", drug_to_idx,
        dataset.atomic_covars_dict["cell_type"],
        max_control_cells=args.max_control_cells, seed=args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = L.Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1,
                        logger=CSVLogger(str(out), name="csv"),
                        max_epochs=args.max_epochs,
                        check_val_every_n_epoch=args.check_every,
                        enable_checkpointing=False, enable_progress_bar=False,
                        num_sanity_val_steps=0)
    trainer.fit(lm, datamodule=dm)
    train_s = time.time() - t0

    if lm.best_state is not None:
        lm.load_state_dict(lm.best_state)
    # Lightning returns the module to CPU when `fit` ends, but `ComPert.predict` moves its inputs
    # to the device string the model was constructed with ("cuda"), so the final evaluation has to
    # put the parameters back on that device explicitly.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lm = lm.to(dev)
    score, P, T, L_ = lm.condition_evaluator.evaluate(lm.model, device=dev)

    peak = {}
    if torch.cuda.is_available():
        peak = dict(peak_alloc_MB=torch.cuda.max_memory_allocated() / 2 ** 20,
                    gpu=torch.cuda.get_device_name(0))
    router = None
    if getattr(lm.model, "last_router_weights", None) is not None:
        pi = lm.model.last_router_weights.detach().float().cpu().numpy()
        ent = -(pi * np.log(np.clip(pi, 1e-12, None))).sum(1)
        router = dict(mean_weight=pi.mean(0).tolist(),
                      entropy_mean=float(ent.mean()), entropy_max=float(np.log(pi.shape[1])),
                      weight_sd_across_inputs=pi.std(0).tolist(),
                      weight_sd_mean=float(pi.std(0).mean()))

    rec = dict(arm=args.arm, lr_scale=args.lr_scale, seed=args.seed,
               hparams=dict(cfg["model"]["hparams"]), params=counts,
               dense_hidden=dense_hidden, parity=parity,
               shared_init_sha256=shared_sha,
               epochs_run=len(lm.history), best_epoch=lm.best["epoch"],
               val_macro_delta_rmse=score, n_val_conditions=len(lm.condition_evaluator.rows),
               train_seconds=round(train_s, 1), router_usage_val=router, **peak,
               max_epochs=args.max_epochs,
               # "budget limited" means the cap stopped a run that was still improving, not simply
               # that the cap was reached. A run whose best epoch is far behind the last one was
               # not short of compute; it was getting worse.
               budget_limited=bool(len(lm.history) >= args.max_epochs
                                   and lm.best["epoch"] >= len(lm.history) - 2),
               early_stopping="none; chemCPA's own patience counter is not wired into the "
                              "Lightning loop, so every run goes to max_epochs and the whole "
                              "validation curve is recorded")
    torch.save(dict(state_dict=lm.best_state, record=rec), out / "best.pt")
    (out / "record.json").write_text(json.dumps(rec, indent=1))
    np.save(out / "val_pred.npy", P.astype(np.float32))
    np.save(out / "val_truth.npy", T.astype(np.float32))
    np.save(out / "val_line.npy", L_.astype(np.int32))
    pd.DataFrame(lm.history).to_csv(out / "history.tsv", sep="\t", index=False)
    print(json.dumps({k: rec[k] for k in ["arm", "lr_scale", "seed", "epochs_run", "best_epoch",
                                          "val_macro_delta_rmse", "train_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
