"""Read the finished official chemCPA example and write down what the model takes and returns.

Goal section 2 asks for one official example to be actually run and for the model's inputs,
outputs and native training procedure to be confirmed. The run itself is
`experiments/chemcpa_moe_v2/chemcpa_example.sbatch`; this loads its checkpoint and records the
shapes, the batch layout and the authors' own R2 read-out, so the description in the report is
measured rather than read off the source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemcpa-src", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--split-key", default="split_random")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.chemcpa_src)
    from chemCPA.data.data import load_dataset_splits
    from chemCPA.lightning_module import ChemCPA
    from chemCPA.train import evaluate_r2

    dparams = dict(dataset_path=a.dataset_path, perturbation_key="condition", dose_key="dose",
                   covariate_keys="cell_type", smiles_key="SMILES", degs_key="all_DEGs",
                   pert_category="cov_drug_dose_name", split_key=a.split_key, use_drugs_idx=True)
    datasets, dataset = load_dataset_splits(**dparams, return_dataset=True)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    lm = ChemCPA(hp["config"], hp["dataset_config"])
    lm.load_state_dict(ck["state_dict"])
    lm = lm.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    m = lm.model

    tr = datasets["training"]
    b = tr[0]
    layout = [f"{i}:{tuple(x.shape) if torch.is_tensor(x) else type(x).__name__}"
              for i, x in enumerate(b)]

    with torch.no_grad():
        genes = datasets["test_control"].genes[:8].to(m.device)
        n = genes.shape[0]
        di = torch.zeros(n, dtype=torch.long, device=m.device)
        do = torch.full((n,), 1.0, device=m.device)
        cov = torch.zeros((n, m.num_covariates[0]), device=m.device)
        cov[:, 0] = 1
        out, cde, (lb, de, lt) = m.predict(genes=genes, drugs_idx=di, dosages=do,
                                           covariates=[cov], return_latent_basal=True)

    r2 = {}
    for name, treated, control in [
            ("train", datasets["training_treated"], datasets["training_control"]),
            ("test", datasets["test_treated"], datasets["test_control"]),
            ("ood", datasets["ood"], datasets["test_control"])]:
        try:
            v = evaluate_r2(m, treated, control.genes)
            r2[name] = dict(zip(["R2_mean", "R2_mean_de", "R2_var", "R2_var_de"],
                                [round(float(x), 5) for x in v]))
        except Exception as e:                                   # noqa: BLE001
            r2[name] = dict(error=f"{type(e).__name__}: {e}")

    rep = dict(
        checkpoint=str(a.checkpoint), split_key=a.split_key,
        dataset=dict(n_genes=int(tr.num_genes), n_drugs=int(tr.num_drugs),
                     num_covariates=[int(x) for x in tr.num_covariates],
                     sizes={k: int(len(v.genes)) for k, v in datasets.items()}),
        hparams=dict(hp["config"]["model"]["hparams"]),
        batch_layout=layout,
        io=dict(input_genes=list(genes.shape),
                output=list(out.shape),
                output_halves="[mean | softplus(variance)], so 2 x n_genes",
                latent_basal=list(lb.shape), drug_embedding=list(de.shape),
                latent_treated=list(lt.shape), cell_drug_embedding=list(cde.shape)),
        params=dict(total=int(sum(p.numel() for p in m.parameters())),
                    encoder=int(sum(p.numel() for p in m.encoder.parameters())),
                    decoder=int(sum(p.numel() for p in m.decoder.parameters())),
                    adversary_drugs=int(sum(p.numel() for p in m.adversary_drugs.parameters())),
                    dosers=int(sum(p.numel() for p in m.dosers.parameters()))),
        objective="GaussianNLLLoss on (mean, var) against the cell's own expression; the "
                  "autoencoder step subtracts reg_adversary * drug-adversary loss and "
                  "reg_adversary_cov * covariate-adversary loss; every adversary_steps-th global "
                  "step updates the adversaries instead, with a gradient penalty",
        authors_r2=r2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
