"""Fresh fit of the upstream PRnet Gaussian generator on Tahoe well means."""

import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.nn import functional as F

from src.goal_tracks.t1_data import DOSES, MIN_CELLS, DoseSource, slice_name, validate_protocol
from src.goal_tracks.t1_io import write_task
from src.r11_chemical_probe.common import config, sha256
from src.r11_chemical_probe.dataset import ENTITY
from src.r12_multiaxis.data import Fit

UPSTREAM = "f19174bde3ed2633f54c7831799cc38c4ffc7a0d"
MODEL = "PRnet-well-adaptation"


def chemical_features(fit, dose):
    entities = pd.read_csv(fit.source.r11 / "features" / "entities.tsv", sep="\t").set_index(ENTITY)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    keys = set(fit.train.rows[ENTITY]) | set(fit.query.rows[ENTITY])
    molecules = {key: Chem.MolFromSmiles(entities.loc[key, "smiles"]) for key in sorted(keys)}
    if any(molecule is None for molecule in molecules.values()):
        raise ValueError("Invalid canonical SMILES in PRnet input")
    coded = {key: generator.GetFingerprintAsNumPy(molecule) for key, molecule in molecules.items()}
    return [
        np.stack([coded[key] for key in block.rows[ENTITY]]).astype(np.float32) * np.log10(dose + 1)
        for block in (fit.train, fit.query)
    ]


def fit_model(fit, dose, upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    relative = "models/PRnet.py"
    if revision != UPSTREAM or subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--", relative], text=True
    ):
        raise ValueError("PRnet source revision is not pristine")
    spec = importlib.util.spec_from_file_location("official_prnet", upstream / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_num_threads(4)

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    train_drug, query_drug = map(tensor, chemical_features(fit, dose))
    train_ctrl, query_ctrl = map(tensor, (fit.train.control, fit.query.control))
    target = tensor(fit.train.treated)
    weights = tensor(fit.weight / np.mean(fit.weight))
    size = len(fit.panel)
    net = module.PGM(size, 64, 10, [128], 64, [128], 1024, 0.05).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-8)
    for epoch in range(100):
        net.train()
        order = torch.randperm(len(target), device=device)
        # Equal-sized splits avoid a singleton batch in upstream BatchNorm.
        chunks = order.tensor_split(max(1, (len(order) + 511) // 512))
        for indices in chunks:
            noise = torch.randn((len(indices), 10), device=device)
            mean, raw_var = net(train_ctrl[indices], train_drug[indices], noise).split(size, dim=1)
            loss = F.gaussian_nll_loss(mean, target[indices], F.softplus(raw_var), reduction="none")
            objective = (loss.mean(dim=1) * weights[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1}/100", flush=True)
    net.eval()
    with torch.inference_mode():
        # The expected conditional Gaussian mean averages over the latent noise
        # without additionally sampling the observation noise.
        noise_grid = torch.randn((32, 10), device=device)
        prediction = []
        for ctrl, drug in zip(query_ctrl.split(256), query_drug.split(256)):
            count = len(ctrl)
            inputs = ctrl.repeat(len(noise_grid), 1)
            structures = drug.repeat(len(noise_grid), 1)
            noise = noise_grid.repeat_interleave(count, dim=0)
            mean = net(inputs, structures, noise)[:, :size].reshape(len(noise_grid), count, size).mean(0)
            prediction.append((mean - ctrl).cpu().numpy())
    return np.concatenate(prediction), {
        "upstream": UPSTREAM,
        "source_sha256": sha256(upstream / relative),
        "input": "same-role logCP10k well means; FCFP4 1024 bits with log10(1+dose_uM)",
        "model": "upstream PRnet PGM Gaussian NLL; group-balanced likelihood",
        "fit_epochs": 100,
        "network_seed": 42,
        "noise_samples": 32,
        "torch": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "config", "staged", "out", "upstream", "protocol"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--task", type=int, choices=range(36), required=True)
    args = parser.parse_args()
    validate_protocol(args.protocol)
    index, role_index = divmod(args.task, 2)
    index, fold = divmod(index, 3)
    dose_index, minimum_index = divmod(index, 2)
    dose, minimum, role = DOSES[dose_index], MIN_CELLS[minimum_index], "AB"[role_index]
    source = DoseSource(config(args.config), args.assets, args.staged, dose, minimum, fold, role)
    fit = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    prediction, receipt = fit_model(fit, dose, args.upstream)
    write_task(args.out / slice_name(dose, minimum, fold, role), source, fit, {MODEL: prediction},
               selection={MODEL: receipt})


if __name__ == "__main__":
    main()
