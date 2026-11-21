"""Fresh-fit State Transition with molecular conditioning and pooled-well supervision."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.goal_tracks.t1_data import DOSES, MIN_CELLS, DoseSource, slice_name, validate_protocol
from src.goal_tracks.t1_io import load_task, write_task
from src.r11_chemical_probe.common import config, read_json, sha256
from src.r11_chemical_probe.dataset import KEY
from src.r12_multiaxis.data import Fit

UPSTREAM = "9bbfe78a434a55205e4de834e1ea99f85f7a3add"
MODEL = "State-Morgan-well-adaptation"
TOKENS_PER_WELL = 8


def load_network(upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError("State source revision changed")
    # The released gene-space ST path does not use the optional VCI decoder.
    # Its package import is deferred in the run-local checkout.
    modified = subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--name-only"], text=True
    ).splitlines()
    if modified != ["src/state/tx/models/decoders.py"]:
        raise ValueError(f"Unexpected State changes: {modified}")
    sys.path.insert(0, str(upstream / "src"))
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    return StateTransitionPerturbationModel


def build_network(network, size, drug_width, cell_set_len=TOKENS_PER_WELL):
    return network(
        input_dim=size, hidden_dim=256, output_dim=size, pert_dim=drug_width, gene_dim=size,
        embed_key="X_hvg", output_space="gene", cell_set_len=cell_set_len,
        gene_decoder_bool=False, finetune_vci_decoder=False, loss="mse", dropout=0.1,
        predict_residual=True, n_encoder_layers=2, n_decoder_layers=2, transformer_backbone_key="GPT2",
        transformer_backbone_kwargs={
            "vocab_size": 1, "n_embd": 256, "n_head": 8, "n_layer": 4, "use_cache": False,
            "resid_pdrop": 0.1, "attn_pdrop": 0.0, "embd_pdrop": 0.0,
        },
    )


def control_sets(fit, cell_folders):
    role = fit.source.role
    wanted = set(fit.train_units) | set(fit.query_units)
    lookup = {}
    for cell_folder in cell_folders:
        row_path, matrix_path = cell_folder / f"{role}_rows.tsv", cell_folder / f"{role}_cells.npz"
        receipt = read_json(cell_folder / "receipt.json")
        for file in (row_path, matrix_path):
            if sha256(file) != receipt["files"][file.name]:
                raise ValueError("State control cell sample differs from its receipt")
        rows = pd.read_csv(row_path, sep="\t")
        if not rows.role.eq(role).all():
            raise ValueError("State saw the other DMSO role")
        counts = sparse.load_npz(matrix_path)
        panel = counts[:, fit.panel]
        libraries = np.asarray(counts.sum(axis=1)).ravel()
        positions = rows.groupby(KEY, sort=False).indices
        for unit in sorted(wanted & positions.keys()):
            if unit in lookup:
                raise ValueError(f"State control set appears in both registered caches: {unit}")
            index = positions[unit]
            if len(index) < TOKENS_PER_WELL:
                raise ValueError(f"State control set too small for {unit}")
            lookup[unit] = np.stack([
                np.log1p(np.asarray(panel[chunk].sum(axis=0)).ravel() * (1e4 / libraries[chunk].sum()))
                for chunk in np.array_split(index, TOKENS_PER_WELL)
            ])
    if missing := wanted.difference(lookup):
        raise ValueError(f"State lacks {len(missing)} control sets, including {sorted(missing)[:3]}")
    units = sorted(wanted)
    features = np.stack([lookup[unit] for unit in units]).astype(np.float32)
    indices = {unit: i for i, unit in enumerate(units)}
    return features, [indices[key] for key in fit.train_units], [indices[key] for key in fit.query_units]


def forward(model, basal, chemical):
    batch = {"ctrl_cell_emb": basal, "pert_emb": chemical.unsqueeze(1).expand(-1, basal.shape[1], -1)}
    return model(batch).reshape(basal.shape).mean(dim=1)


def fit_model(fit, upstream, cell_folders):
    network = load_network(upstream)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda")

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    control, train_control_idx, query_control_idx = control_sets(fit, cell_folders)
    control = tensor(control)
    train_control_idx, query_control_idx = [torch.as_tensor(x, device=device)
                                            for x in (train_control_idx, query_control_idx)]
    train_drug, query_drug = map(tensor, (fit.train.chemical["E"], fit.query.chemical["E"]))
    query_control_mean = tensor(fit.query.control)
    target = tensor(fit.train.treated)
    weight = tensor(fit.weight / np.mean(fit.weight))
    net = build_network(network, len(fit.panel), train_drug.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.001)
    for epoch in range(40):
        net.train()
        order = torch.randperm(len(target), device=device)
        for idx in order.split(128):
            prediction = forward(net, control[train_control_idx[idx]], train_drug[idx])
            loss = ((prediction - target[idx]).square().mean(dim=1) * weight[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/40", flush=True)
    net.eval()
    with torch.inference_mode():
        query = torch.cat([forward(net, control[idx], drug)
                           for idx, drug in zip(query_control_idx.split(128), query_drug.split(128))])
    relative = upstream / "src/state/tx/models"
    return (query - query_control_mean).cpu().numpy(), {
        "upstream": UPSTREAM,
        "architecture_sha256": sha256(relative / "state_transition.py"),
        "decoder_import_patch_sha256": sha256(relative / "decoders.py"),
        "control_cells_receipt_sha256": {str(path): sha256(path / "receipt.json") for path in cell_folders},
        "input": "ECFP4 instead of categorical perturbations; 8 pooled same-role DMSO control tokens",
        "model": "Official State Transition GPT2 freshly fit to 2,000-gene well mean with MSE",
        "fit_epochs": 40,
        "network_seed": 42,
        "torch": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    for key in ("assets", "config", "staged", "out", "protocol", "extra-cells"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--task", type=int, choices=range(36))
    args = parser.parse_args()
    if args.smoke:
        net = build_network(load_network(args.upstream), 20, 16)
        result = forward(net, torch.rand(2, TOKENS_PER_WELL, 20), torch.rand(2, 16))
        result.sum().backward()
        print(f"State smoke passed: {tuple(result.shape)}")
        return
    if any(getattr(args, key) is None for key in ("assets", "config", "staged", "out", "protocol", "extra_cells", "task")):
        parser.error("Production requires all input paths and the task ID")
    validate_protocol(args.protocol)
    index, role_index = divmod(args.task, 2)
    index, fold = divmod(index, 3)
    dose_index, minimum_index = divmod(index, 2)
    dose, minimum, role = DOSES[dose_index], MIN_CELLS[minimum_index], "AB"[role_index]
    destination = args.out / slice_name(dose, minimum, fold, role)
    if (destination / "receipt.json").exists():
        existing, _, _, _ = load_task(destination, args.staged, args.protocol, dose, minimum, fold, role)
        if existing["models"] != [MODEL]:
            raise ValueError("Existing task does not belong to State")
        print(f"Verified existing State prediction: {destination.name}")
        return
    source = DoseSource(config(args.config), args.assets, args.staged, dose, minimum, fold, role)
    fit = Fit(source, source.fit_rows, source.query_rows, panel=source.panel)
    original = args.assets / "cells-v2"
    cell_folders = (original,) if dose == 0.5 else (args.extra_cells,)
    if dose == 0.05:
        cell_folders = (original, args.extra_cells)
    prediction, receipt = fit_model(fit, args.upstream, cell_folders)
    write_task(destination, source, fit,
               {MODEL: prediction}, selection={MODEL: receipt})


if __name__ == "__main__":
    main()
