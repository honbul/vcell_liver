"""Public CellHit, Velodrome and CODE-AE adapters for continuous response."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .external import SOURCE
from .io import sha256

PUBLIC_ARMS = ("cellhit", "velodrome_inductive", "velodrome_transductive",
               "codeae_inductive", "codeae_transductive")
FILES = {
    "CellHit/CellHit/models/model.py": "663695813bf016f522a1cc81561328c270be7e14d5626f74d62dad6604423c6f",
    "Velodrome/Velodrome Train/NetVelo.py": "9380626d9423c044eef0396aad78959b70eef04c2e87b57d9cd2f6c56c876222",
    "Velodrome/Velodrome Train/FuncVelov3.py": "f62597778b66185dc579537548421c78f7ac8248476ea0d2a9b5ed53566c3194",
    **{"CODE-AE/code/" + name + ".py": digest for name, digest in {
        "dsn_ae": "134f6b8dcd323d57f7156ae604c820e0236be939fab7c689d6c3667c250d5c58",
        "loss_and_metrics": "b2d5484844e977a2d578f0259562a15b69db40ea9ec6e826357a815d7c2d8111",
        "mlp": "40205cf27e260affff2c99913eef7946e526029067236df6ce3836080e495442",
        "base_ae": "1dd62c2cd556c5486c8dcf325f563fd2cd1160e9c72452b751980952902f4508",
        "types_": "9ec2a019cf6d22deac8f833f95b78e81fac35ba540cad794bd504e5af07e568d",
        "gradient_reversal": "133848fd67a1b9d7c42e28ebff655c399255cf63b514505a2e47524c71dba8da",
    }.items()},
}


def verify_upstream(root):
    for path, expected in FILES.items():
        if sha256(root / path) != expected:
            raise ValueError(f"Unregistered upstream public model source: {path}")
    return FILES


def upstream_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cellhit_predictions(train, fit_x, query, query_x, root, depth, *, threads=5):
    """Run the public CellHit ensemble for each eligible native assay and drug."""
    upstream = upstream_module(root / "CellHit/CellHit/models/model.py", "cellhit_public_model")
    predicted = np.zeros(len(query))
    support = np.zeros(len(query))
    indices = query.groupby("compound_id").indices
    for (*_, drug), rows in train.groupby([*SOURCE, "compound_id"], sort=True):
        if drug not in indices or rows.lineage_group.nunique() < 35:
            continue
        rows = rows.reset_index(drop=True)
        x = fit_x.loc[rows.cell_id].copy()
        y = rows.residual.to_numpy()
        partitions = [{
            "train_X": x.iloc[f], "train_Y": y[f], "valid_X": x.iloc[v], "valid_Y": y[v],
        } for f, v in GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=811).split(
            x, groups=rows.lineage_group
        )]
        params = {"max_depth": int(depth), "eta": 0.045, "subsample": 0.8, "colsample_bytree": 0.7,
                  "n_estimators": 450, "early_stopping_rounds": 30, "reg_lambda": 6,
                  "objective": "reg:squarederror", "tree_method": "hist", "nthread": threads}
        model = upstream.EnsembleXGBoost(params)
        model.fit(partitions, fix_seed=True)
        positions = indices[drug]
        predicted[positions] += model.predict(query_x.loc[query.iloc[positions].cell_id])["predictions"]
        support[positions] += 1
    return np.divide(predicted, support, out=np.full(len(query), np.nan), where=support > 0)


def device_tensor(frame, device):
    import torch

    return torch.as_tensor(frame.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)


def drugs_and_indices(train, query, fit_x, query_x):
    drugs = pd.Index(sorted(train.compound_id.unique()))
    row_map = {
        "fit_cell": fit_x.index.get_indexer(train.cell_id),
        "fit_drug": drugs.get_indexer(train.compound_id),
        "query_cell": query_x.index.get_indexer(query.cell_id),
        "query_drug": drugs.get_indexer(query.compound_id),
    }
    if any(np.any(pos < 0) for pos in row_map.values()):
        raise ValueError("Missing cell or compound representation for public baseline")
    return drugs, row_map


def sample_indices(n, size, device):
    import torch

    return torch.randperm(n, device=device)[:size]


def velodrome_predictions(train, fit_x, query, query_x, root, alignment, *, device, transductive):
    """Run upstream encoder and CORAL with one head per source and compound outputs."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    root = root / "Velodrome/Velodrome Train"
    structure = upstream_module(root / "NetVelo.py", "velodrome_public_network")
    objective = upstream_module(root / "FuncVelov3.py", "velodrome_public_objectives")
    fit_data, query_data = device_tensor(fit_x, device), device_tensor(query_x, device)
    drugs, position = drugs_and_indices(train, query, fit_x, query_x)
    encoder, _, _ = structure.Network(SimpleNamespace(hd=1, ldr=0.25, seed=811), fit_data)
    encoder = encoder.to(device)
    by_source = train.groupby(SOURCE, sort=True).indices
    heads = nn.ModuleList(nn.Linear(128, len(drugs)) for _ in by_source).to(device)
    y = torch.tensor(train.residual.to_numpy(), dtype=torch.float32, device=device) * 5
    fit_cell = torch.as_tensor(position["fit_cell"], device=device)
    fit_drug = torch.as_tensor(position["fit_drug"], device=device)
    support = np.zeros((len(heads), len(drugs)), dtype=bool)
    for j, indices in enumerate(by_source.values()):
        support[j, position["fit_drug"][indices]] = True
    opt = torch.optim.AdamW([*encoder.parameters(), *heads.parameters()], lr=0.001, weight_decay=0.002)
    unlabelled = query_data if transductive else fit_data
    for epoch in range(50):
        batches = [
            torch.as_tensor(indices, device=device)[torch.randperm(len(indices), device=device)].split(768)
            for indices in by_source.values()
        ]
        for step in range(max(map(len, batches))):
            u = unlabelled[sample_indices(len(unlabelled), 96, device)]
            latent_u = encoder(u)
            out_u = [head(latent_u) for head in heads]
            losses, covariances, consistency = [], [], []
            for j, batch in enumerate(batches):
                ix = batch[step % len(batch)]
                if len(ix) < 2:
                    continue
                latent = encoder(fit_data[fit_cell[ix]])
                p = heads[j](latent)[torch.arange(len(ix), device=device), fit_drug[ix]]
                losses.append(F.mse_loss(p, y[ix]))
                if alignment:
                    covariances.append(objective.coral(latent, latent_u))
                    for k in range(j):
                        common = support[j] & support[k]
                        if common.any():
                            consistency.append(F.mse_loss(out_u[j][:, common], out_u[k][:, common]))
            loss = torch.stack(losses).mean()
            if covariances:
                loss = loss + alignment * torch.stack(covariances).mean()
            if consistency:
                loss = loss + 0.2 * alignment * torch.stack(consistency).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    encoder.eval()
    heads.eval()
    with torch.no_grad():
        features = encoder(query_data)
        predictions = np.stack([head(features).cpu().numpy() for head in heads])
    combined = (predictions * support[:, None, :]).sum(axis=0) / support.sum(axis=0)
    return combined[position["query_cell"], position["query_drug"]] / 5


def codeae_predictions(train, fit_x, query, query_x, root, alignment, *, device, transductive):
    """Run public shared/private encoders and MMD before continuous-response fine-tuning."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    sys.path.insert(0, str(root / "CODE-AE/code"))
    try:
        from dsn_ae import DSNAE
        from loss_and_metrics import mmd_loss
        from mlp import MLP
    finally:
        sys.path.pop(0)
    x, x_query = device_tensor(fit_x, device), device_tensor(query_x, device)
    drugs, index = drugs_and_indices(train, query, fit_x, query_x)
    embedding = 64
    shared = MLP(x.shape[1], embedding, [384, 192], dop=0.15).to(device)
    decoder = MLP(embedding * 2, x.shape[1], [192, 384], dop=0.15).to(device)
    options = dict(shared_encoder=shared, decoder=decoder, input_dim=x.shape[1],
                   latent_dim=embedding, hidden_dims=[192, 96], dop=0.15)
    source, target = DSNAE(**options).to(device), DSNAE(**options).to(device)
    unlabelled = x_query if transductive else x
    opt = torch.optim.AdamW([*shared.parameters(), *decoder.parameters(), *source.private_encoder.parameters(),
                             *target.private_encoder.parameters()], lr=0.001)
    for _ in range(400):
        xx = x[sample_indices(len(x), 64, device)]
        uu = unlabelled[sample_indices(len(unlabelled), 64, device)]
        sx, ux = source(xx), target(uu)
        loss = source.loss_function(*sx)["loss"] + target.loss_function(*ux)["loss"]
        if alignment:
            loss = loss + alignment * mmd_loss(sx[2], ux[2], device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    regressor = nn.Sequential(nn.Linear(embedding, 128), nn.GELU(), nn.Dropout(0.2),
                              nn.Linear(128, len(drugs))).to(device)
    opt = torch.optim.AdamW([*shared.parameters(), *regressor.parameters()], lr=0.0005, weight_decay=0.002)
    y = torch.tensor(train.residual.to_numpy(), dtype=torch.float32, device=device) * 5
    fit_cell = torch.as_tensor(index["fit_cell"], device=device)
    fit_drug = torch.as_tensor(index["fit_drug"], device=device)
    weights = torch.as_tensor(train.compound_id.map(train.compound_id.value_counts()).to_numpy(),
                              dtype=torch.float32, device=device).reciprocal()
    weights /= weights.mean()
    for _ in range(45):
        for ix in torch.randperm(len(train), device=device).split(768):
            values = regressor(shared(x[fit_cell[ix]]))[torch.arange(len(ix), device=device), fit_drug[ix]]
            loss = torch.mean(F.mse_loss(values, y[ix], reduction="none") * weights[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    shared.eval()
    regressor.eval()
    with torch.no_grad():
        predicted = regressor(shared(x_query)).cpu().numpy()
    return predicted[index["query_cell"], index["query_drug"]] / 5


def public_predictions(method, training, query, fit_x, query_x, root: Path, setting, *, gpu):
    if method == "cellhit":
        return cellhit_predictions(training, fit_x, query, query_x, root, setting)
    import torch

    torch.manual_seed(811)
    torch.set_num_threads(4)
    device = torch.device("cuda" if gpu else "cpu")
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("Registered GPU run requires an available CUDA device")
    transductive = method.endswith("_transductive")
    run = velodrome_predictions if method.startswith("velodrome") else codeae_predictions
    return run(training, fit_x, query, query_x, root, setting, device=device, transductive=transductive)
