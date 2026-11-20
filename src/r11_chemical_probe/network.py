"""Small PyTorch mean-state and optionally matched FiLM models."""

import numpy as np
import torch
from torch import nn

from .dataset import group_weights


class Film(nn.Module):
    def __init__(self, n_drug, n_control, n_genes, width):
        super().__init__()
        self.drug = nn.Linear(n_drug, width)
        self.modulate = nn.Linear(n_control, 2 * width)
        self.rest = nn.Sequential(nn.GELU(), nn.Dropout(0.1), nn.Linear(width, width),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(width, n_genes))

    def forward(self, drug, control):
        scale, shift = self.modulate(control).chunk(2, dim=-1)
        return self.rest((1 + scale) * self.drug(drug) + shift)


class Concat(nn.Module):
    def __init__(self, n_inputs, n_genes, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_inputs, width), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(width, width), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(width, n_genes))

    def forward(self, *inputs):
        return self.layers(torch.cat(inputs, dim=-1) if len(inputs) > 1 else inputs[0])


def train_nn(name, train, target, rows, query, lr, cfg, seed, score=None, epochs=None):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    n_inputs = sum(v.shape[1] for v in train)
    if name == "F":
        model = Film(train[0].shape[1], train[1].shape[1], target.shape[1], 128)
    else:
        model = Concat(n_inputs, target.shape[1], 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    weights = group_weights(rows)
    weights *= len(weights) / weights.sum()
    train = tuple(torch.as_tensor(v, dtype=torch.float32) for v in train)
    query = tuple(torch.as_tensor(v, dtype=torch.float32) for v in query)
    target = torch.as_tensor(target, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_score, best_prediction, best_epoch, best_state = float("inf"), None, 0, None
    limit = cfg["max_epochs"] if epochs is None else epochs
    steps = 0
    used_epochs = 0
    for epoch in range(1, limit + 1):
        model.train()
        permutation = rng.permutation(len(target))
        for start in range(0, len(target), cfg["batch_rows"]):
            idx = permutation[start:start + cfg["batch_rows"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*(v[idx] for v in train))
            loss = (weights[idx] * (prediction - target[idx]).square().mean(dim=1)).mean()
            loss.backward()
            optimizer.step()
            steps += 1
        used_epochs = epoch
        if score is None:
            continue
        model.eval()
        with torch.inference_mode():
            prediction = model(*query).numpy()
        value = score(prediction)
        if value < best_score:
            best_score, best_prediction, best_epoch = value, prediction, epoch
            best_state = {name: v.detach().clone() for name, v in model.state_dict().items()}
        if epoch - best_epoch >= cfg["patience"]:
            break
    if score is None:
        model.eval()
        with torch.inference_mode():
            best_prediction = model(*query).numpy()
        best_epoch = limit
        best_state = model.state_dict()
    return best_prediction, best_state, dict(
        epoch=best_epoch, actual_epochs=used_epochs, updates=steps,
        params=sum(p.numel() for p in model.parameters()), lr=lr, seed=seed,
        validation_score=best_score if score is not None else None
    )
