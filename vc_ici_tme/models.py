"""Cohort-weighted classical models and directed edge-conditioned graph attention."""
from dataclasses import dataclass
from math import sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .reference import TYPES


CANDIDATES = {
    "ridge": (5, 50, 500),
    "elasticnet": (.005, .025, .125),
    "lightgbm": ((100, 7), (180, 15), (240, 23)),
    "tme_gnn": ((32, 36, .20), (48, 72, .30), (64, 100, .40)),
    "tme_no_edges": ((32, 36, .20), (48, 72, .30), (64, 100, .40)),
    "tme_no_fractions": ((32, 36, .20), (48, 72, .30), (64, 100, .40)),
    "tme_no_pathways": ((32, 36, .20), (48, 72, .30), (64, 100, .40)),
    "fraction_path_ridge": (5, 50, 500),
}


def sample_weights(cohort):
    _, ids, sizes = np.unique(cohort, return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(sizes[ids])
    return (weights / weights.mean()).astype(np.float32)


@dataclass
class Classical:
    columns: np.ndarray
    scaler: StandardScaler | None
    model: object
    program: bool

    def predict(self, profiles, ids):
        values = profiles.nodes[ids].reshape(len(ids), -1) if self.program else profiles.bulk[ids]
        values = values[:, self.columns]
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(values)
        p = self.model.predict_proba(values)[:, 1].clip(1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


def classical_fit(name, config, profiles, rows, y, cohort, seed):
    program = name == "fraction_path_ridge"
    values = profiles.nodes[rows].reshape(len(rows), -1) if program else profiles.bulk[rows]
    # Per-cohort variance and associations are never computed over the
    # omitted cohort or over the HCC query.
    n = min(750, values.shape[1])
    order = np.argsort(-values.var(axis=0), kind="stable")[:n]
    selected = values[:, order]
    weights = sample_weights(cohort)
    scaler = None
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        trees, leaves = config
        model = LGBMClassifier(n_estimators=trees, num_leaves=leaves,
                               min_child_samples=9, learning_rate=.035, colsample_bytree=.75,
                               reg_lambda=5, n_jobs=4, verbosity=-1, random_state=seed)
    else:
        scaler = StandardScaler().fit(selected)
        selected = scaler.transform(selected)
        if name == "elasticnet":
            model = LogisticRegression(C=config, penalty="elasticnet", solver="saga",
                                       l1_ratio=.30, max_iter=2400, random_state=seed)
        else:
            model = RidgeClassifier(alpha=config)
    model.fit(selected, y, sample_weight=weights)
    return Classical(order, scaler, model, program)


class EdgeAttention(nn.Module):
    def __init__(self, width, edge_width, dropout):
        super().__init__()
        self.width = width
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.relation_gate = nn.Linear(edge_width, 1, bias=False)
        self.relation_message = nn.Linear(edge_width, width, bias=False)
        self.out = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(self, node, edge):
        # From sender i to receiver j. Direct scores and edge messages retain
        # the biological direction of each ligand-receptor pair.
        query, key, value = self.query(node), self.key(node), self.value(node)
        affinity = torch.einsum("bjd,bid->bij", query, key) / sqrt(self.width)
        attention = (affinity + self.relation_gate(edge).squeeze(-1)).softmax(dim=1)
        message = value[:, :, None, :] + self.relation_message(edge)
        return self.norm(node + self.out((attention[..., None] * message).sum(dim=1)))


class TMEGraph(nn.Module):
    def __init__(self, node_width, edge_width, width, dropout, use_edges):
        super().__init__()
        self.encoder = nn.Linear(node_width, width)
        self.type_embeddings = nn.Parameter(torch.randn(len(TYPES), width) * .05)
        self.layers = nn.ModuleList([EdgeAttention(width, edge_width, dropout) for _ in range(2)]
                                    if use_edges else [])
        self.readout = nn.Sequential(nn.Flatten(), nn.Linear(len(TYPES) * width, width),
                                     nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(width, 1))

    def forward(self, node, edge):
        state = F.gelu(self.encoder(node) + self.type_embeddings)
        for layer in self.layers:
            state = layer(state, edge)
        return self.readout(state).ravel()


@dataclass
class GraphModel:
    name: str
    network: TMEGraph
    device: str
    node_center: np.ndarray
    node_scale: np.ndarray
    edge_scale: np.ndarray

    def arrays(self, nodes, edges):
        x = ((nodes - self.node_center) / self.node_scale).clip(-9, 9)
        e = (edges / self.edge_scale).clip(0, 12)
        if self.name == "tme_no_edges":
            e = np.zeros_like(e)
        elif self.name == "tme_no_fractions":
            x[..., -1] = 0
            # The original biological edge includes mixture proportions.
            # The fitter supplies independently regenerated unweighted edges.
        elif self.name == "tme_no_pathways":
            x[..., :-1] = 0
        return tuple(torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in (x, e))

    def predict(self, nodes, edges):
        self.network.eval()
        with torch.inference_mode():
            return self.network(*self.arrays(nodes, edges)).detach().cpu().numpy()


def graph_fit(name, config, nodes, edges, y, cohort, seed, device):
    width, epochs, dropout = config
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    network = TMEGraph(nodes.shape[-1], edges.shape[-1], width, dropout, name != "tme_no_edges").to(device)
    model = GraphModel(name, network, device, nodes.mean(axis=0),
                       nodes.std(axis=0).clip(min=.04),
                       np.maximum(np.quantile(edges, .90, axis=0).astype(np.float32), .05))
    node, edge = model.arrays(nodes, edges)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    weight = torch.as_tensor(sample_weights(cohort), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)
    generator = np.random.default_rng(seed)
    network.train()
    for _ in range(epochs):
        for rows in np.array_split(generator.permutation(len(node)), max(1, len(node) // 48)):
            optimizer.zero_grad(set_to_none=True)
            logits = network(node[rows], edge[rows])
            loss = F.binary_cross_entropy_with_logits(logits, target[rows], weight=weight[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 3)
            optimizer.step()
        scheduler.step()
    return model
