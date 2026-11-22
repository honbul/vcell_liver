"""Source-trained multioutput regression for transcriptome perturbations."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
import torch
from torch import nn

def validation_loss(prediction, observation, assays):
    """Mean per-source assay RMSE after removal of each systematic mean."""
    losses = []
    for assay in np.unique(assays):
        p, y = prediction[assays == assay], observation[assays == assay]
        residual = (p - p.mean(axis=0)) - (y - y.mean(axis=0))
        losses.append(np.sqrt(np.mean(np.square(residual))))
    return float(np.mean(losses))


def equal_target_weights(meta):
    count = meta.groupby("target_ensembl").target_ensembl.transform("size").to_numpy()
    return (len(count) / (meta.target_ensembl.nunique() * count)).astype(np.float32)


def context_transform(train, *other):
    mean = train.mean(axis=0)
    representation = PCA(n_components=min(24, len(train) - 1, train.shape[1]),
                         svd_solver="randomized", random_state=41026)
    representation.fit(train - mean)
    return [representation.transform(block - mean).astype(np.float32) for block in (train, *other)]


def feature_arrays(features, roles, contexts, representation, include_context):
    matrices = [features.encode(role.meta.target_ensembl, representation) for role in roles]
    if include_context:
        matrices = [np.column_stack((x, context)) for x, context in zip(matrices, contexts)]
    scaler = StandardScaler().fit(matrices[0])
    return [scaler.transform(x).clip(-12, 12).astype(np.float32) for x in matrices]


@dataclass
class Selected:
    model: object
    setting: dict
    loss: float


def fit_ridge(inputs, train, valid, values=(3, 30, 300, 3000)):
    train_x, valid_x = inputs
    weight = equal_target_weights(train.meta)
    losses = []
    for alpha in values:
        model = Ridge(alpha=alpha, solver="cholesky").fit(train_x, train.delta, sample_weight=weight)
        loss = validation_loss(model.predict(valid_x), valid.delta, valid.meta.assay.to_numpy())
        losses.append(Selected(model, {"alpha": alpha}, loss))
    return min(losses, key=lambda r: r.loss)


class ReducedRank:
    def __init__(self, y, rank, sample_weight, seed=41026):
        self.center = np.average(y, axis=0, weights=sample_weight)
        centered = y - self.center
        _, _, self.components = randomized_svd(centered * np.sqrt(sample_weight[:, None]),
                                               n_components=min(rank, min(y.shape) - 1),
                                               n_iter=5, random_state=seed)
        self.scores = centered @ self.components.T


@dataclass
class ReducedRankModel:
    center: np.ndarray
    components: np.ndarray
    model: object

    def predict(self, x):
        return self.center + self.model.predict(x) @ self.components


class LightGBMOutputs:
    def __init__(self, params):
        self.params = params
        self.models = []

    def fit(self, x, y, weight):
        from lightgbm import LGBMRegressor
        self.models = [LGBMRegressor(**self.params).fit(x, column, sample_weight=weight) for column in y.T]
        return self

    def predict(self, x):
        return np.column_stack([model.predict(x) for model in self.models]).astype(np.float32)


def fit_reduced(estimator, inputs, train, valid):
    train_x, valid_x = inputs
    weight = equal_target_weights(train.meta)
    centered = ReducedRank(train.delta, rank=32, sample_weight=weight)
    best = None
    for setting in (0, 1):
        if estimator == "elastic":
            alpha = (0.008, 0.04)[setting]
            # This sklearn release requires weighted rows to be supplied
            # explicitly for its multi-task penalized linear estimator.
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=0.2, max_iter=700, random_state=41026)
            root = np.sqrt(weight[:, None])
            model.fit(train_x * root, centered.scores * root)
            params = {"alpha": alpha, "l1_ratio": 0.2}
        else:
            leaves, minimum = ((15, 80), (31, 50))[setting]
            params = {"num_leaves": leaves, "min_child_samples": minimum, "learning_rate": 0.05,
                      "n_estimators": 160, "verbosity": -1, "n_jobs": 4, "random_state": 41026,
                      "colsample_bytree": 0.75, "reg_lambda": 3.0}
            model = LightGBMOutputs(params).fit(train_x, centered.scores, weight)
        fitted = ReducedRankModel(centered.center, centered.components, model)
        loss = validation_loss(fitted.predict(valid_x), valid.delta, valid.meta.assay.to_numpy())
        candidate = Selected(fitted, {"rank": centered.components.shape[0], **params}, loss)
        if best is None or candidate.loss < best.loss:
            best = candidate
    return best


class TranscriptDecoder(nn.Module):
    def __init__(self, dimensions, width, output, graph_block=0):
        super().__init__()
        self.graph_block = graph_block
        primary = dimensions - 2 * graph_block
        self.root = nn.Linear(primary, width)
        if graph_block:
            self.first = nn.Linear(graph_block, width, bias=False)
            self.second = nn.Linear(graph_block, width, bias=False)
        self.decode = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.12),
                                    nn.Linear(width, width), nn.GELU(), nn.Dropout(0.12),
                                    nn.Linear(width, output))

    def forward(self, x):
        n = self.graph_block
        if n:
            root = torch.cat((x[:, :n], x[:, 3 * n:]), dim=1)
            hidden = self.root(root) + self.first(x[:, n:2 * n]) + self.second(x[:, 2 * n:3 * n])
        else:
            hidden = self.root(x)
        return self.decode(hidden)


class TorchPrediction:
    def __init__(self, model, center, device):
        self.model, self.center, self.device = model, center, device

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return np.concatenate([
            self.model(torch.as_tensor(batch, device=self.device)).cpu().numpy()
            for batch in np.array_split(x, max(1, (len(x) + 511) // 512))
        ]) + self.center


def fit_neural(inputs, train, valid, device, seed, epochs, graph_block=0):
    x, vx = inputs
    w = equal_target_weights(train.meta)
    mean = np.average(train.delta, axis=0, weights=w)
    y = torch.as_tensor(train.delta - mean, device=device)
    tx = torch.as_tensor(x, device=device)
    weights = torch.as_tensor(w, device=device)
    variance = y.square().mean(dim=0).sqrt().clip(min=0.045)
    gene_weight = torch.sqrt(torch.median(variance) / variance).clip(0.3, 2.0)
    best = None
    started = perf_counter()
    for width in (128, 256):
        torch.manual_seed(seed)
        random = np.random.default_rng(seed)
        model = TranscriptDecoder(x.shape[1], width, train.delta.shape[1], graph_block).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
        patience = 0
        smallest_loss = float("inf")
        wrapper = TorchPrediction(model, mean, device)
        for epoch in range(epochs):
            model.train()
            for indices in np.array_split(random.permutation(len(x)), max(1, (len(x) + 255) // 256)):
                index = torch.as_tensor(indices, device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tx[index])
                mse = ((prediction - y[index]).square() * gene_weight).mean(dim=1)
                correlation = nn.functional.cosine_similarity(prediction, y[index], dim=1)
                loss = ((mse + 0.005 * (1 - correlation)) * weights[index]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            value = validation_loss(wrapper.predict(vx), valid.delta, valid.meta.assay.to_numpy())
            if best is None or value < best.loss:
                frozen = TranscriptDecoder(x.shape[1], width, train.delta.shape[1], graph_block).to(device)
                frozen.load_state_dict(model.state_dict())
                best = Selected(TorchPrediction(frozen, mean, device), {
                    "width": width, "epoch": epoch + 1, "seed": seed,
                    "wall_seconds_at_selection": perf_counter() - started,
                }, value)
            if value < smallest_loss - 1e-5:
                smallest_loss, patience = value, 0
            else:
                patience += 1
                if patience == 7:
                    break
    return best


def copy_same_gene(train, roles, mean):
    means = {
        gene: train.delta[frame.index.to_numpy()].mean(axis=0)
        for gene, frame in train.meta.groupby("target_ensembl")
    }
    return [np.stack([means.get(gene, mean) for gene in role.meta.target_ensembl]) for role in roles]


def targeted_attenuation(axes, train, valid, roles):
    def features(role):
        columns = axes.get_indexer(role.meta.target_ensembl)
        rows = np.flatnonzero(columns >= 0)
        return rows, columns[rows], role.context[rows, columns[rows]]

    rows, cols, x = features(train)
    raw = train.delta[rows, cols]
    slope = float(np.clip(-np.dot(x, raw) / max(float(np.dot(x, x)), 1e-8), 0, 1))
    output = []
    for role in (valid, *roles):
        rows, cols, x = features(role)
        estimate = np.zeros_like(role.context)
        estimate[rows, cols] = -slope * x
        output.append(estimate)
    return slope, output
