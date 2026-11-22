"""Two-axis pharmacogenomics heads with train-only RNA and frozen FM projections."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch import nn

from vc_build.io import sha256
from vc_build.v0plus_cv import panel_targets

from .adaptive import Projection, sample_rank

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
CELL_MODALITIES = ("scgpt", "geneformer")
DRUG_MODALITIES = ("molformer", "clamp")
HCC_NAMES = ("liver", "hepat", "hepatic")
SOLID_NAMES = ("lung", "breast", "bowel", "colon", "brain", "renal", "pancrea", "skin", "ovary", "stomach")


def fm_read(root):
    path = root / "vectors.npz"
    stamp = json.loads((root / "receipt.json").read_text())["output"]
    if path.stat().st_size != stamp["bytes"] or sha256(path) != stamp["sha256"]:
        raise ValueError(f"Foundation embedding differs from its frozen receipt: {path}")
    with np.load(path, allow_pickle=False) as data:
        ids, values = data["id"].astype(str), data["vectors"].astype(np.float32)
    if not pd.Index(ids).is_unique or not np.isfinite(values).all():
        raise ValueError("Malformed FM embedding keys or matrix")
    return pd.DataFrame(values, index=ids), stamp["sha256"]


def fm_paths(root, *, limore=False):
    cell_root = root / ("limore-v2" if limore else "bulk-v2")
    blocks = {name: fm_read(cell_root / name) for name in CELL_MODALITIES}
    blocks.update({name: fm_read(root / "chemistry-v1" / name) for name in DRUG_MODALITIES})
    return {key: data[0] for key, data in blocks.items()}, {key: data[1] for key, data in blocks.items()}


@dataclass
class FrozenFeatures:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray

    @classmethod
    def fit(cls, matrix, width):
        mask = np.isfinite(matrix).all(axis=1)
        if mask.sum() < 4:
            return None
        train = matrix[mask]
        center = train.mean(axis=0)
        scale = np.maximum(train.std(axis=0), .0005)
        _, _, basis = np.linalg.svd((train-center)/scale, full_matrices=False)
        return cls(center, scale, basis[:min(width, len(basis)-1)])

    def transform(self, bank, ids):
        if bank is None:
            return np.zeros((len(ids), len(self.basis)+1), dtype=np.float32)
        x = bank.reindex(ids).to_numpy(dtype=np.float32)
        used = np.isfinite(x).all(axis=1)
        adjusted = np.zeros((len(x), len(self.basis)+1), dtype=np.float32)
        adjusted[used, :-1] = np.clip((x[used]-self.mean)/self.scale, -8., 8.) @ self.basis.T
        adjusted[used, :-1] /= np.sqrt(max(len(self.basis), 1))
        adjusted[used, -1] = 1
        return adjusted


@dataclass
class Views:
    projection: Projection
    selected: np.ndarray
    cell_fm: dict
    drug_fm: dict
    ids: list
    fingerprints: np.ndarray

    @classmethod
    def fit(cls, expression, fit, active, banks):
        basis = Projection.fit(expression, "rank_z")
        variable = basis.hvg[:active]
        measure = fit[["compound_id", "isomeric_smiles"]].drop_duplicates().sort_values("compound_id")
        if measure.compound_id.duplicated().any():
            raise ValueError("One compound has multiple structural identities")
        compounds = list(measure.compound_id)
        x = np.stack([MORGAN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in measure.isomeric_smiles])
        cell, drugs = ({}, {})
        for modality, container, ids in (
            *((name, cell, expression.index) for name in CELL_MODALITIES),
            *((name, drugs, compounds) for name in (*DRUG_MODALITIES, "cmap")),
        ):
            if modality in banks:
                projection = FrozenFeatures.fit(banks[modality].reindex(ids).to_numpy(np.float32),
                                                48 if modality in CELL_MODALITIES else 24)
                if projection is not None:
                    container[modality] = projection
        return cls(basis, variable, cell, drugs, compounds, x.astype(np.float32))

    def cells(self, rna, banks):
        rna_part = self.projection.transform(rna)[:, self.selected]
        cell_fm = [p.transform(banks.get(name), rna.index) for name, p in self.cell_fm.items()]
        return rna_part, np.concatenate(cell_fm, axis=1) if cell_fm else np.empty((len(rna), 0), np.float32)

    def drugs(self, banks):
        drug_fm = [p.transform(banks.get(name), self.ids) for name, p in self.drug_fm.items()]
        return np.concatenate([self.fingerprints, *drug_fm], axis=1)


class TwoAxisNet(nn.Module):
    def __init__(self, mode, raw_cells, extra_cells, n_drug_features, n_drugs, width, rank, dropout):
        super().__init__()
        self.mode = mode
        self.cell = nn.Sequential(nn.Linear(raw_cells+extra_cells, width), nn.LayerNorm(width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, rank))
        self.drug = nn.Sequential(nn.Linear(n_drug_features, width), nn.LayerNorm(width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, rank))
        self.id_embeddings = nn.Embedding(n_drugs, rank)
        self.additive = nn.Linear(rank, 1)
        self.residual = nn.Sequential(nn.LayerNorm(rank), nn.Linear(rank, rank), nn.GELU(), nn.Linear(rank, 1))
        self.hyper = nn.Linear(rank, 2*rank)
        self.dropout = dropout

    def forward(self, expression, fm, molecule):
        if self.training and fm.shape[1]:
            fm = fm * (torch.rand(len(fm), 1, device=fm.device) > .28)
        cell = self.cell(torch.cat((expression, fm), dim=1))
        drug = self.drug(molecule) + self.id_embeddings.weight
        base = cell @ drug.T / np.sqrt(cell.shape[1])
        if self.mode == "twotower":
            base = base + self.residual(cell[:, None, :] * drug[None, :, :]).squeeze(-1)
        elif self.mode == "hypernet":
            gamma, bias = self.hyper(drug).chunk(2, dim=1)
            interactions = cell[:, None] * (1 + .25*gamma[None].tanh()) + .25*bias[None]
            base = base + self.residual(interactions).squeeze(-1)
        return base + self.additive(cell)


def tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def objective(p, y, mask, anchor, weights, dual):
    w = weights[:, None] * mask
    smooth = torch.nn.functional.smooth_l1_loss(p, y, beta=.45, reduction="none")
    value = (smooth*w).sum()/w.sum()
    if not dual:
        return value
    col_weight = w.sum(0).clamp_min(1e-6)
    px = (p - (p*w).sum(0)/col_weight) * w.sqrt()
    yx = (y - (y*w).sum(0)/col_weight) * w.sqrt()
    corr = (px*yx).sum(0)/(px.square().sum(0)*yx.square().sum(0)).clamp_min(1e-8).sqrt()
    m = mask.sum(0) >= 8
    pearson = -corr[m].mean() if m.any() else p.sum()*0
    truth = ((y + anchor)/.17).masked_fill(~mask, -1e4)
    estimate = ((p + anchor)/.17).masked_fill(~mask, -1e4)
    listnet = -(truth.softmax(1) * estimate.log_softmax(1)).sum(1)
    within_cell = (listnet*weights).sum()/weights.sum()
    return value + .08*pearson + .035*within_cell


@dataclass
class LeanFit:
    params: tuple
    views: Views
    state: dict
    anchor: pd.Series
    outcome_scale: float
    molecule_bank: np.ndarray

    def predict(self, query, expression, banks, *, device="cpu"):
        mode, _, width, rank, dropout, *_ = self.params
        expression, fm = self.views.cells(expression, banks)
        model = TwoAxisNet(mode, expression.shape[1], fm.shape[1], self.molecule_bank.shape[1],
                           len(self.views.ids), width, rank, dropout).to(device)
        model.load_state_dict(self.state)
        model.eval()
        with torch.inference_mode():
            v = model(tensor(expression, device), tensor(fm, device),
                      tensor(self.molecule_bank, device)).cpu().numpy() * self.outcome_scale
        cells = pd.Index(query.cell_id.drop_duplicates())
        # The expression matrix was ordered to match these IDs by the caller.
        i = cells.get_indexer(query.cell_id)
        j = pd.Index(self.views.ids).get_indexer(query.compound_id)
        if (j < 0).any():
            raise ValueError("Compound is outside the locked sensitivity panel")
        return query.compound_id.map(self.anchor).to_numpy() + v[i, j]


CONFIGS = {
    "bilinear_rna": ("bilinear", 512, 96, 16, .18, 36, True, 1., False),
    "two_tower_rna": ("twotower", 768, 128, 20, .20, 44, True, 1., False),
    "hypernet_rna": ("hypernet", 512, 96, 16, .24, 44, True, 1., False),
    "two_tower_fm": ("twotower", 512, 128, 24, .23, 44, True, 1., True),
    "hypernet_fm": ("hypernet", 512, 128, 20, .24, 44, True, 1., True),
    "bilinear_fm": ("bilinear", 512, 96, 24, .23, 44, True, 1., True),
    "two_tower_cmap": ("twotower", 512, 128, 24, .23, 44, True, 1., True),
    "two_tower_weight": ("twotower", 512, 128, 24, .23, 44, True, 2.25, True),
    "two_tower_smooth_only": ("twotower", 512, 128, 24, .23, 44, False, 1., True),
}


def liver_mask(tissues):
    return tissues.fillna("").str.lower().str.contains("|".join(HCC_NAMES), regex=True).to_numpy()


def train(fit, expression, bank, config, *, device, seed):
    mode, active, width, rank, dropout, epochs, dual, domain_weight, use_fm = config
    torch.manual_seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("GPU resource missing for leaning sensitivity model")
        torch.set_float32_matmul_precision("high")
    r = np.random.default_rng(seed)
    banks = bank if use_fm else {}
    panel, anchor, _, _ = panel_targets(fit)
    views = Views.fit(expression, fit, active, banks)
    x, fm = views.cells(expression, banks)
    d = views.drugs(banks)
    target = panel.pivot_table(index="cell_id", columns="compound_id", values="residual",
                               aggfunc="mean").reindex(index=expression.index, columns=views.ids).to_numpy()
    mask = np.isfinite(target)
    scale = max(.05, float(np.std(target[mask])))
    target = np.nan_to_num(target/scale)
    tissue = fit[["cell_id", "tissue_origin"]].drop_duplicates("cell_id").set_index("cell_id").reindex(expression.index)
    solid = tissue.tissue_origin.fillna("").str.lower().str.contains("|".join(SOLID_NAMES), regex=True)
    weight = np.where(liver_mask(tissue.tissue_origin), domain_weight, np.where(solid, np.sqrt(domain_weight), 1.))
    x, fm, d, y, w, center = [tensor(a, device) for a in (
        x, fm, d, target, weight, (anchor.reindex(views.ids).to_numpy()-anchor.mean())/scale)]
    m = torch.as_tensor(mask, device=device)
    model = TwoAxisNet(mode, x.shape[1], fm.shape[1], d.shape[1], d.shape[0], width, rank, dropout).to(device)
    optimization = torch.optim.AdamW(model.parameters(), lr=.0022, weight_decay=.08)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimization, T_max=epochs, eta_min=.00025)
    for _ in range(epochs):
        model.train()
        for i in np.array_split(r.permutation(len(x)), 1+len(x)//112):
            prediction = model(x[i], fm[i], d)
            value = objective(prediction, y[i], m[i], center, w[i], dual)
            optimization.zero_grad(set_to_none=True)
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimization.step()
        schedule.step()
    result = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return LeanFit(config, views, result, anchor, scale, d)


def forecast(model, query, rna, bank, *, device):
    cells = query.cell_id.drop_duplicates()
    return model.predict(query, rna.loc[cells], bank, device=device)
