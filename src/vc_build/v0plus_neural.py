"""RNA-only prediction with Chronos supervision, drug-target gating and domain alignment."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .external import _chemistry
from .io import sha256
from .v0plus_features import ExpressionFeatures

ABLATIONS = {
    "v0plus_inductive": {"paths": True, "chronos": True, "drivers": True, "targets": True,
                         "rank": True, "adapt": "source"},
    "v0plus_transductive": {"paths": True, "chronos": True, "drivers": True, "targets": True,
                            "rank": True, "adapt": "query"},
    "no_chronos": {"chronos": False}, "no_driver": {"drivers": False},
    "no_pathways": {"paths": False}, "no_targets": {"targets": False},
    "no_alignment": {"adapt": "none"}, "no_rank": {"rank": False},
}
GPU_SETTINGS = [(192, 70), (320, 100)]
ABLATIONS["v0plus_fixed"] = ABLATIONS["v0plus_inductive"]


def load_auxiliary(root):
    receipt = json.loads((root / "receipt.json").read_text())
    tables = {}
    for name, card in receipt["endpoints"].items():
        path = root / (name + ".parquet")
        if sha256(path) != card["sha256"]:
            raise ValueError(f"Auxiliary labels fail artifact verification: {name}")
        tables[name] = pd.read_parquet(path)
    return tables, receipt


class DrugAttention(nn.Module):
    def __init__(self, width, target_count):
        super().__init__()
        self.identity = nn.Embedding(target_count + 1, 48)
        self.activity = nn.Linear(2, 48, bias=False)
        self.query = nn.Linear(width, 48)
        self.out = nn.Sequential(nn.Linear(48, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, chemical, indices, rna, dependency, mask):
        # A learned null token is present for every compound, including unknown targets.
        values = torch.stack((rna, dependency), dim=-1)
        target = self.identity(indices) + self.activity(values)
        attention = (target * self.query(chemical)[:, None, :]).sum(dim=-1) / np.sqrt(48)
        attention = attention.masked_fill(~mask, torch.finfo(attention.dtype).min).softmax(dim=-1)
        return self.out((attention[:, :, None] * target).sum(dim=1))


class MolecularPredictor(nn.Module):
    def __init__(self, n_rna, n_paths, n_chemical, n_drugs, n_targets, n_drivers, width, config):
        super().__init__()
        self.config = config
        self.n_rna = n_rna
        self.rna = nn.Sequential(nn.Dropout(0.15), nn.Linear(n_rna, width * 2), nn.LayerNorm(width * 2),
                                 nn.GELU(), nn.Dropout(0.25), nn.Linear(width * 2, width))
        if config["paths"]:
            self.path = nn.Sequential(nn.Linear(n_paths, width), nn.LayerNorm(width), nn.GELU())
        self.cell = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.25), nn.Linear(width, width))
        self.drug = nn.Sequential(nn.Linear(n_chemical, width * 2), nn.LayerNorm(width * 2),
                                  nn.GELU(), nn.Dropout(0.15), nn.Linear(width * 2, width))
        if config.get("drug_identity", True):
            self.drug_id = nn.Embedding(n_drugs, width)
            nn.init.normal_(self.drug_id.weight, std=0.04)
        if config["chronos"]:
            self.dependency = nn.Linear(width, n_targets + 1)
            self.dependency_state = nn.Linear(n_targets + 1, width, bias=False)
        if config["drivers"]:
            self.drivers = nn.Linear(width, n_drivers)
            self.driver_state = nn.Linear(n_drivers, width, bias=False)
        if config["targets"]:
            self.interactions = DrugAttention(width, n_targets)
        self.gate = nn.Linear(width, width)
        self.out = nn.Sequential(nn.LayerNorm(width * 3), nn.Linear(width * 3, width),
                                 nn.GELU(), nn.Dropout(0.25), nn.Linear(width, 1))

    def encode(self, profile):
        profile_state = self.rna(profile[:, :self.n_rna])
        if self.config["paths"]:
            profile_state = profile_state + self.path(profile[:, self.n_rna:])
        return self.cell(profile_state)

    def predict(self, state, chemical, drug, drug_targets, target_rna, has_target):
        interaction_state = state
        dependency = None
        drivers = None
        if self.config["chronos"]:
            dependency = self.dependency(state)
            interaction_state = interaction_state + 0.4 * self.dependency_state(dependency)
        if self.config["drivers"]:
            drivers = self.drivers(state)
            interaction_state = interaction_state + 0.4 * self.driver_state(drivers)
        chemical_state = self.drug(chemical)
        if self.config.get("drug_identity", True):
            chemical_state = chemical_state + self.drug_id(drug)
        if self.config["targets"]:
            target_dependency = dependency.gather(1, drug_targets) if dependency is not None else torch.zeros_like(target_rna)
            chemical_state = chemical_state + self.interactions(
                chemical_state, drug_targets, target_rna, target_dependency, has_target,
            )
        matched = interaction_state * (1. + self.gate(chemical_state).tanh())
        response = self.out(torch.cat([matched, chemical_state, matched * chemical_state], dim=1)).squeeze(1)
        return response, dependency, drivers


@dataclass
class MolecularData:
    features: ExpressionFeatures
    chemical: np.ndarray
    drugs: list
    genes: list
    target_positions: np.ndarray
    target_mask: np.ndarray
    expression_positions: np.ndarray

    @classmethod
    def prepare(cls, fit, fit_rna, edges, targets, configuration, *, registry=None):
        structures = ["compound_id", "isomeric_smiles"]
        rows = fit[structures].drop_duplicates()
        fit_targets = targets
        if registry is not None:
            known_drugs = set(rows.compound_id)
            fit_targets = {drug: genes for drug, genes in targets.items() if drug in known_drugs}
            unfamiliar = registry.loc[~registry.compound_id.isin(known_drugs), structures]
            rows = pd.concat([rows, unfamiliar], ignore_index=True).drop_duplicates("compound_id")
        features = ExpressionFeatures.fit(fit_rna, edges, fit_targets, n_variable=768, rank=configuration["rank"])
        drugs = sorted(rows.compound_id.unique())
        chemistry = _chemistry(rows, fp_size=2048).loc[drugs].to_numpy(dtype="float32")
        genes = sorted(set(features.columns) & set().union(*fit_targets.values()))
        indexed = {gene: i + 1 for i, gene in enumerate(genes)}
        items = [[indexed[gene] for gene in sorted(set(targets.get(drug, ())) & set(genes))] for drug in drugs]
        target_positions = np.zeros((len(drugs), max(map(len, items), default=0) + 1), dtype=np.int64)
        target_mask = np.zeros_like(target_positions, dtype=bool)
        for j, item in enumerate(items):
            target_positions[j, 1:len(item) + 1] = item
            target_mask[j, :len(item) + 1] = True
        expr = pd.Index(features.columns).get_indexer(genes)
        return cls(features, chemistry, drugs, genes, target_positions, target_mask, expr)

    def inputs(self, rna, device):
        state = self.features.transform(rna).to_numpy(dtype=np.float32)
        target_rna = np.column_stack((np.zeros(len(state), dtype=np.float32), state[:, self.expression_positions]))
        return torch.as_tensor(state, device=device), torch.as_tensor(target_rna, device=device)

    def drug_inputs(self, device):
        return (torch.as_tensor(self.chemical, device=device),
                torch.as_tensor(self.target_positions, device=device),
                torch.as_tensor(self.target_mask, device=device))


def masked_standardize(table, cells, columns, device):
    selected = table.reindex(index=cells, columns=columns)
    scale = selected.std().clip(lower=0.1)
    z = ((selected - selected.mean()) / scale).clip(-8, 8)
    values = torch.tensor(z.fillna(0).to_numpy(dtype=np.float32), device=device)
    present = torch.tensor(z.notna().to_numpy(), device=device)
    return values, present


def masked_loss(prediction, truth, mask):
    error = F.smooth_l1_loss(prediction, truth, reduction="none")
    return (error * mask).sum() / mask.sum().clamp_min(1)


def correlation_alignment(first, second):
    first_mean, second_mean = first.mean(dim=0), second.mean(dim=0)
    z1 = first - first_mean
    z2 = second - second_mean
    covariance = (z1.T @ z1) / (len(first) - 1) - (z2.T @ z2) / (len(second) - 1)
    return (first_mean - second_mean).square().mean() + 2 * covariance.square().mean()


def grouped_pair_indices(frame, fields, rng):
    pair = np.arange(len(frame))
    for indices in frame.groupby(fields, sort=False).indices.values():
        shuffled = rng.permutation(indices)
        pair[shuffled] = np.roll(shuffled, 1)
    return pair


@dataclass
class FittedMolecular:
    network: MolecularPredictor
    molecular: MolecularData
    anchor: pd.Series

    @torch.no_grad()
    def predict(self, query, rna):
        device = next(self.network.parameters()).device
        profile, target_rna = self.molecular.inputs(rna, device)
        chemical, targets, known = self.molecular.drug_inputs(device)
        cell_index = rna.index.get_indexer(query.cell_id)
        drug_index = pd.Index(self.molecular.drugs).get_indexer(query.compound_id)
        if (cell_index < 0).any():
            raise ValueError("Missing query RNA")
        valid = np.flatnonzero(drug_index >= 0)
        residual = np.full(len(query), np.nan)
        self.network.eval()
        latent = self.network.encode(profile)
        for start in range(0, len(valid), 1024):
            rows = valid[start:start + 1024]
            cell = torch.as_tensor(cell_index[rows], device=device)
            drug = torch.as_tensor(drug_index[rows], device=device)
            indices = targets[drug]
            cell_targets = target_rna[cell].gather(1, indices)
            output, _, _ = self.network.predict(latent[cell], chemical[drug], drug, indices,
                                                 cell_targets, known[drug])
            residual[rows] = output.cpu().numpy() / 5.
        return query.compound_id.map(self.anchor).to_numpy(), residual


def fit_molecular(arm, width, epochs, source, anchor, fit_rna, query_rna, edges, targets, auxiliary, *,
                  gpu, interaction_target=False, registry=None, options=None, batch_size=1024):
    configs = ABLATIONS["v0plus_inductive"] | ABLATIONS[arm] | (options or {})
    if registry is not None and (configs.get("drug_identity", True) or configs["adapt"] == "query"):
        raise ValueError("A cold compound requires an ID-free inductive model")
    if not gpu or not torch.cuda.is_available():
        raise RuntimeError("V0+ molecular model requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(11711)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(11711)
    molecular = MolecularData.prepare(source, fit_rna, edges, targets, configs, registry=registry)
    state, target_rna = molecular.inputs(fit_rna, device)
    chemical, target_positions, target_mask = molecular.drug_inputs(device)
    feature_count = len(molecular.features.indices)
    dtypes = auxiliary["drivers"].columns
    network = MolecularPredictor(feature_count, state.shape[1] - feature_count,
                                 chemical.shape[1], len(molecular.drugs), len(molecular.genes),
                                 len(dtypes), width, configs).to(device)
    chronos, has_chronos = masked_standardize(auxiliary["chronos"], fit_rna.index, ["", *molecular.genes], device)
    drivers, has_drivers = masked_standardize(auxiliary["drivers"], fit_rna.index, dtypes, device)
    row_cells = torch.as_tensor(fit_rna.index.get_indexer(source.cell_id), device=device)
    row_drugs = torch.as_tensor(pd.Index(molecular.drugs).get_indexer(source.compound_id), device=device)
    response = torch.tensor(source.residual.to_numpy(dtype=np.float32), device=device) * 5.
    absolute = torch.tensor(source.model_value.to_numpy(dtype=np.float32), device=device) * 5.
    potency = torch.tensor(source.compound_id.map(anchor).to_numpy(dtype=np.float32), device=device) * 5.
    counts = source.groupby(["compound_id", "source_id"]).residual.transform("size")
    weights = torch.tensor((counts.mean() / counts).clip(0.25, 4.).to_numpy(dtype=np.float32), device=device)
    source_map = [torch.as_tensor(fit_rna.index.get_indexer(pd.Index(rows.cell_id.unique())), device=device)
                  for _, rows in source.groupby("source_id", sort=True)]
    unlabelled = molecular.inputs(query_rna, device)[0] if configs["adapt"] == "query" else None
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0008, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00012)
    for epoch in range(epochs):
        axis = ["source_id", "compound_id" if epoch % 2 == 0 else "cell_id"]
        pair = torch.as_tensor(grouped_pair_indices(source, axis, rng), device=device)
        for primary in torch.randperm(len(source), device=device).split(batch_size):
            rows = torch.cat([primary, pair[primary]])
            cell, drug = row_cells[rows], row_drugs[rows]
            latent = network.encode(state[cell])
            genes = target_positions[drug]
            pred, dep, genotype = network.predict(latent, chemical[drug], drug, genes,
                                                   target_rna[cell].gather(1, genes), target_mask[drug])
            size = len(primary)
            error = F.smooth_l1_loss(pred[:size], response[primary], beta=0.5, reduction="none")
            use_residual = interaction_target or epoch % 2 == 0
            observed = response if use_residual else absolute
            value = pred if use_residual else pred + potency[rows]
            delta = observed[primary] - observed[pair[primary]]
            sign_loss = F.softplus(-(value[:size] - value[size:]) * delta.sign())
            valid = delta.abs() > 0.06
            loss = (error * weights[primary]).mean() + 0.10 * (sign_loss * valid).mean()
            if dep is not None:
                loss += 0.25 * masked_loss(dep[:size], chronos[cell[:size]], has_chronos[cell[:size]])
            if genotype is not None:
                loss += 0.12 * masked_loss(genotype[:size], drivers[cell[:size]], has_drivers[cell[:size]])
            if configs["adapt"] != "none":
                domains = [state[ix[torch.randperm(len(ix), device=device)[:96]]] for ix in source_map]
                if unlabelled is not None:
                    domains.append(unlabelled[torch.randperm(len(unlabelled), device=device)[:96]])
                embeddings = [network.encode(domain) for domain in domains if len(domain) > 1]
                target_domain = embeddings[-1]
                alignment = torch.stack([correlation_alignment(z, target_domain) for z in embeddings[:-1]]).mean()
                loss += 0.08 * alignment
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step()
        scheduler.step()
    return FittedMolecular(network.eval(), molecular, anchor)
