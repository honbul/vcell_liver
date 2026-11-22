"""Directed bond-message regression with a separate, masked auxiliary assay head."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .features import structure

ELEMENTS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 26, 29, 30, 33, 34, 35, 53, 78)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
HYBRID = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3)
CHIRAL = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
STEREO = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
          Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)
NODE_DIM = len(ELEMENTS) + 1 + len(HYBRID) + len(CHIRAL) + 7
EDGE_DIM = len(BOND_TYPES) + len(STEREO) + 2


@lru_cache(maxsize=30000)
def parse(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid standard parent structure")
    nodes = []
    for a in mol.GetAtoms():
        number = a.GetAtomicNum()
        nodes.append([
            *(number == n for n in ELEMENTS), number not in ELEMENTS,
            *(a.GetHybridization() == h for h in HYBRID), *(a.GetChiralTag() == h for h in CHIRAL),
            a.GetDegree() / 4, a.GetTotalNumHs() / 4, a.GetFormalCharge() / 2,
            a.GetMass() / 100, a.GetIsAromatic(), a.IsInRing(), a.GetNumRadicalElectrons(),
        ])
    sources, ends, reverse, bonds = [], [], [], []
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge = [*(b.GetBondType() == kind for kind in BOND_TYPES), *(b.GetStereo() == x for x in STEREO),
                b.GetIsConjugated(), b.IsInRing()]
        i = len(sources)
        sources.extend([a, c])
        ends.extend([c, a])
        reverse.extend([i+1, i])
        bonds.extend([edge, edge])
    return (
        np.asarray(nodes, dtype=np.float32).reshape(-1, NODE_DIM), np.asarray(bonds, dtype=np.float32).reshape(-1, EDGE_DIM),
        np.asarray(sources, dtype=np.int64), np.asarray(ends, dtype=np.int64), np.asarray(reverse, dtype=np.int64),
    )


def batch(graphs, physicochemical, device):
    atoms, bonds, sources, destinations, reverses, membership = [], [], [], [], [], []
    a_offset = e_offset = 0
    for j, (node, bond, start, end, rev) in enumerate(graphs):
        atoms.append(node)
        bonds.append(bond)
        sources.append(start + a_offset)
        destinations.append(end + a_offset)
        reverses.append(rev + e_offset)
        membership.append(np.full(len(node), j, dtype=np.int64))
        a_offset += len(node)
        e_offset += len(bond)
    return [torch.as_tensor(np.concatenate(parts), device=device)
            for parts in (atoms, bonds, sources, destinations, reverses, membership)] + [
        torch.as_tensor(physicochemical, dtype=torch.float32, device=device)
    ]


class DirectedModel(nn.Module):
    def __init__(self, width, depth, drop, properties, auxiliary):
        super().__init__()
        self.depth = depth
        self.edge_input = nn.Linear(NODE_DIM + EDGE_DIM, width)
        self.edge_update = nn.Linear(width, width, bias=False)
        self.atom_out = nn.Linear(NODE_DIM + width, width)
        self.head = nn.Sequential(nn.Linear(2 * width + properties, width), nn.LayerNorm(width),
                                  nn.SiLU(), nn.Dropout(drop), nn.Linear(width, 1 + auxiliary))
        self.drop = nn.Dropout(drop)

    def forward(self, atom, bond, src, dst, rev, mol, properties):
        base = self.edge_input(torch.cat((atom[src], bond), dim=1))
        h = torch.nn.functional.silu(base)
        for _ in range(self.depth - 1):
            incoming = h.new_zeros((len(atom), h.shape[1])).index_add(0, dst, h)
            h = self.drop(torch.nn.functional.silu(base + self.edge_update(incoming[src] - h[rev])))
        node_h = torch.nn.functional.silu(self.atom_out(torch.cat((
            atom, h.new_zeros((len(atom), h.shape[1])).index_add(0, dst, h)
        ), dim=1)))
        n = len(properties)
        counts = node_h.new_zeros(n).index_add(0, mol, node_h.new_ones(len(mol)))
        mean = node_h.new_zeros((n, node_h.shape[1])).index_add(0, mol, node_h) / counts[:, None]
        maxpool = node_h.new_full(mean.shape, -float("inf")).scatter_reduce(
            0, mol[:, None].expand_as(node_h), node_h, "amax", include_self=True)
        return self.head(torch.cat((mean, maxpool, properties), dim=1))


@dataclass
class GraphFit:
    config: tuple
    scale: object
    weights: dict
    names: tuple

    def raw(self, smiles, *, device="cpu"):
        net = DirectedModel(*self.config[:3], self.scale.n_features_in_, len(self.names)).to(device)
        net.load_state_dict(self.weights)
        net.eval()
        data = graph_data(smiles, self.scale)
        with torch.inference_mode():
            result = [net(*select(data, ids, device)).cpu().numpy()
                      for ids in np.array_split(np.arange(len(smiles)), 1+len(smiles)//128) if len(ids)]
        return np.concatenate(result, axis=0)

    def mechanisms(self, smiles, *, device="cpu"):
        return expit(self.raw(smiles, device=device)[:, 1:])


def physics(smiles):
    return np.stack([structure(s)[0][2048:] for s in smiles])


def graph_data(smiles, scale):
    active = list(smiles)
    return [parse(s) for s in active], np.clip(scale.transform(physics(active)), -8, 8)


def select(data, ids, device):
    return batch([data[0][i] for i in ids], data[1][ids], device)


def train_graph(smiles, labels, auxiliary, config, *, seed, device, sample_weights=None):
    width, depth, dropout, coefficient, epochs = config
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        if not torch.cuda.is_available():
            raise RuntimeError("The D-MPNN GPU allocation has no CUDA device")
    scale = StandardScaler().fit(physics(smiles))
    data = graph_data(smiles, scale)
    y = torch.as_tensor(np.asarray(labels), device=device, dtype=torch.float32)
    w = torch.ones_like(y) if sample_weights is None else torch.as_tensor(sample_weights, device=device, dtype=y.dtype)
    names = ()
    if coefficient:
        names = tuple(c for c in auxiliary if "__" in c)
        aux_x = graph_data(auxiliary.smiles, scale)
        aux_y = torch.as_tensor(auxiliary[list(names)].to_numpy(), dtype=y.dtype, device=device)
        support = torch.isfinite(aux_y)
        positives = torch.nansum(aux_y, dim=0)
        task_weights = (support.sum(dim=0) / (2 * positives.clamp_min(1))).clip(1., 10.)
    model = DirectedModel(width, depth, dropout, scale.n_features_in_, len(names)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0018, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=.0002)
    fn = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(y)), 1+len(y)//72):
            output = model(*select(data, ids, device))[:, 0]
            objective = (fn(output, y[ids], reduction="none") * w[ids]).mean()
            if coefficient:
                ix = rng.choice(len(aux_y), 96, replace=len(aux_y) < 96)
                prediction = model(*select(aux_x, ix, device))[:, 1:]
                target = aux_y[ix]
                mask = support[ix]
                term = fn(prediction, target.nan_to_num(), pos_weight=task_weights, reduction="none")
                objective = objective + coefficient * ((term * mask).sum() / mask.sum().clamp_min(1))
            optim.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optim.step()
        scheduler.step()
    return GraphFit(config, scale, {key: val.detach().cpu() for key, val in model.state_dict().items()}, names)
