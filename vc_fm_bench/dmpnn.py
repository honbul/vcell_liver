"""Directed bond-message-passing comparator based on Yang et al. (2019)."""
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
import torch
from torch import nn
from torch.nn import functional as F


ATOM_NUMBERS = (5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
CHIRAL = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
          Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
HYBRID = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
          Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2)
BONDS = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
STEREO = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)


def one_hot(value, choices):
    return [int(value == i) for i in choices] + [int(value not in choices)]


def atom_features(atom):
    return (
        one_hot(atom.GetAtomicNum(), ATOM_NUMBERS) +
        one_hot(atom.GetTotalDegree(), range(6)) +
        one_hot(atom.GetFormalCharge(), (-2, -1, 0, 1, 2)) +
        one_hot(atom.GetTotalNumHs(), range(5)) +
        one_hot(atom.GetChiralTag(), CHIRAL) +
        one_hot(atom.GetHybridization(), HYBRID) +
        [float(atom.GetIsAromatic()), atom.GetMass() * .01]
    )


def bond_features(bond):
    return (one_hot(bond.GetBondType(), BONDS) + one_hot(bond.GetStereo(), STEREO)
            + [float(bond.IsInRing()), float(bond.GetIsConjugated())])


@dataclass
class Graph:
    nodes: torch.Tensor
    edges: torch.Tensor
    sources: torch.Tensor
    receivers: torch.Tensor
    reverse: torch.Tensor


def molecule(text):
    compound = Chem.MolFromSmiles(text)
    if compound is None or not compound.GetNumAtoms():
        raise ValueError("Registered canonical structure is not an RDKit molecular graph")
    nodes = [atom_features(atom) for atom in compound.GetAtoms()]
    edges, sources, receivers, reverse = [], [], [], []
    for bond in compound.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        j = len(sources)
        sources.extend((a, b))
        receivers.extend((b, a))
        reverse.extend((j + 1, j))
        edges.extend([bond_features(bond)] * 2)
    feature_count = len(BONDS) + len(STEREO) + 4
    return Graph(torch.tensor(nodes, dtype=torch.float32),
                 torch.tensor(edges, dtype=torch.float32).reshape(-1, feature_count),
                 *(torch.tensor(i, dtype=torch.long) for i in (sources, receivers, reverse)))


def batch(graphs):
    node_counts = torch.tensor([len(m.nodes) for m in graphs])
    atoms = torch.cat([m.nodes for m in graphs], dim=0)
    edges = torch.cat([m.edges for m in graphs], dim=0)
    node_offsets = node_counts.cumsum(0) - node_counts
    edge_counts = np.array([len(m.edges) for m in graphs])
    edge_offsets = np.cumsum(edge_counts) - edge_counts
    source = torch.cat([m.sources + offset for m, offset in zip(graphs, node_offsets)])
    dest = torch.cat([m.receivers + offset for m, offset in zip(graphs, node_offsets)])
    reverse = torch.cat([m.reverse + offset for m, offset in zip(graphs, edge_offsets)])
    atom_graph = torch.repeat_interleave(torch.arange(len(graphs)), node_counts)
    return atoms, edges, source, dest, reverse, atom_graph, node_counts


class DMPNN(nn.Module):
    def __init__(self, n_atoms, n_bonds, width, depth):
        super().__init__()
        self.width, self.depth = width, depth
        self.initial = nn.Linear(n_atoms + n_bonds, width, bias=False)
        self.update = nn.Linear(width, width, bias=False)
        self.atom = nn.Linear(n_atoms + width, width)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(.15), nn.Linear(width, 1))

    def forward(self, atoms, edges, source, dest, reverse, atom_graph, counts):
        initial = self.initial(torch.cat([atoms[source], edges], dim=1))
        messages = F.relu(initial)
        for _ in range(self.depth - 1):
            incoming = atoms.new_zeros((len(atoms), self.width)).index_add(0, dest, messages)
            messages = F.relu(initial + self.update(incoming[source] - messages[reverse]))
        incoming = atoms.new_zeros((len(atoms), self.width)).index_add(0, dest, messages)
        hidden = F.relu(self.atom(torch.cat([atoms, incoming], dim=1)))
        pooled = atoms.new_zeros((len(counts), self.width)).index_add(0, atom_graph, hidden)
        return self.readout(pooled / counts[:, None]).ravel()


def chemistry_predictor(structures):
    if not structures.id.is_unique:
        raise ValueError("Duplicate molecular identity")
    graph_index = {}

    def sample(key):
        if key not in graph_index:
            graph_index[key] = molecule(smiles.loc[key])
        return graph_index[key]

    smiles = structures.set_index("id").smiles

    def run(fit, query, config, seed):
        width, depth, epochs = config
        torch.manual_seed(seed)
        torch.set_num_threads(3)
        rng = np.random.default_rng(seed)
        fit_graph = [sample(key) for key in fit.drug_id]
        query_graph = [sample(key) for key in query.drug_id]
        network = DMPNN(fit_graph[0].nodes.shape[1], fit_graph[0].edges.shape[1], width, depth)
        target = torch.tensor(fit.outcome.to_numpy(), dtype=torch.float32)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.0002)
        network.train()
        for _ in range(epochs):
            for indices in np.array_split(rng.permutation(len(fit_graph)), max(1, len(fit_graph) // 64)):
                optimizer.zero_grad(set_to_none=True)
                logits = network(*batch([fit_graph[i] for i in indices]))
                F.binary_cross_entropy_with_logits(logits, target[indices]).backward()
                optimizer.step()
            scheduler.step()
        network.eval()
        with torch.inference_mode():
            parts = np.array_split(np.arange(len(query_graph)), max(1, len(query_graph) // 128))
            return torch.cat([network(*batch([query_graph[i] for i in ids])) for ids in parts]).numpy()

    return run
