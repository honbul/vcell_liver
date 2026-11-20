"""GEARS paper architecture with graphs estimated without held-out RNA."""
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from artifact_io import digest, write
from common_data import ROOT, PUBLIC

SOURCE = '9d27bd619935cebe6f1519c3b53f61d2777dee59'


def coexpression(treated, single_mask):
    # Same family as the paper: absolute Pearson of train single-treatment cells.
    x = np.asarray(treated[single_mask].toarray(), dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError('Insufficient train single-intervention cells')
    x -= x.mean(axis=0)
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = 1
    x /= norm
    cor = np.abs(x.T @ x)
    src, tgt, weight = [], [], []
    for j, scores in enumerate(cor):
        top = np.argsort(-scores, kind='stable')[:21]
        for i in top[scores[top] > 0.4]:
            src.append(i)
            tgt.append(j)
            weight.append(scores[i])
    return np.asarray([src, tgt], dtype=np.int64), np.asarray(weight, dtype=np.float32)


def ontology(genes):
    path = PUBLIC/'norman/go.csv'
    receipt = {r['file']: r['sha256'] for r in json.loads(
        (ROOT/'manifests/public_receipt.json').read_text())}
    if digest(path) != receipt['norman/go.csv']:
        raise ValueError('GO hash mismatch')
    index = {g: i for i, g in enumerate(genes)}
    neighbors = [[] for _ in genes]
    with path.open() as f:
        for row in csv.DictReader(f):
            i, j = index.get(row['source']), index.get(row['target'])
            if i is not None and j is not None:
                neighbors[j].append((float(row['importance']), i))
    src, tgt, weight = [], [], []
    for j, pairs in enumerate(neighbors):
        for w, i in sorted(pairs, key=lambda p: (-p[0], p[1]))[:21]:
            src.append(i)
            tgt.append(j)
            weight.append(w)
    return np.asarray([src, tgt], dtype=np.int64), np.asarray(weight, dtype=np.float32)


def build(data, out, device, graph_path=None):
    src = ROOT/'.upstream/gears-paper'
    command = ['git', '-C', str(src)]
    assert subprocess.check_output(command + ['rev-parse','HEAD'], text=True).strip() == SOURCE
    assert not subprocess.check_output(command + ['diff','HEAD','--','gears'])
    sys.path.insert(0, str(src))
    from gears.model import GEARS_Model

    if graph_path is None:
        graph_path = out/'train_graphs.npz'
    elif not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if not graph_path.exists():
        mask = np.array([len(data.train[c]['targets']) == 1 for c in data.train_names])
        edges, weights = coexpression(data.treated, mask)
        go, go_w = ontology(data.genes)
        temp = graph_path.with_suffix('.tmp.npz')
        np.savez(temp, co_edges=edges, co_weights=weights, go_edges=go, go_weights=go_w,
                 train_single_cells=int(mask.sum()))
        temp.replace(graph_path)
    with np.load(graph_path, allow_pickle=False) as graph:
        co, co_w = (torch.as_tensor(graph[k], device=device) for k in ('co_edges','co_weights'))
        go, go_w = (torch.as_tensor(graph[k], device=device) for k in ('go_edges','go_weights'))
        n = int(graph['train_single_cells'])

    class CommonGEARS(GEARS_Model):
        def __init__(self):
            super().__init__(dict(
                num_genes=len(data.genes), hidden_size=64, device=device, uncertainty=False,
                num_go_gnn_layers=1, num_gene_gnn_layers=1, decoder_hidden_size=16,
                G_coexpress=co, G_coexpress_weight=co_w, G_go=go, G_go_weight=go_w))
            self._batched_graphs = {}

        def forward(self, controls, flags):
            b, g = controls.shape
            if b not in self._batched_graphs:
                offsets = (torch.arange(b, device=device) * g).view(1,b,1)
                self._batched_graphs[b] = ((co[:,None,:] + offsets).reshape(2,-1),
                                           co_w.repeat(b))
            # Original graph is indexed in one gene set. Expand its indices per
            # cell so the same convolution is applied throughout the batch.
            self.G_coexpress, self.G_coexpress_weight = self._batched_graphs[b]
            return super().forward(SimpleNamespace(
                x=torch.stack((controls, flags), dim=-1).reshape(-1,2),
                batch=torch.arange(b, device=device).repeat_interleave(g)))

    model = CommonGEARS().to(device)
    write(out/'model_init.json', dict(
        source=SOURCE, graphs_sha256=digest(graph_path), train_single_cells=n,
        coexpression_edges=int(co.shape[1]), go_edges=int(go.shape[1]),
        n_parameters=sum(p.numel() for p in model.parameters()),
        n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model
