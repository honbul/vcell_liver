"""RDKit atom and bond views and differentiable molecular message passing."""
from collections.abc import Sequence

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F

ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 11, 12, 19, 20, 26, 29, 30)
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def onehot(value, values: Sequence):
    return [value == x for x in values] + [value not in values]


def atom_features(atom):
    return [*onehot(atom.GetAtomicNum(), ELEMENTS), *onehot(atom.GetDegree(), range(6)),
            *onehot(atom.GetFormalCharge(), range(-2, 3)), *onehot(atom.GetTotalNumHs(), range(5)),
            *onehot(int(atom.GetHybridization()), range(2, 7)), *onehot(int(atom.GetChiralTag()), range(4)),
            atom.GetIsAromatic(), atom.IsInRing(), atom.GetMass()/100, atom.GetTotalValence()/6]


def bond_features(bond):
    return [*onehot(bond.GetBondType(), BOND_TYPES), *onehot(int(bond.GetStereo()), range(6)),
            bond.IsInRing(), bond.GetIsConjugated(), 0.]


def make_graphs(smiles):
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None or not m.GetNumAtoms() for m in molecules):
        raise ValueError("Invalid registered exact-parent SMILES")
    max_nodes = max(m.GetNumAtoms() for m in molecules)
    max_edges = max(m.GetNumBonds()*2 + m.GetNumAtoms() for m in molecules)
    atom_dim, bond_dim = len(atom_features(molecules[0].GetAtomWithIdx(0))), 15
    atoms = np.zeros((len(molecules),max_nodes,atom_dim), dtype=np.float32)
    bonds = np.zeros((len(molecules),max_edges,bond_dim), dtype=np.float32)
    edges = np.zeros((len(molecules),max_edges,2), dtype=np.int64)
    node_mask = np.zeros((len(molecules),max_nodes), dtype=bool)
    edge_mask = np.zeros((len(molecules),max_edges), dtype=bool)
    for i,mol in enumerate(molecules):
        n=mol.GetNumAtoms()
        atoms[i,:n] = [atom_features(a) for a in mol.GetAtoms()]
        node_mask[i,:n] = True
        self_edge = np.zeros(bond_dim, dtype=np.float32)
        self_edge[-1] = 1
        bonds[i,:n] = self_edge
        edges[i,:n] = np.stack([np.arange(n)]*2, axis=1)
        for j,bond in enumerate(mol.GetBonds()):
            a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            pos=n+2*j
            bonds[i,pos:pos+2] = bond_features(bond)
            edges[i,pos],edges[i,pos+1]=(a,b),(b,a)
        edge_mask[i,:n+mol.GetNumBonds()*2] = True
    return {"atoms": atoms,"bonds":bonds,"edges":edges,"node_mask":node_mask,"edge_mask":edge_mask}


def aggregate(value, indices, n):
    out = value.new_zeros((n,value.shape[1]))
    return out.index_add_(0,indices,value)


def segment_softmax(logit, dest, n):
    maximum = torch.full((n,logit.shape[1]), -10000., dtype=logit.dtype, device=logit.device)
    maximum.scatter_reduce_(0,dest[:,None].expand_as(logit),logit.detach(),reduce="amax",include_self=True)
    exp = (logit-maximum[dest]).exp()
    return exp / aggregate(exp,dest,n)[dest].clamp_min(1e-8)


class Gine(nn.Module):
    def __init__(self, width, edge_dim):
        super().__init__()
        self.edge=nn.Linear(edge_dim,width)
        self.update=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.epsilon=nn.Parameter(torch.zeros(()))

    def forward(self,x,e,src,dst):
        m=F.relu(x[src]+self.edge(e))
        return self.update((1+self.epsilon)*x+aggregate(m,dst,len(x)))


class GatV2(nn.Module):
    def __init__(self,width,edge_dim):
        super().__init__()
        self.left=nn.Linear(width,width,bias=False)
        self.right=nn.Linear(width,width,bias=False)
        self.edge=nn.Linear(edge_dim,width,bias=False)
        self.attn=nn.Parameter(torch.randn(4,width//4)*.1)
        self.out=nn.Linear(width,width)

    def forward(self,x,e,src,dst):
        a,b=self.left(x),self.right(x)
        vals=a[src]+self.edge(e)
        scores=(F.leaky_relu(vals+b[dst],.2).reshape(len(src),4,-1)*self.attn).sum(-1)
        w=segment_softmax(scores,dst,len(x))
        return self.out(aggregate((vals.reshape(len(src),4,-1)*w[...,None]).flatten(1),dst,len(x)))


class AttentiveFP(nn.Module):
    def __init__(self,width,edge_dim):
        super().__init__()
        self.value=nn.Linear(width+edge_dim,width)
        self.attn=nn.Linear(width*2+edge_dim,1)
        self.gru=nn.GRUCell(width,width)

    def forward(self,x,e,src,dst):
        v=self.value(torch.cat([x[src],e],dim=1))
        logit=F.leaky_relu(self.attn(torch.cat((x[src],x[dst],e),dim=1)),.1)
        weight=segment_softmax(logit,dst,len(x))
        return self.gru(F.elu(aggregate(v*weight,dst,len(x))),x)


class MoleculeNet(nn.Module):
    def __init__(self,atom_dim,bond_dim,width,depth,dropout,kind):
        super().__init__()
        layer={"gine":Gine,"gatv2":GatV2,"attentivefp":AttentiveFP}[kind]
        self.atom=nn.Linear(atom_dim,width)
        self.convs=nn.ModuleList([layer(width,bond_dim) for _ in range(depth)])
        self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.readout=nn.Linear(width,1)
        self.dropout=nn.Dropout(dropout)

    def forward(self,bank,ids):
        atoms, mask=bank["atoms"][ids],bank["node_mask"][ids]
        valid=bank["edge_mask"][ids]
        edge=bank["edges"][ids]+torch.arange(len(ids),device=ids.device)[:,None,None]*atoms.shape[1]
        src,dst=edge[valid].unbind(1)
        bond=bank["bonds"][ids][valid]
        x=self.atom(atoms).flatten(0,1)
        for conv,norm in zip(self.convs,self.norm):
            x=norm(x+self.dropout(conv(x,bond,src,dst)))
        x=x.reshape(len(ids),atoms.shape[1],-1)
        score=self.readout(x).squeeze(-1).masked_fill(~mask,-10000.)
        return (x*score.softmax(1)[...,None]).sum(1)
