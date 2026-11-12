"""The three models. Everything outside the residual response transform R(h) is shared verbatim.

    basal (2000) -> cell encoder  -> 128
    morgan (2048) -> drug encoder -> 128
    concat (256)  -> trunk        -> h (256)
    decoder(h + R(h))             -> Delta (2000)

A  R_A(h) = W2 GELU(W1 h)                        single bottleneck, width solved for parameter parity
C  R_C(h) = sum_k softmax(router([z,d]))_k A_k(h)  four bottleneck-32 experts, input-dependent weights
B  R_B(h) = sum_k softmax(a)_k A_k(h)              the same four experts, one global weight vector

B is not C with the router switched off at inference; it is trained from scratch (plan s6.4). A fixed
convex combination of parallel MLPs is re-expressible as one wider Dense MLP, so B is the control that
removes input dependence from C, not a third independent model family.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _encoder(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class DenseResidual(nn.Module):
    """R_A: one bottleneck MLP. `hidden` is fixed by parameter counting, never by a score."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, h, z=None, dr=None):
        return self.fc2(torch.nn.functional.gelu(self.fc1(h))), None


class ExpertBank(nn.Module):
    """K independent bottleneck experts. Separate init per expert so their outputs start apart."""

    def __init__(self, d: int, k: int, r: int):
        super().__init__()
        self.k, self.r = k, r
        self.w1 = nn.Parameter(torch.empty(k, d, r))
        self.b1 = nn.Parameter(torch.zeros(k, r))
        self.w2 = nn.Parameter(torch.empty(k, r, d))
        self.b2 = nn.Parameter(torch.zeros(k, d))
        for i in range(k):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            bound1 = 1 / math.sqrt(d)
            bound2 = 1 / math.sqrt(r)
            nn.init.uniform_(self.b1[i], -bound1, bound1)
            nn.init.uniform_(self.b2[i], -bound2, bound2)

    def forward(self, h):                      # [B, d] -> [B, K, d]
        u = torch.einsum("bd,kdr->bkr", h, self.w1) + self.b1
        u = torch.nn.functional.gelu(u)
        return torch.einsum("bkr,krd->bkd", u, self.w2) + self.b2


class ConditionalExperts(nn.Module):
    """R_C: soft routing on the cell and drug embeddings jointly. All K experts are computed."""

    def __init__(self, d: int, k: int, r: int, d_gate_in: int, d_gate_hidden: int = 32):
        super().__init__()
        self.experts = ExpertBank(d, k, r)
        self.router = nn.Sequential(nn.Linear(d_gate_in, d_gate_hidden), nn.GELU(),
                                    nn.Linear(d_gate_hidden, k))

    def forward(self, h, z, dr):
        pi = torch.softmax(self.router(torch.cat([z, dr], -1)), -1)     # [B, K]
        return torch.einsum("bk,bkd->bd", pi, self.experts(h)), pi


class FixedExperts(nn.Module):
    """R_B: the same experts, one global softmax weight vector shared by every input."""

    def __init__(self, d: int, k: int, r: int):
        super().__init__()
        self.experts = ExpertBank(d, k, r)
        self.logits = nn.Parameter(torch.zeros(k))

    def forward(self, h, z=None, dr=None):
        pi = torch.softmax(self.logits, -1)
        out = torch.einsum("k,bkd->bd", pi, self.experts(h))
        return out, pi.expand(h.shape[0], -1)


class ResponseModel(nn.Module):
    def __init__(self, n_genes: int, n_bits: int, kind: str, d_emb: int = 128, d_trunk: int = 256,
                 dropout: float = 0.1, n_experts: int = 4, expert_rank: int = 32,
                 dense_hidden: int | None = None):
        super().__init__()
        self.kind = kind
        self.cell_encoder = _encoder(n_genes, d_emb, d_emb)
        self.drug_encoder = _encoder(n_bits, d_emb, d_emb)
        self.drop = nn.Dropout(dropout)
        self.trunk = nn.Sequential(nn.Linear(d_trunk, d_trunk), nn.LayerNorm(d_trunk), nn.GELU(),
                                   nn.Linear(d_trunk, d_trunk))
        if kind == "A":
            assert dense_hidden is not None
            self.response = DenseResidual(d_trunk, dense_hidden)
        elif kind == "C":
            self.response = ConditionalExperts(d_trunk, n_experts, expert_rank, 2 * d_emb)
        elif kind == "B":
            self.response = FixedExperts(d_trunk, n_experts, expert_rank)
        else:
            raise ValueError(kind)
        self.decoder = nn.Linear(d_trunk, n_genes)

    def forward(self, basal, fp):
        z = self.drop(self.cell_encoder(basal))
        dr = self.drop(self.drug_encoder(fp))
        h = self.drop(self.trunk(torch.cat([z, dr], -1)))
        r, pi = self.response(h, z, dr)
        return self.decoder(h + r), pi

    def param_counts(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        resp = sum(p.numel() for p in self.response.parameters())
        router = 0
        if self.kind == "C":
            router = sum(p.numel() for p in self.response.router.parameters())
        elif self.kind == "B":
            router = self.response.logits.numel()
        return dict(total=tot, response=resp, router=router, shared=tot - resp)


def solve_dense_hidden(n_genes: int, n_bits: int, **kw) -> tuple[int, dict]:
    """Pick A's bottleneck width by counting C's parameters, not by looking at any score.

    Scans integer widths and keeps the one whose total parameter count is closest to C's, requiring
    total within 1% and the response-transform part within 5% (plan s6.5).
    """
    ref = ResponseModel(n_genes, n_bits, "C", **kw).param_counts()
    best = None
    for hdim in range(1, 4096):
        a = ResponseModel(n_genes, n_bits, "A", dense_hidden=hdim, **kw).param_counts()
        d_tot = abs(a["total"] - ref["total"]) / ref["total"]
        d_res = abs(a["response"] - ref["response"]) / ref["response"]
        if best is None or d_tot < best[1]:
            best = (hdim, d_tot, d_res, a)
        if a["total"] > ref["total"] * 1.02:
            break
    hdim, d_tot, d_res, a = best
    assert d_tot <= 0.01, f"total parameter mismatch {d_tot:.4%} at hidden={hdim}"
    assert d_res <= 0.05, f"response parameter mismatch {d_res:.4%} at hidden={hdim}"
    return hdim, dict(dense_hidden=hdim, A=a, C=ref,
                      total_rel_diff=d_tot, response_rel_diff=d_res)


def build(kind: str, n_genes: int, n_bits: int, dense_hidden: int | None = None, **kw) -> ResponseModel:
    return ResponseModel(n_genes, n_bits, kind, dense_hidden=dense_hidden, **kw)
