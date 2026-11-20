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

Two initialisation defects of the first build are fixed here; both are recorded in
`reports/moe_drug_v1/INIT_FIX.md`.

1.  `solve_dense_hidden` instantiated hundreds of trial models to count parameters, and every one of
    them drew from the global RNG. Only A called it, so after a common `torch.manual_seed(seed)` the
    A and C runs built their *shared* encoders, trunk and decoder from different RNG states. The
    parameter scan now runs inside `torch.random.fork_rng`, and -- separately, because an isolated
    scan alone would still leave the two kinds' draw orders coupled to their head shapes -- the
    shared modules are built once per seed by `shared_init_state` and copied into every model with
    `load_shared`. `shared_fingerprint` lets a run assert that the copy took.

2.  `ExpertBank` stores w1 as [K, d, r] and contracts it as `bd,kdr->bkr`, so the fan-in of that map
    is d. `nn.init.kaiming_uniform_` on a [d, r] slice reads fan-in off dimension 1 and used r. The
    two bounds were therefore swapped relative to an equivalent `nn.Linear`: with d = 256 and r = 32
    the first layer was initialised 2.8x too wide and the second 2.8x too narrow. The bounds are now
    written out explicitly from the contraction that actually happens.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn


def _encoder(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class SharedCore(nn.Module):
    """Everything A, B and C have in common, in one module so it can be initialised once.

    Built in a fixed order with shapes that do not depend on the response head, so for a given seed
    its initial values are identical for every kind -- which is the property the comparison needs
    and the property the first build silently lost.
    """

    def __init__(self, n_genes: int, n_bits: int, d_emb: int = 128, d_trunk: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.cell_encoder = _encoder(n_genes, d_emb, d_emb)
        self.drug_encoder = _encoder(n_bits, d_emb, d_emb)
        self.trunk = nn.Sequential(nn.Linear(d_trunk, d_trunk), nn.LayerNorm(d_trunk), nn.GELU(),
                                   nn.Linear(d_trunk, d_trunk))
        self.decoder = nn.Linear(d_trunk, n_genes)
        self.drop = nn.Dropout(dropout)

    def embed(self, basal, fp):
        z = self.drop(self.cell_encoder(basal))
        dr = self.drop(self.drug_encoder(fp))
        h = self.drop(self.trunk(torch.cat([z, dr], -1)))
        return z, dr, h


def shared_init_state(n_genes: int, n_bits: int, seed: int, d_emb: int = 128, d_trunk: int = 256,
                      dropout: float = 0.1) -> dict:
    """The one set of initial values for the shared modules, for this seed, for every kind."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        core = SharedCore(n_genes, n_bits, d_emb, d_trunk, dropout)
    return {k: v.detach().clone() for k, v in core.state_dict().items()}


class DenseResidual(nn.Module):
    """R_A: one bottleneck MLP. `hidden` is fixed by parameter counting, never by a score."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, h, z=None, dr=None):
        return self.fc2(torch.nn.functional.gelu(self.fc1(h))), None


class ExpertBank(nn.Module):
    """K independent bottleneck experts, batched into one tensor per layer.

    Each expert is the exact equivalent of `nn.Linear(d, r) -> GELU -> nn.Linear(r, d)`, including
    its initialisation: PyTorch's default `kaiming_uniform_(a=sqrt(5))` reduces to
    U(-1/sqrt(fan_in), +1/sqrt(fan_in)), and that is what is written below. The fan-in of the first
    map is d and of the second is r, read off the einsum contractions in `forward`, not off the
    storage layout.
    """

    def __init__(self, d: int, k: int, r: int):
        super().__init__()
        self.k, self.r, self.d = k, r, d
        self.w1 = nn.Parameter(torch.empty(k, d, r))
        self.b1 = nn.Parameter(torch.empty(k, r))
        self.w2 = nn.Parameter(torch.empty(k, r, d))
        self.b2 = nn.Parameter(torch.empty(k, d))
        bound1 = 1.0 / math.sqrt(d)            # fan-in of  bd,kdr->bkr
        bound2 = 1.0 / math.sqrt(r)            # fan-in of  bkr,krd->bkd
        for i in range(k):                     # per expert, so they do not start as copies
            nn.init.uniform_(self.w1[i], -bound1, bound1)
            nn.init.uniform_(self.b1[i], -bound1, bound1)
            nn.init.uniform_(self.w2[i], -bound2, bound2)
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
        self.core = SharedCore(n_genes, n_bits, d_emb, d_trunk, dropout)
        if kind == "A":
            assert dense_hidden is not None
            self.response = DenseResidual(d_trunk, dense_hidden)
        elif kind == "C":
            self.response = ConditionalExperts(d_trunk, n_experts, expert_rank, 2 * d_emb)
        elif kind == "B":
            self.response = FixedExperts(d_trunk, n_experts, expert_rank)
        else:
            raise ValueError(kind)

    def load_shared(self, state: dict) -> None:
        """Copy the common initial values in. Explicit, so nothing depends on RNG draw order."""
        self.core.load_state_dict(state)

    def shared_fingerprint(self) -> str:
        """SHA-256 over the shared parameters, in a fixed order. Equal across kinds or the run is
        not a controlled comparison."""
        hsh = hashlib.sha256()
        for name, value in sorted(self.core.state_dict().items()):
            hsh.update(name.encode())
            hsh.update(value.detach().cpu().contiguous().numpy().tobytes())
        return hsh.hexdigest()

    def forward(self, basal, fp):
        z, dr, h = self.core.embed(basal, fp)
        r, pi = self.response(h, z, dr)
        return self.core.decoder(h + r), pi

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
    total within 1% and the response-transform part within 5% (plan s6.5). The whole scan runs in a
    forked RNG so that counting parameters cannot change what any model is initialised to.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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


def build(kind: str, n_genes: int, n_bits: int, dense_hidden: int | None = None,
          seed: int | None = None, **kw) -> ResponseModel:
    """Build one model. With `seed`, the shared modules are copied from the common initial state for
    that seed, so A, B and C differ only in their response head."""
    model = ResponseModel(n_genes, n_bits, kind, dense_hidden=dense_hidden, **kw)
    if seed is not None:
        model.load_shared(shared_init_state(
            n_genes, n_bits, seed,
            d_emb=kw.get("d_emb", 128), d_trunk=kw.get("d_trunk", 256),
            dropout=kw.get("dropout", 0.1)))
    return model
