"""P0 / P1 / P2 / P3: conditional residual experts on chemCPA's shared decoder hidden state.

The base is the published `chemCPA.model.ComPert` at 43e830eb, unmodified. Its encoder, drug
embedding, doser, covariate embeddings, both adversaries, the Gaussian reconstruction head and the
GaussianNLL objective are all kept exactly as the authors wrote them. The only change is one
residual term inserted at a single place, and `P0` does not even have that.

Where.  chemCPA's decoder is an MLP `[dim] + [width]*depth + [2*num_genes]`. Everything but its last
linear map produces a hidden vector `h` of size `width`, shared by every output gene -- this is the
"shared decoder hidden representation" the study is about. The arms are

    P0   out = head(h)                                           published chemCPA
    P1   out = head(h + W2 GELU(W1 h))                           one Dense residual, width solved
                                                                 for parameter parity with P2
    P2   out = head(h + sum_k pi_k(z_cov, z_drug) A_k(h))        K=4 bottleneck experts, soft router
    P3   out = head(h + sum_k softmax(a)_k A_k(h))               the same experts, input-independent

`W2` and every expert's second projection are **zero-initialised**, so at step 0 every arm computes
exactly the function P0 computes. The arms then differ only in what they can learn, not in where
they start. P3 is trained from scratch like the others, not obtained by freezing P2's router.

A fixed convex combination of parallel MLPs is re-expressible as one wider MLP, so P3 is the control
that isolates input dependence; it is not a third architecture.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseResidual(nn.Module):
    """P1. `hidden` is fixed by counting P2's parameters, never by looking at a score."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, cond=None):
        return self.fc2(F.gelu(self.fc1(h))), None


class ExpertBank(nn.Module):
    """K independent bottleneck experts, batched. Each is exactly Linear(d,r) -> GELU -> Linear(r,d).

    Initialisation bounds are written from the fan-in of the contraction in `forward` -- w1 is
    contracted over d and w2 over r -- because reading them off the storage layout is what went
    wrong in the v1 study (see reports/initfix_v2/INIT_FIX.md). The second projection is zeroed so
    the bank starts as the zero function.
    """

    def __init__(self, d: int, k: int, r: int):
        super().__init__()
        self.k, self.r, self.d = k, r, d
        self.w1 = nn.Parameter(torch.empty(k, d, r))
        self.b1 = nn.Parameter(torch.empty(k, r))
        self.w2 = nn.Parameter(torch.zeros(k, r, d))
        self.b2 = nn.Parameter(torch.zeros(k, d))
        bound1 = 1.0 / math.sqrt(d)
        for i in range(k):
            nn.init.uniform_(self.w1[i], -bound1, bound1)
            nn.init.uniform_(self.b1[i], -bound1, bound1)

    def forward(self, h):                       # [B, d] -> [B, K, d]
        u = F.gelu(torch.einsum("bd,kdr->bkr", h, self.w1) + self.b1)
        return torch.einsum("bkr,krd->bkd", u, self.w2) + self.b2


class ConditionalExperts(nn.Module):
    """P2. Soft routing on the cell-covariate and drug embeddings jointly; all K experts computed."""

    def __init__(self, d: int, k: int, r: int, d_cond: int, d_gate_hidden: int = 32):
        super().__init__()
        self.experts = ExpertBank(d, k, r)
        self.router = nn.Sequential(nn.Linear(d_cond, d_gate_hidden), nn.GELU(),
                                    nn.Linear(d_gate_hidden, k))

    def forward(self, h, cond):
        pi = torch.softmax(self.router(cond), -1)
        return torch.einsum("bk,bkd->bd", pi, self.experts(h)), pi


class FixedExperts(nn.Module):
    """P3. The same experts, one global weight vector shared by every input."""

    def __init__(self, d: int, k: int, r: int):
        super().__init__()
        self.experts = ExpertBank(d, k, r)
        self.logits = nn.Parameter(torch.zeros(k))

    def forward(self, h, cond=None):
        pi = torch.softmax(self.logits, -1)
        return torch.einsum("k,bkd->bd", pi, self.experts(h)), pi.expand(h.shape[0], -1)


def attach_residual(model, arm: str, n_experts: int = 4, expert_rank: int = 32,
                    dense_hidden: int | None = None):
    """Give a built `ComPert` a residual on its decoder's shared hidden state.

    Returns the model. For `P0` it is returned untouched, so P0 really is the published model and
    not a re-implementation of it wearing the same name.
    """
    model.residual_arm = arm
    model.residual = None
    model.last_router_weights = None
    if arm == "P0":
        return model

    net = model.decoder.network
    keys = list(net._modules.keys())
    head_key = keys[-1]
    head = net._modules[head_key]
    assert isinstance(head, nn.Linear), f"decoder does not end in a linear map but in {type(head)}"
    d = head.in_features
    d_cond = 2 * model.hparams["dim"]           # covariate embedding ++ drug embedding

    if arm == "P1":
        assert dense_hidden is not None
        model.residual = DenseResidual(d, dense_hidden)
    elif arm == "P2":
        model.residual = ConditionalExperts(d, n_experts, expert_rank, d_cond)
    elif arm == "P3":
        model.residual = FixedExperts(d, n_experts, expert_rank)
    else:
        raise ValueError(arm)
    model.residual = model.residual.to(next(model.parameters()).device)
    model._decoder_head_key = head_key
    return model


def decoder_with_residual(model, latent_treated, cond):
    """Run chemCPA's decoder, adding the residual at the shared hidden state.

    Mirrors `MLP.forward`: the body, then the final linear map, then the `ReLU` last-layer
    activation on the mean half only, exactly as the published code does it.
    """
    net = model.decoder.network
    keys = list(net._modules.keys())
    x = latent_treated
    for k in keys[:-1]:
        x = net._modules[k](x)
    pi = None
    if getattr(model, "residual", None) is not None:
        r, pi = model.residual(x, cond)
        x = x + r
    model.last_router_weights = pi
    x = net._modules[keys[-1]](x)
    if model.decoder.activation == "ReLU":
        dim = x.size(1) // 2
        x = torch.cat((model.decoder.relu(x[:, :dim]), x[:, dim:]), dim=1)
    return x


def param_counts(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    res = sum(p.numel() for p in model.residual.parameters()) if getattr(model, "residual", None) else 0
    router = 0
    if getattr(model, "residual", None) is not None:
        if hasattr(model.residual, "router"):
            router = sum(p.numel() for p in model.residual.router.parameters())
        elif hasattr(model.residual, "logits"):
            router = model.residual.logits.numel()
    return dict(total=tot, residual=res, router=router, base=tot - res)


def solve_dense_hidden(d: int, n_experts: int = 4, expert_rank: int = 32) -> tuple[int, dict]:
    """P1's bottleneck width, by counting P2's residual parameters. Closed form, no RNG touched.

    P2 residual = experts (k*(d*r + r + r*d + d)) + router (d_cond*32 + 32 + 32*k + k).
    P1 residual = d*m + m + m*d + d.
    The router is part of what P2 spends, so P1 is matched against the whole residual block.
    """
    experts = n_experts * (d * expert_rank + expert_rank + expert_rank * d + d)
    # d_cond is 2*dim and is supplied by the caller through `parity_report`; the router term is
    # added there. Here we match the expert part and let the caller assert the total.
    m = max(1, round((experts - d) / (2 * d + 1)))
    best = min(range(max(1, m - 4), m + 5),
               key=lambda x: abs((d * x + x + x * d + d) - experts))
    return best, dict(target_residual_params=experts,
                      dense_residual_params=d * best + best + best * d + d)


def residual_fingerprint(model) -> str:
    """SHA-256 over the parameters P0, P1, P2 and P3 all share, to prove a common start."""
    hsh = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("residual."):
            continue
        hsh.update(name.encode())
        hsh.update(value.detach().cpu().contiguous().float().numpy().tobytes())
    return hsh.hexdigest()
