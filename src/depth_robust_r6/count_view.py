"""Observation views of control cells: exact binomial thinning of integer UMI counts.

A cell is (c, c_rest): integer counts on the 2,000-gene panel and the sum of all other genes, so
that library = c.sum() + c_rest is the cell's full library size. A view at keep-rate p draws
    c'      ~ Binomial(c, p)        per panel gene
    c_rest' ~ Binomial(c_rest, p)
and its library size is c'.sum() + c_rest' -- the thinned total, never an independently thinned
library and never the original one. p = 1 returns the inputs unchanged (identity, no draw).
The normalisation is the same at every p:  x = log1p(1e4 * c' / library').

Thinning a cell is NOT a new biological replicate: the thinned view and its parent cell always
stay on the same side of every split (see data.py).
"""
from __future__ import annotations

import numpy as np
import torch

SCALE = 1e4
TAUS = (1.0, 4.0)        # fixed soft-confidence constants c / (c + tau); an empirical feature, not a posterior


def thin_np(c: np.ndarray, c_rest: np.ndarray, p: float, rng: np.random.Generator):
    """Exact thinning with numpy. c [n, G] and c_rest [n] non-negative integers."""
    assert np.issubdtype(c.dtype, np.integer) and np.issubdtype(c_rest.dtype, np.integer)
    assert 0.0 < p <= 1.0
    if p == 1.0:
        return c.copy(), c_rest.copy()
    return rng.binomial(c, p).astype(c.dtype), rng.binomial(c_rest, p).astype(c_rest.dtype)


def thin_torch(c: torch.Tensor, c_rest: torch.Tensor, p, gen: torch.Generator | None = None):
    """Exact thinning on the tensors' device. c [..., G], c_rest [...] hold integer values (any dtype).

    p is a float or a tensor broadcastable to c_rest (one keep-rate per cell or per set). Entries
    with p == 1 are returned unchanged.
    """
    cf, rf = c.float(), c_rest.float()
    p = torch.as_tensor(p, dtype=torch.float32, device=c.device)
    if p.ndim == 0:
        if float(p) == 1.0:
            return cf, rf
        p = p.expand(rf.shape)
    pg = p.unsqueeze(-1).expand(cf.shape)
    tc = torch.binomial(cf, pg, generator=gen)
    tr = torch.binomial(rf, p.expand(rf.shape), generator=gen)
    full = (p >= 1.0)
    tc = torch.where(full.unsqueeze(-1).expand(cf.shape), cf, tc)
    tr = torch.where(full.expand(rf.shape), rf, tr)
    return tc, tr


def library(c, c_rest):
    return c.sum(-1) + c_rest


def lognorm(c, lib):
    """log1p(1e4 * c / lib); a cell whose thinned library is 0 is all-zero."""
    if isinstance(c, np.ndarray):
        return np.log1p(SCALE * c / np.maximum(lib, 1)[..., None])
    return torch.log1p(SCALE * c / lib.clamp(min=1).unsqueeze(-1))


def set_features(c: torch.Tensor, c_rest: torch.Tensor) -> dict:
    """Per-cell tensors every model input is built from. c [B, N, G] float (integer-valued)."""
    lib = library(c, c_rest)
    x = lognorm(c, lib)
    return dict(c=c, lib=lib, x=x, log1p_c=torch.log1p(c),
                conf=torch.stack([c / (c + t) for t in TAUS], -1), mask=(c > 0),
                log_lib=torch.log(lib.clamp(min=1)), n_det=(c > 0).sum(-1).float())
