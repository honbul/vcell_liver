"""R7 models: chemistry backbone + (none | concat | gated graph | fixed graph) mechanism correction.

delta_hat = m_c + MLP([ECFP, X0_std])  +  m_available * r(...)
The correction r has a zero-initialised output layer, so every arm starts from the chemistry prediction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SEED = 9
N_EDGE = 11
N_REL = 6
N_STATUS = 4


def zero_linear(i, o):
    l = nn.Linear(i, o)
    nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
    return l


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2048 + 978, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.out = zero_linear(512, 978)

    def forward(self, fp, x0s, cmean):
        return cmean + self.out(self.body(torch.cat([fp, x0s], 1)))


class Cond(nn.Module):
    """q_b from cell state, drug structure and drug-level evidence status."""

    def __init__(self, d=64):
        super().__init__()
        self.cell = nn.Linear(978, d); self.drug = nn.Linear(2048, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d + N_STATUS, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, fp, x0s, status):
        return self.mlp(torch.cat([F.relu(self.cell(x0s)), F.relu(self.drug(fp)), status], 1))


class GraphCorrection(nn.Module):
    """Relation-typed message passing over the KG; edge gates conditioned on edge evidence and q_b (gated) or on the
    relation type only (fixed). Measured nodes carry X0, unmeasured nodes a missing-expression mask."""

    def __init__(self, G, gated=True, h=32, k=8, d=64):
        super().__init__()
        self.gated = gated
        self.register_buffer("src", torch.as_tensor(G["edge_index"][0]))
        self.register_buffer("dst", torch.as_tensor(G["edge_index"][1]))
        self.register_buffer("rel", torch.as_tensor(G["edge_rel"]))
        self.register_buffer("efeat", torch.as_tensor(G["edge_feat"]))
        self.register_buffer("measured", torch.as_tensor(G["measured"]))
        self.register_buffer("lm", torch.as_tensor(G["lm_idx"]))
        n = len(G["measured"])
        indeg = np.bincount(G["edge_index"][1], minlength=n).astype(np.float32)
        self.register_buffer("inv_sqrt_indeg", torch.as_tensor(1.0 / np.sqrt(np.maximum(indeg, 1))))
        self.n = n
        self.ident = nn.Embedding(n, h)
        nn.init.normal_(self.ident.weight, std=0.1)
        self.wx = nn.Linear(3, h)
        self.ws = nn.Linear(N_SEED, h)
        self.cond = Cond(d)
        self.w_rel = nn.ParameterList([nn.Parameter(torch.randn(N_REL, h, h) / np.sqrt(h)) for _ in range(2)])
        self.ln = nn.ModuleList([nn.LayerNorm(h) for _ in range(2)])
        self.b_rel = nn.Parameter(torch.zeros(N_REL))
        if gated:
            self.pq = nn.Linear(d, k); self.ke = nn.Linear(N_EDGE, k)
        self.read = nn.Sequential(nn.Linear(h + d, h), nn.ReLU())
        self.out = zero_linear(h, 1)

    def gate(self, q):
        b = self.b_rel[self.rel][None, :]                                    # (1, E)
        if not self.gated:
            return torch.sigmoid(b).expand(q.shape[0], -1)
        return torch.sigmoid(b + self.pq(q) @ self.ke(self.efeat).T)        # (B, E)

    def forward(self, fp, x0s, status, seeds):
        """seeds: (b_index, node_index, feats) long/long/float tensors for the batch (already augmented)."""
        B = fp.shape[0]
        q = self.cond(fp, x0s, status)
        xn = torch.zeros(B, self.n, device=fp.device)
        xn[:, self.lm] = x0s
        meas = self.measured[None, :].expand(B, -1)
        nf = torch.stack([xn * meas, meas, 1 - meas], -1)                     # (B, N, 3)
        H = self.ident.weight[None] + self.wx(nf)
        bi, ni, sf = seeds
        if len(bi):
            H = H.index_put((bi, ni), self.ws(sf), accumulate=True)
        a = self.gate(q)
        for l in range(2):
            HW = torch.einsum("bnh,rhk->bnrk", H, self.w_rel[l])               # (B, N, R, h)
            m = HW[:, self.src, self.rel] * a[..., None]                       # (B, E, h)
            agg = torch.zeros_like(H).index_add_(1, self.dst, m) * self.inv_sqrt_indeg[None, :, None]
            H = self.ln[l](H + F.relu(agg))
        hl = H[:, self.lm]                                                    # (B, 978, h)
        r = self.out(self.read(torch.cat([hl, q[:, None].expand(-1, hl.shape[1], -1)], -1))).squeeze(-1)
        return r, a


class ConcatCorrection(nn.Module):
    """Same knowledge pooled into fixed vectors: target-embedding pool and a fixed-adjacency diffusion profile."""

    def __init__(self, G, h=32, d=64):
        super().__init__()
        n = len(G["measured"])
        self.n = n
        self.register_buffer("lm", torch.as_tensor(G["lm_idx"]))
        self.ident = nn.Embedding(n, h)
        nn.init.normal_(self.ident.weight, std=0.1)
        self.ws = nn.Linear(N_SEED, h)
        self.cond = Cond(d)
        src, dst = G["edge_index"]; rel = G["edge_rel"]
        sgn = np.where(np.isin(rel, [0, 3]), 1.0, np.where(np.isin(rel, [1, 4]), -1.0, 0.0)).astype(np.float32)
        outd = np.bincount(src, minlength=n).astype(np.float32); ind = np.bincount(dst, minlength=n).astype(np.float32)
        w = 1.0 / np.sqrt(np.maximum(outd[src], 1) * np.maximum(ind[dst], 1))
        # propagation x_next[v] = sum_u A[u, v] x[u]  ->  sparse matrix of shape (N, N) indexed [v, u]
        idx = torch.as_tensor(np.stack([dst, src]))
        self.register_buffer("A_all", torch.sparse_coo_tensor(idx, torch.as_tensor(w), (n, n)).coalesce())
        self.register_buffer("A_sgn", torch.sparse_coo_tensor(idx, torch.as_tensor(w * sgn), (n, n)).coalesce())
        self.prof = nn.Linear(978 * 3, d)
        self.head = nn.Sequential(nn.Linear(d + h + d, 256), nn.ReLU())
        self.out = zero_linear(256, 978)

    def diffusion(self, B, seeds, device):
        bi, ni, sf = seeds
        x1 = torch.zeros(self.n, B, device=device); xs = torch.zeros(self.n, B, device=device)
        if len(bi):
            x1.index_put_((ni, bi), torch.ones_like(bi, dtype=torch.float32), accumulate=True)
            act = sf[:, 3] - sf[:, 4]                                          # action sign (pos - neg), 0 if unknown
            xs.index_put_((ni, bi), act, accumulate=True)
        outs = []
        for A, x in [(self.A_all, x1), (self.A_sgn, x1), (self.A_sgn, xs)]:
            y1 = torch.sparse.mm(A, x); y2 = torch.sparse.mm(A, y1)
            outs.append((y1 + y2)[self.lm].T)                                  # (B, 978)
        return torch.cat(outs, 1)

    def forward(self, fp, x0s, status, seeds):
        B = fp.shape[0]
        q = self.cond(fp, x0s, status)
        bi, ni, sf = seeds
        pool = torch.zeros(B, self.ident.weight.shape[1], device=fp.device)
        if len(bi):
            pool = pool.index_add(0, bi, self.ident(ni) + self.ws(sf))
        with torch.no_grad():
            prof = self.diffusion(B, seeds, fp.device)
        r = self.out(self.head(torch.cat([q, pool, self.prof(prof)], 1)))
        return r, None


class R7Model(nn.Module):
    def __init__(self, arm, G=None):
        super().__init__()
        self.arm = arm
        self.backbone = Backbone()
        if arm == "gated":
            self.corr = GraphCorrection(G, gated=True)
        elif arm == "fixed":
            self.corr = GraphCorrection(G, gated=False)
        elif arm == "concat":
            self.corr = ConcatCorrection(G)
        else:
            assert arm == "chem"
            self.corr = None

    def forward(self, fp, x0s, cmean, status=None, m_avail=None, seeds=None):
        dc = self.backbone(fp, x0s, cmean)
        if self.corr is None:
            return dc, torch.zeros_like(dc), None
        r, a = self.corr(fp, x0s, status, seeds)
        r = r * m_avail[:, None]
        return dc + r, r, a
