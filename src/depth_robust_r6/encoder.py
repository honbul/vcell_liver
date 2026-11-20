"""The small explicit state encoder and the shared delta head.

    cells of one control set [N, G]  --gene-token encoder (2 blocks, d=128)-->  per-cell vector
    per-cell vectors  --mean + attention pooling-->  set vector
    full-panel set summary  --MLP skip (every neural arm has it)-->  skip vector
    z = fuse(set vector, skip vector);   delta = head(z, drug ECFP4)

Two input contracts, same encoder size:
    basic   token value = [log-normalised expression, within-cell rank]
            skip        = mean log-normalised expression over the set (G)
    obs     token value = basic + [log1p raw count, c/(c+tau) for two fixed tau]
            cell token  = + [log observed library size, log1p detected panel genes]
            skip        = basic + detection fraction per gene (G) + mean log1p raw count (G)
                          + [mean log library, mean log1p detected genes]
Nothing here is told the keep-rate p: at deployment only the observed library size is known.
Zero-count genes are never fed as tokens (they are padding), but they are never removed from the
skip summary or from the response target either; single-UMI genes are kept.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, n_genes: int, obs_input: bool, d: int = 128, n_tok: int = 256,
                 n_blocks: int = 2, heads: int = 4, skip_hidden: int = 256, dropout: float = 0.1,
                 x_mean=None, x_sd=None, tiebreak=None):
        super().__init__()
        self.G, self.d, self.n_tok, self.obs_input = n_genes, d, n_tok, obs_input
        self.gene_emb = nn.Embedding(n_genes, d)
        self.val_proj = nn.Sequential(nn.Linear(5 if obs_input else 2, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cell_proj = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d)) if obs_input else None
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_att = nn.MultiheadAttention(d, heads, batch_first=True)
        n_skip = n_genes * (3 if obs_input else 1) + (2 if obs_input else 0)
        self.skip = nn.Sequential(nn.Linear(n_skip, skip_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(skip_hidden, d))
        self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.LayerNorm(d))
        # fixed (training-data) constants: gene standardisation of the skip, token tie-break order
        self.register_buffer("x_mean", torch.zeros(n_genes) if x_mean is None else torch.as_tensor(x_mean).float())
        self.register_buffer("x_sd", torch.ones(n_genes) if x_sd is None else torch.as_tensor(x_sd).float())
        tb = torch.linspace(0.5, 0.0, n_genes) if tiebreak is None else torch.as_tensor(tiebreak).float()
        self.register_buffer("tiebreak", tb)          # in [0, 0.5): orders equal counts, never across counts

    def cells(self, f: dict) -> torch.Tensor:
        """Per-cell vectors [B, N, d] from the gene-token encoder."""
        c, x = f["c"], f["x"]
        B, N, G = c.shape
        k = min(self.n_tok, G)
        score = c + self.tiebreak                      # integer counts + fixed sub-integer tie-break
        top, idx = score.topk(k, dim=-1)               # [B, N, k]
        pad = top < 1.0                                # zero-count genes are padding, not tokens
        xv = x.gather(-1, idx)
        rank = torch.arange(k, device=c.device, dtype=torch.float32).expand(B, N, k) / k
        feats = [xv, rank]
        if self.obs_input:
            feats += [f["log1p_c"].gather(-1, idx),
                      f["conf"][..., 0].gather(-1, idx), f["conf"][..., 1].gather(-1, idx)]
        tok = self.gene_emb(idx) + self.val_proj(torch.stack(feats, -1))
        cls = self.cls.expand(B, N, 1, self.d)
        if self.obs_input:
            cf = torch.stack([f["log_lib"] / 10.0, torch.log1p(f["n_det"]) / 7.0], -1)
            cls = cls + self.cell_proj(cf).unsqueeze(2)
        h = torch.cat([cls, tok], 2).reshape(B * N, k + 1, self.d)
        m = torch.cat([torch.zeros(B, N, 1, dtype=torch.bool, device=c.device), pad], 2).reshape(B * N, k + 1)
        h = self.blocks(h, src_key_padding_mask=m)
        return self.norm(h[:, 0]).reshape(B, N, self.d)

    def skip_summary(self, f: dict) -> torch.Tensor:
        s = [(f["x"].mean(1) - self.x_mean) / self.x_sd]
        if self.obs_input:
            s += [f["mask"].float().mean(1) - 0.5, f["log1p_c"].mean(1),
                  (f["log_lib"].mean(1, keepdim=True) / 10.0),
                  (torch.log1p(f["n_det"]).mean(1, keepdim=True) / 7.0)]
        return torch.cat(s, -1)

    def forward(self, f: dict):
        """Returns (z [B, d], per-cell vectors [B, N, d])."""
        hc = self.cells(f)
        q = self.pool_q.expand(hc.shape[0], 1, self.d)
        att, _ = self.pool_att(q, hc, hc, need_weights=False)
        z = self.fuse(torch.cat([hc.mean(1), att[:, 0], self.skip(self.skip_summary(f))], -1))
        return z, hc


class ResponseModel(nn.Module):
    """state z (or a learned constant when `use_state` is False) + drug ECFP4 -> signed delta [G]."""

    def __init__(self, n_genes: int, n_fp: int = 2048, obs_input: bool = False, use_state: bool = True,
                 d: int = 128, d_drug: int = 256, hidden: int = 512, dropout: float = 0.1,
                 recon_head: bool = False, latent_proj: bool = False, head_mode: str = "joint", **enc_kw):
        super().__init__()
        assert head_mode in ("joint", "additive")
        self.use_state, self.d, self.head_mode = use_state, d, head_mode
        self.encoder = StateEncoder(n_genes, obs_input, d=d, dropout=dropout, **enc_kw) if use_state else None
        self.z0 = None if use_state else nn.Parameter(torch.zeros(1, d))
        self.drug = nn.Sequential(nn.Linear(n_fp, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, d_drug), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + d_drug, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_genes))
        # "additive" (revision 1): delta = g(z) + h(z, drug); g never sees the drug, h's last layer starts at 0
        self.state_head = None
        if head_mode == "additive":
            self.state_head = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_genes))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.recon = nn.Linear(d, n_genes) if recon_head else None      # denoise-control only
        self.proj = nn.Linear(d, d) if latent_proj else None            # student -> teacher latent

    def state(self, f: dict | None, batch: int = 1):
        if not self.use_state:
            return self.z0.expand(batch, self.d), None
        return self.encoder(f)

    def delta_parts(self, z: torch.Tensor, fp: torch.Tensor):
        """z [B, d], fp [B, K, n_fp] -> (state-only part [B, 1, G] or None, drug-specific part [B, K, G])."""
        e = self.drug(fp)
        h = self.head(torch.cat([z.unsqueeze(1).expand(-1, e.shape[1], -1), e], -1))
        g = self.state_head(z).unsqueeze(1) if self.state_head is not None else None
        return g, h

    def delta(self, z: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """z [B, d], fp [B, K, n_fp] -> [B, K, G] (K drugs per control set)."""
        g, h = self.delta_parts(z, fp)
        return h if g is None else g + h

    def forward(self, f, fp):
        z, hc = self.state(f, fp.shape[0])
        return self.delta(z, fp), z, hc


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
