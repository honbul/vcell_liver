"""Measure the two initialisation defects of the first build, and show the fix removes them.

This runs on shapes alone -- no data, no GPU, no training -- so it is cheap to re-run and it is the
evidence behind the numbers quoted in `reports/initfix_v2/INIT_FIX.md`.

Defect (a).  `solve_dense_hidden` built hundreds of trial models to count parameters and every one
of them drew from the global RNG. Only kind A called it. So after a common `torch.manual_seed(seed)`
the A run and the C run initialised their *shared* encoders, trunk and decoder from different RNG
states, and the "identical modules" premise of the comparison did not hold.

Defect (b).  `ExpertBank` stores w1 as [K, d, r] and contracts `bd,kdr->bkr`, so that map's fan-in is
d. `kaiming_uniform_` reads fan-in off dimension 1 of a 2-D slice, which is r. The two layers'
bounds were therefore swapped.

The old code is reproduced here rather than imported, because it no longer exists in the package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import torch
import torch.nn as nn

from .models import ResponseModel, build, shared_init_state, solve_dense_hidden


# ---------------------------------------------------------------------------------------------
# the first build, reproduced exactly as it was at c667041
# ---------------------------------------------------------------------------------------------
def _old_encoder(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class _OldExpertBank(nn.Module):
    def __init__(self, d, k, r):
        super().__init__()
        self.k, self.r, self.d = k, r, d
        self.w1 = nn.Parameter(torch.empty(k, d, r))
        self.b1 = nn.Parameter(torch.zeros(k, r))
        self.w2 = nn.Parameter(torch.empty(k, r, d))
        self.b2 = nn.Parameter(torch.zeros(k, d))
        for i in range(k):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            nn.init.uniform_(self.b1[i], -1 / math.sqrt(d), 1 / math.sqrt(d))
            nn.init.uniform_(self.b2[i], -1 / math.sqrt(r), 1 / math.sqrt(r))


class _OldModel(nn.Module):
    """Shared modules in the old construction order, with the old per-kind head."""

    def __init__(self, n_genes, n_bits, kind, d_emb=128, d_trunk=256, dense_hidden=None,
                 n_experts=4, expert_rank=32):
        super().__init__()
        self.kind = kind
        self.cell_encoder = _old_encoder(n_genes, d_emb, d_emb)
        self.drug_encoder = _old_encoder(n_bits, d_emb, d_emb)
        self.drop = nn.Dropout(0.1)
        self.trunk = nn.Sequential(nn.Linear(d_trunk, d_trunk), nn.LayerNorm(d_trunk), nn.GELU(),
                                   nn.Linear(d_trunk, d_trunk))
        if kind == "A":
            self.response = nn.ModuleDict(dict(fc1=nn.Linear(d_trunk, dense_hidden),
                                               fc2=nn.Linear(dense_hidden, d_trunk)))
        else:
            self.response = nn.ModuleDict(dict(
                experts=_OldExpertBank(d_trunk, n_experts, expert_rank),
                router=nn.Sequential(nn.Linear(2 * d_emb, 32), nn.GELU(), nn.Linear(32, n_experts))))
        self.decoder = nn.Linear(d_trunk, n_genes)

    def shared_fingerprint(self):
        hsh = hashlib.sha256()
        for mod in ["cell_encoder", "drug_encoder", "trunk", "decoder"]:
            for name, value in sorted(getattr(self, mod).state_dict().items()):
                hsh.update(f"{mod}.{name}".encode())
                hsh.update(value.detach().cpu().contiguous().numpy().tobytes())
        return hsh.hexdigest()


def _old_solve_dense_hidden(n_genes, n_bits):
    """The old scan: it instantiates models on the *global* RNG, which is the defect."""
    ref = sum(p.numel() for p in _OldModel(n_genes, n_bits, "C").parameters())
    best = None
    for hdim in range(1, 4096):
        tot = sum(p.numel() for p in _OldModel(n_genes, n_bits, "A", dense_hidden=hdim).parameters())
        d_tot = abs(tot - ref) / ref
        if best is None or d_tot < best[1]:
            best = (hdim, d_tot)
        if tot > ref * 1.02:
            break
    return best[0]


def audit(n_genes: int = 2000, n_bits: int = 2048, seeds=(42, 43, 44)) -> dict:
    d_trunk, rank = 256, 32
    out: dict = {}

    # ---- (a) shared initial values, before and after -----------------------------------------
    before, after = {}, {}
    for seed in seeds:
        torch.manual_seed(seed)
        hidden_old = _old_solve_dense_hidden(n_genes, n_bits)
        fa = _OldModel(n_genes, n_bits, "A", dense_hidden=hidden_old).shared_fingerprint()
        torch.manual_seed(seed)
        fc = _OldModel(n_genes, n_bits, "C").shared_fingerprint()
        before[str(seed)] = dict(A=fa[:16], C=fc[:16], identical=bool(fa == fc))

        hidden = solve_dense_hidden(n_genes, n_bits)[0]
        ga = build("A", n_genes, n_bits, dense_hidden=hidden, seed=seed).shared_fingerprint()
        gc = build("C", n_genes, n_bits, seed=seed).shared_fingerprint()
        gb = build("B", n_genes, n_bits, seed=seed).shared_fingerprint()
        after[str(seed)] = dict(A=ga[:16], C=gc[:16], B=gb[:16],
                                identical=bool(ga == gc == gb))
    # how far apart the old A and C shared weights actually were, in units of their own spread
    torch.manual_seed(42)
    _old_solve_dense_hidden(n_genes, n_bits)
    oa = _OldModel(n_genes, n_bits, "A", dense_hidden=146)
    torch.manual_seed(42)
    oc = _OldModel(n_genes, n_bits, "C")
    diffs = []
    for (na, pa), (nc, pc) in zip(oa.trunk.named_parameters(), oc.trunk.named_parameters()):
        assert na == nc
        diffs.append(float((pa - pc).abs().mean()) / max(float(pa.abs().mean()), 1e-12))
    out["shared_init"] = dict(
        before=before, after=after,
        old_trunk_mean_abs_relative_difference=round(sum(diffs) / len(diffs), 4),
        rng_draws_consumed_by_old_scan=_count_old_scan_draws(n_genes, n_bits),
        note="the old scan ran only for kind A, so only kind A's shared weights were displaced")

    # ---- (b) expert initialisation scale ------------------------------------------------------
    torch.manual_seed(0)
    old = _OldExpertBank(d_trunk, 4, rank)
    torch.manual_seed(0)
    new = ResponseModel(n_genes, n_bits, "C").response.experts
    exp_w1 = 1.0 / math.sqrt(3 * d_trunk)      # sd of U(-1/sqrt(d), 1/sqrt(d))
    exp_w2 = 1.0 / math.sqrt(3 * rank)
    out["expert_init"] = dict(
        fan_in_w1=d_trunk, fan_in_w2=rank,
        old_w1_sd=round(float(old.w1.std()), 6), new_w1_sd=round(float(new.w1.std()), 6),
        correct_w1_sd=round(exp_w1, 6),
        old_w2_sd=round(float(old.w2.std()), 6), new_w2_sd=round(float(new.w2.std()), 6),
        correct_w2_sd=round(exp_w2, 6),
        old_w1_overscaled_by=round(float(old.w1.std()) / exp_w1, 3),
        old_w2_underscaled_by=round(float(old.w2.std()) / exp_w2, 3))

    # what the swap did to the expert output at initialisation, relative to A's Dense residual
    torch.manual_seed(0)
    h = torch.randn(512, d_trunk)
    def _bank_out(bank):
        u = torch.nn.functional.gelu(torch.einsum("bd,kdr->bkr", h, bank.w1) + bank.b1)
        return torch.einsum("bkr,krd->bkd", u, bank.w2) + bank.b2
    with torch.no_grad():
        out["expert_init"]["old_output_sd"] = round(float(_bank_out(old).std()), 6)
        out["expert_init"]["new_output_sd"] = round(float(_bank_out(new).std()), 6)
    return out


def _count_old_scan_draws(n_genes: int, n_bits: int) -> int:
    """Number of parameter tensors the old scan drew before returning -- the size of the shift."""
    n = 0
    ref = sum(p.numel() for p in _OldModel(n_genes, n_bits, "C").parameters())
    n += 1
    for hdim in range(1, 4096):
        tot = sum(p.numel() for p in _OldModel(n_genes, n_bits, "A", dense_hidden=hdim).parameters())
        n += 1
        if tot > ref * 1.02:
            break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = audit()
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
