"""Follow-up correction heads. The chemistry backbone and the graph correction are imported unchanged from the
original study (src/evidence_gated_r7/models.py); only the controls and the branch heads are new.

delta_hat = delta_chem + m_available * r

  M   annotation-metadata residual MLP: r = MLP(X0, ECFP, meta) - no target identity, no graph
  G0  the original gated graph correction called with an empty seed list while status / m_available keep the
      (augmented) values of the sample -> generic path of the same scaffold
  GS  the original gated graph correction with the null drug-target mapping
  GR  the original gated graph correction with the real mapping (= B11 of the original study)
  TD  target-difference head (plan 11.2): r = F(S, q, G) - F(empty, q, G), exactly 0 for an empty seed list
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evidence_gated_r7.models import Backbone, GraphCorrection, zero_linear, N_STATUS

N_META = 6     # status(4: mechanism, activity, log target count, signed fraction) + mapping(2: exact-match fraction, multi-gene fraction)


class MetaCorrection(nn.Module):
    """Same condition encoder shape as Cond of the graph arm (cell 978->64, drug 2048->64), metadata instead of status,
    then a 978-wide output head. h2 is set so that the number of operating parameters matches GR within 10 %."""

    def __init__(self, h2, d=64):
        super().__init__()
        self.cell = nn.Linear(978, d); self.drug = nn.Linear(2048, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d + N_META, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, h2), nn.ReLU())
        self.out = zero_linear(h2, 978)

    def forward(self, fp, x0s, meta):
        return self.out(self.mlp(torch.cat([F.relu(self.cell(x0s)), F.relu(self.drug(fp)), meta], 1)))


class FollowupModel(nn.Module):
    ARMS = ("chem", "M", "G0", "GS", "GR", "TD")

    def __init__(self, arm, G=None, h2=None):
        super().__init__()
        assert arm in self.ARMS
        self.arm = arm
        self.backbone = Backbone()
        if arm == "chem":
            self.corr = None
        elif arm == "M":
            self.corr = MetaCorrection(h2)
        else:
            self.corr = GraphCorrection(G, gated=True)

    def correction(self, fp, x0s, status, meta, seeds):
        if self.arm == "M":
            return self.corr(fp, x0s, meta), None
        if self.arm == "TD":
            empty = (seeds[0][:0], seeds[1][:0], seeds[2][:0])
            r1, a = self.corr(fp, x0s, status, seeds)
            r0, _ = self.corr(fp, x0s, status, empty)
            return r1 - r0, a
        return self.corr(fp, x0s, status, seeds)       # G0 receives an empty seed list from the bank

    def forward(self, fp, x0s, cmean, status=None, meta=None, m_avail=None, seeds=None):
        dc = self.backbone(fp, x0s, cmean)
        if self.corr is None:
            return dc, torch.zeros_like(dc), None
        r, a = self.correction(fp, x0s, status, meta, seeds)
        r = r * m_avail[:, None]
        return dc + r, r, a


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))
