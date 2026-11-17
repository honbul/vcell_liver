"""Upstream WAVE (unchanged module) and the R4 variants built from its parts."""
import importlib.util, os
import torch
import torch.nn as nn
import torch.nn.functional as F


def wave_module():
    path = os.path.join(os.environ["R4_WAVE"], "wave", "model.py")
    spec = importlib.util.spec_from_file_location("wave_upstream_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class W0Native(nn.Module):
    """Upstream WAVE unchanged. delta_hat = (VAE_recon(X0) + fusion) - X0."""

    def __init__(self):
        super().__init__()
        self.net = wave_module().WAVE()

    def forward(self, x0, fp):
        return self.net(x0, fp)            # final, mu, logvar, delta

    @torch.no_grad()
    def predict_delta(self, x0, fp):
        final, _, _, _ = self.net(x0, fp)
        return final - x0


class W1DeltaAnchor(nn.Module):
    """Same GeneVAE encoder, DrugNN, GeneDrugFusion; prediction = X0 + fusion(encoded(X0), DrugNN(fp)).
    The VAE latent heads / decoder are kept as parameters but unused. A 256->64 projection head on the
    fusion's penultimate activation gives z (used only by W2 / latent regression)."""

    def __init__(self, z_dim=64):
        super().__init__()
        self.net = wave_module().WAVE()
        self.head = nn.Linear(256, z_dim)

    def forward(self, x0, fp):
        enc = self.net.gene_vae.encoder(x0)
        drug = self.net.drug_feature(fp)
        h = self.net.fusion.fusion_layer[:-1](torch.cat([enc, drug], 1))
        delta = self.net.fusion.fusion_layer[-1](h)
        return delta, self.head(h)

    @torch.no_grad()
    def predict_delta(self, x0, fp):
        return self.forward(x0, fp)[0]


class DeltaMLP(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2048 + 978, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.out = nn.Linear(512, 978)

    def forward(self, x0, fp):
        h = self.body(torch.cat([fp, x0], 1))
        return self.out(h), None

    @torch.no_grad()
    def predict_delta(self, x0, fp):
        return self.forward(x0, fp)[0]


def wave_loss(pred, target, mu, logvar, alpha=1.0, beta=0.01):
    """Upstream loss_fct, copied verbatim in content (wave/train.py)."""
    mse_loss = F.mse_loss(pred, target)
    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_mean * target_mean, dim=1)
    denominator = torch.sqrt(torch.sum(pred_mean ** 2, dim=1) * torch.sum(target_mean ** 2, dim=1))
    pearson_loss = 1 - torch.mean(numerator / denominator)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return mse_loss + alpha * pearson_loss + beta * kld_loss


def cross_half_similarity(u, ua, ub, has):
    """Teacher similarity from independent plate views: for two replicated units, mean of cos(uA_i, uB_j) and cos(uB_i, uA_j)
    (different plates, hence different DMSO pools); otherwise the consensus cosine."""
    un = F.normalize(u, dim=1); an = F.normalize(ua, dim=1); bn = F.normalize(ub, dim=1)
    cons = un @ un.T
    cross = 0.5 * (an @ bn.T + bn @ an.T)
    both = has[:, None] & has[None, :]
    return torch.where(both, cross, cons)


def soft_targets(u, r, rho, tau_t, sim=None):
    """q_i = (1 - rho) onehot(i) + rho * p~_i; p_ij = softmax_{j != i}(cos(u_i,u_j)/tau_T), off-diagonal scaled by sqrt(r_i r_j),
    the removed mass returned to the diagonal. u, r are constants (no gradient)."""
    n = u.shape[0]
    if sim is None:
        un = F.normalize(u, dim=1)
        sim = un @ un.T
    s = sim / tau_t
    eye = torch.eye(n, dtype=torch.bool, device=u.device)
    p = torch.softmax(s.masked_fill(eye, float("-inf")), 1) if n > 1 else torch.zeros_like(s)
    p = p * torch.sqrt(r[:, None] * r[None, :])
    q = rho * p.masked_fill(eye, 0.0)
    q = q + torch.diag(1.0 - q.sum(1))
    return q


def rel_loss(z, u, q, r, tau_s):
    logits = F.normalize(z, dim=1) @ F.normalize(u, dim=1).T / tau_s
    return -(r * (q * torch.log_softmax(logits, 1)).sum(1)).mean()
