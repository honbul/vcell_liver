"""Acquisition policies. Every policy sees only: episode inputs (fingerprints, f0, control features), the revealed groups
and their revealed values. It never sees query groups' values or unrevealed values (fixtures check this)."""
import numpy as np

from .common import sha_hex
from .models import GP, make_gps, mixture_weights, nu_vector, noise_features

BATCH = 4


class Context:
    def __init__(self, ep, ek, hp, S, Y, Z, round_idx, rng_seed):
        self.ep, self.ek, self.hp = ep, ek, hp
        self.S = list(S)
        self.Y, self.Z = Y, Z
        self.round = round_idx
        self.rng = np.random.RandomState(rng_seed)
        rev = set(self.S)
        self.cand = np.array([int(i) for i in ep.pool if int(i) not in rev])
        # reference pool: unrevealed pool groups, capped at 256 by hash (input features only)
        if len(self.cand) > 256:
            h = [sha_hex(f"{ep.cell}|{ep.groups[i]}|{ep.seed}") for i in self.cand]
            self.R = np.sort(self.cand[np.argsort(h)[:256]])
        else:
            self.R = self.cand.copy()


def _greedy_vr(Sig, nu, cand, R, b):
    """Batch-greedy global variance reduction on a fixed Gaussian posterior over the pool positions.
    Sig is indexed by position in `cand` (cand covers R)."""
    pos = {g: i for i, g in enumerate(cand)}
    Ri = np.array([pos[g] for g in R])
    Sig = Sig.copy()
    chosen = []
    avail = np.ones(len(cand), bool)
    for _ in range(min(b, len(cand))):
        den = nu[cand] + np.diag(Sig)
        A = (Sig[Ri] ** 2).sum(0) / den / len(Ri)
        A[~avail] = -np.inf
        j = int(np.argmax(A))
        chosen.append(int(cand[j]))
        avail[j] = False
        col = Sig[:, j].copy()
        Sig -= np.outer(col, col) / (nu[cand[j]] + col[j])
    return chosen


def random_policy(ctx):
    return [int(x) for x in ctx.rng.choice(ctx.cand, size=min(BATCH, len(ctx.cand)), replace=False)]


def diversity_policy(ctx):
    T = ctx.ek.tan
    chosen = []
    ref = list(ctx.S)
    cand = list(ctx.cand)
    for _ in range(min(BATCH, len(cand))):
        dmin = 1 - T[np.ix_(cand, ref)].max(1) if ref else np.ones(len(cand))
        # maximise the distance to the nearest revealed / already chosen group; ties by the smallest group name
        m = dmin.max()
        ties = [i for i in range(len(cand)) if dmin[i] == m]
        best = min(ties, key=lambda i: ctx.ep.groups[cand[i]])
        chosen.append(int(cand[best]))
        ref.append(cand[best])
        cand.pop(best)
    return chosen


def _fixed_post(ctx, hetero=False):
    hp = dict(ctx.hp)
    if not hetero:
        hp = {**hp, "hetero": False}
    nu = nu_vector(ctx.ep, ctx.ek, hp)
    g = make_gps(ctx.ek, nu, hp, "fixed")[0][1]
    mu, Sig = g.posterior(ctx.S, ctx.Z, ctx.cand)
    return nu, g, mu, Sig


def latent_variance_policy(ctx):
    _, _, _, Sig = _fixed_post(ctx)
    v = np.diag(Sig)
    order = np.lexsort((ctx.ep.groups[ctx.cand], -v))
    return [int(ctx.cand[i]) for i in order[:BATCH]]


def total_variance_policy(ctx):
    nu, g, _, Sig = _fixed_post(ctx, hetero=True)
    v = np.diag(Sig) + g.nu[ctx.cand]
    order = np.lexsort((ctx.ep.groups[ctx.cand], -v))
    return [int(ctx.cand[i]) for i in order[:BATCH]]


def fixed_vr_policy(ctx):
    nu, g, _, Sig = _fixed_post(ctx)
    return _greedy_vr(Sig, g.nu, ctx.cand, ctx.R, BATCH)


# ------------------------------------------------------------------ mixture-based policies
def _mixture(ctx, hetero, adapt=True):
    hp = ctx.hp if hetero else {**ctx.hp, "hetero": False}
    nu = nu_vector(ctx.ep, ctx.ek, hp)
    gps = make_gps(ctx.ek, nu, hp, "mixture")
    w = mixture_weights(gps, ctx.S, ctx.Z, hp["T"]) if adapt else np.full(len(gps), 1.0 / len(gps))
    posts = [g.posterior(ctx.S, ctx.Z, ctx.cand) for _, g in gps]
    return gps, w, posts


def _shortlist(ctx, gps, w, posts, n=64):
    pos = {g: i for i, g in enumerate(ctx.cand)}
    Ri = np.array([pos[g] for g in ctx.R])
    A = np.zeros(len(ctx.cand))
    for (name, g), wh, (mu, Sig) in zip(gps, w, posts):
        A += wh * (Sig[Ri] ** 2).sum(0) / (g.nu[ctx.cand] + np.diag(Sig)) / len(Ri)
    order = np.lexsort((ctx.ep.groups[ctx.cand], -A))
    return order[:n], Ri


def proposed_policy(ctx, hetero=True, adapt=True, batch_condition=True, n_short=64, n_fant=8):
    """Noise-aware outcome-adaptive mixture design (A_mix with fantasies, batch-greedy conditioning)."""
    gps, w, posts = _mixture(ctx, hetero, adapt)
    T = ctx.hp["T"]
    d = ctx.Z.shape[1]
    short, Ri = _shortlist(ctx, gps, w, posts, n_short)
    H = len(gps)
    mus = np.stack([p[0] for p in posts])                 # H x P x d
    Sigs = [p[1].copy() for p in posts]                    # H x (P x P)
    nus = np.stack([g.nu[ctx.cand] for _, g in gps])       # H x P
    rs = np.random.RandomState(int(sha_hex(f"fant|{ctx.ep.cell}|{ctx.ep.seed}|{ctx.round}")[:8], 16))
    eps = rs.randn(n_fant, d)
    uh = rs.rand(n_fant)
    chosen = []
    avail = list(short)
    for _ in range(min(BATCH, len(avail))):
        best, best_val = None, np.inf
        for j in avail:
            vj = np.array([Sigs[h][j, j] + nus[h, j] for h in range(H)])          # predictive var per axis
            # fantasies: h ~ w, y ~ N(mu_h[j], vj[h])
            cw = np.cumsum(w)
            hs = np.minimum(np.searchsorted(cw, uh * cw[-1]), H - 1)
            Y = mus[hs, j, :] + np.sqrt(vj[hs])[:, None] * eps                    # F x d
            # tempered weight update
            diff = Y[:, None, :] - mus[None, :, j, :]                                # F x H x d
            ll = -0.5 * (diff ** 2).sum(2) / vj[None, :] - 0.5 * d * np.log(2 * np.pi * vj)[None, :]
            lw = np.log(np.maximum(w, 1e-300))[None, :] + T * ll if adapt else np.log(np.maximum(w, 1e-300))[None, :] + 0 * ll
            lw -= lw.max(1, keepdims=True)
            wn = np.exp(lw)
            wn /= wn.sum(1, keepdims=True)                                           # F x H
            kRj = np.stack([Sigs[h][Ri, j] for h in range(H)])                       # H x R
            var_new = np.stack([np.diag(Sigs[h])[Ri] for h in range(H)]) - kRj ** 2 / vj[:, None]   # H x R
            gain = kRj / vj[:, None]                                                 # H x R
            mu_new = mus[:, Ri, :][None] + gain[None, :, :, None] * (Y[:, None, :] - mus[None, :, j, :])[:, :, None, :]  # F x H x R x d
            mbar = (wn[:, :, None, None] * mu_new).sum(1)                           # F x R x d
            spread = ((mu_new - mbar[:, None]) ** 2).sum(3).mean(2) / d              # F x H
            V = (wn * (var_new.mean(1)[None, :] + spread)).sum(1)                   # F
            val = V.mean()
            if val < best_val - 1e-15 or (abs(val - best_val) <= 1e-15 and ctx.ep.groups[ctx.cand[j]] < ctx.ep.groups[ctx.cand[best]]):
                best, best_val = j, val
        chosen.append(int(ctx.cand[best]))
        avail.remove(best)
        if batch_condition:
            for h in range(H):
                col = Sigs[h][:, best].copy()
                Sigs[h] -= np.outer(col, col) / (nus[h, best] + col[best])
        else:   # top-b: no conditioning between picks of the batch
            pass
    return chosen


def pdbal_policy(ctx, n_samples=32, n_cand=64):
    """Adapted PDBAL (BATCHIE): minimise sum over posterior-sample pairs of distance x prod of Bhattacharyya overlaps."""
    gps, w, posts = _mixture(ctx, hetero=True, adapt=True)
    d = ctx.Z.shape[1]
    rs = np.random.RandomState(int(sha_hex(f"pdbal|{ctx.ep.cell}|{ctx.ep.seed}|{ctx.round}")[:8], 16))
    ci = np.arange(len(ctx.cand))
    if len(ci) > n_cand:
        ci = np.sort(rs.choice(ci, n_cand, replace=False))
    pos = {g: i for i, g in enumerate(ctx.cand)}
    Ri = np.array([pos[g] for g in ctx.R])
    pts = np.unique(np.concatenate([Ri, ci]))
    ppos = {p: i for i, p in enumerate(pts)}
    cw = np.cumsum(w)
    hs = np.minimum(np.searchsorted(cw, rs.rand(n_samples) * cw[-1]), len(gps) - 1)
    chols = {}
    F = np.empty((n_samples, len(pts), d))
    noise = np.empty((n_samples, len(pts)))
    for s, h in enumerate(hs):
        mu, Sig = posts[h]
        if h not in chols:
            M = Sig[np.ix_(pts, pts)]
            M = 0.5 * (M + M.T) + 1e-8 * np.eye(len(pts)) * max(1.0, np.trace(M) / len(pts))
            ev, evec = np.linalg.eigh(M)
            chols[h] = evec * np.sqrt(np.clip(ev, 0, None))
        F[s] = mu[pts] + chols[h] @ rs.randn(len(pts), d)
        noise[s] = gps[h][1].nu[ctx.cand[pts]]
    Rp = np.array([ppos[r] for r in Ri])
    Cp = np.array([ppos[c] for c in ci])
    iu, ju = np.triu_indices(n_samples, 1)
    dist = np.sqrt(((F[iu][:, Rp] - F[ju][:, Rp]) ** 2).mean((1, 2)))           # pairs
    logcur = np.log(np.maximum(dist, 1e-300))
    nv = 0.5 * (noise[iu][:, Cp] + noise[ju][:, Cp])                                  # pairs x C
    logbc = -((F[iu][:, Cp] - F[ju][:, Cp]) ** 2).sum(2) / (8 * nv)                 # pairs x C
    chosen = []
    avail = np.ones(len(ci), bool)
    for _ in range(min(BATCH, len(ci))):
        tot = logcur[:, None] + logbc
        m = tot.max(0)
        score = m + np.log(np.exp(tot - m).sum(0))
        score[~avail] = np.inf
        k = int(np.argmin(score))
        chosen.append(int(ctx.cand[ci[k]]))
        avail[k] = False
        logcur = logcur + logbc[:, k]
    return chosen


def learned_loss_features(ctx):
    nu, g, _, Sig = _fixed_post(ctx, hetero=True)
    ep = ctx.ep
    nf = noise_features(ep, ctx.ek)
    T = ctx.ek.tan
    maxsim = T[np.ix_(ctx.cand, ctx.S)].max(1) if len(ctx.S) else np.zeros(len(ctx.cand))
    meansim = T[np.ix_(ctx.cand, ctx.cand)].mean(1)
    return np.column_stack([np.diag(Sig), g.nu[ctx.cand], nf[ctx.cand, 1], nf[ctx.cand, 3], nf[ctx.cand, 4], maxsim, meansim,
                            np.full(len(ctx.cand), len(ctx.S))])


def learned_loss_policy(ctx):
    model = ctx.hp["learned_loss_model"]
    X = np.nan_to_num(learned_loss_features(ctx))
    p = model.predict(X)
    order = np.lexsort((ctx.ep.groups[ctx.cand], -p))
    return [int(ctx.cand[i]) for i in order[:BATCH]]


POLICIES = {
    "random": random_policy,
    "diversity": diversity_policy,
    "latent_variance": latent_variance_policy,
    "fixed_vr": fixed_vr_policy,
    "total_variance": total_variance_policy,
    "proposed_nam": lambda c: proposed_policy(c, hetero=bool(c.hp.get("hetero")), adapt=True, batch_condition=True),
    "abl_homoscedastic": lambda c: proposed_policy(c, hetero=False, adapt=True, batch_condition=True),
    "abl_prior_weights": lambda c: proposed_policy(c, hetero=bool(c.hp.get("hetero")), adapt=False, batch_condition=True),
    "abl_top_b": lambda c: proposed_policy(c, hetero=bool(c.hp.get("hetero")), adapt=True, batch_condition=False),
    "pdbal_rna": pdbal_policy,
    "learned_loss": learned_loss_policy,
}
