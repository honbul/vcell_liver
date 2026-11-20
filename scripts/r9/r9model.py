"""R9 heads: weighted logistic/MLP baselines, diagonal-Gaussian prototype head (section 6) and capacity allocation rules.

Loss weights (section 8):
  'cell'  : w_i = 1/n
  'class' : w_i = 1/(C n_k)                     (L_class)
  'donor' : w_i = 1/(C D_k n_kd)                (L_donor,class)
Weighted loss is the only correction (no additional resampling), so there is no double correction.

Prototype head: s_k(z) = logsumexp_j [log w_kj + log N(z; mu_kj, diag var_kj)], sum_j w_kj = 1 per class (within-class
softmax), class prior uniform and identical for every arm; p(k|z) = softmax_k s_k(z).
"""
import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

SHRINK = 0.3          # diagonal variance shrinkage toward the pooled within-class variance
VAR_FLOOR = 1e-3
EPOCHS = 300          # identical optimisation budget for every prototype arm
LR = 0.02
LOGVAR_REG = 1e-2
SUPPORT_MIN_CELLS = 10
SUPPORT_MIN_FRAC = 0.05
KMEANS_SUB = 4000
torch.set_num_threads(8)


def loss_weights(y, donor, scheme, C=None):
    y = np.asarray(y)
    donor = np.asarray(donor)
    classes = np.unique(y)
    C = C or len(classes)
    n = len(y)
    w = np.zeros(n, float)
    if scheme == 'cell':
        w[:] = 1.0 / n
    elif scheme == 'class':
        for k in classes:
            m = y == k
            w[m] = 1.0 / (C * m.sum())
    elif scheme == 'donor':
        for k in classes:
            mk = y == k
            ds = np.unique(donor[mk])
            for d in ds:
                m = mk & (donor == d)
                w[m] = 1.0 / (C * len(ds) * m.sum())
    else:
        raise ValueError(scheme)
    return w / w.sum()


# ---------------- baselines ----------------
def fit_lr(Z, y, w, seed=0):
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(Z, y, sample_weight=w * len(w))
    return clf


class MLPHead:
    def __init__(self, C, d, hidden=128, seed=0):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(hidden, C))

    def fit(self, Z, yi, w, epochs=EPOCHS, lr=3e-3):
        Z = torch.as_tensor(Z, dtype=torch.float32)
        yi = torch.as_tensor(yi, dtype=torch.long)
        w = torch.as_tensor(w, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = (torch.nn.functional.cross_entropy(self.net(Z), yi, reduction='none') * w).sum()
            loss.backward()
            opt.step()
        return self

    def predict_logp(self, Z):
        self.net.eval()
        with torch.no_grad():
            return torch.log_softmax(self.net(torch.as_tensor(Z, dtype=torch.float32)), 1).numpy()


# ---------------- Gaussian components ----------------
def comp_stats(Z, idx, pooled_var):
    x = Z[idx]
    mu = x.mean(0)
    var = x.var(0) if len(idx) > 1 else np.ones(Z.shape[1])
    var = (1 - SHRINK) * var + SHRINK * pooled_var + VAR_FLOOR
    return mu, var


def loggauss(Z, mu, var):
    """Z (n,d), mu/var (K,d) -> (n,K) diagonal Gaussian log density."""
    iv = 1.0 / var
    q = (Z ** 2) @ iv.T - 2 * Z @ (mu * iv).T + (mu ** 2 * iv).sum(1)
    return -0.5 * (q + np.log(var).sum(1) + Z.shape[1] * math.log(2 * math.pi))


def gen_class_scores(Z, comps, C):
    """comps: list of dicts(k, mu, var, w) with sum_j w_kj = 1 within class -> (n,C) class log-scores."""
    mu = np.stack([c['mu'] for c in comps])
    var = np.stack([c['var'] for c in comps])
    lg = loggauss(Z, mu, var) + np.log(np.array([c['w'] for c in comps]))[None]
    S = np.full((len(Z), C), -np.inf)
    ks = np.array([c['k'] for c in comps])
    for k in range(C):
        m = ks == k
        if m.any():
            S[:, k] = np.logaddexp.reduce(lg[:, m], axis=1)
    return S


def log_softmax(S):
    m = S.max(1, keepdims=True)
    return S - m - np.log(np.exp(S - m).sum(1, keepdims=True))


def weighted_ce(S, yi, w):
    lp = log_softmax(S)
    return float(-(lp[np.arange(len(yi)), yi] * w).sum() / w.sum())


def kmeans_labels(X, m, seed):
    if m == 1 or len(X) <= m:
        return np.zeros(len(X), int)
    rng = np.random.default_rng(seed)
    sub = X if len(X) <= KMEANS_SUB else X[rng.choice(len(X), KMEANS_SUB, replace=False)]
    km = KMeans(n_clusters=m, n_init=3, random_state=seed).fit(sub)
    return km.predict(X)


class Allocation:
    """Cell-to-component membership per class; components are (class, member index array)."""

    def __init__(self, Z, yi, donor, C):
        self.Z, self.yi, self.donor, self.C = Z, yi, donor, C
        self.pooled = {k: Z[yi == k].var(0) + VAR_FLOOR for k in range(C)}
        self.members = [(k, np.where(yi == k)[0]) for k in range(C)]
        self.log = []

    def comps(self, wscheme_w=None, exclude_donor=None):
        """Gaussian parameters from member cells (optionally excluding one donor); mixture weights = member mass."""
        out = []
        for k, idx in self.members:
            if exclude_donor is not None:
                idx = idx[self.donor[idx] != exclude_donor]
            if len(idx) == 0:
                continue
            mu, var = comp_stats(self.Z, idx, self.pooled[k])
            mass = len(idx) if wscheme_w is None else wscheme_w[idx].sum()
            out.append(dict(k=k, mu=mu, var=var, w=mass, n=len(idx)))
        for k in range(self.C):
            tot = sum(c['w'] for c in out if c['k'] == k)
            for c in out:
                if c['k'] == k:
                    c['w'] = c['w'] / tot
        return out

    def count(self):
        return len(self.members)

    def split(self, j, seed):
        k, idx = self.members[j]
        lab = kmeans_labels(self.Z[idx], 2, seed)
        if lab.min() == lab.max():
            return False
        self.members[j] = (k, idx[lab == 0])
        self.members.append((k, idx[lab == 1]))
        return True

    def support(self, idx):
        dc = np.bincount(np.unique(self.donor[idx], return_inverse=True)[1])
        return int(((dc >= SUPPORT_MIN_CELLS) & (dc >= SUPPORT_MIN_FRAC * len(idx))).sum())


def alloc_fixed(Z, yi, donor, C, m, seed):
    al = Allocation(Z, yi, donor, C)
    members = []
    for k, idx in al.members:
        lab = kmeans_labels(Z[idx], m, seed)
        members += [(k, idx[lab == c]) for c in range(m) if (lab == c).any()]
    al.members = members
    return al


def _donor_ce(al, comps_ex, d, cand=None):
    """Class-balanced CE on held-out donor d's cells with components fitted without d."""
    m = al.donor == d
    Zd, yd = al.Z[m], al.yi[m]
    w = np.zeros(len(yd))
    for k in np.unique(yd):
        w[yd == k] = 1.0 / (yd == k).sum()
    return weighted_ce(gen_class_scores(Zd, comps_ex, al.C), yd, w)


def alloc_greedy(Z, yi, donor, C, K, rule, seed):
    """rule: 'ds' donor-supported (inner leave-one-source-donor-out gain + child donor support),
             'tl' training-loss (in-sample CE reduction, no donor criteria), 'rand' random class."""
    al = Allocation(Z, yi, donor, C)
    rng = np.random.default_rng(seed)
    donors = np.unique(donor)
    step = 0
    while al.count() < K:
        step += 1
        if rule == 'rand':
            k = rng.integers(C)
            js = [j for j, (kk, idx) in enumerate(al.members) if kk == k]
            j = max(js, key=lambda j: len(al.members[j][1]))
            ok = al.split(j, seed + step)
            al.log.append(dict(step=step, rule='rand', k=int(k), j=int(j), ok=ok))
            if not ok:
                break
            continue
        cands = []
        if rule == 'tl':
            w_all = loss_weights(yi, donor, 'class', C)
            base_tl = weighted_ce(gen_class_scores(Z, al.comps(), C), yi, w_all)
        for j, (k, idx) in enumerate(al.members):
            if len(idx) < 2 * SUPPORT_MIN_CELLS:
                continue
            if rule == 'tl':
                lab = kmeans_labels(Z[idx], 2, seed + step)
                if lab.min() == lab.max():
                    continue
                trial = list(al.members)
                trial[j] = (k, idx[lab == 0])
                trial.append((k, idx[lab == 1]))
                saved = al.members
                al.members = trial
                new = weighted_ce(gen_class_scores(Z, al.comps(), C), yi, w_all)
                al.members = saved
                cands.append(dict(j=j, k=k, score=base_tl - new, admissible=True, sup0=np.nan, sup1=np.nan,
                                  frac_pos=np.nan, n_eval=0))
                continue
            # donor-supported rule
            lab = kmeans_labels(Z[idx], 2, seed + step)
            if lab.min() == lab.max():
                continue
            s0, s1 = al.support(idx[lab == 0]), al.support(idx[lab == 1])
            gains = []
            for d in donors:
                if ((yi == k) & (donor == d)).sum() < SUPPORT_MIN_CELLS:
                    continue
                keep = donor[idx] != d
                if keep.sum() < 2 * SUPPORT_MIN_CELLS:
                    continue
                lab_ex = kmeans_labels(Z[idx[keep]], 2, seed + step)   # split defined without donor d
                if lab_ex.min() == lab_ex.max():
                    continue
                base = _donor_ce(al, al.comps(exclude_donor=d), d)
                saved = al.members
                ix = idx[keep]
                trial = list(saved)
                trial[j] = (k, ix[lab_ex == 0])
                trial.append((k, ix[lab_ex == 1]))
                al.members = trial
                new = _donor_ce(al, al.comps(exclude_donor=d), d)
                al.members = saved
                gains.append(base - new)
            if not gains:
                continue
            g = np.array(gains)
            adm = (g.mean() > 0) and ((g > 0).mean() >= 0.5) and min(s0, s1) >= 2
            cands.append(dict(j=j, k=k, score=float(g.mean()), admissible=bool(adm), sup0=s0, sup1=s1,
                              frac_pos=float((g > 0).mean()), n_eval=len(g)))
        adm = [c for c in cands if c['admissible']]
        fallback = False
        if adm:
            best = max(adm, key=lambda c: c['score'])
        else:
            # Fixed fallback keeps the total budget: class with fewest components, split its largest component.
            fallback = True
            cnt = np.bincount([k for k, _ in al.members], minlength=C)
            order = np.argsort(cnt, kind='stable')
            best = None
            for k in order:
                js = [j for j, (kk, idx) in enumerate(al.members) if kk == k and len(idx) >= 2]
                if js:
                    best = dict(j=max(js, key=lambda j: len(al.members[j][1])), k=int(k), score=np.nan,
                                sup0=np.nan, sup1=np.nan, frac_pos=np.nan, n_eval=0)
                    break
            if best is None:
                break
        ok = al.split(best['j'], seed + step)
        al.log.append(dict(step=step, rule=rule, fallback=fallback, n_candidates=len(cands),
                           n_admissible=len(adm), **{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                                     for k, v in best.items() if k != 'admissible'}, ok=ok))
        if not ok:
            break
    return al


# ---------------- discriminative prototype head ----------------
class ProtoHead:
    def __init__(self, comps, C):
        self.C = C
        self.k = torch.as_tensor([c['k'] for c in comps], dtype=torch.long)
        self.mu = torch.nn.Parameter(torch.as_tensor(np.stack([c['mu'] for c in comps]), dtype=torch.float32))
        lv = torch.as_tensor(np.log(np.stack([c['var'] for c in comps])), dtype=torch.float32)
        self.logvar0 = lv.clone()
        self.logvar = torch.nn.Parameter(lv)
        self.mixlogit = torch.nn.Parameter(torch.log(torch.as_tensor([c['w'] for c in comps], dtype=torch.float32)))
        self.mask = torch.zeros(C, len(comps), dtype=torch.bool)
        self.mask[self.k, torch.arange(len(comps))] = True

    def scores(self, Z):
        var = torch.exp(self.logvar).clamp_min(VAR_FLOOR)
        iv = 1.0 / var
        q = (Z ** 2) @ iv.T - 2 * Z @ (self.mu * iv).T + (self.mu ** 2 * iv).sum(1)
        lg = -0.5 * (q + torch.log(var).sum(1) + Z.shape[1] * math.log(2 * math.pi))
        # within-class normalised mixture weights: sum_j w_kj = 1
        ml = self.mixlogit.unsqueeze(0).expand(self.C, -1).masked_fill(~self.mask, -float('inf'))
        logw = torch.log_softmax(ml, 1)                     # (C,K), -inf off-class
        logw_j = logw[self.k, torch.arange(len(self.k))]     # (K,)
        x = (lg + logw_j).unsqueeze(1).expand(-1, self.C, -1).masked_fill(~self.mask.unsqueeze(0), -float('inf'))
        return torch.logsumexp(x, 2)                         # (n,C) class log-scores; uniform prior

    def fit(self, Z, yi, w, epochs=EPOCHS, lr=LR):
        Z = torch.as_tensor(Z, dtype=torch.float32)
        yi = torch.as_tensor(yi, dtype=torch.long)
        w = torch.as_tensor(w, dtype=torch.float32)
        opt = torch.optim.Adam([self.mu, self.logvar, self.mixlogit], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            ce = (torch.nn.functional.cross_entropy(self.scores(Z), yi, reduction='none') * w).sum()
            loss = ce + LOGVAR_REG * ((self.logvar - self.logvar0) ** 2).mean()
            loss.backward()
            opt.step()
        return self

    def predict(self, Z):
        with torch.no_grad():
            S = self.scores(torch.as_tensor(Z, dtype=torch.float32)).numpy()
        return log_softmax(S), np.logaddexp.reduce(S, axis=1)   # log p(k|z), unnormalised log density


def gmm_head(Z, yi, C, m, seed):
    comps = []
    for k in range(C):
        x = Z[yi == k]
        mm = min(m, max(1, len(x) // 5))
        g = GaussianMixture(n_components=mm, covariance_type='diag', reg_covar=1e-3, random_state=seed, n_init=1).fit(x)
        for j in range(mm):
            comps.append(dict(k=k, mu=g.means_[j], var=g.covariances_[j], w=g.weights_[j]))
    return comps


def comps_from_alloc(al, w_scheme):
    """Initial component parameters; mixture weights = arm-weighted member mass (normalised within class)."""
    return al.comps(wscheme_w=w_scheme)


def describe_alloc(al, donor_names):
    rows = []
    for j, (k, idx) in enumerate(al.members):
        dn, dc = np.unique(al.donor[idx], return_counts=True)
        rows.append(dict(proto=j, class_idx=int(k), n_cells=int(len(idx)), n_donors=int(len(dn)),
                         support=al.support(idx), max_donor_share=float(dc.max() / dc.sum()),
                         top_donor=str(donor_names[dn[dc.argmax()]]) if donor_names is not None else str(dn[dc.argmax()])))
    return rows
