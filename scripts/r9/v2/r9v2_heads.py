"""R9 v2 head utilities. Computation is copied from the v1 modules (r9arms.run_models, r9model.alloc_greedy) without
changing any numeric step; additions are (i) returned parameters (checkpoints), (ii) a frozen-variance prototype head,
(iii) candidate-level logging of the donor-supported greedy allocation with a snapshot at the first fallback,
(iv) a per-class matched-count k-means allocation, and (v) prototype diagnostics.
"""
import copy
import math
import time

import numpy as np
import torch

from r9model import (EPOCHS, LR, LOGVAR_REG, SUPPORT_MIN_CELLS, VAR_FLOOR, Allocation, MLPHead, ProtoHead,
                     _donor_ce, alloc_fixed, comp_stats, comps_from_alloc, fit_lr, gen_class_scores, gmm_head,
                     kmeans_labels, log_softmax, loss_weights, weighted_ce)
from r9arms import M_PER_CLASS


# ---------------------------------------------------------------- A4 refit with checkpoints (copy of r9arms.run_models)
def run_models_ckpt(Ztr, ytr, dtr, Zte, C, seed, models, Zextra=None):
    K = M_PER_CLASS * C
    w = {s: loss_weights(ytr, dtr, s, C) for s in ('cell', 'class', 'donor')}
    out, cache = {}, {}

    def lr_logp(clf, X):
        lp = np.full((len(X), C), -np.inf)
        lp[:, clf.classes_] = np.log(np.clip(clf.predict_proba(X), 1e-12, 1))
        return lp

    def get_alloc(kind):
        if kind not in cache:
            if kind == 'fixed':
                cache[kind] = alloc_fixed(Ztr, ytr, dtr, C, M_PER_CLASS, seed)
            elif kind == 'p1':
                cache[kind] = alloc_fixed(Ztr, ytr, dtr, C, 1, seed)
            else:
                from r9model import alloc_greedy
                cache[kind] = alloc_greedy(Ztr, ytr, dtr, C, K, kind, seed)
        return cache[kind]

    for m in models:
        t0 = time.process_time()
        res = dict(logdens=None, alloc=None)
        Xs = [Zte] + ([Zextra] if Zextra is not None else [])
        if m.startswith('LR_'):
            clf = fit_lr(Ztr, ytr, w[m.split('_')[1]], seed)
            res['logp'] = [lr_logp(clf, X) for X in Xs]
            res['params'] = dict(coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)
        elif m == 'MLP_donor':
            hd = MLPHead(C, Ztr.shape[1], seed=seed).fit(Ztr, ytr, w['donor'])
            res['logp'] = [hd.predict_logp(X) for X in Xs]
            res['params'] = {k: v.detach().numpy() for k, v in hd.net.state_dict().items()}
        elif m == 'GMM_K':
            comps = gmm_head(Ztr, ytr, C, M_PER_CLASS, seed)
            res['logp'], res['logdens'] = [], []
            for X in Xs:
                S = gen_class_scores(X, comps, C)
                res['logp'].append(log_softmax(S))
                res['logdens'].append(np.logaddexp.reduce(S, axis=1))
            res['params'] = dict(k=np.array([c['k'] for c in comps]), mu=np.stack([c['mu'] for c in comps]),
                                 var=np.stack([c['var'] for c in comps]), w=np.array([c['w'] for c in comps]))
        else:
            kind, scheme = {'P1_class': ('p1', 'class'), 'P1_donor': ('p1', 'donor'), 'A00': ('fixed', 'class'),
                            'A10': ('fixed', 'donor'), 'A01': ('ds', 'class'), 'A11': ('ds', 'donor'),
                            'TL_donor': ('tl', 'donor'), 'RA_donor': ('rand', 'donor')}[m]
            al = get_alloc(kind)
            torch.manual_seed(seed)
            ph = ProtoHead(comps_from_alloc(al, w[scheme]), C).fit(Ztr, ytr, w[scheme])
            res['logp'], res['logdens'] = [], []
            for X in Xs:
                lp, ld = ph.predict(X)
                res['logp'].append(lp)
                res['logdens'].append(ld)
            res['alloc'] = al
            res['params'] = proto_params(ph)
        res['cpu_seconds'] = time.process_time() - t0
        out[m] = res
    return out


def proto_params(ph):
    return dict(k=ph.k.numpy(), mu=ph.mu.detach().numpy(), logvar=ph.logvar.detach().numpy(),
                logvar0=ph.logvar0.numpy(), mixlogit=ph.mixlogit.detach().numpy())


# ---------------------------------------------------------------- frozen-variance prototype head (E1)
class FrozenVarProtoHead(ProtoHead):
    """Identical to ProtoHead except that log-variances are excluded from the optimizer (kept at their initial value).
    The LOGVAR_REG term is then identically 0 and is dropped."""

    def fit(self, Z, yi, w, epochs=EPOCHS, lr=LR):
        Z = torch.as_tensor(Z, dtype=torch.float32)
        yi = torch.as_tensor(yi, dtype=torch.long)
        w = torch.as_tensor(w, dtype=torch.float32)
        self.logvar.requires_grad_(False)
        opt = torch.optim.Adam([self.mu, self.mixlogit], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            ce = (torch.nn.functional.cross_entropy(self.scores(Z), yi, reduction='none') * w).sum()
            ce.backward()
            opt.step()
        return self


def class_fixed_var(Z, yi, C):
    """Per-class diagonal variance of the single-component (P1) initialisation: comp_stats over all class cells."""
    al = Allocation(Z, yi, np.zeros(len(yi), int), C)
    out = {}
    for k, idx in al.members:
        out[k] = comp_stats(Z, idx, al.pooled[k])[1]
    return out


# ---------------------------------------------------------------- donor-supported greedy with full logging (E2)
def alloc_greedy_ds_logged(Z, yi, donor, C, K, seed):
    """Exact copy of r9model.alloc_greedy(rule='ds') decisions. Extra outputs:
       cand_log : one row per examined component per step with rejection_reason
       snapshot : deep copy of the membership at the first step without an admissible candidate (supported-stop, K*)
    Returns (Allocation at K (A11 allocation), snapshot members or None if no fallback occurred, cand_log)."""
    al = Allocation(Z, yi, donor, C)
    donors = np.unique(donor)
    step = 0
    cand_log = []
    snapshot = None
    while al.count() < K:
        step += 1
        cands = []
        for j, (k, idx) in enumerate(al.members):
            base_row = dict(step=step, j=j, k=int(k), n_cells=int(len(idx)))
            if len(idx) < 2 * SUPPORT_MIN_CELLS:
                cand_log.append(dict(base_row, rejection_reason='too_few_cells'))
                continue
            lab = kmeans_labels(Z[idx], 2, seed + step)
            if lab.min() == lab.max():
                cand_log.append(dict(base_row, rejection_reason='kmeans_degenerate'))
                continue
            s0, s1 = al.support(idx[lab == 0]), al.support(idx[lab == 1])
            gains = []
            for d in donors:
                if ((yi == k) & (donor == d)).sum() < SUPPORT_MIN_CELLS:
                    continue
                keep = donor[idx] != d
                if keep.sum() < 2 * SUPPORT_MIN_CELLS:
                    continue
                lab_ex = kmeans_labels(Z[idx[keep]], 2, seed + step)
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
                cand_log.append(dict(base_row, sup0=s0, sup1=s1, rejection_reason='no_evaluable_donor'))
                continue
            g = np.array(gains)
            gm, fp, ms = float(g.mean()), float((g > 0).mean()), min(s0, s1)
            adm = (gm > 0) and (fp >= 0.5) and ms >= 2
            reasons = [r for r, bad in (('mean_gain<=0', not gm > 0), ('frac_pos<0.5', not fp >= 0.5),
                                        ('child_support<2', not ms >= 2)) if bad]
            cand_log.append(dict(base_row, sup0=s0, sup1=s1, mean_gain=gm, frac_pos=fp, n_eval=len(g),
                                 admissible=bool(adm), rejection_reason='admissible' if adm else '+'.join(reasons)))
            cands.append(dict(j=j, k=k, score=gm, admissible=bool(adm), sup0=s0, sup1=s1, frac_pos=fp, n_eval=len(g)))
        adm = [c for c in cands if c['admissible']]
        fallback = False
        if adm:
            best = max(adm, key=lambda c: c['score'])
        else:
            if snapshot is None:
                snapshot = copy.deepcopy(al.members)
            fallback = True
            cnt = np.bincount([k for k, _ in al.members], minlength=C)
            order = np.argsort(cnt, kind='stable')
            best = None
            for k in order:
                js = [j for j, (kk, idx) in enumerate(al.members) if kk == k and len(idx) >= 2]
                if js:
                    best = dict(j=max(js, key=lambda j: len(al.members[j][1])), k=int(k), score=np.nan)
                    break
            if best is None:
                break
        ok = al.split(best['j'], seed + step)
        al.log.append(dict(step=step, fallback=fallback, n_candidates=len(cands), n_admissible=len(adm),
                           j=int(best['j']), k=int(best['k']), score=float(best['score']), ok=ok))
        if not ok:
            break
    return al, snapshot, cand_log


def alloc_from_members(Z, yi, donor, C, members):
    al = Allocation(Z, yi, donor, C)
    al.members = list(members)
    return al


def alloc_fixed_counts(Z, yi, donor, C, counts, seed):
    """Per-class k-means partition with counts[k] components (same k-means routine/seed as alloc_fixed)."""
    al = Allocation(Z, yi, donor, C)
    members = []
    for k, idx in al.members:
        lab = kmeans_labels(Z[idx], int(counts[k]), seed)
        members += [(k, idx[lab == c]) for c in range(int(counts[k])) if (lab == c).any()]
    al.members = members
    return al


# ---------------------------------------------------------------- diagnostics
def proto_diagnostics(ph, Ztr, ytr, dtr, wtr, Zte, yte, C, target_k=None):
    """Variance-floor fraction, log-variance change, component effective sample size (within-class responsibility
    mass of training cells), max donor share (responsibility weighted), train weighted CE, held-out CE."""
    with torch.no_grad():
        Zt = torch.as_tensor(Ztr, dtype=torch.float32)
        var = torch.exp(ph.logvar).clamp_min(VAR_FLOOR)
        iv = 1.0 / var
        q = (Zt ** 2) @ iv.T - 2 * Zt @ (ph.mu * iv).T + (ph.mu ** 2 * iv).sum(1)
        lg = -0.5 * (q + torch.log(var).sum(1) + Zt.shape[1] * math.log(2 * math.pi))
        ml = ph.mixlogit.unsqueeze(0).expand(ph.C, -1).masked_fill(~ph.mask, -float('inf'))
        logw = torch.log_softmax(ml, 1)[ph.k, torch.arange(len(ph.k))]
        lgw = (lg + logw).numpy()
        S_tr = ph.scores(Zt).numpy()
        S_te = ph.scores(torch.as_tensor(Zte, dtype=torch.float32)).numpy()
    kk = ph.k.numpy()
    lv, lv0 = ph.logvar.detach().numpy(), ph.logvar0.numpy()
    ess, dshare = np.zeros(len(kk)), np.zeros(len(kk))
    donors = np.unique(dtr)
    for k in range(C):
        js = np.where(kk == k)[0]
        cells = np.where(ytr == k)[0]
        if len(js) == 0 or len(cells) == 0:
            continue
        r = lgw[np.ix_(cells, js)]
        r = np.exp(r - r.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        ess[js] = r.sum(0)
        for jj, j in enumerate(js):
            mass = np.array([r[dtr[cells] == d, jj].sum() for d in donors])
            dshare[j] = mass.max() / max(mass.sum(), 1e-12)
    lp_te = log_softmax(S_te)
    wte = np.zeros(len(yte))
    for k in np.unique(yte):
        wte[yte == k] = 1.0 / (yte == k).sum()
    d = dict(n_comp=len(kk), floor_frac=float((np.exp(lv) <= VAR_FLOOR * (1 + 1e-6)).mean()),
             mean_abs_dlogvar=float(np.abs(lv - lv0).mean()), max_abs_dlogvar=float(np.abs(lv - lv0).max()),
             ess_min=float(ess.min()), ess_median=float(np.median(ess)), max_donor_share=float(dshare.max()),
             mean_donor_share=float(dshare.mean()),
             train_ce_weighted=weighted_ce(S_tr, ytr, wtr), test_ce_classbal=float(-(lp_te[np.arange(len(yte)), yte] * wte).sum() / wte.sum()),
             test_nll=float(-np.clip(lp_te[np.arange(len(yte)), yte], -50, 0).mean()))
    d['ce_gap'] = d['test_ce_classbal'] - d['train_ce_weighted']
    if target_k is not None:
        js = np.where(kk == target_k)[0]
        d.update(target_n_comp=len(js), target_ess_min=float(ess[js].min()) if len(js) else np.nan,
                 target_max_donor_share=float(dshare[js].max()) if len(js) else np.nan,
                 target_floor_frac=float((np.exp(lv[js]) <= VAR_FLOOR * (1 + 1e-6)).mean()) if len(js) else np.nan,
                 target_mean_abs_dlogvar=float(np.abs(lv[js] - lv0[js]).mean()) if len(js) else np.nan)
    return d
