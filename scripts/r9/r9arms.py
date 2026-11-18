"""Model zoo shared by A2 (2x2 + baselines), A3 (calibration / pseudo-unknown) and A4 (external)."""
import numpy as np

from r9model import *

# Same total prototype budget for every multi-prototype arm: K = M_PER_CLASS * C.
M_PER_CLASS = 3

MODELS = ['LR_cell', 'LR_class', 'LR_donor', 'MLP_donor', 'P1_class', 'P1_donor', 'GMM_K',
          'A00', 'A10', 'A01', 'A11', 'TL_donor', 'RA_donor']
DESCR = {
    'LR_cell': 'multinomial logistic regression, unweighted cells',
    'LR_class': 'logistic, class-balanced loss L_class',
    'LR_donor': 'logistic, donor-class balanced loss L_donor,class',
    'MLP_donor': 'MLP 50-128-C, L_donor,class',
    'P1_class': 'single prototype per class (K=C), L_class',
    'P1_donor': 'single prototype per class (K=C), L_donor,class',
    'GMM_K': 'generative per-class diagonal GMM, K total (no discriminative training)',
    'A00': 'fixed allocation K/C per class (k-means init), L_class',
    'A10': 'fixed allocation K/C per class (k-means init), L_donor,class',
    'A01': 'donor-supported adaptive allocation (total K), L_class',
    'A11': 'donor-supported adaptive allocation (total K), L_donor,class',
    'TL_donor': 'training-loss greedy allocation (total K), L_donor,class',
    'RA_donor': 'random class allocation (total K), L_donor,class',
}


def run_models(Ztr, ytr, dtr, Zte, C, seed, models=MODELS, Zextra=None):
    """Returns {model: dict(logp=(n,C), logdens=(n,) or None, alloc=Allocation or None)} on Zte (and Zextra if given)."""
    K = M_PER_CLASS * C
    w = {s: loss_weights(ytr, dtr, s, C) for s in ('cell', 'class', 'donor')}
    out = {}
    cache = {}

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
                cache[kind] = alloc_greedy(Ztr, ytr, dtr, C, K, kind, seed)
        return cache[kind]

    for m in models:
        res = dict(logdens=None, alloc=None)
        Xs = [Zte] + ([Zextra] if Zextra is not None else [])
        if m.startswith('LR_'):
            clf = fit_lr(Ztr, ytr, w[m.split('_')[1]], seed)
            res['logp'] = [lr_logp(clf, X) for X in Xs]
        elif m == 'MLP_donor':
            h = MLPHead(C, Ztr.shape[1], seed=seed).fit(Ztr, ytr, w['donor'])
            res['logp'] = [h.predict_logp(X) for X in Xs]
        elif m == 'GMM_K':
            comps = gmm_head(Ztr, ytr, C, M_PER_CLASS, seed)
            res['logp'], res['logdens'] = [], []
            for X in Xs:
                S = gen_class_scores(X, comps, C)
                res['logp'].append(log_softmax(S))
                res['logdens'].append(np.logaddexp.reduce(S, axis=1))
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
            res['n_proto'] = al.count()
        out[m] = res
    return out
