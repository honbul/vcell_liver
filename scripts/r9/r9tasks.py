"""Task definitions shared by every runner so that all heads see exactly the same training cells."""
import numpy as np
import pandas as pd

from r9common import BROAD, CACHE


def load_obs():
    return pd.read_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t', index_col=0)


def rep_name(cond):
    target, lineage = cond.get('target'), cond.get('lineage')
    return f'lin_{lineage}' if lineage else ('full' if not target else f'excl_{target.replace("/", "")}')


def select(obs, cond, test, seed):
    """-> LABELS, yi (all cells; -1 outside task), src idx, te idx, design dict.
    Sparse condition: target restricted to N cells from D source donors (pool: >=128 target cells; N/D each)."""
    target, lineage = cond.get('target'), cond.get('lineage')
    donor = obs.patient.values
    if lineage:
        LABELS = sorted(obs.res3[obs.broad == lineage].unique())
        yi = obs.res3.map({k: i for i, k in enumerate(LABELS)}).fillna(-1).astype(int).values
        inlin = obs.broad.values == lineage
    else:
        LABELS = list(BROAD)
        yi = obs.broad.map({k: i for i, k in enumerate(BROAD)}).values
        inlin = np.ones(len(obs), bool)
    src = np.where((donor != test) & inlin)[0]
    te = np.where((donor == test) & inlin)[0]
    design = {}
    if target and not lineage:
        ti = BROAD.index(target)
        tc = {d: src[(donor[src] == d) & (yi[src] == ti)] for d in np.unique(donor[src])}
        pool = sorted(d for d, ix in tc.items() if len(ix) >= 128)
        rng = np.random.default_rng([BROAD.index(target), int(test[3:]) if test.startswith('HCC') else 0, seed, cond['N'], cond['D']])
        chosen = sorted(rng.choice(pool, cond['D'], replace=False))
        idx = np.concatenate([rng.choice(tc[d], cond['N'] // cond['D'], replace=False) for d in chosen])
        src = np.concatenate([src[yi[src] != ti], idx])
        design = dict(target=target, N=cond['N'], D=cond['D'], pool=pool, chosen=chosen)
    return LABELS, yi, src, te, design
