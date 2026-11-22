"""Vectorized evaluation of source-specific rankings with the registered tie policy."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vc_build.external import SOURCE, native_sensitivity

from .source import correlation


def top_prob(rank, k):
    terminal = np.sort(rank, axis=0)[-k]
    gt, eq = rank > terminal, rank == terminal
    return gt + eq*((k-gt.sum(axis=0))/eq.sum(axis=0))


def cell_metrics(observed, predictions):
    y = rankdata(observed)
    n = len(y)
    if n < 2 or np.ptp(y) == 0:
        return None
    p = rankdata(predictions, axis=0)
    k = min(n,5)
    gains = (y-1)/(n-1)
    order = np.argsort(-p, axis=0, kind="stable")
    above = gains[order[:k]]
    for col in range(p.shape[1]):
        if np.unique(p[:,col]).size==n:
            continue
        ranks = p[:,col]
        for position, i in enumerate(order[:k,col]):
            matched = ranks==ranks[i]
            if matched.sum()>1:
                above[position,col] = gains[matched].mean()
    down = 1./np.log2(np.arange(2,k+2))
    ndcg = (down @ above)/(down @ np.sort(gains)[-k:][::-1])
    precision = top_prob(y,k) @ top_prob(p,k)/k
    spearman = correlation(p,y)
    return np.stack([ndcg,precision,spearman],axis=1)


def compare(wide, observations, names):
    """Return one matrix of five metrics per native screening source."""
    source_labels = observations.assign(outcome=native_sensitivity(observations))
    joined = source_labels[["cell_id","lineage_group","compound_id",*SOURCE,"outcome"]].merge(
        wide[["cell_id","compound_id",*names]], on=["cell_id","compound_id"], validate="many_to_one")
    joined = joined.loc[np.isfinite(joined[names]).all(axis=1)]
    cells, drugs = {}, {}
    for assay, piece in joined.groupby(SOURCE):
        cell_rows = cells.setdefault(assay,[])
        drug_rows = drugs.setdefault(assay,[])
        for _,sample in piece.groupby("cell_id"):
            values = cell_metrics(sample.outcome.to_numpy(), sample[names].to_numpy())
            if values is not None:
                cell_rows.append(values)
        by_family = piece.groupby(["lineage_group","compound_id"],as_index=False)[["outcome",*names]].mean()
        for _,rows in by_family.groupby("compound_id"):
            y = rows.outcome.to_numpy()
            if len(y)<5 or np.ptp(y)==0:
                continue
            p = rows[names].to_numpy()
            drug_rows.append(np.stack([correlation(rankdata(p,axis=0),rankdata(y)),correlation(p,y)],axis=1))
    return cells,drugs


def pooled(all_results, names, metrics):
    cell_results, drug_results = {},{}
    for cells,drugs in all_results:
        for key,record in cells.items():
            cell_results.setdefault(key,[]).extend(record)
        for key,record in drugs.items():
            drug_results.setdefault(key,[]).extend(record)
    pieces = [np.column_stack((np.mean(drug_results[k],axis=0),np.mean(cell_results[k],axis=0)))
              for k in cell_results if cell_results[k] and drug_results[k]]
    if not pieces:
        raise ValueError("Insufficient lineages to evaluate both V0+ axes")
    return pd.DataFrame(np.mean(pieces,axis=0),index=names,columns=metrics)
