"""Scores for independently frozen cell and compound prediction panels."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


def response_metrics(frame, score):
    within_drug, within_cell = [], []
    for _, group in frame.groupby("drug_id", sort=False):
        if len(group) < 5 or group.outcome.nunique() < 2:
            continue
        rho = (spearmanr(group.outcome, group[score]).statistic
               if group[score].nunique() > 1 else 0.0)
        within_drug.append(float(rho))
    for _, group in frame.groupby("cell_id", sort=False):
        if len(group) < 5 or group.outcome.nunique() < 2:
            continue
        relevance = group.outcome.to_numpy()
        # Shifting only when needed preserves the native nonnegative AA/AUC
        # scales and allows native assay values below zero in source checks.
        relevance = relevance - min(0.0, relevance.min())
        within_cell.append(float(ndcg_score(relevance[None, :],
                                          group[score].to_numpy()[None, :], k=5)))
    return {
        "within_drug_spearman": float(np.mean(within_drug)) if within_drug else None,
        "cell_macro_ndcg5": float(np.mean(within_cell)) if within_cell else None,
        "spearman_drugs": len(within_drug), "ndcg_cells": len(within_cell),
    }


def classification_metrics(y, prediction):
    if len(np.unique(y)) != 2:
        return {"auroc": None, "auprc": None, "prevalence": float(np.mean(y))}
    return {"auroc": float(roc_auc_score(y, prediction)),
            "auprc": float(average_precision_score(y, prediction)),
            "prevalence": float(np.mean(y))}
