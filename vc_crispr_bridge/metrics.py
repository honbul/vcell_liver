"""Measure gene-specific deviations on separately frozen CRISPRi predictions."""
import numpy as np


def row_correlation(pred, truth):
    pred = pred - pred.mean(axis=1, keepdims=True)
    truth = truth - truth.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.einsum("ij,ij->i", pred, pred) *
                          np.einsum("ij,ij->i", truth, truth))
    numerator = np.einsum("ij,ij->i", pred, truth)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def top_effect_metrics(pred, truth, depth=20):
    if depth > truth.shape[1] or depth < 1:
        raise ValueError("Invalid DEG depth")
    top_truth = np.argsort(-np.abs(truth), axis=1, kind="stable")[:, :depth]
    top_prediction = np.argsort(-np.abs(pred), axis=1, kind="stable")[:, :depth]
    recovery, sign_recovery, direction_truth = [], [], []
    for i, (known, found) in enumerate(zip(top_truth, top_prediction)):
        shared = np.intersect1d(known, found[pred[i, found] != 0], assume_unique=True)
        observed_top = np.sign(truth[i, known])
        recovered = np.sign(pred[i, shared]) == np.sign(truth[i, shared])
        direction_truth.append(float(np.mean((np.sign(pred[i, known]) == observed_top) & (observed_top != 0))))
        recovery.append(len(shared) / depth)
        sign_recovery.append(float(np.sum(recovered & (truth[i, shared] != 0)) / depth))
    return {
        f"top{depth}_recall": float(np.mean(recovery)),
        f"top{depth}_signed_recall": float(np.mean(sign_recovery)),
        f"top{depth}_direction_at_observed": float(np.mean(direction_truth)),
    }


def perturbation_metrics(prediction, observation, source_training_mean):
    if prediction.shape != observation.shape or prediction.ndim != 2:
        raise ValueError("Predicted and observed effect tables are misaligned")
    if source_training_mean.shape != (prediction.shape[1],):
        raise ValueError("Centering vector must come from the registered training gene axis")
    if not all(np.isfinite(a).all() for a in (prediction, observation, source_training_mean)):
        raise ValueError("Nonfinite centered effect")
    # Center after freezing, separately within each assay's scored panel.
    # The fitted model and its hyperparameters never receive the observed center.
    predicted_residual = prediction - prediction.mean(axis=0, keepdims=True)
    observed_residual = observation - observation.mean(axis=0, keepdims=True)
    values = {
        "centered_pearson": float(row_correlation(predicted_residual, observed_residual).mean()),
        "centered_rmse": float(np.sqrt(np.mean(np.square(predicted_residual - observed_residual)))),
        "source_reference_pearson": float(row_correlation(
            prediction - source_training_mean, observation - source_training_mean,
        ).mean()),
        "uncentered_pearson": float(row_correlation(prediction, observation).mean()),
        "uncentered_rmse": float(np.sqrt(np.mean(np.square(prediction - observation)))),
        "measured_targets": len(observation),
        "measured_expression_genes": observation.shape[1],
    }
    values.update(top_effect_metrics(prediction, observation))
    return values
