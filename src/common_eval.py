"""One condition-weighted evaluator for all common benchmark predictions."""
import json
import numpy as np


def evaluate(dataset, role, predictions, allow_partial=False):
    if role not in {'val', 'test'}:
        raise ValueError('This evaluator supports validation and frozen test only')
    genes = json.loads((dataset / 'genes.json').read_text())['output']
    controls = np.load(dataset / 'eval_controls.npy', allow_pickle=False)
    reference = controls.mean(axis=0, dtype=np.float64)
    with np.load(dataset / f'outcomes_{role}.npz', allow_pickle=False) as measured:
        names = measured['condition'].tolist()
        truth = measured['expression_mean'].astype(np.float64)
    if not set(predictions) <= set(names) or not predictions:
        raise ValueError('Unknown conditions or missing all predictions')
    if not allow_partial and set(predictions) != set(names):
        raise ValueError('Incomplete prediction coverage')
    rows = []
    for i, c in enumerate(names):
        if c not in predictions:
            continue
        expr = np.asarray(predictions[c], dtype=np.float64)
        if expr.shape != (len(genes),) or not np.isfinite(expr).all():
            raise ValueError(f'Invalid expression output: {c}')
        true_delta, delta = truth[i] - reference, expr - reference
        rmse = np.sqrt(np.mean((delta - true_delta) ** 2))
        no_change = np.sqrt(np.mean(true_delta ** 2))
        tc, pc = true_delta - true_delta.mean(), delta - delta.mean()
        denom = np.linalg.norm(tc) * np.linalg.norm(pc)
        pearson = float(np.clip((tc @ pc) / denom, -1, 1)) if denom else None
        absolute = np.abs(true_delta)
        top = np.argsort(-absolute, kind='stable')[:20]
        top = top[absolute[top] > 0]
        # Only positive magnitudes provide ranked predicted evidence.
        pred_abs = np.abs(delta)
        pred_top = np.argsort(-pred_abs, kind='stable')[:len(top)]
        pred_top = pred_top[pred_abs[pred_top] > 0]
        rows.append(dict(
            condition=c, delta_rmse=float(rmse), delta_pearson=pearson,
            no_change_delta_rmse=float(no_change), gain_vs_no_change=float(no_change-rmse),
            direction_top20=float(np.mean(np.sign(delta[top]) == np.sign(true_delta[top]))) if len(top) else None,
            absolute_delta_top20_recall=len(set(top) & set(pred_top))/len(top) if len(top) else None))
    keys = ('delta_rmse', 'delta_pearson', 'no_change_delta_rmse', 'gain_vs_no_change',
            'direction_top20', 'absolute_delta_top20_recall')
    values = {k: [r[k] for r in rows if r[k] is not None] for k in keys}
    return dict(
        role=role, n_scored=len(rows), n_available=len(names),
        metrics={k: float(np.mean(v)) if len(v) else None for k, v in values.items()},
        n_defined={k: len(v) for k, v in values.items()}, per_condition=rows)
