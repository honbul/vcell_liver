"""Shared native prediction path for validation and frozen test inference."""
import numpy as np
import torch


def flags_for(targets, n_genes, device, panel=None):
    flags = np.zeros((len(targets), n_genes), dtype=np.int64)
    for i, ix in enumerate(targets):
        flags[i, ix] = 1
    if panel is not None:
        flags = flags[:, panel]
    if not np.array_equal(flags.sum(axis=1), [len(t) for t in targets]):
        raise ValueError('A target flag was lost')
    return torch.from_numpy(flags).to(device)


def expression_model(data, model_type, mode, out, device, graph_path=None):
    if model_type == 'scgpt':
        from common_scgpt import build, predict as native_predict
        model, gene_ids, _ = build(data.genes, mode, out, device)

        def predict(control, flags, panel=None):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return native_predict(model, gene_ids if panel is None else gene_ids[panel],
                                      control, flags)

    elif model_type == 'gears':
        from common_gears import build
        model = build(data, out, device, graph_path=graph_path)

        def predict(control, flags, panel=None):
            if panel is not None:
                raise ValueError('GEARS expects the full output panel')
            return model(control, flags)
    else:
        raise ValueError(f'Unknown model: {model_type}')
    return model, predict


def predict_conditions(predict, eval_control, meta, indices, device, cap):
    predictions = {}
    values = torch.as_tensor(eval_control, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for condition, item in meta.items():
            genes = indices(item)
            pred = []
            for ctrl in values.split(cap):
                flags = flags_for([genes]*len(ctrl), ctrl.shape[1], device)
                pred.append(predict(ctrl, flags).float().cpu().numpy())
            predictions[condition] = np.concatenate(pred, axis=0).mean(axis=0, dtype=np.float64)
    return predictions
