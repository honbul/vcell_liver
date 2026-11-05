"""Check actual manuscript scGPT LoRA gradients, updates, and state restoration."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from artifact_io import digest, write
from common_scgpt import build, gene_panel, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    torch.manual_seed(26)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    genes = json.loads((a.dataset/'genes.json').read_text())['output']
    tr_meta = json.loads((a.dataset/'train_conditions.json').read_text())
    with np.load(a.dataset/'train_index.npz', allow_pickle=False) as ix:
        names = ix['condition'][:2]
        pairs = ix['fixed_control_index'][:2]
    indices = {g: i for i, g in enumerate(genes)}
    target_indices = [[indices[g] for g in tr_meta[c]['targets']] for c in names]
    panel = gene_panel(target_indices, len(genes), 128, np.random.default_rng(26))
    flags = np.zeros((2, len(genes)), dtype=np.int64)
    for i, ix in enumerate(target_indices):
        flags[i, ix] = 1
    np.testing.assert_array_equal(flags[:, panel].sum(1), flags.sum(1))
    flags = torch.as_tensor(flags[:, panel], device=device)
    control = torch.from_numpy(sparse.load_npz(a.dataset/'controls_train.npz')[
        pairs][:, panel].toarray()).to(device)
    treated = torch.from_numpy(sparse.load_npz(a.dataset/'treated_train.npz')[
        :2, panel].toarray()).to(device)

    model, ids, adapters = build(genes, 'M3', a.output, device)
    ids = ids[panel]
    assert len(adapters) == len(model.transformer_encoder.layers)
    attention = [layer.self_attn for layer in model.transformer_encoder.layers]
    frozen = [att.parametrizations.in_proj_weight.original for att in attention]
    assert not any(p.requires_grad for p in frozen)
    # Save only small summaries of frozen matrices before the updates.
    original_signatures = [p.detach().cpu().numpy().tobytes() for p in frozen]
    initial = {name: (mod.down.detach().clone(), mod.up.detach().clone())
               for name, mod in adapters.items()}
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    losses = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = (predict(model, ids, control, flags) - treated).square().mean()
        loss.backward()
        losses.append(float(loss.detach()))
        opt.step()
    audits = {}
    for name, mod in adapters.items():
        grad_down, grad_up = mod.down.grad.norm().item(), mod.up.grad.norm().item()
        delta_down = (mod.down.detach()-initial[name][0]).norm().item()
        delta_up = (mod.up.detach()-initial[name][1]).norm().item()
        assert min(grad_down, grad_up, delta_down, delta_up) > 0, name
        audits[name] = dict(grad_down=grad_down, grad_up=grad_up,
                            update_down=delta_down, update_up=delta_up)
    assert all(p.detach().cpu().numpy().tobytes() == original_signatures[i] and p.grad is None
               for i, p in enumerate(frozen))
    model.eval()
    with torch.inference_mode():
        before = predict(model, ids, control, flags).clone()
    state = a.output/'smoke_model.pt'
    torch.save(model.state_dict(), state)
    restore_dir = a.output/'restore'
    restore_dir.mkdir()
    loaded, tokens, _ = build(genes, 'M3', restore_dir, device)
    loaded.load_state_dict(torch.load(state, map_location='cpu', weights_only=True), strict=True)
    loaded.eval()
    with torch.inference_mode():
        after = predict(loaded, tokens[panel], control, flags)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    write(a.output/'result.json', {
        'status': 'smoke_verified', 'sample': 'first two training pairs only', 'panel_genes': len(panel),
        'targets_preserved': True, 'frozen_projection_weights_unchanged': True,
        'state_reload_exact': True, 'state_sha256': digest(state),
        'losses': losses, 'mounts': audits, 'device': str(device),
        'dataset_sha256': digest(a.dataset/'receipt.json')
    })


if __name__ == '__main__':
    main()
