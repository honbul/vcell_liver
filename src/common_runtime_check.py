"""GPU preflight of one train batch and the full-panel inference shape, without scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from artifact_io import digest
from common_runtime import expression_model, flags_for
from common_selection import NEURAL, immutable
from common_train import batches
from common_train_data import TrainingData
from run_support import seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['gears','scgpt'], required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    config = json.loads(NEURAL.read_text())
    seed_all(0)
    device = torch.device('cuda:0')
    d = TrainingData(a.dataset)
    rows, pairs = d.schedule('D0',0,1)
    ix = batches(rows,config['training_batch_size'])[0]
    gs = [d.train_targets[c] for c in d.train_names[ix]]
    model, predict = expression_model(d,a.model,'M0',a.output,device)
    panel = None
    if a.model == 'scgpt':
        from common_scgpt import gene_panel
        panel = gene_panel(gs,len(d.genes),config['scgpt']['max_train_genes'],
                           np.random.default_rng(0))
    control, truth, _ = d.batch(ix,pairs,panel)
    flags = flags_for(gs,len(d.genes),device,panel)
    model.train()
    pred = predict(torch.from_numpy(control).to(device),flags,panel)
    loss = (pred.float()-torch.from_numpy(truth).to(device)).square().mean()
    if pred.shape != truth.shape or not torch.isfinite(loss):
        raise RuntimeError('Incorrect native expression training output')
    loss.backward()
    trained = [p for p in model.parameters() if p.grad is not None]
    if not trained or not all(torch.isfinite(p.grad).all() for p in trained):
        raise RuntimeError('Missing or nonfinite gradient')
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=config[a.model]['default_lr'])
    opt.step()
    model.eval()
    with torch.inference_mode():
        controls = torch.from_numpy(d.eval_control[:config['validation_control_chunk']]).to(device)
        full = predict(controls,flags_for([gs[0]]*len(controls),len(d.genes),device))
        if full.shape != controls.shape or not torch.isfinite(full).all():
            raise RuntimeError('Incorrect full-panel expression inference')
    immutable(a.output/'result.json',dict(
        status='smoke_verified', model=a.model, dataset_sha256=digest(a.dataset/'receipt.json'),
        training_shape=list(pred.shape), full_panel_inference_shape=list(full.shape),
        gradient_tensors=len(trained), target_flags_preserved=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu=torch.cuda.get_device_name(), no_validation_or_test_scoring=True))


if __name__ == '__main__':
    main()
