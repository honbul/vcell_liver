"""Run a validation-only common GEARS or scGPT training configuration."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from common_data import ROOT
from common_eval import evaluate
from common_train_data import TrainingData
from run_support import claim_run, checkpoint, digest, rng_state, restore_rng, seed_all, write

NEURAL = ROOT/'protocols/common/neural-v1.json'
ADAPTERS = ('common_data.py', 'common_train_data.py', 'common_eval.py',
            'common_gears.py', 'common_scgpt.py', 'scgpt_author.py',
            'artifact_io.py', 'common_train.py', 'run_support.py')
TERMINAL_REPRODUCTION = {'modified_reproduction_executed', 'quantitative_match',
                         'quantitative_mismatch', 'asset_unavailable', 'execution_failed'}


def require_paper_status(path):
    status = json.loads(path.read_text())
    for track in ('R1', 'R2'):
        r = status[track]
        if r['status'] not in TERMINAL_REPRODUCTION or not r['evidence']:
            raise ValueError(f'Paper-track status has not been recorded for {track}')
    return digest(path)


def batches(rows, cap):
    cuts = list(range(cap, len(rows), cap))
    if cuts and len(rows)-cuts[-1] == 1:
        # Keep every training cell without a singleton BatchNorm batch.
        cuts[-1] -= 1
    return np.split(rows, cuts)


def flags_for(targets, n_genes, device, panel=None):
    flags = np.zeros((len(targets), n_genes), dtype=np.int64)
    for i, ix in enumerate(targets):
        flags[i, ix] = 1
    if panel is not None:
        flags = flags[:, panel]
    if not np.array_equal(flags.sum(axis=1), [len(t) for t in targets]):
        raise ValueError('A target flag was lost')
    return torch.from_numpy(flags).to(device)


def predict_conditions(predict, eval_control, meta, indices, device, cap):
    predictions = {}
    values = torch.as_tensor(eval_control, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for c, item in meta.items():
            gs = indices(item)
            pred = []
            for ctrl in values.split(cap):
                flags = flags_for([gs]*len(ctrl), ctrl.shape[1], device)
                pred.append(predict(ctrl, flags, None).float().cpu().numpy())
            predictions[c] = np.concatenate(pred, axis=0).mean(axis=0, dtype=np.float64)
    return predictions


def run(a):
    config = json.loads(NEURAL.read_text())
    status_hash = require_paper_status(a.paper_status)
    if a.model == 'gears' and a.mode != 'M0':
        raise ValueError('scGPT adaptation is not a GEARS option')
    if a.phase == 'C0' and (a.mode != 'M0' or a.data_mode != 'D0'):
        raise ValueError('C0 uses only native baseline conditions')
    if a.phase == 'M' and (a.model != 'scgpt' or a.data_mode != 'D0'):
        raise ValueError('Model screening uses the fixed data protocol')
    if a.phase == 'D' and a.model != 'scgpt':
        raise ValueError('Data screening uses a selected scGPT model')
    mode_cfg = config[a.model]
    lr = a.lr if a.lr is not None else mode_cfg['default_lr']
    if lr not in mode_cfg.get('learning_rates', [mode_cfg['default_lr']]):
        raise ValueError('Learning rate outside preregistered choices')
    if a.phase in ('C0', 'M') and a.mode in ('M0', 'M-random') and lr != mode_cfg['default_lr']:
        raise ValueError('Native and random-init controls use the same default learning rate')
    if a.seed not in config['confirmation_training_seeds']:
        raise ValueError('Training seed not declared')
    if a.phase != 'F' and a.seed != config['train_seed']:
        raise ValueError('Screening must use the same single training seed')
    if not torch.cuda.is_available():
        raise RuntimeError('Request a GPU using the Slurm wrapper')
    if a.model == 'scgpt' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Common scGPT requires bf16 CUDA')
    device = torch.device('cuda:0')
    seed_all(a.seed)
    data = TrainingData(a.dataset)
    started = time.monotonic()
    dataset_sha = digest(a.dataset/'receipt.json')
    out, lock = claim_run(a.phase, a.run_id, NEURAL, {
        'adapter_sha256': {n: digest(Path(__file__).with_name(n)) for n in ADAPTERS},
        'dataset_sha256': dataset_sha, 'paper_status_sha256': status_hash, 'model': a.model,
        'mode': a.mode, 'data_mode': a.data_mode, 'lr': lr, 'seed': a.seed,
    }, a.resume)
    if a.model == 'scgpt':
        from common_scgpt import build, gene_panel, predict as native_predict

        model, gene_ids, _ = build(data.genes, a.mode, out, device)

        def predict(control, flags, panel):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return native_predict(model, gene_ids if panel is None else gene_ids[panel],
                                      control, flags)

        optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        gamma = 0.9
    else:
        from common_gears import build
        model = build(data, out, device)

        def predict(control, flags, panel):
            return model(control, flags)

        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        gamma = 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)
    state_path = out/'resume.pt'
    start_epoch, best, best_epoch, best_val, history = 1, None, 0, float('inf'), []
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['current'], strict=True)
        optim.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best, best_epoch, best_val = state['best'], state['best_epoch'], state['best_val']
        history = state['history']
        start_epoch = state['epoch'] + 1
        restore_rng(state['rng'])
        del state
    for epoch in range(start_epoch, config['epochs']+1):
        if epoch - best_epoch > config['patience'] + 1 and best_epoch:
            break
        epoch_start = time.monotonic()
        rows, pairs = data.schedule(a.data_mode, a.seed, epoch)
        panel_rng = np.random.default_rng(np.random.SeedSequence([a.seed, epoch, 2731]))
        model.train()
        n_steps, n_flags, batch_losses = 0, 0, []
        for ix in batches(rows, config['training_batch_size']):
            targets = [data.train_targets[c] for c in data.train_names[ix]]
            panel = (gene_panel(targets, len(data.genes), mode_cfg['max_train_genes'], panel_rng)
                     if a.model == 'scgpt' else None)
            x, y, _ = data.batch(ix, pairs, panel)
            control = torch.as_tensor(x, device=device, dtype=torch.float32)
            truth = torch.as_tensor(y, device=device, dtype=torch.float32)
            flags = flags_for(targets, len(data.genes), device, panel)
            optim.zero_grad(set_to_none=True)
            loss = (predict(control, flags, panel).float()-truth).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch} update {n_steps}')
            loss.backward()
            if a.model == 'scgpt':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            else:
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optim.step()
            n_steps += 1
            n_flags += sum(map(len, targets))
            batch_losses.append(float(loss.detach()))
        model.eval()
        val_preds = predict_conditions(predict, data.eval_control, data.val, data.indices,
                                       device, config['validation_control_chunk'])
        scored = evaluate(a.dataset, 'val', val_preds)
        val = scored['metrics']['delta_rmse']
        if val < best_val:
            best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_val, best_epoch = val, epoch
            checkpoint(out/'best_model.pt', best)
            write(out/'best_validation.json', scored)
        scheduler.step()
        exposures = np.unique(data.train_names[rows], return_counts=True)[1]
        stats = dict(epoch=epoch, val_delta_rmse=val, train_loss=float(np.mean(batch_losses)),
                     examples=len(rows), updates=n_steps, target_flags=n_flags,
                     changed_pairs=float(np.mean(pairs != data.paired)),
                     distinct_examples=len(np.unique(rows)), intervention_min=int(exposures.min()),
                     intervention_max=int(exposures.max()), seconds=time.monotonic()-epoch_start)
        history.append(stats)
        checkpoint(state_path, dict(
            epoch=epoch, current=model.state_dict(), best=best, best_epoch=best_epoch,
            best_val=best_val, optimizer=optim.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng_state(), history=history))
        print('C0_M_D_EPOCH', json.dumps(stats), flush=True)
    # Re-export the committed best state after resume, even if interrupted while
    # overwriting best_model.pt for a later epoch before its checkpoint completed.
    if best is None:
        raise RuntimeError('No validation result')
    checkpoint(out/'best_model.pt', best)
    model.load_state_dict(best, strict=True)
    model.eval()
    val_preds = predict_conditions(predict, data.eval_control, data.val, data.indices,
                                   device, config['validation_control_chunk'])
    scored = evaluate(a.dataset, 'val', val_preds)
    if not np.isclose(scored['metrics']['delta_rmse'], best_val, atol=1e-6, rtol=1e-5):
        raise RuntimeError('Validation score did not replay after reloading')
    write(out/'best_validation.json', scored)
    write(out/'result.json', {
        'status': 'validation_complete', 'phase': a.phase, 'run_id': a.run_id,
        'dataset_sha256': dataset_sha, 'model': a.model, 'mode': a.mode, 'data_mode': a.data_mode,
        'seed': a.seed, 'lr': lr, 'best_epoch': best_epoch, 'validation': scored['metrics'],
        'best_model_sha256': digest(out/'best_model.pt'), 'history': history,
        'epoch_seconds': sum(row['seconds'] for row in history),
        'total_seconds_this_invocation': time.monotonic()-started,
        'updates': sum(row['updates'] for row in history), 'examples': sum(row['examples'] for row in history),
        'gpu': torch.cuda.get_device_name(), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'model_init': json.loads((out/'model_init.json').read_text())
    })
    print('COMMON_VALIDATION_FINISHED', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--paper-status', type=Path, required=True)
    p.add_argument('--phase', choices=['C0','M','D','F'], required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--model', choices=['gears','scgpt'], required=True)
    p.add_argument('--mode', choices=['M0','M1','M2','M3','M-random'], default='M0')
    p.add_argument('--data-mode', choices=['D0','D1','D2'], default='D0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
