"""Validation-only F1 ablation training (A6-A9) of scGPT on the common Norman split.

The epoch loop mirrors src/common_train.py exactly (same protocol file, batches, gene panel
RNG, Adam/StepLR, gradient-norm clipping, per-epoch validation, best-epoch replay). Only the
registered options of experiments/feedback/f1_ablation_runs.json change: learning rate,
adaptation capacity (common_scgpt.configure keyword hooks), control input (Dmean) and
exposure schedule (Dcap). Test/holdout outcomes are never loaded.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fcntl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common_scgpt  # noqa: E402
import scgpt_author  # noqa: E402
from common_eval import evaluate  # noqa: E402
from common_runtime import flags_for, predict_conditions  # noqa: E402
from common_train import batches  # noqa: E402
from f1_ablation_data import AblationData, schedule_stats  # noqa: E402
from run_support import checkpoint, digest, restore_rng, rng_state, seed_all, write  # noqa: E402

ROOT = SRC.parent
NEURAL = ROOT / 'protocols/common/neural-v1.json'
REGISTRY = ROOT / 'experiments/feedback/f1_ablation_runs.json'
PRIOR = (ROOT / 'prior_assets').resolve()
# prior_assets -> <prior worktree>/assets (itself a link to pubdata); author sources live in that worktree.
PRIOR_ROOT = Path(os.readlink(ROOT / 'prior_assets')).parent
CODE = [SRC / n for n in ('common_data.py', 'common_train_data.py', 'common_eval.py',
                          'common_scgpt.py', 'scgpt_author.py', 'artifact_io.py', 'common_train.py',
                          'common_runtime.py', 'run_support.py')] + [
    Path(__file__), Path(__file__).with_name('f1_ablation_data.py')]


def registered(run_id):
    reg = json.loads(REGISTRY.read_text())
    match = [r for r in reg['runs'] if r['run_id'] == run_id]
    if len(match) != 1:
        raise ValueError(f'Run not registered: {run_id}')
    return reg, match[0]


def use_prior_assets():
    """Point the hash-checked public scGPT assets and author sources at the prior study."""
    common_scgpt.PUBLIC = PRIOR / 'public'
    scgpt_author.ROOT = PRIOR_ROOT


def claim(out, settings, resume):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'result.json').exists():
        raise RuntimeError(f'Immutable finished run: {out}')
    record = out / 'settings.json'
    if record.exists():
        if not resume or json.loads(record.read_text())['contract'] != settings:
            raise RuntimeError('Existing run requires explicit resume and identical settings')
    else:
        try:
            commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                             text=True).strip()
        except Exception:  # pragma: no cover - provenance only
            commit = None
        write(record, dict(contract=settings, worktree_head=commit,
                           slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                           hostname=os.uname().nodename))
    return lock


def frozen_backbone_check(model, init):
    """For PEFT modes: loaded backbone tensors that are not trainable must equal the public file."""
    public = torch.load(common_scgpt.PUBLIC / 'scgpt' / 'best_model.pt', map_location='cpu', weights_only=True)
    public = {k.replace('Wqkv.', 'in_proj_'): v for k, v in public.items()}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    trainable = set(init['trainable_parameters'])
    checked, changed = 0, []
    for name in init['backbone_keys_loaded']:
        if name in trainable:
            continue
        mounted = None
        for suffix in ('in_proj_weight', 'weight'):
            if name.endswith(suffix):
                cand = name[:-len(suffix)] + 'parametrizations.' + suffix + '.original'
                if cand in state:
                    mounted = cand
        key = mounted or name
        checked += 1
        if not torch.equal(state[key].to(public[name].dtype), public[name]):
            changed.append(name)
    return dict(frozen_backbone_tensors_checked=checked, frozen_backbone_tensors_changed=changed)


def run(a):
    config = json.loads(NEURAL.read_text())
    reg, spec = registered(a.run_id)
    if spec['data_mode'] not in ('D0', 'Dmean', 'Dcap'):
        raise ValueError('Unexpected data mode')
    mode, lr, adapt, data_mode = spec['mode'], spec['lr'], spec['adapt'], spec['data_mode']
    seed = config['train_seed']
    epochs = config['epochs'] if a.smoke_updates is None else a.smoke_epochs
    dataset = a.dataset.resolve()
    dataset_sha = digest(dataset / 'receipt.json')
    if dataset_sha != reg['dataset_receipt_sha256']:
        raise ValueError('Dataset does not match the registered split')
    if json.loads((dataset / 'receipt.json').read_text())['split_seed'] != 26:
        raise ValueError('Ablations use the initial split only')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Request a bf16 GPU using the Slurm wrapper')
    use_prior_assets()
    device = torch.device('cuda:0')
    seed_all(seed)
    data = AblationData(dataset)
    started = time.monotonic()
    settings = {
        'registry_sha256': digest(REGISTRY), 'protocol_sha256': digest(NEURAL),
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in CODE},
        'dataset_sha256': dataset_sha, 'public_receipt_sha256': digest(PRIOR / 'public/receipt.json'),
        'spec': spec, 'seed': seed, 'epochs': epochs, 'smoke_updates': a.smoke_updates}
    if a.smoke_updates is None:
        out = ROOT / 'assets/runs/A' / a.run_id
    else:
        out = ROOT / 'assets/runs/A-smoke' / a.run_id
        if out.exists():
            shutil.rmtree(out)
    lock = claim(out, settings, a.resume)  # noqa: F841 - held for the process lifetime
    model, gene_ids, mounts = common_scgpt.build(data.genes, mode, out, device, adapt=adapt or None)

    def predict(control, flags, panel=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return common_scgpt.predict(model, gene_ids if panel is None else gene_ids[panel],
                                        control, flags)

    mode_cfg = config['scgpt']
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)
    state_path = out / 'resume.pt'
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
    smoke = {}
    mean_input = None
    if data_mode == 'Dmean':
        # Diagnostic only: the same model fed the mean of the 32 evaluation controls.
        mean_input = data.eval_control.mean(axis=0, dtype=np.float64)[None].astype(np.float32)
    for epoch in range(start_epoch, epochs + 1):
        if epoch - best_epoch > config['patience'] + 1 and best_epoch:
            break
        epoch_start = time.monotonic()
        lr_epoch = scheduler.get_last_lr()[0]
        rows, pairs = data.schedule(data_mode, seed, epoch)
        panel_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2731]))
        model.train()
        n_steps, n_flags, batch_losses = 0, 0, []
        for ix in batches(rows, config['training_batch_size']):
            if a.smoke_updates is not None and n_steps >= a.smoke_updates:
                break
            targets = [data.train_targets[c] for c in data.train_names[ix]]
            panel = common_scgpt.gene_panel(targets, len(data.genes), mode_cfg['max_train_genes'],
                                            panel_rng)
            x, y, _ = data.batch(ix, pairs, panel, mode=data_mode)
            if a.smoke_updates is not None and n_steps == 0 and epoch == start_epoch:
                smoke['first_batch'] = dict(
                    rows=len(ix), panel_genes=len(panel),
                    control_rows_identical=bool(np.all(x == x[:1])),
                    control_equals_pool_mean=bool(np.allclose(x, data.control_mean[panel][None])),
                    control_equals_paired_cells=bool(np.array_equal(
                        x, data.control[pairs[ix]][:, panel].toarray())),
                    control_row_std_mean=float(x.std(axis=0).mean()))
            control = torch.as_tensor(x, device=device, dtype=torch.float32)
            truth = torch.as_tensor(y, device=device, dtype=torch.float32)
            flags = flags_for(targets, len(data.genes), device, panel)
            optim.zero_grad(set_to_none=True)
            loss = (predict(control, flags, panel).float() - truth).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch} update {n_steps}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optim.step()
            n_steps += 1
            n_flags += sum(map(len, targets))
            batch_losses.append(float(loss.detach()))
        model.eval()
        val_preds = predict_conditions(predict, data.eval_control, data.val, data.indices,
                                       device, config['validation_control_chunk'])
        scored = evaluate(dataset, 'val', val_preds)
        val = scored['metrics']['delta_rmse']
        diag = None
        if mean_input is not None:
            diag = evaluate(dataset, 'val', predict_conditions(
                predict, mean_input, data.val, data.indices, device, 1))['metrics']['delta_rmse']
        if val < best_val:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val, best_epoch = val, epoch
            checkpoint(out / 'best_model.pt', best)
            write(out / 'best_validation.json', scored)
        scheduler.step()
        exposures = np.unique(data.train_names[rows], return_counts=True)[1]
        repeats = np.unique(rows, return_counts=True)[1]
        stats = dict(epoch=epoch, val_delta_rmse=val, train_loss=float(np.mean(batch_losses)),
                     lr=lr_epoch, examples=len(rows), updates=n_steps, target_flags=n_flags,
                     changed_pairs=float(np.mean(pairs != data.paired)),
                     distinct_examples=len(repeats), max_cell_repeats=int(repeats.max()),
                     intervention_min=int(exposures.min()), intervention_max=int(exposures.max()),
                     val_delta_pearson=scored['metrics']['delta_pearson'],
                     val_top20_recall=scored['metrics']['absolute_delta_top20_recall'],
                     seconds=time.monotonic() - epoch_start)
        if diag is not None:
            stats['val_delta_rmse_mean_control_input'] = diag
        history.append(stats)
        checkpoint(state_path, dict(
            epoch=epoch, current=model.state_dict(), best=best, best_epoch=best_epoch,
            best_val=best_val, optimizer=optim.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng_state(), history=history))
        write(out / 'history.json', history)
        print('F1_ABLATION_EPOCH', json.dumps(stats), flush=True)
    if best is None:
        raise RuntimeError('No validation result')
    init = json.loads((out / 'model_init.json').read_text())
    if a.smoke_updates is not None:
        ups = {n: float(m.up.detach().norm()) for n, m in mounts.items()}
        smoke.update(
            n_trainable=init['n_trainable'], n_parameters=init['n_parameters'],
            adapter_mounts=list(mounts),
            adapter_shapes={n: [list(m.down.shape), list(m.up.shape), m.scale] for n, m in mounts.items()},
            adapter_up_norm_min=min(ups.values()) if ups else None,
            trainable_blocks=sorted({int(n.split('.')[2]) for n in init['trainable_parameters']
                                     if n.startswith('transformer_encoder.layers.')}),
            optimizer_lr_initial=lr, lr_per_epoch=[h['lr'] for h in history],
            updates_run=[h['updates'] for h in history])
        if mode != 'M0':
            smoke.update(frozen_backbone_check(model, init))
        smoke['schedule'] = [schedule_stats(data, data_mode, seed, e) for e in (1, 2, 15)]
        write(out / 'smoke_evidence.json', smoke)
        print('F1_ABLATION_SMOKE', json.dumps(smoke), flush=True)
    checkpoint(out / 'best_model.pt', best)
    model.load_state_dict(best, strict=True)
    model.eval()
    val_preds = predict_conditions(predict, data.eval_control, data.val, data.indices,
                                   device, config['validation_control_chunk'])
    scored = evaluate(dataset, 'val', val_preds)
    if not np.isclose(scored['metrics']['delta_rmse'], best_val, atol=1e-6, rtol=1e-5):
        raise RuntimeError('Validation score did not replay after reloading')
    write(out / 'best_validation.json', scored)
    write(out / 'result.json', {
        'status': 'validation_complete' if a.smoke_updates is None else 'smoke_only',
        'run_id': a.run_id, 'experiment': spec['experiment'], 'change': spec['change'],
        'dataset_sha256': dataset_sha, 'model': 'scgpt', 'mode': mode, 'adapt': adapt,
        'data_mode': data_mode, 'seed': seed, 'lr': lr, 'best_epoch': best_epoch,
        'validation': scored['metrics'], 'best_model_sha256': digest(out / 'best_model.pt'),
        'history': history, 'epoch_seconds': sum(row['seconds'] for row in history),
        'total_seconds_this_invocation': time.monotonic() - started,
        'updates': sum(row['updates'] for row in history),
        'examples': sum(row['examples'] for row in history),
        'gpu': torch.cuda.get_device_name(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'model_init': init})
    print('F1_ABLATION_FINISHED', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'prior_assets/common/C0-Norman-v1-s26')
    p.add_argument('--run-id', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-updates', type=int, help='Smoke test only: updates per epoch')
    p.add_argument('--smoke-epochs', type=int, default=2)
    run(p.parse_args())


if __name__ == '__main__':
    main()
