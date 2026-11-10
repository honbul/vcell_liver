"""F1-diag A2: parameter change and Adam-state evidence from saved scGPT checkpoints.

Gradient norms were NOT logged during training (``clip_grad_norm_`` return value was
discarded). What *is* saved:
  * ``best_model.pt`` (best validation epoch) and ``resume.pt`` (epoch-15 ``current``
    state, Adam ``exp_avg``/``exp_avg_sq``, scheduler, history).
  * The init is not saved, but it is deterministic: training called ``seed_all(seed)``
    and then built the model on CPU. We rebuild with the same seed and verify the
    reconstruction against a tensor that no mode ever trains (``cls_decoder``) and,
    for partial modes, against every frozen tensor.

Per module group we report ||theta_T - theta_0||_F, ||theta_0||_F, the relative change,
and for the fused QKV projection the singular-value energy of the update (how much of
M0's update a rank-8 adapter could in principle express). From the epoch-15 Adam
state we report sqrt(sum v) (EMA of squared *clipped* gradients, beta2=0.999, i.e.
roughly the last ~1000 updates) as a gradient-norm proxy, and the norm of the last
Adam step lr*m_hat/(sqrt(v_hat)+eps). Run one label per process so torch's RNG
stream matches training exactly.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUNS = {
    'M0': 'C0/C0-Norman-v1-scgpt-M0-D0-s26-r0',
    'M1': 'M/M-Norman-v1-scgpt-M1-D0-s26-r0',
    'M2': 'M/M-Norman-v1-scgpt-M2-D0-s26-r0',
    'M3': 'M/M-Norman-v1-scgpt-M3-D0-s26-r0',
    'M-random': 'M/M-Norman-v1-scgpt-M-random-D0-s26-r0',
    'D1': 'D/D-Norman-v1-scgpt-M0-D1-s26-r0',
    'D2': 'D/D-Norman-v1-scgpt-M0-D2-s26-r0',
}
LAYER = re.compile(r'transformer_encoder\.layers\.(\d+)\.(.*)')
LORA = re.compile(r'(transformer_encoder\.layers\.\d+\.self_attn)\.parametrizations\.in_proj_weight\.(original|0\.down|0\.up)')


def group_of(name):
    m = LAYER.match(name)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        if 'in_proj' in rest:
            kind = 'qkv_lora' if ('0.down' in rest or '0.up' in rest) else 'qkv'
        elif 'out_proj' in rest:
            kind = 'attn_out'
        elif rest.startswith('linear'):
            kind = 'ffn'
        elif rest.startswith('norm'):
            kind = 'layernorm'
        else:
            kind = 'other'
        return f'transformer_{kind}', layer
    for prefix, g in (('encoder.embedding', 'gene_embedding'), ('encoder.', 'gene_embedding_norm'),
                      ('value_encoder.', 'value_encoder'), ('pert_encoder.', 'pert_encoder'),
                      ('cls_decoder.', 'cls_decoder_unused'), ('decoder.', 'expr_decoder')):
        if name.startswith(prefix):
            return g, None
    return 'other', None


def effective(state, scale):
    """Replace LoRA parametrization keys by the effective fused in_proj weight."""
    out, lora = {}, {}
    for k, v in state.items():
        m = LORA.match(k)
        if m:
            lora.setdefault(m.group(1), {})[m.group(2)] = v.float()
        else:
            out[k] = v.float()
    for base, parts in lora.items():
        out[f'{base}.in_proj_weight'] = parts['original'] + scale * (parts['0.up'] @ parts['0.down'])
    return out


def svd_energy(delta):
    s = torch.linalg.svdvals(delta.double())
    e = s.square()
    tot = float(e.sum())
    if tot == 0:
        return dict(top8=None, top16=None, stable_rank=None, sigma1=0.0)
    return dict(top8=float(e[:8].sum() / tot), top16=float(e[:16].sum() / tot),
                stable_rank=float(tot / e[0]), sigma1=float(s[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', choices=list(RUNS), required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--public', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    import scgpt_author
    import common_scgpt
    from run_support import seed_all
    from feedback.f1_diag_infer import fake_root

    a.out.mkdir(parents=True, exist_ok=True)
    scgpt_author.ROOT = fake_root(a.prior_root, a.out)
    common_scgpt.PUBLIC = a.public
    run = a.runs / RUNS[a.label]
    result = json.loads((run / 'result.json').read_text())
    init_meta = json.loads((run / 'model_init.json').read_text())
    genes = json.loads((a.dataset / 'genes.json').read_text())['output']
    work = a.out / 'model_build' / f'params-{a.label}'
    work.mkdir(parents=True, exist_ok=True)

    seed_all(result['seed'])  # identical RNG stream to common_train before build()
    model, gene_ids, mounts = common_scgpt.build(genes, result['mode'], work, torch.device('cpu'))
    init_raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scale = next(iter(mounts.values())).scale if mounts else None
    trainable = list(init_meta['trainable_parameters'])
    del model

    best_raw = torch.load(run / 'best_model.pt', map_location='cpu', weights_only=True)
    resume = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
    final_raw = resume['current']
    assert set(best_raw) == set(init_raw) == set(final_raw)

    # Verify the init reconstruction: frozen tensors must be bit-identical to init.
    frozen = [k for k in init_raw if k not in trainable]
    mismatched = [k for k in frozen if not torch.equal(init_raw[k], best_raw[k])]
    check = dict(n_frozen_tensors=len(frozen), frozen_mismatched=mismatched[:10],
                 n_frozen_mismatched=len(mismatched),
                 cls_decoder_identical=all(torch.equal(init_raw[k], best_raw[k])
                                           for k in init_raw if k.startswith('cls_decoder.')))

    used_rows = gene_ids.cpu().numpy()
    rows = []
    qkv_svd = []
    for tag, raw in (('best', best_raw), ('epoch15', final_raw)):
        init_eff, cur_eff = effective(init_raw, scale), effective(raw, scale)
        acc = {}
        for k in cur_eff:
            d = (cur_eff[k] - init_eff[k])
            g, layer = group_of(k)
            if g == 'transformer_qkv' and k.endswith('in_proj_weight'):
                if float(d.norm()) > 0:
                    qkv_svd.append(dict(label=a.label, state=tag, layer=layer, **svd_energy(d),
                                        delta_fro=float(d.norm()), init_fro=float(init_eff[k].norm())))
            for key in ((g, layer), (g, 'all')):
                s = acc.setdefault(key, [0.0, 0.0, 0])
                s[0] += float(d.double().square().sum())
                s[1] += float(init_eff[k].double().square().sum())
                s[2] += d.numel()
            if k == 'encoder.embedding.weight':
                du = d[used_rows]
                s = acc.setdefault(('gene_embedding_used_rows', 'all'), [0.0, 0.0, 0])
                s[0] += float(du.double().square().sum())
                s[1] += float(init_eff[k][used_rows].double().square().sum())
                s[2] += du.numel()
        for (g, layer), (dsq, isq, n) in sorted(acc.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            rows.append(dict(label=a.label, state=tag, group=g, layer=layer,
                             delta_fro=dsq ** 0.5, init_fro=isq ** 0.5,
                             relative=(dsq ** 0.5 / isq ** 0.5) if isq else None,
                             delta_rms_per_param=(dsq / n) ** 0.5 if n else None, n_values=n))

    # Adam state at the end of epoch 15 (post-clipping gradients).
    opt = resume['optimizer']
    ids = [i for grp in opt['param_groups'] for i in grp['params']]
    if len(ids) != len(trainable):
        raise ValueError('Optimizer/param mapping mismatch')
    lr_now = opt['param_groups'][0]['lr']
    betas = opt['param_groups'][0]['betas']
    eps = opt['param_groups'][0]['eps']
    lr_used = lr_now / resume['scheduler'].get('gamma', 0.9)  # scheduler stepped after epoch 15
    gacc = {}
    steps = set()
    for name, i in zip(trainable, ids):
        st = opt['state'][i]
        t = int(st['step'])
        steps.add(t)
        m, v = st['exp_avg'].double(), st['exp_avg_sq'].double()
        m_hat, v_hat = m / (1 - betas[0] ** t), v / (1 - betas[1] ** t)
        step = lr_used * m_hat / (v_hat.sqrt() + eps)
        g, layer = group_of(name)
        for key in (g, 'ALL'):
            s = gacc.setdefault(key, [0.0, 0.0, 0.0, 0])
            s[0] += float(v_hat.sum())
            s[1] += float(m_hat.square().sum())
            s[2] += float(step.square().sum())
            s[3] += m.numel()
    adam_rows = [dict(label=a.label, group=g, grad_rms_norm_proxy=s[0] ** 0.5,
                      momentum_norm=s[1] ** 0.5, last_step_norm=s[2] ** 0.5, n_values=s[3])
                 for g, s in sorted(gacc.items())]
    hist = resume['history']
    out = dict(
        label=a.label, run=RUNS[a.label], mode=result['mode'], data_mode=result['data_mode'],
        best_epoch=result['best_epoch'], n_trainable=init_meta['n_trainable'],
        init_check=check, lora_scale=scale, adam_steps=sorted(steps), lr_epoch15=lr_used,
        lr_after=lr_now, betas=list(betas), eps=eps,
        epoch_seconds=result['epoch_seconds'], peak_allocated_gib=result['peak_allocated_gib'],
        gpu=result['gpu'], updates=result['updates'],
        history=[dict(epoch=h['epoch'], train_loss=h['train_loss'], val=h['val_delta_rmse']) for h in hist],
        groups=rows, qkv_svd=qkv_svd, adam=adam_rows,
        not_logged=['per-step gradient norm (clip_grad_norm_ return value discarded)',
                    'per-step or per-epoch parameter snapshots other than best/epoch-15',
                    'clipping frequency', 'per-epoch train/val time split beyond epoch seconds'])
    (a.out / 'params').mkdir(exist_ok=True)
    (a.out / 'params' / f'{a.label}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('F1_DIAG_PARAMS', a.label, json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
