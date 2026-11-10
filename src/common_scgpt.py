"""Native scGPT expression model and explicit adaptation modes for common data."""
import json
import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from artifact_io import digest, write
from common_data import ROOT, PUBLIC
from scgpt_author import author_sources, GeneVocab

MODES = ('M0', 'M1', 'M2', 'M3', 'M-random')
PREFIXES = ('encoder.', 'value_encoder.', 'transformer_encoder.')


class FusedQKVLoRA(nn.Module):
    """Low rank update to PyTorch MultiheadAttention's fused in_proj_weight."""
    def __init__(self, weight, rank=8, alpha=16):
        super().__init__()
        self.down = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.up = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, original):
        return original + self.scale * (self.up @ self.down)


def configure(model, mode, partial_blocks=2, lora_rank=8, lora_alpha=16, lora_ffn=False):
    """Defaults reproduce the registered M0-M3 modes; keyword hooks serve feedback ablations."""
    if mode not in MODES:
        raise ValueError(mode)
    if mode != 'M2' and partial_blocks != 2:
        raise ValueError('partial_blocks applies only to M2')
    if mode != 'M3' and (lora_rank, lora_alpha, lora_ffn) != (8, 16, False):
        raise ValueError('LoRA options apply only to M3')
    layers = model.transformer_encoder.layers
    if not 1 <= partial_blocks <= len(layers) or lora_rank < 1:
        raise ValueError('Invalid adaptation capacity')
    if mode in ('M0', 'M-random'):
        model.requires_grad_(True)
        # The native expression path never executes this classification head.
        model.cls_decoder.requires_grad_(False)
    else:
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)
        model.pert_encoder.requires_grad_(True)
    mounts = {}
    if mode == 'M2':
        for block in layers[-partial_blocks:]:
            block.requires_grad_(True)
    if mode == 'M3':
        for i, block in enumerate(layers):
            attn = block.self_attn
            if not isinstance(attn, nn.MultiheadAttention) or attn.in_proj_weight is None:
                raise ValueError('Expected actual fused attention projection')
            adapter = FusedQKVLoRA(attn.in_proj_weight, lora_rank, lora_alpha)
            parametrize.register_parametrization(attn, 'in_proj_weight', adapter)
            mounts[f'transformer_encoder.layers.{i}.self_attn.in_proj_weight'] = adapter
            if lora_ffn:
                # Same low-rank update on both feed-forward projections of the block.
                for name in ('linear1', 'linear2'):
                    linear = getattr(block, name)
                    if not isinstance(linear, nn.Linear):
                        raise ValueError('Expected feed-forward linear projections')
                    adapter = FusedQKVLoRA(linear.weight, lora_rank, lora_alpha)
                    parametrize.register_parametrization(linear, 'weight', adapter)
                    mounts[f'transformer_encoder.layers.{i}.{name}.weight'] = adapter
    return mounts


def build(genes, mode, out, device, adapt=None):
    author_sources(out)
    from scgpt.model import TransformerGenerator

    # Use SDPA for the long gene sequence. All genes in C0 have valid tokens,
    # so we pass no padding mask and avoid the separate fused eval-only path.
    torch.backends.mha.set_fastpath_enabled(False)
    public = PUBLIC / 'scgpt'
    receipt = {r['file']: r['sha256'] for r in json.loads(
        (ROOT / 'manifests/public_receipt.json').read_text())}
    files = ['vocab.json', 'args.json']
    if mode != 'M-random':
        files.append('best_model.pt')
    for name in files:
        if digest(public / name) != receipt[f'scgpt/{name}']:
            raise ValueError(f'Unexpected scGPT artifact: {name}')
    vocab = GeneVocab.from_file(public / 'vocab.json')
    for token in ('<pad>', '<cls>', '<eoc>'):
        if token not in vocab:
            vocab.append_token(token)
    if not all(g in vocab and vocab[g] != vocab['<pad>'] for g in genes):
        raise ValueError('Unknown or padding token in shared gene set')
    args = json.loads((public / 'args.json').read_text())
    model = TransformerGenerator(
        len(vocab), args['embsize'], args['nheads'], args['d_hid'], args['nlayers'],
        nlayers_cls=args['n_layers_cls'], n_cls=1, vocab=vocab, dropout=0.0,
        pad_token='<pad>', pad_value=0, pert_pad_id=0, use_fast_transformer=False)
    loaded = {}
    if mode != 'M-random':
        state = torch.load(public / 'best_model.pt', map_location='cpu', weights_only=True)
        loaded = {k.replace('Wqkv.', 'in_proj_'): v for k, v in state.items()
                  if k.startswith(PREFIXES)}
        current = model.state_dict()
        expected = {k for k in current if k.startswith(PREFIXES)}
        if set(loaded) != expected:
            raise ValueError(f'Checkpoint backbone key mismatch: {set(loaded) ^ expected}')
        current.update(loaded)
        model.load_state_dict(current, strict=True)
        assert all(torch.equal(model.state_dict()[k], v) for k, v in loaded.items())
    # Author Fig. 3a run used dropout 0 and loads only these backbone prefixes.
    mounts = configure(model, mode, **(adapt or {}))
    write(out / 'model_init.json', {**({'adaptation': adapt} if adapt else {}),
        'mode': mode, 'checkpoint_sha256': receipt['scgpt/best_model.pt'] if loaded else None,
        'backbone_keys_loaded': sorted(loaded), 'attention_projections': list(mounts),
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'n_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_parameters': {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    })
    return model.to(device), torch.tensor([vocab[g] for g in genes], device=device), mounts


def gene_panel(target_indices, n_genes, cap, rng):
    mandatory = np_unique(target_indices)
    if len(mandatory) > cap:
        raise ValueError('Too many intervention targets for a single training panel')
    if len(mandatory) and (mandatory[0] < 0 or mandatory[-1] >= n_genes):
        raise ValueError('Invalid target position')
    import numpy as np
    rest = np.setdiff1d(np.arange(n_genes), mandatory, assume_unique=True)
    sampled = rng.choice(rest, min(cap-len(mandatory), len(rest)), replace=False)
    return np.sort(np.concatenate([mandatory, sampled]))


def np_unique(target_indices):
    import numpy as np
    return np.unique(np.fromiter((g for targets in target_indices for g in targets), dtype=np.int64))


def predict(model, gene_ids, control, flags):
    src = gene_ids.expand(control.shape[0], -1)
    return model(src, control, flags, src_key_padding_mask=None)['mlm_output']
