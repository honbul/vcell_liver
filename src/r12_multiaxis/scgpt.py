"""Frozen whole-human scGPT on the pre-registered DMSO sample."""

import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
from scipy import sparse
import torch

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import KEY, table


def official_model(repo, checkpoint):
    # Run the original encoder without importing the unrelated tokenizer/training stack.
    for name, path in (("scgpt", repo / "scgpt"), ("scgpt.model", repo / "scgpt/model")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    transformer = importlib.import_module("scgpt.model.model").TransformerModel
    c = json.loads((checkpoint / "args.json").read_text())
    vocab = json.loads((checkpoint / "vocab.json").read_text())
    model = transformer(ntoken=len(vocab), d_model=c["embsize"], nhead=c["nheads"], d_hid=c["d_hid"],
                        nlayers=c["nlayers"], nlayers_cls=c["n_layers_cls"], n_cls=1, vocab=vocab,
                        pad_token=c["pad_token"], pad_value=c["pad_value"], dropout=c["dropout"],
                        do_mvc=True, use_fast_transformer=False)
    state = torch.load(checkpoint / "best_model.pt", map_location="cpu", weights_only=True)
    state = {re.sub(r"self_attn\.(?:_impl\.)?Wqkv\.", "self_attn.in_proj_", name): value
             for name, value in state.items()}
    prefixes = ("encoder.", "value_encoder.", "transformer_encoder.", "bn.", "dsbn.")
    required = {k for k in model.state_dict() if k.startswith(prefixes)}
    absent = required.difference(state)
    extra = {k for k in state if k.startswith(prefixes) and k not in required}
    if absent or extra:
        raise ValueError(f"Encoder weight coverage missing={sorted(absent)} extra={sorted(extra)}")
    model.load_state_dict({k: state[k] for k in required}, strict=False)
    return model.cuda().eval().requires_grad_(False), vocab, c, len(required)


def sequence(row, vocab_ids, opts, rng):
    gene_ids = vocab_ids[row.indices]
    use = gene_ids >= 0
    genes = gene_ids[use]
    values = row.data[use]
    if not len(values):
        return None, None, None
    bins = np.quantile(values, np.linspace(0.0, 1.0, opts["n_bins"] - 1))
    left = np.digitize(values, bins)
    right = np.digitize(values, bins, right=True)
    bins = np.ceil(left + rng.random(len(values)) * (right - left))
    maximum = opts["max_seq_len"] - 1
    truncate = len(values) > maximum
    if truncate:
        choice = rng.choice(len(genes), maximum, replace=False)
        genes, bins = genes[choice], bins[choice]
    return genes, bins, truncate


def read_sample(cfg, assets, role):
    root = assets / "cells-v2"
    path = root / f"{role}_cells.npz"
    rows = table(root / f"{role}_rows.tsv")
    if not rows.role.eq(role).all() or sha256(path) != read_json(root / "receipt.json")["files"][path.name]:
        raise ValueError("Unverified role-specific samples")
    genes = table(Path(cfg["r11_assets"]) / "metadata/genes.tsv")
    return rows, sparse.load_npz(path), genes


def output_units(out, role, rows, counts, embedded, model_tokens):
    rows = rows.reset_index(drop=True)
    groups = rows.groupby(KEY, sort=True).indices
    library = np.asarray(counts.sum(axis=1)).ravel()
    x = counts[:, model_tokens].tocsr()
    raw = np.stack([np.log1p(np.asarray(x[p].sum(axis=0)).ravel() * (1e4 / library[p].sum()))
                    for p in groups.values()]).astype(np.float32)
    vec = np.stack([embedded[p].mean(axis=0) for p in groups.values()])
    views = np.stack([[embedded[p[rows.view.iloc[p].to_numpy() == j]].mean(axis=0) for p in groups.values()]
                      for j in (0, 1)])
    if not np.isfinite(vec).all():
        raise ValueError("Nonfinite model embeddings")
    well, line = map(np.asarray, zip(*groups))
    np.savez(out / f"{role}.npz", well=well, line=line, raw=raw, embedding=vec, token=model_tokens, views=views)


def embed(cfg, assets, out, repo, sample_only=False, batch_size=32):
    start = time.monotonic()
    checkpoint = Path(cfg["scgpt_checkpoint"])
    model, vocab, opts, loaded = official_model(repo, checkpoint)
    torch.backends.mha.set_fastpath_enabled(False)
    model.transformer_encoder.enable_nested_tensor = False
    pad, cls = vocab[opts["pad_token"]], vocab["<cls>"]
    n_all = 0
    report = {}
    out.mkdir(parents=True, exist_ok=False)
    for role in ("A", "B"):
        rows, cells, metadata = read_sample(cfg, assets, role)
        symbols = metadata.gene_symbol.map(vocab)
        valid = symbols.notna().to_numpy()
        mapping = np.full(cells.shape[1], -1)
        mapping[metadata.loc[valid, "token_id"]] = symbols[valid].astype(int)
        model_tokens = np.flatnonzero(mapping >= 0)
        n = min(len(rows), 64) if sample_only else len(rows)
        rng = np.random.default_rng(cfg["optional"]["cell_seed"])
        output = []
        dropped, truncated = 0, 0
        ids = []
        for first in range(0, n, batch_size):
            seqs = []
            selections = range(first, min(first + batch_size, n))
            for i in selections:
                genes, values, cut = sequence(cells[i], mapping, opts, rng)
                if genes is None:
                    dropped += 1
                    continue
                ids.append(i)
                seqs.append((genes, values))
                truncated += int(cut)
            if seqs:
                length = max(len(g) + 1 for g, _ in seqs)
                tokens = torch.full((len(seqs), length), pad, dtype=torch.long)
                features = torch.full((len(seqs), length), opts["pad_value"], dtype=torch.float32)
                tokens[:, 0] = cls
                for j, (genes, values) in enumerate(seqs):
                    tokens[j, 1:len(genes) + 1] = torch.as_tensor(genes)
                    features[j, 1:len(genes) + 1] = torch.as_tensor(values)
                tokens = tokens.cuda()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    encoded = model._encode(tokens, features.cuda(), tokens.eq(pad))[:, 0, :]
                    encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
                output.append(encoded.cpu().numpy())
            if first % (64 * batch_size) == 0:
                print(f"scGPT role={role} cells={first}/{n} elapsed={time.monotonic()-start:.1f}s", flush=True)
        if not sample_only:
            output_units(out, role, rows.iloc[ids], cells[ids], np.concatenate(output), model_tokens)
        n_all += n
        report[role] = dict(cells=n, unmapped_cells=dropped, truncated=truncated, matched_genes=len(model_tokens))
    write_json(out / "receipt.json", dict(
        scgpt_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        checkpoint={name: sha256(checkpoint / name) for name in ("args.json", "vocab.json", "best_model.pt")},
        encoder_tensors=loaded, elapsed_seconds=time.monotonic() - start, n_cells=n_all,
        torch=torch.__version__, device=torch.cuda.get_device_name(), max_tokens=opts["max_seq_len"],
        binning="scGPT 51 bins, randomized ranks for tied quantiles, fixed per-role seed",
        batch_size=batch_size, sample_only=sample_only, roles=report,
        output_sha256={p.name: sha256(p) for p in out.glob("*.npz")},
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    embed(config(args.config), args.assets, args.out, args.repo, args.sample_only, args.batch_size)


if __name__ == "__main__":
    main()
