"""Extract scGPT or Geneformer CLS embeddings from offline basal RNA parquet."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from .artifacts import file_record, new_directory, write_json
from . import embed  # activates the offline Hugging Face environment

SCGPT_PREFIXES = ("encoder.", "value_encoder.", "transformer_encoder.", "bn.", "dsbn.")


class GeneEncoder(nn.Module):
    def __init__(self, size, width, pad):
        super().__init__()
        self.embedding = nn.Embedding(size, width, padding_idx=pad)
        self.enc_norm = nn.LayerNorm(width)

    def forward(self, tokens):
        return self.enc_norm(self.embedding(tokens))


class ContinuousValues(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear1 = nn.Linear(1, width)
        self.linear2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, values):
        return self.norm(self.linear2(self.linear1(values.clamp(max=512).unsqueeze(-1)).relu()))


class ScGPTEncoder(nn.Module):
    """Encoder-only adapter with scGPT's published continuous-value architecture."""
    def __init__(self, dimensions, vocabulary, embedding_rows):
        super().__init__()
        width = dimensions["embsize"]
        self.pad = vocabulary[dimensions["pad_token"]]
        self.encoder = GeneEncoder(embedding_rows, width, self.pad)
        self.value_encoder = ContinuousValues(width)
        layer = nn.TransformerEncoderLayer(
            width, dimensions["nheads"], dimensions["d_hid"], dropout=0, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer, dimensions["nlayers"], enable_nested_tensor=False,
        )

    def forward(self, tokens, values):
        hidden = self.encoder(tokens) + self.value_encoder(values)
        return self.transformer_encoder(hidden, src_key_padding_mask=tokens == self.pad)[:, 0]


def scgpt_checkpoint(checkpoint):
    import re

    configuration = json.loads((checkpoint / "args.json").read_text())
    vocabulary = json.loads((checkpoint / "vocab.json").read_text())
    raw = torch.load(checkpoint / "best_model.pt", map_location="cpu", weights_only=True)
    for special in ("<pad>", "<cls>", "<eoc>"):
        if special not in vocabulary:
            vocabulary[special] = max(vocabulary.values()) + 1
    embedding_rows = raw["encoder.embedding.weight"].shape[0]
    if max(vocabulary.values()) >= embedding_rows:
        raise ValueError("Vocabulary IDs are outside the checkpoint embedding table")
    encoder = ScGPTEncoder(configuration, vocabulary, embedding_rows)
    state = {
        re.sub(r"self_attn\.(?:_impl\.)?Wqkv\.", "self_attn.in_proj_", name): tensor
        for name, tensor in raw.items() if name.startswith(SCGPT_PREFIXES)
    }
    encoder.load_state_dict(state, strict=True)
    return encoder, vocabulary, configuration


def scgpt_batch(rows, genes, ids, vocabulary, options):
    tokens = np.full((len(rows), options["max_seq_len"]), vocabulary["<pad>"], dtype=np.int64)
    bins = np.full(tokens.shape, options["pad_value"], dtype=np.float32)
    tokens[:, 0], bins[:, 0] = vocabulary["<cls>"], 0
    valid, truncated = [], 0
    encoded = np.asarray([vocabulary.get(gene, -1) for gene in genes], dtype=np.int64)
    for i, (row, key) in enumerate(zip(rows, ids)):
        used = np.flatnonzero((row > 0) & (encoded >= 0))
        if used.size == 0:
            continue
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        random = np.random.default_rng(seed)
        selected = row[used]
        thresholds = np.quantile(selected, np.linspace(0, 1, options["n_bins"] - 1))
        low = np.digitize(selected, thresholds, right=True)
        high = np.digitize(selected, thresholds, right=False)
        values = np.maximum(1, np.ceil(high + random.random(len(used)) * (low - high)))
        limit = options["max_seq_len"] - 1
        if len(used) > limit:
            subsample = np.sort(random.choice(len(used), limit, replace=False))
            used, values = used[subsample], values[subsample]
            truncated += 1
        tokens[i, 1:1 + len(used)] = encoded[used]
        bins[i, 1:1 + len(used)] = values
        valid.append(i)
    return tokens[valid], bins[valid], valid, truncated


def load_geneformer_assets(assets):
    names = ["token_dictionary", "gene_median_dictionary", "gene_name_id_dict",
             "ensembl_mapping_dict"]
    loaded = {}
    for key in names:
        with (assets / f"{key}_gc104M.pkl").open("rb") as stream:
            loaded[key] = pickle.load(stream)
    return loaded


def geneformer_mapping(genes, asset):
    valid_cols, tokens, medians = [], [], []
    vocabulary = asset["token_dictionary"]
    reference = asset["gene_median_dictionary"]
    names, aliases = asset["gene_name_id_dict"], asset["ensembl_mapping_dict"]
    for i, gene in enumerate(genes):
        ensembl = gene.split(".")[0] if gene.startswith("ENSG") else names.get(gene)
        ensembl = aliases.get(ensembl, ensembl)
        if ensembl in vocabulary and ensembl in reference and reference[ensembl] > 0:
            valid_cols.append(i)
            tokens.append(vocabulary[ensembl])
            medians.append(reference[ensembl])
    unique, indices = np.unique(tokens, return_inverse=True)
    median = np.zeros(len(unique), dtype=np.float64)
    median[indices] = medians
    if not len(unique):
        raise ValueError("No valid genes for Geneformer")
    return np.asarray(valid_cols), unique, indices, median


def geneformer_batch(rows, mapping, vocabulary, length, scale):
    cols, unique, indices, medians = mapping
    arrays, valid, truncated = [], [], 0
    for i, row in enumerate(rows):
        abundance = row[cols]
        if scale == "log2":
            abundance = np.expm1(abundance * np.log(2))
        summed = np.bincount(indices, weights=abundance, minlength=len(unique)) / medians
        nonzero = np.flatnonzero(summed > 0)
        if not nonzero.size:
            continue
        # Dividing by total library size multiplies all nonzero values by
        # the same scalar, leaving the median-adjusted rank unchanged.
        order = nonzero[np.argsort(-summed[nonzero], kind="stable")]
        truncated += len(order) > length - 2
        arrays.append([vocabulary["<cls>"], *unique[order[:length - 2]], vocabulary["<eos>"]])
        valid.append(i)
    if not arrays:
        return np.empty((0, 0), dtype=np.int64), [], 0
    values = np.full((len(arrays), max(map(len, arrays))), vocabulary["<pad>"], dtype=np.int64)
    for i, row in enumerate(arrays):
        values[i, :len(row)] = row
    return values, valid, truncated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, nargs="+")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", required=True, choices=("scgpt", "geneformer"))
    p.add_argument("--scale", required=True, choices=("log2", "linear"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1 or (args.model == "geneformer" and args.assets is None):
        p.error("Provide a positive batch size and local Geneformer assets when applicable")
    tables = [pd.read_parquet(path) for path in args.input]
    genes = tables[0].columns.astype(str).to_numpy()
    if any(not table.columns.equals(tables[0].columns) for table in tables):
        raise ValueError("Transcriptome queries have incompatible gene axes")
    data = pd.concat(tables)
    if not data.index.is_unique or not data.columns.is_unique:
        raise ValueError("Duplicate RNA profile IDs or feature names")
    keys = data.index.to_numpy(dtype=str)
    raw = data.to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("RNA input requires finite nonnegative abundance")
    gpu = args.device == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    if args.model == "scgpt":
        model, vocab, options = scgpt_checkpoint(args.checkpoint)
        mapped = len(set(genes) & set(vocab))
        torch.backends.mha.set_fastpath_enabled(False)
    else:
        from transformers import BertForMaskedLM

        asset = load_geneformer_assets(args.assets)
        vocab = asset["token_dictionary"]
        mapping = geneformer_mapping(genes, asset)
        mapped = len(mapping[0])
        model = BertForMaskedLM.from_pretrained(args.checkpoint, local_files_only=True).bert
    model = model.eval().requires_grad_(False).to(args.device)
    vectors, used = [], []
    truncated = 0
    with torch.inference_mode():
        for start_row in range(0, len(raw), args.batch_size):
            rows = raw[start_row:start_row + args.batch_size]
            if args.model == "scgpt":
                tokens, values, valid, cut = scgpt_batch(
                    rows, genes, keys[start_row:start_row + len(rows)], vocab, options,
                )
                if valid:
                    result = model(torch.as_tensor(tokens, device=args.device),
                                   torch.as_tensor(values, device=args.device))
                    result = nn.functional.normalize(result, dim=1)
            else:
                tokens, valid, cut = geneformer_batch(rows, mapping, vocab, 4096, args.scale)
                if valid:
                    x = torch.as_tensor(tokens, device=args.device)
                    result = model(input_ids=x, attention_mask=x.ne(vocab["<pad>"]).long(),
                                   output_hidden_states=True).hidden_states[-2][:, 0, :]
            truncated += cut
            if valid:
                used.extend(start_row + i for i in valid)
                vectors.append(result.float().cpu().numpy())
    if not vectors:
        raise ValueError("No valid profiles for this encoder")
    output = np.concatenate(vectors)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite cell embedding")
    if gpu:
        torch.cuda.synchronize()
    with new_directory(args.output) as out:
        np.savez(out / "vectors.npz", id=keys[used], vectors=output)
        write_json(out / "receipt.json", {
            "model": args.model, "input_scale": args.scale,
            "input": [file_record(path) for path in args.input],
            "genes": len(genes), "matched_genes": mapped,
            "rows": len(data), "embedded": len(used), "truncated": truncated,
            "dropped_ids": sorted(set(keys) - set(keys[used])),
            "dimension": output.shape[1],
            "checkpoint": {p.name: file_record(p) for p in sorted(args.checkpoint.iterdir()) if p.is_file()},
            "tokenization_assets": {p.name: file_record(p) for p in sorted(args.assets.iterdir()) if p.is_file()}
            if args.assets else {},
            "device": torch.cuda.get_device_name() if gpu else "cpu",
            "wall_seconds": perf_counter() - start,
            "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else 0,
            "output": file_record(out / "vectors.npz"),
        })


if __name__ == "__main__":
    main()
