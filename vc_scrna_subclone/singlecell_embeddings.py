"""Stream registered single cells through the Track 6 frozen scGPT encoder."""
import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn.functional import normalize

from .cohorts import file_record
from .singlecell_inputs import COHORTS, files_receipt
from .source_panels import frozen


def embed_shard(data, genes, keys, model, vocab, options, device, batch_size):
    from vc_fm_bench.transcriptome import scgpt_batch

    outputs = []
    pos = []
    truncations = 0
    with torch.inference_mode():
        for offset in range(0, len(keys), batch_size):
            rows = data[offset:offset + batch_size]
            if sparse.issparse(rows):
                rows = rows.toarray()
            rows = rows.astype(np.float32, copy=False)
            if not np.isfinite(rows).all() or (rows < 0).any():
                raise ValueError("Invalid RNA input scale")
            tokens, bins, valid, truncation = scgpt_batch(
                rows, genes, keys[offset:offset + len(rows)], vocab, options,
            )
            truncations += truncation
            if valid:
                x = torch.as_tensor(tokens, device=device)
                y = torch.as_tensor(bins, device=device)
                outputs.append(normalize(model(x, y), dim=1).cpu().numpy().astype(np.float32))
                pos.extend(offset + i for i in valid)
    if not outputs:
        raise ValueError("No cells overlap the verified scGPT vocabulary")
    embedding = np.concatenate(outputs)
    if not np.isfinite(embedding).all() or len(embedding) != len(pos):
        raise ValueError("Invalid vector or scGPT result cardinality")
    return embedding, keys[pos], truncations


def prepare(a):
    from vc_fm_bench.transcriptome import scgpt_checkpoint

    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    in_receipt = a.singlecell / "receipt.json"
    source_receipt = a.sources / "receipt.json"
    inputs = json.loads(in_receipt.read_text())
    source = json.loads(source_receipt.read_text())
    model, tokens, options = scgpt_checkpoint(a.checkpoint)
    model = model.eval().requires_grad_(False).to(a.device)
    torch.backends.mha.set_fastpath_enabled(False)
    common_genes = pd.read_csv(frozen(a.singlecell, "pseudo/genes.tsv", inputs), sep="\t").symbol.to_numpy()
    scopes = ("pseudo", "tahoe", *COHORTS)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as directory:
        stage = Path(directory)
        summaries = {}

        def write(name, data, axis, keys):
            started = perf_counter()
            vector, valid, truncated = embed_shard(data, axis, keys, model, tokens, options, a.device, a.batch_size)
            np.savez(stage / f"{name}.npz", id=valid.astype(str), vectors=vector)
            summaries[name] = {
                "input_rows": len(keys), "embedded": len(vector), "dimension": vector.shape[1],
                "truncated_rows": truncated, "seconds": perf_counter() - started,
                "missing_ids": sorted(set(keys) - set(valid)),
            }
            print(f"{name}: {len(vector)}/{len(keys)} scGPT embeddings", flush=True)

        for name in scopes:
            axis = pd.read_csv(frozen(a.singlecell, f"{name}/genes.tsv", inputs), sep="\t").symbol.to_numpy()
            if not np.array_equal(axis, common_genes):
                raise ValueError(f"Inconsistent shared gene axis for {name}")
            counts = sparse.load_npz(frozen(a.singlecell, f"{name}/counts.npz", inputs))
            meta = pd.read_parquet(frozen(a.singlecell, f"{name}/rows.parquet", inputs))
            write(name, counts, axis, meta.id.to_numpy(dtype=str))
        for name, filename in (("source_bulk", "cell_lines_rna.parquet"), ("organoids", "organoids/rna.parquet")):
            frame = pd.read_parquet(frozen(a.sources, filename, source))
            header = frame.columns.intersection(common_genes, sort=False)
            write(name, frame.loc[:, header].to_numpy(np.float32), header.to_numpy(dtype=str),
                  frame.index.to_numpy(dtype=str))
        (stage / "receipt.json").write_text(json.dumps({
            "queries": file_record(in_receipt), "source_rna": file_record(source_receipt),
            "checkpoint": {path.name: file_record(path) for path in a.checkpoint.iterdir() if path.is_file()},
            "device": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if a.device == "cuda" else 0,
            "main_maximum_tokens": options["max_seq_len"], "scopes": summaries, "files": files_receipt(stage),
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("singlecell", "sources", "checkpoint", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=12)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    prepare(a)


if __name__ == "__main__":
    main()
