"""Run the official four-layer UCE checkpoint on the same frozen DMSO cells."""

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.r11_chemical_probe.common import config, sha256, write_json

from .scgpt import output_units, read_sample


SPECIES = "human"
CHROM_OFFSET = 143574
PROTEIN_FILE = "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"


def verify_assets(root):
    import hashlib

    release = json.loads((root / "uce-figshare.json").read_text())
    by_name = {file["name"]: file for file in release["files"]}
    output = {}
    for name in ("species_offsets.pkl", "species_chrom.csv", "4layer_model.torch",
                 "all_tokens.torch", "protein_embeddings.tar.gz"):
        path = root / name
        entry = by_name[name]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Unexpected UCE asset length for {name}")
        md5 = hashlib.md5()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(8 << 20), b""):
                md5.update(chunk)
        if md5.hexdigest() != entry["computed_md5"]:
            raise ValueError(f"UCE release checksum mismatch for {name}")
        output[name] = dict(figshare_id=entry["id"], bytes=entry["size"], md5=md5.hexdigest(), sha256=sha256(path))
    output[PROTEIN_FILE] = dict(sha256=sha256(root / PROTEIN_FILE))
    return output


def official_model(repo, root):
    spec = importlib.util.spec_from_file_location("r12_official_uce_model", repo / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TransformerModel(token_dim=5120, d_model=1280, nhead=20,
                                    d_hid=5120, nlayers=4, output_dim=1280, dropout=0.05)
    state = torch.load(root / "4layer_model.torch", map_location="cpu", weights_only=True, mmap=True)
    official_token_weights = torch.load(root / "all_tokens.torch", map_location="cpu", weights_only=True, mmap=True)
    # Official evaluate.py retains the trained embeddings for the original species.
    if official_token_weights.shape[0] not in (CHROM_OFFSET, 145469):
        raise ValueError("Unexpected official UCE embedding index")
    model.pe_embedding = nn.Embedding.from_pretrained(state["pe_embedding.weight"])
    model.load_state_dict(state, strict=True, assign=True)
    return model.cuda().eval().requires_grad_(False)


def gene_mapping(metadata, root):
    with (root / "species_offsets.pkl").open("rb") as file:
        offsets = pickle.load(file)
    pe = torch.load(root / PROTEIN_FILE, map_location="cpu", weights_only=True)
    token = pd.Series(np.arange(len(pe)) + offsets[SPECIES], index=pd.Index(pe, name="symbol"))
    all_chrom = pd.read_csv(root / "species_chrom.csv")
    all_chrom["chrom_code"] = pd.Categorical(all_chrom.species + "_" + all_chrom.chromosome).codes
    human = all_chrom.loc[all_chrom.species.eq(SPECIES), ["gene_symbol", "chrom_code", "start"]]
    human = human.drop_duplicates().loc[lambda df: ~df.gene_symbol.duplicated(keep=False)]
    table = metadata.assign(gene_symbol=metadata.gene_symbol.str.upper())
    joined = table.merge(human, on="gene_symbol", validate="many_to_one")
    joined["model_token"] = joined.gene_symbol.map(token)
    joined = joined.dropna(subset=["model_token"]).sort_values("token_id")
    return tuple(joined[c].to_numpy(dtype=np.int64) for c in ("token_id", "model_token", "chrom_code", "start"))


def sentence(row, mp, rng):
    ids, chromosome, position = mp
    # Match the released sentence construction with sparse support.
    weights = np.log1p(row.data.astype(np.float64))
    if weights.sum() <= 0:
        return None
    selected = rng.choice(row.indices, size=1024, replace=True, p=weights / weights.sum())
    chrom = np.unique(chromosome[selected])
    rng.shuffle(chrom)
    sequence = [3]
    for c in chrom:
        in_chrom = selected[chromosome[selected] == c]
        sorted_genes = in_chrom[np.argsort(position[in_chrom])]
        sequence.extend([CHROM_OFFSET + c, *ids[sorted_genes], 2])
    return sequence


def embed(cfg, assets, root, repo, out, batch_size):
    started = time.monotonic()
    inventory = verify_assets(root)
    model = official_model(repo, root)
    torch.backends.mha.set_fastpath_enabled(False)
    roles = {}
    out.mkdir(parents=True, exist_ok=False)
    for role in ("A", "B"):
        rows, cells, metadata = read_sample(cfg, assets, role)
        token, *mapping = gene_mapping(metadata, root)
        matched = cells[:, token].tocsr()
        rng = np.random.default_rng(cfg["optional"]["cell_seed"])
        embeds, used = [], []
        for first in range(0, len(rows), batch_size):
            sequences = []
            for i in range(first, min(first + batch_size, len(rows))):
                s = sentence(matched[i], mapping, rng)
                if s is not None:
                    sequences.append(s)
                    used.append(i)
            if sequences:
                width = max(map(len, sequences))
                pos = torch.zeros((len(sequences), width), dtype=torch.long)
                mask = torch.zeros((len(sequences), width), dtype=torch.float32)
                for j, seq in enumerate(sequences):
                    pos[j, :len(seq)] = torch.as_tensor(seq)
                    mask[j, :len(seq)] = 1
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    seq_embedding = nn.functional.normalize(model.pe_embedding(pos.cuda()), dim=2)
                    _, result = model(seq_embedding, mask=mask.cuda())
                embeds.append(result.float().cpu().numpy())
            if first % (batch_size * 64) == 0:
                print(f"UCE role={role} cells={first}/{len(rows)} elapsed={time.monotonic()-started:.1f}s",
                      flush=True)
        output_units(out, role, rows.iloc[used], cells[used], np.concatenate(embeds), token)
        roles[role] = dict(cells=len(rows), missing=len(rows) - len(used), model_genes=len(token))
    write_json(out / "receipt.json", dict(repository="https://github.com/snap-stanford/UCE",
               code_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
               release_article=24320806, checkpoint_layers=4, species=SPECIES, assets=inventory,
               roles=roles, device=torch.cuda.get_device_name(), torch=torch.__version__,
               elapsed_seconds=time.monotonic() - started, batch_size=batch_size,
               inference="Frozen CLS, 1024 draws per cell with log1p(count) weight, chromosome-position ordering",
               output_sha256={p.name: sha256(p) for p in out.glob("*.npz")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--uce-assets", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    embed(config(args.config), args.assets, args.uce_assets, args.repo, args.out, args.batch_size)


if __name__ == "__main__":
    main()
