"""Compute deterministic, frozen molecular embeddings using a local ChemBERTa release."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import digest, receipt
from .chemistry import identity


class ChemBERTa:
    def __init__(self, model):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.encoder = AutoModel.from_pretrained(str(model), local_files_only=True,
                                                 trust_remote_code=False, add_pooling_layer=False).eval().to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        self.capacity = int(self.encoder.config.max_position_embeddings) - 2

    def encode(self, smiles):
        result, truncated = [], []
        for start in range(0, len(smiles), 32):
            batch = list(smiles[start: start + 32])
            lengths = [len(tokens) for tokens in self.tokenizer(batch, add_special_tokens=True)["input_ids"]]
            truncated.extend(length > self.capacity for length in lengths)
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=self.capacity,
                                    return_tensors="pt", return_special_tokens_mask=True)
            valid = inputs["attention_mask"] * (1 - inputs.pop("special_tokens_mask"))
            with self.torch.inference_mode():
                hidden = self.encoder(**inputs).last_hidden_state
                feature = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1).clamp(min=1).unsqueeze(1)
            result.append(feature.numpy())
        return np.concatenate(result), np.asarray(truncated)


def prepare(model, structures, output):
    rows = []
    for path in structures:
        table = pd.read_csv(path, sep="\t", usecols=lambda c: c in ["smiles", "inchikey"]).dropna()
        if "inchikey" not in table:
            normalized = []
            for raw in sorted(set(table.smiles)):
                try:
                    canonical, key, *_ = identity(raw)
                except (ValueError, RuntimeError):
                    continue
                normalized.append((canonical, key))
            table = pd.DataFrame(normalized, columns=["smiles", "inchikey"])
        rows.append(table)
    ids = pd.concat(rows, ignore_index=True).drop_duplicates("inchikey").sort_values("inchikey")
    encoder = ChemBERTa(model)
    values, truncated = encoder.encode(ids.smiles.tolist())
    frame = pd.DataFrame(values, index=pd.Index(ids.inchikey, name="inchikey")).add_prefix("cb_")
    frame["chemberta_truncated"] = truncated
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output / "chemberta.parquet")
    receipt(output / "receipt.json", dict(checkpoint={p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()},
        structures={str(p): digest(p) for p in structures}, output=digest(output / "chemberta.parquet"),
        rows=len(frame), dimensions=values.shape[1], truncated=int(truncated.sum())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    prepare(**vars(parser.parse_args()))
