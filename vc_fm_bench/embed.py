"""Extract frozen representations from local model files without network access."""
import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_record, new_directory, write_json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("USE_TF", "0")

MODELS = ("ecfp4", "chemberta2", "molformer", "unimol", "esm2")


def read_queries(path, value_column):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"id", value_column} <= set(reader.fieldnames or ()):
            raise ValueError(f"Expected id and {value_column} columns")
        rows = [(row["id"].strip(), row[value_column].strip()) for row in reader]
    ids = [key for key, _ in rows]
    if not rows or len(set(ids)) != len(rows) or any(not key or not value for key, value in rows):
        raise ValueError("Query requires nonempty, unique IDs and nonempty inputs")
    return rows


def molecular_queries(queries):
    from rdkit import Chem

    valid, failures = [], {}
    for key, value in queries:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            failures[key] = "RDKit parse failure"
            continue
        valid.append((key, Chem.MolToSmiles(molecule, isomericSmiles=True)))
    return valid, failures


def ecfp4(values):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

    properties = (
        Descriptors.MolWt, Crippen.MolLogP, rdMolDescriptors.CalcTPSA,
        Lipinski.NumHDonors, Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds,
        Lipinski.RingCount, Lipinski.NumAromaticRings, rdMolDescriptors.CalcFractionCSP3,
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    vectors = []
    for value in values:
        molecule = Chem.MolFromSmiles(value)
        fingerprint = generator.GetFingerprintAsNumPy(molecule)
        vectors.append(np.concatenate((fingerprint, [f(molecule) for f in properties])))
    return np.asarray(vectors, dtype=np.float32)


def transformer_embeddings(model_name, checkpoint, values, device, batch_size, max_length):
    import torch
    from transformers import AutoModel, AutoTokenizer

    custom_code = model_name == "molformer"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=custom_code, local_files_only=True,
    )
    model_options = {"deterministic_eval": True} if custom_code else {}
    model = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=custom_code, local_files_only=True, **model_options,
    )
    model = model.eval().to(device)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            chunk = values[start:start + batch_size]
            batch = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt", return_special_tokens_mask=True,
            ).to(device)
            special = batch.pop("special_tokens_mask").bool()
            outputs = model(**batch)
            if model_name == "molformer":
                vectors.append(outputs.pooler_output.float().cpu().numpy())
                continue
            keep = batch["attention_mask"].bool() & ~special
            if not keep.any(dim=1).all():
                raise ValueError("At least one input has no valid tokens")
            hidden = outputs.last_hidden_state
            pooled = (hidden * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True)
            vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors)


def protein_embeddings(checkpoint, sequences, device, batch_size):
    # No sequence is discarded or truncated at the model position limit.
    # Chunk pooling is weighted by the represented residues and has no
    # cross-chunk attention. Buckets of nearby lengths avoid long padding.
    chunks = []
    for i, sequence in enumerate(sequences):
        for pos in range(0, len(sequence), 1022):
            part = sequence[pos:pos + 1022]
            chunks.append((i, part))
    chunks.sort(key=lambda item: len(item[1]))
    indices, parts = zip(*chunks)
    hidden = transformer_embeddings("esm2", checkpoint, parts, device, batch_size, 1024)
    series = np.zeros((len(sequences), hidden.shape[1]), dtype=np.float64)
    lengths = np.asarray(list(map(len, parts)))
    np.add.at(series, np.asarray(indices), hidden * lengths[:, None])
    return (series / np.asarray(list(map(len, sequences)))[:, None]).astype(np.float32)


def unimol_embeddings(checkpoint, queries, device, batch_size):
    # Author package constructs an ETKDG conformer and uses a zero-coordinate
    # fallback on conformer errors. Precompute conformers here to exclude failures.
    from rdkit import Chem
    from rdkit.Chem import AllChem
    os.environ["UNIMOL_WEIGHT_DIR"] = str(checkpoint)
    from unimol_tools import UniMolRepr

    inputs, keys, errors = [], [], {}
    for key, smiles in queries:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        options = AllChem.ETKDGv3()
        options.randomSeed = 42
        if AllChem.EmbedMolecule(mol, options) != 0:
            errors[key] = "RDKit ETKDGv3 did not generate a 3D conformer"
            continue
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        mol = Chem.RemoveHs(mol)
        keys.append(key)
        inputs.append((list(atom.GetSymbol() for atom in mol.GetAtoms()),
                       mol.GetConformer().GetPositions()))
    if not inputs:
        raise ValueError("No valid 3D conformers")
    model = UniMolRepr(
        data_type="molecule", remove_hs=True, use_cuda=device.startswith("cuda"),
        batch_size=batch_size, model_name="unimolv1",
    )
    result = model.get_repr({"atoms": [a for a, _ in inputs], "coordinates": [c for _, c in inputs]})
    return keys, np.asarray(result["cls_repr"], dtype=np.float32), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    gpu = args.device == "cuda" and args.model != "ecfp4"
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.model != "ecfp4" and (args.checkpoint is None or not args.checkpoint.is_dir()):
        parser.error("A local checkpoint directory is required")
    field = "sequence" if args.model == "esm2" else "smiles"
    original = read_queries(args.input, field)
    queries, errors = molecular_queries(original) if field == "smiles" else (original, {})
    if not queries:
        raise ValueError("No valid queries")
    keys, values = map(list, zip(*queries))
    if gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    if args.model == "ecfp4":
        vectors = ecfp4(values)
    elif args.model == "unimol":
        keys, vectors, invalid_conformers = unimol_embeddings(
            args.checkpoint, queries, args.device, args.batch_size,
        )
        errors.update(invalid_conformers)
    elif args.model == "esm2":
        vectors = protein_embeddings(args.checkpoint, values, args.device, args.batch_size)
    else:
        vectors = transformer_embeddings(
            args.model, args.checkpoint, values, args.device, args.batch_size,
            max_length=300,
        )
    if gpu:
        torch.cuda.synchronize()
    # Device or model failures abort the extraction. Numeric failures drop only
    # the affected query and are included explicitly in the coverage receipt.
    if vectors.ndim != 2 or len(keys) != len(vectors) or len(set(keys)) != len(vectors):
        raise ValueError("Model did not return one feature vector per key")
    finite = np.isfinite(vectors).all(axis=1)
    for key, ok in zip(keys, finite):
        if not ok:
            errors[key] = "Nonfinite embedding"
    if not finite.any():
        raise ValueError("No finite embedding vectors")
    with new_directory(args.output) as staging:
        file = staging / "vectors.npz"
        np.savez(file, id=np.asarray(keys)[finite], vectors=vectors[finite].astype(np.float32))
        write_json(staging / "receipt.json", {
            "model": args.model,
            "device": torch.cuda.get_device_name() if gpu else "cpu",
            "input": file_record(args.input),
            "checkpoint": {str(p.relative_to(args.checkpoint)): file_record(p)
                           for p in sorted(args.checkpoint.rglob("*")) if p.is_file()}
            if args.checkpoint else {},
            "output": file_record(file),
            "n_requested": len(original),
            "n_embedded": int(finite.sum()),
            "dimension": vectors.shape[1],
            "proteins_chunked": sum(len(s) > 1022 for _, s in queries) if field == "sequence" else None,
            "wall_seconds": perf_counter() - start,
            "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else 0,
            "vram_peak_reserved_bytes": torch.cuda.max_memory_reserved() if gpu else 0,
            "failures": errors,
        })


if __name__ == "__main__":
    main()
