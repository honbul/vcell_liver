"""Generate one verified, frozen CLS vector and one ECFP4 vector per canonical entity."""

import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

from .common import arguments, config, sha256, write_json
from .prepare import unique


def encode(cfg, sample_manifest, model_dir, out):
    import torch
    from transformers import AutoModel, AutoTokenizer

    settings = cfg["features"]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if sha256(model_dir / "pytorch_model.bin") != settings["weights_sha256"]:
        raise ValueError("Checkpoint hash differs from the model repository")
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt", "pytorch_model.bin")
    receipt = {name: sha256(model_dir / name) for name in files}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model, loading = AutoModel.from_pretrained(
        model_dir, local_files_only=True, output_loading_info=True, add_pooling_layer=False
    )
    unused_head = {"norm_mean", "norm_std", "regression.dense.bias", "regression.dense.weight",
                   "regression.out_proj.bias", "regression.out_proj.weight"}
    if loading["missing_keys"] or set(loading["unexpected_keys"]) != unused_head or loading["mismatched_keys"]:
        raise ValueError(f"Checkpoint backbone load is incomplete: {loading}")
    model.eval().requires_grad_(False)
    entities = pd.read_csv(sample_manifest, sep="\t", keep_default_na=False)
    entities = entities[entities.split.isin(["train", "test"]) & ~entities.quarantined]
    entities = entities[["compound_entity_id", "compound_group_id", "smiles", "split"]].drop_duplicates()
    entities = entities.sort_values("compound_entity_id").reset_index(drop=True)
    unique(entities, ["compound_entity_id"])
    if not entities.smiles.astype(bool).all():
        raise ValueError("Eligible entity has no validated SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=settings["ecfp_radius"], fpSize=settings["ecfp_bits"], includeChirality=settings["ecfp_chirality"]
    )
    mols = [Chem.MolFromSmiles(s) for s in entities.smiles]
    if any(m is None for m in mols):
        raise ValueError("Invalid canonical SMILES")
    fps = [generator.GetFingerprint(m) for m in mols]
    ecfp = np.vstack([generator.GetFingerprintAsNumPy(m) for m in mols])
    entities["scaffold"] = [
        scaffold or "<acyclic>" for scaffold in (MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols)
    ]

    vectors, lengths = [], []
    with torch.inference_mode():
        for start in range(0, len(entities), 32):
            texts = entities.smiles.iloc[start:start+32].tolist()
            batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            usable_length = model.config.max_position_embeddings - tokenizer.pad_token_id - 1
            if batch["input_ids"].shape[1] > usable_length or not torch.all(batch["input_ids"][:, 0] == tokenizer.cls_token_id):
                raise ValueError("Truncation or missing CLS in chemical tokenizer")
            lengths.extend(batch["attention_mask"].sum(dim=1).tolist())
            vectors.append(model(**batch).last_hidden_state[:, 0].numpy())
        chem = np.concatenate(vectors)
        first = tokenizer(entities.smiles.iloc[:2].tolist(), padding=True, return_tensors="pt")
        repeat = model(**first).last_hidden_state[:, 0].numpy()
    if chem.shape != (len(entities), model.config.hidden_size) or not np.isfinite(chem).all():
        raise ValueError("Nonfinite or unexpected embedding output")
    repeat_diff = float(np.max(np.abs(chem[:2] - repeat)))
    if repeat_diff > 1e-5:
        raise ValueError(f"Repeated CLS encoding difference {repeat_diff}")

    train = entities.split.eq("train").to_numpy()
    known_scaffolds = set(entities.loc[train, "scaffold"]) - {"<acyclic>"}
    similarity = entities.loc[~train].copy()
    fit_fp = [fp for fp, fit in zip(fps, train) if fit]
    similarity["max_tanimoto_vs_source"] = [max(DataStructs.BulkTanimotoSimilarity(fps[i], fit_fp)) for i in similarity.index]
    similarity["scaffold_seen"] = similarity.scaffold.isin(known_scaffolds)
    similarity["num_source_entities"] = train.sum()
    entities["n_tokens"] = lengths

    out.mkdir(parents=True, exist_ok=False)
    entities.to_csv(out / "entities.tsv", sep="\t", index=False)
    similarity.to_csv(out / "test_similarity.tsv", sep="\t", index=False)
    np.savez(out / "features.npz", E=ecfp, C=chem, entity=entities.compound_entity_id.to_numpy(dtype=str))
    write_json(out / "receipt.json", dict(
        model=settings["checkpoint"], revision=settings["revision"], input_hash=sha256(sample_manifest),
        checkpoint_files=receipt, unused_regression_head_keys=sorted(unused_head),
        backbone_parameters=sum(p.numel() for p in model.parameters()), embedding_dim=chem.shape[1],
        ecfp_dim=ecfp.shape[1], entities=len(entities), missing=0, truncation=0,
        repeat_max_absolute_difference=repeat_diff, candidate_pooling=settings["pooling"], device="CPU",
        source_model_card_license="not declared on the checkpoint card as checked for this revision; weights are referenced only",
        response_pretraining_overlap="not established; structure exposure possible",
        versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers", "rdkit")},
        outputs={f: sha256(out / f) for f in ("entities.tsv", "features.npz", "test_similarity.tsv")}
    ))


def main():
    parser = arguments(__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    encode(config(args.config), args.assets / "metadata" / "samples.tsv", args.checkpoint_dir, args.assets / "features")


if __name__ == "__main__":
    main()
