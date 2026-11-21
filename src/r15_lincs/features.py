"""Rebuild structure-only ChemBERTa features and retain the actual landmark identities."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json

from .data import STUDIES, table


def metadata(cache, output, lincs_map):
    genes = pd.read_csv(cache/STUDIES[0]/"landmarks.tsv",sep="\t")
    available = pd.read_csv(lincs_map,sep="\t")
    genes = genes.merge(available[["gene_id","ensembl_id"]],left_on="pr_gene_id",right_on="gene_id",how="left",
                        validate="one_to_one").rename(columns={"pr_gene_id":"token_id","pr_gene_symbol":"gene_symbol"})
    entities = pd.concat([table(cache/study/"chemicals.tsv") for study in STUDIES],ignore_index=True)
    parents = entities.loc[entities.structure.ne(""),"parent"].drop_duplicates().sort_values().to_numpy(dtype=str)
    p = output/"metadata"
    p.mkdir(parents=True,exist_ok=True)
    genes[["token_id","gene_symbol","ensembl_id"]].to_csv(p/"genes.tsv",sep="\t",index=False)
    f = output/"features"
    f.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"compound_entity_id":parents,"smiles":parents}).to_csv(f/"entities.tsv",sep="\t",index=False)
    return parents


def embed(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    parents = metadata(args.cache,args.out,args.lincs_map)
    expected = read_json(args.reference)["checkpoint_files"]
    for name,digest in expected.items():
        if sha256(args.checkpoint/name)!=digest:
            raise ValueError(f"ChemBERTa structure-only checkpoint asset changed: {name}")
    torch.set_num_threads(4)
    encoder = AutoModel.from_pretrained(args.checkpoint,local_files_only=True,add_pooling_layer=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    limit = encoder.config.max_position_embeddings-tokenizer.pad_token_id-1
    use = []
    with torch.inference_mode():
        for batch in range(0,len(parents),32):
            payload = tokenizer(parents[batch:batch+32].tolist(),padding=True,truncation=False,return_tensors="pt")
            if payload["input_ids"].shape[1]>limit or not torch.all(payload["input_ids"][:,0]==tokenizer.cls_token_id):
                raise ValueError("Structure encoder would truncate a canonical SMILES")
            use.append(encoder(**payload).last_hidden_state[:,0].numpy())
    matrix = np.concatenate(use)
    artifact = args.out/"features/chemberta.npz"
    np.savez_compressed(artifact,parents=parents,embedding=matrix)
    write_json(args.out/"features/receipt.json",dict(chemberta_sha256=sha256(artifact),
        reference_sha256=sha256(args.reference),metadata_sha256=sha256(args.out/"metadata/genes.tsv"),
        parent_count=len(parents),training_expression_used=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("cache","out","lincs-map","reference","checkpoint"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    args=parser.parse_args()
    embed(args)


if __name__=="__main__":
    main()
