"""Expose the audited CMap 2020 cohort to the fixed R16 fitting procedure."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE
from src.r15_lincs.design import Bank, chemical_features, fold_id
from src.r15_lincs.forecast import position


STUDY = "CMap2020_clean24"


class ConfirmBank(Bank):
    def __init__(self, root, protocol, scenario, cell):
        self.root, self.scenario, self.cell = Path(root),scenario,cell
        receipt_path = root/"receipt.json"
        receipt = read_json(receipt_path)
        if receipt["protocol_sha256"] != sha256(protocol):
            raise ValueError("Confirmation cache differs from its registered protocol")
        for name,digest in receipt["files"].items():
            if sha256(root/name) != digest:
                raise ValueError(f"Confirmation cache artifact changed: {name}")
        self.receipts = {STUDY:sha256(receipt_path)}
        rows = pd.read_csv(root/"target_rows.tsv",sep="\t",keep_default_na=False)
        rows["study"] = STUDY
        rows[GROUP], rows[ENTITY], rows[LINE] = rows.parent,rows.parent,rows.cell_id
        rows[KEY[0]] = STUDY+":"+rows.inst_id
        rows["plate_id"] = STUDY+":"+rows.cell_id+":"+rows.rna_plate
        rows[CONTROL] = rows.plate_id
        rows["time_h"], rows["dose_uM"] = rows.hours,rows.dose
        self.rows = rows
        self.landmarks = pd.read_csv(root/"landmarks.tsv",sep="\t")
        self.y = {STUDY:root/"target_wells.npy"}
        self.controls = {STUDY:np.load(root/"vehicle.npy",mmap_mode="r",allow_pickle=False)}
        self.specs = pd.read_csv(root/"chemicals.tsv",sep="\t").set_index("parent",verify_integrity=True)
        self.fps = chemical_features(self.specs.index)
        self.inputs = {}
        if scenario == "B":
            names = pd.read_csv(root/"transfer_rows.tsv",sep="\t").parent
            values = np.load(root/"transfer.npy",allow_pickle=False)
            self.inputs = {(p,24):v for p,v in zip(names,values,strict=True)}

    def eligible(self):
        technical = self.rows[LINE].eq(self.cell) & self.cell.startswith("JHH")
        rows = self.rows.loc[self.rows.quality | technical]
        if self.scenario == "B":
            rows = rows.loc[rows.parent.map(lambda p:(p,24) in self.inputs)]
        return rows

    def target(self):
        rows = self.eligible()
        return rows.loc[rows[LINE].eq(self.cell)].reset_index(drop=True)


def confirm_split(bank, fold):
    eligible = bank.eligible()
    target = eligible[LINE].eq(bank.cell)
    labels = eligible.parent.map(lambda p:fold_id(p,"R15-outer",3))
    training = eligible.loc[labels.ne(fold) & (target|eligible[LINE].eq("HEPG2"))]
    held = eligible.loc[labels.eq(fold)&target]
    flag = training.parent.map(lambda p:fold_id(p,f"R15-inner-{fold}",5)).eq(0)
    # Validation contains target wells only. Other-cell labels for those parents are
    # removed from the inner training side before hyperparameter and gain selection.
    return tuple(x.reset_index(drop=True) for x in (
        training.loc[~flag], training.loc[flag & training[LINE].eq(bank.cell)], training, held))


def confirm_position(bank, train, query, role):
    raw = bank.response(train,role)
    train_pos = np.zeros_like(raw)
    query_pos = np.zeros((len(query),len(bank.landmarks)),dtype=np.float32)
    query_groups = query.groupby(LINE).indices
    for line,indices in train.groupby(LINE).indices.items():
        idx = query_groups.get(line,np.empty(0,dtype=int))
        y,tr,te = position(bank,train.iloc[indices],query.iloc[idx],role)
        if not np.array_equal(y,raw[indices]):
            raise ValueError("Position training axis diverged")
        train_pos[indices],query_pos[idx] = tr,te
    return raw,train_pos,query_pos


def embed(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    receipt = read_json(args.cache/"receipt.json")
    for name in ("landmarks.tsv","chemicals.tsv"):
        if sha256(args.cache/name)!=receipt["files"][name]:
            raise ValueError(f"Confirmation structure or gene axis changed: {name}")
    checkpoint = read_json(args.reference)["checkpoint_files"]
    for name,digest in checkpoint.items():
        if sha256(args.checkpoint/name)!=digest:
            raise ValueError(f"ChemBERTa encoder pin changed: {name}")
    parents = pd.read_csv(args.cache/"chemicals.tsv",sep="\t").parent.to_numpy(dtype=str)
    features = args.out/"features"
    features.mkdir(parents=True,exist_ok=False)
    metadata = args.out/"metadata"
    metadata.mkdir()
    pd.DataFrame({ENTITY:parents,"smiles":parents}).to_csv(features/"entities.tsv",sep="\t",index=False)
    land = pd.read_csv(args.cache/"landmarks.tsv",sep="\t").rename(
        columns={"pr_gene_id":"token_id","pr_gene_symbol":"gene_symbol"})
    gene_map = pd.read_csv(args.gene_map,sep="\t").rename(columns={"gene_id":"token_id"})
    land.merge(gene_map[["token_id","ensembl_id"]],on="token_id",how="left",validate="one_to_one").to_csv(
        metadata/"genes.tsv",sep="\t",index=False)
    torch.set_num_threads(4)
    encoder = AutoModel.from_pretrained(args.checkpoint,local_files_only=True,add_pooling_layer=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    limit = encoder.config.max_position_embeddings-tokenizer.pad_token_id-1
    vectors = []
    with torch.inference_mode():
        for i in range(0,len(parents),32):
            batch = tokenizer(parents[i:i+32].tolist(),padding=True,truncation=False,return_tensors="pt")
            if batch["input_ids"].shape[1]>limit or not torch.all(batch["input_ids"][:,0]==tokenizer.cls_token_id):
                raise ValueError("Confirmation embedding would truncate the input")
            vectors.append(encoder(**batch).last_hidden_state[:,0].numpy())
    artifact = features/"chemberta.npz"
    np.savez_compressed(artifact,parents=parents,embedding=np.concatenate(vectors))
    write_json(features/"receipt.json",dict(chemberta_sha256=sha256(artifact),parent_count=len(parents),
        target_cache_sha256=sha256(args.cache/"receipt.json"),reference_sha256=sha256(args.reference),
        metadata_sha256=sha256(metadata/"genes.tsv"),training_expression_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","out","reference","checkpoint","gene-map"):
        parser.add_argument("--"+name,type=Path,required=True)
    embed(parser.parse_args())


if __name__ == "__main__":
    main()
