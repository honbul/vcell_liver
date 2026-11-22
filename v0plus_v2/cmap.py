"""Frozen non-liver 24-hour CMap drug features for the response sensitivity panel."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from vc_build.io import publish_json, sha256
from vc_build.v0plus_cv import training_inputs

SEED = 28818


def file_rows(path, samples, landmark):
    with h5py.File(path, "r") as h5:
        index = pd.Index(h5["0/META/COL/id"][:].astype(str))
        ids = index.get_indexer(samples)
        columns = pd.Index(h5["0/META/ROW/id"][:].astype(str)).get_indexer(landmark)
        if not index.is_unique or (ids<0).any() or (columns<0).any():
            raise ValueError("CMap signature/landmark axes changed")
        result = np.empty((len(ids), len(columns)), dtype=np.float32)
        matrix = h5["0/DATA/0/matrix"]
        order = ids.argsort()
        for ix in np.array_split(order, int(np.ceil(len(order)/512))):
            result[ix] = matrix[ids[ix], :][:,columns]
    return result


def build(args):
    train, _, receipt = training_inputs(args.bundle)
    eligible = set(train.compound_id)
    meta = pd.read_csv(args.cmap/"compoundinfo_beta.txt", sep="\t",
                       usecols=["pert_id","inchi_key"], dtype=str)
    meta = meta.loc[meta.inchi_key.isin(eligible)].drop_duplicates()
    if meta.pert_id.duplicated().any():
        raise ValueError("A CMap parent had conflicting exact stereo registry values")
    metadata = ("cell_iname","cell_lineage","primary_disease")
    cells = pd.read_csv(args.cmap/"cellinfo_beta.txt", sep="\t", usecols=metadata).fillna("")
    tissue = cells.cell_lineage + " " + cells.primary_disease
    nonhepatic = set(cells.loc[~tissue.str.contains("liver|hepat|hepatic", case=False, regex=True), "cell_iname"])
    keep = ["sig_id","cell_iname","pert_type","pert_id","pert_time","pert_time_unit","pert_dose","pert_dose_unit",
            "nsample","cc_q75","qc_pass"]
    chunks = []
    for rows in pd.read_csv(args.cmap/"siginfo_beta.txt", sep="\t", usecols=keep, chunksize=160_000, low_memory=False):
        valid = (rows.cell_iname.isin(nonhepatic) & rows.pert_type.eq("trt_cp") &
                 rows.pert_id.isin(meta.pert_id) & rows.pert_time_unit.eq("h") &
                 pd.to_numeric(rows.pert_time,errors="coerce").eq(24) &
                 rows.pert_dose_unit.eq("uM") & pd.to_numeric(rows.pert_dose,errors="coerce").between(.01,100) &
                 pd.to_numeric(rows.qc_pass,errors="coerce").eq(1) &
                 pd.to_numeric(rows.nsample,errors="coerce").ge(2) &
                 pd.to_numeric(rows.cc_q75,errors="coerce").ge(.2))
        chunks.append(rows.loc[valid,["sig_id","cell_iname","pert_id","pert_dose"]])
    admitted = pd.concat(chunks,ignore_index=True).drop_duplicates("sig_id").merge(meta,on="pert_id",validate="many_to_one")
    table = pd.read_csv(args.landmarks,sep="\t")
    genes = table.pr_gene_id.astype(str).to_numpy()
    matrix, = args.cmap.glob("level5_beta_trt_cp_*.gctx")
    x = file_rows(matrix,admitted.sig_id,genes)
    dose_group = admitted.groupby(["inchi_key","cell_iname","pert_dose"]).indices
    contrasts = pd.DataFrame(np.stack([x[i].mean(axis=0) for i in dose_group.values()]),
                             index=pd.MultiIndex.from_tuples(dose_group, names=["compound","cell","dose"]))
    # Give cell contexts equal mass irrespective of repeated signatures or dose counts.
    final = contrasts.groupby(level=["compound","cell"]).mean().groupby(level="compound").mean()
    args.out.mkdir(parents=True,exist_ok=False)
    path = args.out/"vectors.npz"
    np.savez_compressed(path,id=final.index.to_numpy(dtype=str),vectors=final.to_numpy(dtype=np.float32))
    publish_json(args.out/"receipt.json",{"output":{"sha256":sha256(path),"bytes":path.stat().st_size},
        "input_metadata_sha256":{name:sha256(args.cmap/name) for name in ("compoundinfo_beta.txt","cellinfo_beta.txt","siginfo_beta.txt")},
        "source_matrix":str(matrix),"source_matrix_size":matrix.stat().st_size, "source_genes_sha256":sha256(args.landmarks),
        "registered_rows": receipt["train_rows"],"signatures":len(admitted), "nonhepatocyte_cells":int(admitted.cell_iname.nunique()),
        "requested":len(eligible),"covered":len(final),"required_qc":"24h, two or more replicates, cc_q75 >= 0.2",
        "model_fit_or_outcome_access":False})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("cmap","bundle","landmarks","out"):
        cli.add_argument("--"+key,type=Path,required=True)
    build(cli.parse_args())
