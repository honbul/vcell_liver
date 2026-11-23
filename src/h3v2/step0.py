"""Verify actual atlas schemas and OOF train/query isolation before selection."""
import json

import lightgbm
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from nextgen_h2.run import read_json
from vc_build.io import publish_json,sha256

from .data import ROOT, SCREENS, SPLITS, grouped_split, nested, read_data, register, table_hash


def main():
    source=read_json(ROOT/"splits/manifest.json")
    registry=register()
    audits={}
    for split in SPLITS:
        mode=split.rsplit("-",1)[0]
        fit,test=[read_data(split,role) for role in ("fit","query")]
        if mode in SCREENS and (mode in set(fit.source_id) or set(test.source_id)!={mode}):
            raise ValueError("Held screen entered sampling")
        k=read_json(ROOT/"kernels"/split/"receipt.json")
        if k["outer_query_outcomes_loaded"]:
            raise ValueError("OOF operator received unsealed test outcomes")
        observations=[]
        partitions=[nested(fit,split,registry,j) for j in range(3)]
        views={f"inner-{i}":v for i,v in enumerate(partitions)}|{"outer":(fit,test)}
        for role,(tr,va) in views.items():
            if set(grouped_split(tr,mode,registry))&set(grouped_split(va,mode,registry)):
                raise ValueError(f"Group leakage in the actual feature basis: {split}/{role}")
            root=ROOT/"features"/split/role
            a=read_json(root/"receipt.json")
            wanted={"source_rna_width":17871,"h2_rna_width":14421,"pathway_tf":338,"fingerprint":2048,
                    "rdkit":4,"chembl_gene_axis":1451,"lincs":978,"context_pathways":44,"direct_target_summary":5,
                    "masked_profiles_all_nan":True,"fit_only_centering":True}
            if any(a["dimensions"].get(n)!=d for n,d in wanted.items()):
                raise ValueError("A required feature modality is absent")
            if a["held_endpoint_values_used"] or (a["fit_id_sha256"],a["held_id_sha256"])!=(
                table_hash(tr),table_hash(va)
            ):
                raise ValueError("Feature construction joined a different fit/query pair")
            if len(a["pretrained"])!=2 or a["dimensions"]["pca"]<256:
                raise ValueError("Insufficient feature dimension")
            observations.append({"role":role,**a})
        for j in range(3):
            not_this=pd.concat([item[1] for i,item in enumerate(partitions) if i!=j],ignore_index=True)
            this=partitions[j][1]
            for rule,left,right in ((f"one-{j}",this,not_this),(f"two-{j}",not_this,this)):
                part=k["parts"][rule]
                truth=(table_hash(left),table_hash(right),0)
                observed=tuple(part[n] for n in ("fit_row_key_sha256","query_row_key_sha256","fit_query_group_overlap"))
                if truth!=observed:
                    raise ValueError("An OOF fit gained access to the wrong validation group")
        if k["parts"]["outer"]["query_row_key_sha256"]!=table_hash(test):
            raise ValueError("Bank's outer bases forecast a different roster")
        audits[split]={"feature_receipts":observations,"kernel_receipt_sha256":sha256(ROOT/"kernels"/split/"receipt.json"),
                       "outer_support":source["split_support"][split]}
    specs={"torch":torch.__version__,"sklearn":sklearn.__version__,"lightgbm":lightgbm.__version__,
           "numpy":np.__version__,"pandas":pd.__version__,"RDKit":rdkit.__version__,
           "lightgbm_so_sha256":sha256(ROOT/"deps/lightgbm/lib/lib_lightgbm.so")}
    publish_json(ROOT/"step0_verification.json",{
        "feature_models_share_one_atlas":True,"audited_atlas_count":4*len(audits),
        "full_partition_sha256":sha256(ROOT/"splits/manifest.json"),
        "folds":audits,"versions":specs,"query_outcomes_for_outer_consumed":False,
    })
    print("Verified 60 fit-only shared feature atlases and 15 authentic OOF banks",json.dumps(specs))


if __name__=="__main__":
    main()
