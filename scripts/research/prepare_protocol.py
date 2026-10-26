"""Freeze donor folds and derive development files without changing legacy artifacts."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from common import DATA,sha,write_json
import anndata as ad
import numpy as np
import pandas as pd

DEVELOPMENT=['HCC01','HCC03','HCC04','HCC05','HCC06','HCC07','HCC08','HCC10']
LEGACY=['HCC02','HCC09']
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
out=DATA/'research_phase2/data';out.mkdir(parents=True,exist_ok=True)
manifest_dir=ROOT/'manifests/research_phase2';manifest_dir.mkdir(parents=True,exist_ok=True)
folds=[];patients=np.random.default_rng(20261026).permutation(DEVELOPMENT).tolist()
for k in range(4):
    test=patients[2*k:2*k+2]
    outer_train=[d for d in patients if d not in test]
    val=np.random.default_rng(20261026+k).choice(outer_train,2,replace=False).tolist()
    for d in DEVELOPMENT:
        role='evaluation' if d in test else 'inner_validation' if d in val else 'inner_training'
        folds.append(dict(split_id='development_donor4_v2',fold=k,patient_id=d,role=role,outer_training=d not in test))
frame=pd.DataFrame(folds)
target=manifest_dir/'development_donor4_v2.tsv'
if target.exists():assert target.read_text()==frame.to_csv(sep='\t',index=False),'Existing folds cannot be silently replaced'
else:frame.to_csv(target,sep='\t',index=False)
for name,source in [('pilot',DATA/'subsets/comparison.h5ad')]+([('full',DATA/'processed/GSE149614.h5ad')] if args.full else []):
    dest=out/f'development_{name}.h5ad'
    if dest.exists():
        a=ad.read_h5ad(dest,backed='r')
        assert set(a.obs.patient_id)==set(DEVELOPMENT)
        print('Existing immutable development data:',dest,a.shape,flush=True);a.file.close();continue
    a=ad.read_h5ad(source)
    a=a[a.obs.patient_id.isin(DEVELOPMENT)].copy()
    assert not a.obs.patient_id.isin(LEGACY).any()
    a.obs['evaluation_role']='development_repeatedly_inspectable'
    a.uns['research_protocol']='research_annotation_v2_20261026'
    a.uns['legacy_donors_excluded']=LEGACY
    a.write_h5ad(dest,compression='gzip')
    a.obs[['cell_id','patient_id','cell_type_coarse']].to_csv(manifest_dir/f'development_{name}_cells.tsv',sep='\t',index=False)
    counts=pd.crosstab(a.obs.patient_id,a.obs.cell_type_coarse)
    counts.to_csv(ROOT/'reports/research_phase2'/f'development_{name}_support.tsv',sep='\t')
    write_json(ROOT/'reports/research_phase2'/f'development_{name}_data.json',dict(dataset_id=f'GSE149614_development_{name}',path=str(dest),n_cells=a.n_obs,n_genes=a.n_vars,sha256=sha(dest),source=str(source),source_sha256=sha(source),donors=DEVELOPMENT,legacy_excluded=LEGACY,split_sha256=sha(target),role='development_not_a_fresh_blind_test'))
    print(name,a.shape,counts.to_string(),flush=True)
print(frame.to_string(index=False),flush=True)
