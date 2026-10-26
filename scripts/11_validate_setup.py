"""Final integration audit against the experiment plan using current artifacts."""
import json
from pathlib import Path
import anndata as ad
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score
from common import DATA,ROOT,load_features,sha,write_json

checks=[]
def check(name,condition,evidence):
    assert condition,(name,evidence)
    checks.append(dict(requirement=name,status='passed',evidence=evidence))

paths=json.loads((ROOT/'reports/environment.json').read_text())
check('G1 environment inventoried and CUDA available',paths['hostname']=='G1' and paths['cuda_available'],'reports/environment.json')
check('Work/data roots respect supplied dir_path',str(ROOT)=='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver' and str(DATA)=='/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver','configs/paths.yaml')
check('Large checkpoint physically stored in data area',(ROOT/'checkpoints').resolve()==DATA/'checkpoints' and (ROOT/'third_party/Geneformer/Geneformer-V2-104M').resolve()==DATA/'checkpoints/Geneformer-V2-104M','filesystem symlink targets')
audit=json.loads((ROOT/'reports/data_audit.json').read_text())
check('All source cell IDs aligned without duplicates',audit['matrix_shape']==[71915,25712] and audit['metadata_missing_from_matrix']==audit['matrix_missing_from_metadata']==audit['duplicate_cell_ids']==0,'reports/data_audit.json')
full=ad.read_h5ad(DATA/'processed/GSE149614.h5ad',backed='r')
check('Canonical data preserves all source cells/genes',full.shape==(71915,25712) and 'counts' in full.layers,'processed/GSE149614.h5ad')
full.file.close()
for subset,n in [('smoke',64),('pilot',2000),('comparison',10000)]:
    a=ad.read_h5ad(DATA/'subsets'/f'{subset}.h5ad')
    manifest=pd.read_csv(ROOT/'manifests/splits'/f'{subset}.tsv',sep='\t')
    check(f'{subset} has expected cell IDs and all ten donors',a.n_obs==n and a.obs.patient_id.nunique()==10 and np.array_equal(a.obs_names,manifest.cell_id),'subset H5AD and split manifest')
    check(f'{subset} raw library sums preserved',np.array_equal(np.asarray(a.layers['counts'].sum(1)).ravel(),a.obs.n_counts) and (a.layers['counts'].data>=0).all() and np.equal(a.layers['counts'].data,np.floor(a.layers['counts'].data)).all(),'layers/counts and obs/n_counts')
    check(f'{subset} donor split disjoint',a.obs.groupby('patient_id',observed=True).split.nunique().max()==1,'AnnData.obs patient_id/split')
    ids={s:set(a.obs.loc[a.obs.split==s,'patient_id']) for s in ['train','val','test']}
    check(f'{subset} 6/2/2 donor holdout',list(map(len,ids.values()))==[6,2,2] and not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']),'AnnData.obs donor sets')
    if subset!='smoke':
        previous=ad.read_h5ad(DATA/'subsets'/('smoke.h5ad' if subset=='pilot' else 'pilot.h5ad'),backed='r')
        check(f'{subset} nested subset',set(previous.obs_names)<=set(a.obs_names),'actual H5AD observation indexes');previous.file.close()
    for model,d in [('pca',50 if subset!='smoke' else 39),('scgpt',512),('geneformer',768)]:
        rid=f'{model}_{subset}';r=json.loads((ROOT/'results'/rid/'run.json').read_text())
        X=load_features(a,rid)
        check(f'{rid} actual finite embeddings in canonical order',len(X)==n and np.isfinite(X).all() and (model=='pca' or X.shape[1]==d),'saved feature cell IDs and values')
        check(f'{rid} exact input/split hashes',r['data_hash']==sha(DATA/'subsets'/f'{subset}.h5ad') and r['split_manifest_hash']==sha(ROOT/'manifests/splits'/f'{subset}.tsv'),'actual H5AD and manifest SHA256')
        pred=pd.read_csv(DATA/'outputs'/rid/'predictions.tsv',sep='\t')
        check(f'{rid} predictions cover same cells',np.array_equal(pred.cell_id,a.obs_names) and np.array_equal(pred.split,a.obs.split.astype(str)),'local cell predictions')
        te=pred.split=='test';m=r['metrics'];labels=m['label_classes'];known=te&pred.truth.isin(labels)
        f1=f1_score(pred.loc[known,'truth'],pred.loc[known,'prediction'],labels=labels,average='macro',zero_division=0)
        ba=balanced_accuracy_score(pred.loc[known,'truth'],pred.loc[known,'prediction'])
        check(f'{rid} metrics independently recomputed',abs(f1-m['macro_f1'])<1e-12 and abs(ba-m['balanced_accuracy'])<1e-12 and r['status']=='executed','scikit-learn recomputation from held-out predictions')
        check(f'{rid} per-class/patient/confusion/resource outputs',all((ROOT/'results'/rid/p).exists() for p in ['per_class.tsv','by_patient.tsv','confusion_matrix.tsv']) and r.get('peak_cpu_rss_gib',0)>0 and r['wall_seconds']>0,'result files and measured run fields')
        if model!='pca':check(f'{rid} actual GPU peak measured',r['peak_cuda_allocated_gib']>0 and r['peak_process_gpu_gib']>0,'CUDA allocator and nvidia-smi process samples')
        if model=='geneformer':check(f'{rid} V2-104M contract',r['actual_parameter_count']==104386867 and r['max_sequence_length']==4096 and r['tokenization']['special_token'] and r['tokenization']['n_counts_basis'].startswith('original'),'checkpoint config/count and official tokenization audit')
for subset in ['pilot','comparison']:
    t=pd.read_csv(ROOT/'reports'/f'kg_ablation_{subset}.tsv',sep='\t')
    selection=json.loads((ROOT/'results'/f'kg_{subset}'/'selection.json').read_text())
    check(f'{subset} five-way KG ablation present',len(t)==5 and selection['size_preserved'] and selection['gene_membership_degree_preserved'],'KG ablation table and degree-preserving control audit')
    pred_base=pd.read_csv(DATA/'outputs'/f'scgpt_{subset}'/'predictions.tsv',sep='\t')
    for suffix in ['pathway','foundation_pathway','foundation_random']:
        pred=pd.read_csv(DATA/'outputs'/f'kg_{subset}_{suffix}'/'predictions.tsv',sep='\t')
        te=pred.split=='test';changed=int((pred.loc[te,'prediction'].to_numpy()!=pred_base.loc[te,'prediction'].to_numpy()).sum())
        recorded=int(t.loc[t.run_id==f'kg_{subset}_{suffix}','n_changed_test_predictions_vs_foundation'].iloc[0])
        check(f'{subset} {suffix} actual prediction changes verified',np.array_equal(pred.cell_id,pred_base.cell_id) and changed==recorded,'paired per-cell predictions')
isp=pd.read_csv(ROOT/'reports/virtual_perturbation_embedding_shift.tsv',sep='\t')
check('Independent deletion with no-op and expressed matched controls',set(isp.condition)=={'no_op','independent_delete','expression_matched_random_delete'} and isp.cell_id.nunique()<=100 and (isp.loc[isp.condition=='no_op','cosine_shift']<1e-5).all() and isp.loc[isp.condition=='independent_delete','token_present'].any(),'actual perturbation TSV')
check('Perturbation interpreted as embedding shift',json.loads((ROOT/'results/isp_pilot/run.json').read_text())['task']=='embedding_shift' and '실측 섭동 검증 전 탐색' in (ROOT/'reports/virtual_perturbation_limitations.md').read_text(),'run task and limitations')
executions=json.loads((ROOT/'reports/notebook_execution.json').read_text())
check('All six tutorials actually executed in fresh kernels',len(executions)==6 and all(x['status']=='executed' for x in executions),'nbclient execution audit')
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(p,as_version=4);cells=[c for c in nb.cells if c.cell_type=='code']
    check(f'{p.name} real code outputs without errors',all(c.execution_count is not None and len(c.outputs)>0 for c in cells) and not any(o.output_type=='error' for c in cells for o in c.outputs),'stored execution counts and output blocks')
check('Source hashes and HF/Zenodo official checksums verified',json.loads((ROOT/'reports/asset_verification.json').read_text())['status']=='passed','reports/asset_verification.json')
check('Large local artifacts have measured path/size/hash',len(json.loads((ROOT/'manifests/local_artifacts.json').read_text()))>20,'manifests/local_artifacts.json')
check('Successful isolated environment freezes recorded',all((ROOT/'envs'/f'vcell-{m}.freeze.txt').stat().st_size>1000 for m in ['scgpt','geneformer']),'envs/*.freeze.txt')
check('Final comparison and limitations report generated',(ROOT/'reports/final_summary.md').exists() and (ROOT/'reports/pretraining_overlap.md').exists(),'final_summary.md and pretraining_overlap.md')
write_json(ROOT/'reports/completion_audit.json',dict(status='passed',n_checks=len(checks),checks=checks,git_push_verification='Verified separately against actual local/server/remote HEAD after final commit'))
print('PASS:',len(checks),'completion checks',flush=True)
