"""Paired donor-level factorial effects from actual disjoint OOF predictions across three seeds."""
import argparse
from research_core import *
from sklearn.metrics import confusion_matrix

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['frozen','all'],default='all');args=p.parse_args()
manifest=pd.read_csv(ROOT/'manifests/research_phase2/frozen_confirmation_runs.tsv',sep='\t')
if args.mode=='all':manifest=pd.concat([manifest,pd.read_csv(ROOT/'manifests/research_phase2/adapted_confirmation_runs_v2.tsv',sep='\t')],ignore_index=True)
factors=['DM00','DM10'] if args.mode=='frozen' else ['DM00','DM10','DM01','DM11'];seeds=[42,43,44];labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];donors=['HCC01','HCC03','HCC04','HCC05','HCC06','HCC07','HCC08','HCC10']
assert len(manifest)==len(factors)*12 and not manifest.duplicated(['factor','seed','fold']).any()
counts=np.zeros((3,len(factors),8,6,6),dtype=np.int64);references={};records=[];sources=[];all_runs={}
for row in manifest.itertuples():
    path=REPORT/'runs'/row.run_id/'run.json';assert path.exists(),f'Incomplete confirmation: {row.run_id}';run=json.loads(path.read_text());assert run['status']=='executed' and run['seed']==row.seed and run['fold']==row.fold and run['n_outer_training']==4096
    assert run['data_variant']==row.data_variant
    frame=pd.read_csv(run['prediction_path'],sep='\t');assert sha(run['prediction_path'])==run['prediction_sha256'] and not frame.cell_id.duplicated().any()
    key=(row.seed,row.fold)
    if key in references:
        ref_frame,ref_run=references[key];assert frame[['cell_id','patient_id','truth']].equals(ref_frame[['cell_id','patient_id','truth']])
        assert run['validation_manifest_sha256']==ref_run['validation_manifest_sha256'] and run['n_inner_training']==ref_run['n_inner_training'] and run['n_inner_validation']==ref_run['n_inner_validation']
        assert run['trainable_head_parameters']==ref_run['trainable_head_parameters'] and run['head_architecture']==ref_run['head_architecture']
    else:references[key]=(frame,run)
    all_runs[(row.seed,row.fold,row.factor)]=run
    for donor,part in frame.groupby('patient_id'):
        assert donor in donors
        counts[seeds.index(row.seed),factors.index(row.factor),donors.index(donor)]+=confusion_matrix(part.truth,part.prediction,labels=labels)
    sources.append(dict(run_id=row.run_id,run_sha256=sha(path),prediction_sha256=sha(run['prediction_path'])))
for seed in seeds:
 for fold in range(4):
  if args.mode=='all':
   for pair in [('DM00','DM01'),('DM10','DM11')]:
    assert all_runs[(seed,fold,pair[0])]['training_manifest_sha256']==all_runs[(seed,fold,pair[1])]['training_manifest_sha256'],'Model-only pair must have identical actual training IDs'
assert np.all(counts.sum(axis=(-1,-2,-3))==63036),'Every factor/seed must retain the entire development population'

def f1(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2)
    return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=np.float64),where=den>0).mean(-1)
pooled=f1(counts.sum(2));per_donor=f1(counts)
for si,seed in enumerate(seeds):
 for fi,factor in enumerate(factors):records.append(dict(seed=seed,factor=factor,n_evaluation=63036,pooled_macro_f1=float(pooled[si,fi])))
rng=np.random.default_rng(20261026);draws=rng.integers(0,8,size=(5000,8));weights=np.stack([(draws==donor).sum(1) for donor in range(8)],axis=1)
bootstrap=f1(np.einsum('bd,sfdij->bsfij',weights,counts)).mean(1)
def effects(values):
    result={'Data gain':values[...,1]-values[...,0]}
    if args.mode=='all':result.update({'Model gain':values[...,2]-values[...,0],'Combined gain':values[...,3]-values[...,0],'Interaction':values[...,3]-values[...,1]-values[...,2]+values[...,0]})
    return result
point=effects(pooled.mean(0));intervals=effects(bootstrap);donor_effects=effects(np.moveaxis(per_donor.mean(0),0,-1));effect_rows=[]
for name,value in point.items():
    delta=donor_effects[name];effect_rows.append(dict(effect=name,pooled_mean_difference=float(value),donor_bootstrap_lower95=float(np.quantile(intervals[name],.025)),donor_bootstrap_upper95=float(np.quantile(intervals[name],.975)),n_nonnegative_donors=int((delta>=0).sum()),n_donors=8))
out=REPORT/('factorial_frozen_controls' if args.mode=='frozen' else 'factorial_confirmation');out.mkdir(exist_ok=True)
pd.DataFrame(records).to_csv(out/'pooled_by_seed.tsv',sep='\t',index=False)
pd.DataFrame(records).groupby('factor').pooled_macro_f1.agg(['mean','std']).to_csv(out/'pooled_mean_std.tsv',sep='\t')
pd.DataFrame(effect_rows).to_csv(out/'paired_effects.tsv',sep='\t',index=False)
pd.DataFrame([dict(seed=seed,factor=factor,patient_id=donor,macro_f1_common_ontology=float(per_donor[si,fi,di])) for si,seed in enumerate(seeds) for fi,factor in enumerate(factors) for di,donor in enumerate(donors)]).to_csv(out/'by_donor.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed_partial_factorial' if args.mode=='frozen' else 'executed',project_commit=git_commit(),n_executed_runs=len(manifest),sources=sources,
    validation='Actual prediction hashes and identical ordered evaluation cells checked; shared head and validation manifests checked; actual training IDs matched within model-only pairs',
    confidence_interval='5000 paired bootstrap samples of8 donors, average across3 seeds; fixed-six F1 recomputed from resampled donor confusion matrices',limitations=['Only8 independent development donors','Overlapping training folds and fixed trained models; bootstrap does not include retraining uncertainty','Development OOF is repeatedly inspected and is not an external final result'],external_final_scores_observed=False))
print(pd.DataFrame(records).groupby('factor').pooled_macro_f1.agg(['mean','std']).to_string());print(pd.DataFrame(effect_rows).to_string(index=False))
