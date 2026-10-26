"""Paired donor-level factorial effects from actual disjoint OOF predictions across three seeds."""
import argparse
from research_core import *
from sklearn.metrics import confusion_matrix

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['frozen','all'],default='all');args=p.parse_args()
manifest=pd.read_csv(ROOT/'manifests/research_phase2/frozen_confirmation_runs.tsv',sep='\t')
if args.mode=='all':manifest=pd.concat([manifest,pd.read_csv(ROOT/'manifests/research_phase2/adapted_confirmation_runs_v2.tsv',sep='\t')],ignore_index=True)
factors=['DM00','DM10'] if args.mode=='frozen' else ['DM00','DM10','DM01','DM11'];seeds=[42,43,44];labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];donors=['HCC01','HCC03','HCC04','HCC05','HCC06','HCC07','HCC08','HCC10']
assert len(manifest)==len(factors)*12 and not manifest.duplicated(['factor','seed','fold']).any()
counts=np.zeros((3,len(factors),8,6,6),dtype=np.int64);references={};evaluation_references={};records=[];sources=[];all_runs={};resources=[]
for row in manifest.itertuples():
    path=REPORT/'runs'/row.run_id/'run.json';assert path.exists(),f'Incomplete confirmation: {row.run_id}';run=json.loads(path.read_text());assert run['status']=='executed' and run['seed']==row.seed and run['fold']==row.fold and run['n_outer_training']==4096
    assert run['data_variant']==row.data_variant
    frame=pd.read_csv(run['prediction_path'],sep='\t');assert sha(run['prediction_path'])==run['prediction_sha256'] and not frame.cell_id.duplicated().any()
    evaluated=frame[['cell_id','patient_id','truth']]
    if row.fold in evaluation_references:assert evaluated.equals(evaluation_references[row.fold]),'Evaluation population/order/truth must also match across all seeds'
    else:evaluation_references[row.fold]=evaluated
    key=(row.seed,row.fold)
    if key in references:
        ref_frame,ref_run=references[key];assert frame[['cell_id','patient_id','truth']].equals(ref_frame[['cell_id','patient_id','truth']])
        assert run['validation_manifest_sha256']==ref_run['validation_manifest_sha256'] and run['n_inner_training']==ref_run['n_inner_training'] and run['n_inner_validation']==ref_run['n_inner_validation']
        assert run['trainable_head_parameters']==ref_run['trainable_head_parameters'] and run['head_architecture']==ref_run['head_architecture']
    else:references[key]=(frame,run)
    all_runs[(row.seed,row.fold,row.factor)]=run
    resources.append(dict(run_id=row.run_id,factor=row.factor,fold=row.fold,seed=row.seed,wall_training_seconds=run['wall_training_seconds'],wall_inference_seconds=run['wall_inference_seconds'],wall_total_seconds=run['wall_total_seconds'],peak_process_gpu_gib=run.get('peak_process_gpu_gib'),trainable_encoder_parameters=run['trainable_encoder_parameters'],trainable_head_parameters=run['trainable_head_parameters'],inner_seen_examples=run.get('update_counts',{}).get('inner',{}).get('seen_examples'),outer_seen_examples=run.get('update_counts',{}).get('outer',{}).get('seen_examples')))
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
def class_scores(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);truth_support=cm.sum(-1);predicted_support=cm.sum(-2)
    return dict(precision=np.divide(tp,predicted_support,out=np.zeros_like(tp,dtype=float),where=predicted_support>0),recall=np.divide(tp,truth_support,out=np.zeros_like(tp,dtype=float),where=truth_support>0),f1=np.divide(2*tp,truth_support+predicted_support,out=np.zeros_like(tp,dtype=float),where=(truth_support+predicted_support)>0))
class_point=class_scores(counts.sum(2));class_boot=class_scores(np.einsum('bd,sfdij->bsfij',weights,counts));class_rows=[];class_effects=[]
for fi,factor in enumerate(factors):
    for li,label in enumerate(labels):
        row=dict(factor=factor,label=label,n_evaluation_support=int(counts[0,fi,:,li,:].sum()))
        for metric,values in class_point.items():row[metric+'_mean']=float(values[:,fi,li].mean());row[metric+'_std']=float(values[:,fi,li].std(ddof=1))
        class_rows.append(row)
        if fi:
            for metric,values in class_point.items():
                delta=float((values[:,fi,li]-values[:,0,li]).mean());samples=(class_boot[metric][:,:,fi,li]-class_boot[metric][:,:,0,li]).mean(1);class_effects.append(dict(factor=factor,reference='DM00',label=label,metric=metric,mean_difference=delta,lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975))))
pd.DataFrame(class_rows).to_csv(out/'per_class_mean_std.tsv',sep='\t',index=False);pd.DataFrame(class_effects).to_csv(out/'paired_class_effects.tsv',sep='\t',index=False)
resource_frame=pd.DataFrame(resources);resource_frame.to_csv(out/'resources_by_run.tsv',sep='\t',index=False);resource_frame.groupby('factor').agg(n_runs=('run_id','size'),mean_training_seconds=('wall_training_seconds','mean'),mean_inference_seconds=('wall_inference_seconds','mean'),sum_current_process_wall_seconds=('wall_total_seconds','sum'),max_process_gpu_gib=('peak_process_gpu_gib','max'),median_process_gpu_gib=('peak_process_gpu_gib','median'),trainable_encoder_parameters=('trainable_encoder_parameters','max'),trainable_head_parameters=('trainable_head_parameters','max')).to_csv(out/'resources_by_factor.tsv',sep='\t')
pd.DataFrame([dict(seed=seed,factor=factor,patient_id=donor,macro_f1_common_ontology=float(per_donor[si,fi,di])) for si,seed in enumerate(seeds) for fi,factor in enumerate(factors) for di,donor in enumerate(donors)]).to_csv(out/'by_donor.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed_partial_factorial' if args.mode=='frozen' else 'executed',project_commit=git_commit(),n_executed_runs=len(manifest),sources=sources,
    validation='Actual prediction hashes and identical ordered evaluation cells checked; shared head and validation manifests checked; actual training IDs matched within model-only pairs',
    confidence_interval='5000 paired bootstrap samples of8 donors, average across3 seeds; fixed-six F1 and per-class precision/recall recomputed from resampled donor confusion matrices',resource_scope='Current process times per immutable run; shared frozen embedding extraction excluded and linked separately. Training includes inner/outer optimizer fits; inference includes validation and outer prediction/reload checks, not pure test latency.',limitations=['Only8 independent development donors','Overlapping training folds and fixed trained models; bootstrap does not include retraining uncertainty','Development OOF is repeatedly inspected and is not an external final result'],external_final_scores_observed=False))
print(pd.DataFrame(records).groupby('factor').pooled_macro_f1.agg(['mean','std']).to_string());print(pd.DataFrame(effect_rows).to_string(index=False))
