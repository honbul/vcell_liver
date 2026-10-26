"""One fixed-seed grouped-development KG diagnostic, with actual matched prediction checks."""
from sklearn.metrics import confusion_matrix
from research_core import *

labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];conditions=['embedding_only','real128','real32','random32','pathway32'];donors=['HCC01','HCC03','HCC04','HCC05','HCC06','HCC07','HCC08','HCC10'];counts=np.zeros((5,8,6,6),dtype=np.int64);sources=[];class_rows=[];frames={name:[] for name in conditions}
for fold in range(4):
    base_id='D00_full4096_fold0_seed42_v1' if fold==0 else f'D04_control_full4096_fold{fold}_seed42_v1';reference=None;base=json.loads((REPORT/'runs'/base_id/'run.json').read_text())
    for ci,condition in enumerate(conditions):
        rid=base_id if condition=='embedding_only' else f'D04_{condition}_full4096_fold{fold}_seed42_v1';path=REPORT/'runs'/rid/'run.json';run=json.loads(path.read_text());assert run['status']=='executed' and run['seed']==42 and run['fold']==fold and run['n_outer_training']==4096
        pred=pd.read_csv(run['prediction_path'],sep='\t');assert sha(run['prediction_path'])==run['prediction_sha256'] and pred.cell_id.is_unique
        if reference is None:reference=pred[['cell_id','patient_id','truth']]
        else:
            assert pred[['cell_id','patient_id','truth']].equals(reference) and run['training_manifest_sha256']==base['training_manifest_sha256'] and run['validation_manifest_sha256']==base['validation_manifest_sha256']
            for key in ['selected_C','selected_class_weight']:assert run['head_selection'][key]==base['head_selection'][key]
        frames[condition].append(pred);sources.append(dict(condition=condition,run_id=rid,run_metadata_sha256=sha(path),prediction_sha256=run['prediction_sha256']))
        for donor,part in pred.groupby('patient_id'):counts[ci,donors.index(donor)]+=confusion_matrix(part.truth,part.prediction,labels=labels)
assert np.all(counts.sum((1,2,3))==63036)
def f1(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2);return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)
pooled=f1(counts.sum(1));donor_scores=f1(counts);rng=np.random.default_rng(42);draws=rng.integers(0,8,size=(5000,8));weights=np.stack([(draws==i).sum(1) for i in range(8)],axis=1);boot=f1(np.einsum('bd,mdij->bmij',weights,counts));effects=[]
for title,left,right in [('Compression gain','real32','real128'),('Real prior vs matched random','real32','random32'),('Compressed prior vs embedding only','real32','embedding_only')]:
    li=conditions.index(left);ri=conditions.index(right);samples=boot[:,li]-boot[:,ri];effects.append(dict(comparison=title,pooled_macro_f1_difference=float(pooled[li]-pooled[ri]),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),n_nonnegative_donors=int((donor_scores[li]>=donor_scores[ri]).sum()),n_donors=8))
out=REPORT/'kg_compression_diagnostic';out.mkdir(exist_ok=True);pd.DataFrame([dict(condition=name,n_evaluation=63036,seed=42,pooled_macro_f1=float(pooled[i])) for i,name in enumerate(conditions)]).to_csv(out/'pooled_metrics.tsv',sep='\t',index=False);pd.DataFrame(effects).to_csv(out/'paired_donor_effects.tsv',sep='\t',index=False)
for name,parts in frames.items():
    frame=pd.concat(parts);assert frame.cell_id.is_unique and frame.patient_id.nunique()==8 and len(frame)==63036;report=classification_report(frame.truth,frame.prediction,labels=labels,output_dict=True,zero_division=0)
    class_rows.extend(dict(condition=name,label=label,**report[label]) for label in labels)
pd.DataFrame(class_rows).to_csv(out/'per_class.tsv',sep='\t',index=False);pd.DataFrame([dict(condition=name,patient_id=donor,macro_f1=float(donor_scores[i,j])) for i,name in enumerate(conditions) for j,donor in enumerate(donors)]).to_csv(out/'by_donor.tsv',sep='\t',index=False)
success=effects[0]['pooled_macro_f1_difference']>0 and effects[1]['pooled_macro_f1_difference']>0 and effects[1]['n_nonnegative_donors']>=5
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),hypothesis_sha256=sha(ROOT/'configs/research/hypothesis_DH3_kg_compression_v1.json'),sources=sources,n_development_cells=63036,n_donors=8,n_seeds=1,n_executed_heads=20,preregistered_point_rule_met=success,interpretation='Exploratory fixed32 compression diagnostic; single seed, no promotion into current D1/partial confirmation or fixed external model batch',uncertainty='5000 paired donor bootstrap samples, fixed fitted models; overlapping training folds and no retraining/seed uncertainty'))
print('KG_COMPRESSION_DIAGNOSTIC_COMPLETE',pd.DataFrame(effects).to_string(index=False),flush=True)
