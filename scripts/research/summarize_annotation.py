"""Aggregate actual disjoint donor OOF predictions, without treating them as a blind test."""
import argparse
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['pilot','full'],default='pilot');args=p.parse_args()
labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK']
rows=[];donors=[];classes=[];sources=[]
for model,code in [('pca','A00'),('scgpt','A01'),('geneformer','A02'),('geneformer_clcancer','A05'),('scvi','A04')]:
    files=[REPORT/'runs'/f'{code}_{args.dataset}_fold{fold}_seed42_v2/run.json' for fold in range(4)]
    if not all(f.exists() for f in files):continue
    runs=[json.loads(f.read_text()) for f in files]
    assert all(r['status']=='executed' and r['regime']=='inductive' for r in runs)
    frames=[pd.read_csv(r['prediction_path'],sep='\t') for r in runs];frame=pd.concat(frames,ignore_index=True)
    assert not frame.cell_id.duplicated().any() and frame.patient_id.nunique()==8
    assert not frame.patient_id.isin(['HCC02','HCC09']).any()
    y,pred=frame.truth,frame.prediction
    rows.append(dict(model=model,n_cells=len(frame),pooled_macro_f1=f1_score(y,pred,labels=labels,average='macro',zero_division=0),balanced_accuracy=balanced_accuracy_score(y,pred),mean_fold_macro_f1=np.mean([r['metrics']['macro_f1'] for r in runs]),development_only=True))
    report=classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)
    classes.extend(dict(model=model,label=label,**report[label]) for label in labels)
    for donor,sub in frame.groupby('patient_id'):
        donors.append(dict(model=model,patient_id=donor,n_cells=len(sub),macro_f1_common_ontology=f1_score(sub.truth,sub.prediction,labels=labels,average='macro',zero_division=0),n_errors=int((sub.truth!=sub.prediction).sum())))
    sources.extend(dict(run_id=r['run_id'],prediction_sha256=sha(r['prediction_path']),run_json_sha256=sha(f)) for r,f in zip(runs,files))
assert rows,'No complete four-fold execution found'
out=REPORT/f'annotation_{args.dataset}_summary';out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/'pooled_metrics.tsv',sep='\t',index=False)
pd.DataFrame(classes).to_csv(out/'per_class.tsv',sep='\t',index=False)
pd.DataFrame(donors).to_csv(out/'by_donor.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),dataset=args.dataset,sources=sources,interpretation='Repeatedly inspected development OOF; no external final score; pooled F1 is not average fold F1'))
print(pd.DataFrame(rows).to_string(index=False))
