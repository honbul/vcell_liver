"""Donor-paired intervals and actual pooled calibration from existing complete OOF predictions."""
import argparse
from sklearn.metrics import confusion_matrix
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--regime',choices=['inductive','query_adaptive'],required=True);args=p.parse_args();labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];donors=['HCC01','HCC03','HCC04','HCC05','HCC06','HCC07','HCC08','HCC10']
patterns={name:f'{code}_full_fold{{fold}}_seed42_v2' for name,code in [('pca','A00'),('scgpt','A01'),('geneformer','A02'),('geneformer_clcancer','A05'),('scvi','A04')]}
if args.regime=='query_adaptive':patterns={'LiverCT_native':'S00_liverct_full_fold{fold}_seed42_v4','LiverCT_SCANVI_control':'S00_liverct_full_fold{fold}_seed42_v4_scanvi_control'}
counts=[];sources=[];rows=[];reference=None
for name,pattern in patterns.items():
    parts=[];probabilities=[]
    for fold in range(4):
        rid=pattern.format(fold=fold);path=REPORT/'runs'/rid/'run.json';run=json.loads(path.read_text());frame=pd.read_csv(run['prediction_path'],sep='\t');assert run['status']=='executed' and sha(run['prediction_path'])==run['prediction_sha256'];parts.append(frame);source=dict(run_id=rid,run_metadata_sha256=sha(path),prediction_sha256=run['prediction_sha256'])
        if args.regime=='inductive':
            pp=ASSETS/'runs'/rid/'probabilities.npz';d=np.load(pp);assert np.array_equal(d['cell_ids'],frame.cell_id.astype(str)) and list(d['labels'])==labels;probabilities.append(d['probabilities']);source['probability_sha256']=sha(pp)
        sources.append(source)
    frame=pd.concat(parts,ignore_index=True);assert len(frame)==63036 and frame.cell_id.is_unique and set(frame.patient_id)==set(donors);evaluated=frame[['cell_id','patient_id','truth']]
    if reference is None:reference=evaluated
    else:assert evaluated.equals(reference),'Methods must have identical ordered population and truth'
    counts.append(np.stack([confusion_matrix(frame.loc[frame.patient_id==donor,'truth'],frame.loc[frame.patient_id==donor,'prediction'],labels=labels+['Unknown']) for donor in donors]));prob=np.concatenate(probabilities) if probabilities else None;summary=metric_summary(frame.truth,frame.prediction,prob,labels);rows.append(dict(model=name,**{key:value for key,value in summary.items() if not isinstance(value,(list,dict))}))
counts=np.stack(counts)
def f1(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)[...,:6];den=cm.sum(-1)[...,:6]+cm.sum(-2)[...,:6];return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(-1)
point=f1(counts.sum(1));assert np.allclose(point,[row['macro_f1'] for row in rows],atol=1e-12,rtol=0)
rng=np.random.default_rng(42);draws=rng.integers(0,8,size=(5000,8));weights=np.stack([(draws==i).sum(1) for i in range(8)],axis=1);boot=f1(np.einsum('bd,mdij->bmij',weights,counts));names=list(patterns);pairs=[('scgpt','pca'),('geneformer','pca'),('geneformer_clcancer','geneformer'),('scvi','pca')] if args.regime=='inductive' else [('LiverCT_native','LiverCT_SCANVI_control')];comparisons=[]
for candidate,base in pairs:
    ci=names.index(candidate);bi=names.index(base);delta=boot[:,ci]-boot[:,bi];comparisons.append(dict(candidate=candidate,reference=base,pooled_macro_f1_difference=float(point[ci]-point[bi]),lower95=float(np.quantile(delta,.025)),upper95=float(np.quantile(delta,.975)),n_donors=8))
out=REPORT/f'annotation_uncertainty_{args.regime}';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'pooled_metrics_and_calibration.tsv',sep='\t',index=False);pd.DataFrame(comparisons).to_csv(out/'paired_donor_bootstrap.tsv',sep='\t',index=False);pd.DataFrame([dict(model=name,macro_f1=float(point[i]),lower95=float(np.quantile(boot[:,i],.025)),upper95=float(np.quantile(boot[:,i],.975))) for i,name in enumerate(names)]).to_csv(out/'absolute_donor_bootstrap.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),regime=args.regime,n_evaluation=63036,n_donors=8,n_seeds=1,sources=sources,uncertainty='5000 paired donor bootstrap samples; fixed-six F1 recomputed from seven-column confusion matrices so ontology Unknown predictions remain penalized',calibration='Actual pooled probability vectors across folds, not an average of fold ECE; native hard-label pipelines have no reconstructed/calibrated coarse probabilities',selection='Post-selection descriptive development summary; no external data, no training or retuning',limitations=['Only8 donors; training folds overlap','Bootstrap holds trained models fixed and does not estimate retraining uncertainty','Reference label amount and query adaptation differ across regimes; compare within each table']))
print('ANNOTATION_DONOR_UNCERTAINTY',args.regime,pd.DataFrame(comparisons).to_string(index=False),flush=True)
