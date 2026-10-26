"""Reuse complete development OOF predictions with the original site→tissue_site mapping."""
import anndata as ad
from research_core import *

out=REPORT/'tissue_error_analysis';assert not (out/'provenance.json').exists();out.mkdir(exist_ok=True)
dataset=ASSETS/'data/development_full.h5ad';a=ad.read_h5ad(dataset,backed='r');obs=a.obs.copy();assert len(obs)==63036 and not obs.patient_id.isin(['HCC02','HCC09']).any()
source=Path('/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614/GSE149614_HCC.metadata.updated.txt.gz');original=pd.read_csv(source,sep='\t',index_col='Cell');assert original.index.is_unique
assert np.array_equal(original.loc[obs.index,'site'].astype(str),obs.tissue_site.astype(str))
assert obs.tissue_site.notna().all();labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];records=[];tissue_rows=[];confusions=[];sources=[]
for code,model in [('A00','pca'),('A01','scgpt'),('A02','geneformer'),('A05','geneformer_clcancer'),('A04','scvi')]:
    parts=[]
    for fold in range(4):
        path=REPORT/'runs'/f'{code}_full_fold{fold}_seed42_v2/run.json';run=json.loads(path.read_text());assert run['status']=='executed' and sha(run['prediction_path'])==run['prediction_sha256'];parts.append(pd.read_csv(run['prediction_path'],sep='\t'));sources.append(dict(run_id=run['run_id'],run_sha256=sha(path),prediction_sha256=run['prediction_sha256']))
    pred=pd.concat(parts).set_index('cell_id');assert pred.index.is_unique and set(pred.index)==set(obs.index);pred=pred.loc[obs.index];assert np.array_equal(pred.truth.astype(str),obs.cell_type_coarse.astype(str))
    frame=obs[['patient_id','tissue_site','cell_type_coarse']].copy();frame['prediction']=pred.prediction.to_numpy();frame['correct']=frame.cell_type_coarse.astype(str)==frame.prediction
    for (donor,tissue),part in frame.groupby(['patient_id','tissue_site'],observed=True):
        for lineage in labels:
            truth=part.cell_type_coarse.astype(str)==lineage;estimated=part.prediction==lineage;tp=int((truth&estimated).sum());n=int(truth.sum());p=int(estimated.sum())
            records.append(dict(model=model,patient_id=str(donor),tissue_site=str(tissue),lineage=lineage,n_truth=n,n_predicted=p,true_positive=tp,false_negative=n-tp,false_positive=p-tp,recall=tp/n if n else None,precision=tp/p if p else None,n_group=len(part)))
    for tissue,part in frame.groupby('tissue_site',observed=True):
        metrics=metric_summary(part.cell_type_coarse,part.prediction,None,labels);tissue_rows.append(dict(model=model,tissue_site=str(tissue),n_donors=part.patient_id.nunique(),**{k:v for k,v in metrics.items() if not isinstance(v,(list,dict))}))
        counts=part.groupby(['patient_id','cell_type_coarse','prediction'],observed=True).size()
        confusions.extend(dict(model=model,tissue_site=str(tissue),patient_id=str(donor),truth=str(truth),prediction=str(estimate),n_cells=int(n)) for (donor,truth,estimate),n in counts.items())
pd.DataFrame(records).to_csv(out/'donor_tissue_lineage.tsv',sep='\t',index=False);pd.DataFrame(tissue_rows).to_csv(out/'pooled_by_tissue.tsv',sep='\t',index=False);pd.DataFrame(confusions).to_csv(out/'donor_tissue_confusion.tsv',sep='\t',index=False)
obs.groupby(['patient_id','tissue_site','cell_type_coarse'],observed=True).size().rename('n_cells').to_csv(out/'population_support.tsv',sep='\t')
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),n_cells=len(obs),n_donors=obs.patient_id.nunique(),metadata_mapping=dict(original_field='site',canonical_field='tissue_site',source_path=str(source),source_sha256=sha(source),n_exact_value_matches=len(obs)),tissue_support=obs.tissue_site.value_counts().to_dict(),sources=sources,new_fitting=False,external_data_used=False,limitations=['Descriptive repeatedly inspected development OOF; no model selection or retraining','Tissue sites and lineage support differ by donor; pooled tissue scores do not identify an independent tissue effect','Absent lineages contribute0 to fixed-six macroF1; inspect support and lineage recall','Source Normal is the author tissue label, not proof of healthy independent donors']))
print('TISSUE_ERROR_ANALYSIS_COMPLETE',len(records),flush=True)
