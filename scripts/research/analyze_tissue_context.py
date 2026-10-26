"""V3 context supplement: alias resolution, sample support, within-donor and UMI-stratified OOF errors."""
import itertools
import anndata as ad
from research_core import *

out=REPORT/'tissue_error_analysis';assert not (out/'context_provenance.json').exists();a=ad.read_h5ad(ASSETS/'data/development_full.h5ad',backed='r');obs=a.obs.copy();assert len(obs)==63036
available=[key for key in ['tissue_site','site','tissue'] if key in obs];assert available;selected=available[0];conflicts=[]
for other in available[1:]:
    left=obs[selected].astype(object).fillna('Unknown').astype(str);right=obs[other].astype(object).fillna('Unknown').astype(str);bad=left!=right
    if bad.any():conflicts.append(dict(left=selected,right=other,n_different=int(bad.sum()),examples=obs.loc[bad,[selected,other]].head(10).astype(str).to_dict('records')))
schema=dict(status='resolved' if not conflicts else 'conflicting_aliases_stop',project_commit=git_commit(),canonical_field='tissue_site',selected_field=selected,available_aliases=available,original_field='site',mapping_source='scripts/01_prepare_hcc.py schema_mapping; original metadata value equality already checked in tissue_error_analysis/provenance.json',conflicts=conflicts,no_inferred_labels=True)
write_json(out/'metadata_schema_resolution.json',schema);assert not conflicts,'Record conflicts without overwriting either metadata column'
obs['tissue_site']=obs[selected].astype(str);support=obs.groupby('tissue_site',observed=True).agg(n_cells=('cell_id','size'),n_donors=('patient_id','nunique'),n_samples=('sample_id','nunique'));support.to_csv(out/'support_by_tissue_site.tsv',sep='\t')
donor_support=obs.groupby(['patient_id','tissue_site'],observed=True).agg(n_cells=('cell_id','size'),n_samples=('sample_id','nunique'));donor_support.to_csv(out/'support_by_donor_tissue_site.tsv',sep='\t')
existing=pd.read_csv(out/'donor_tissue_lineage.tsv',sep='\t');existing['error_rate']=1-existing.recall;existing.to_csv(out/'errors_by_donor_tissue_lineage.tsv',sep='\t',index=False)
lineage=existing.groupby(['model','tissue_site','lineage'],observed=True)[['n_truth','n_predicted','true_positive','false_negative','false_positive']].sum().reset_index();lineage['recall']=lineage.true_positive/lineage.n_truth.replace(0,np.nan);lineage['error_rate']=1-lineage.recall;lineage.to_csv(out/'errors_by_tissue_lineage.tsv',sep='\t',index=False)
pairs=[]
for (model,donor,label),part in existing[existing.n_truth>0].groupby(['model','patient_id','lineage'],observed=True):
    part=part.sort_values('tissue_site')
    for (_,left),(_,right) in itertools.combinations(part.iterrows(),2):pairs.append(dict(model=model,patient_id=donor,lineage=label,left_tissue=left.tissue_site,right_tissue=right.tissue_site,left_n_truth=int(left.n_truth),right_n_truth=int(right.n_truth),left_recall=left.recall,right_recall=right.recall,right_minus_left_recall=right.recall-left.recall))
pd.DataFrame(pairs).to_csv(out/'within_donor_tissue_comparisons.tsv',sep='\t',index=False)
obs['umi_quartile']=pd.qcut(obs.n_counts,4,labels=['Q1_low','Q2','Q3','Q4_high']);models=['pca','scgpt','geneformer','geneformer_clcancer','scvi'];sources=[]
for code,name in zip(['A00','A01','A02','A05','A04'],models):
    parts=[]
    for fold in range(4):
        path=REPORT/'runs'/f'{code}_full_fold{fold}_seed42_v2/run.json';run=json.loads(path.read_text());assert sha(run['prediction_path'])==run['prediction_sha256'];parts.append(pd.read_csv(run['prediction_path'],sep='\t'));sources.append(dict(run_id=run['run_id'],prediction_sha256=run['prediction_sha256']))
    pred=pd.concat(parts).set_index('cell_id');assert pred.index.is_unique and set(pred.index)==set(obs.index);pred=pred.loc[obs.index];assert np.array_equal(pred.truth.astype(str),obs.cell_type_coarse.astype(str));obs[name+'_correct']=pred.prediction.to_numpy()==pred.truth.to_numpy()
obs['all_three_wrong']=~obs[[name+'_correct' for name in models[:3]]].any(axis=1)
columns=[name+'_correct' for name in models]+['all_three_wrong']
for keys,name in [(['tissue_site','cell_type_coarse','umi_quartile'],'errors_by_tissue_lineage_umi_quartile.tsv'),(['patient_id','tissue_site','cell_type_coarse','umi_quartile'],'errors_by_donor_tissue_lineage_umi_quartile.tsv')]:
    grouped=obs.groupby(keys,observed=True);frame=grouped[columns].mean().join(grouped.size().rename('n_cells'));frame.to_csv(out/name,sep='\t')
obs[obs.cell_type_coarse.astype(str)=='B'].groupby(['patient_id','tissue_site','umi_quartile'],observed=True)[columns].agg(['mean','size']).to_csv(out/'B_errors_by_donor_tissue_site_umi_quartile.tsv',sep='\t')
context=[];minimum=5
for keys,part in obs.groupby(['patient_id','tissue_site','cell_type_coarse'],observed=True):
    low=part[part.umi_quartile=='Q1_low'];high=part[part.umi_quartile=='Q4_high']
    if min(len(low),len(high))>=minimum:
        row=dict(patient_id=str(keys[0]),tissue_site=str(keys[1]),lineage=str(keys[2]),n_low=len(low),n_high=len(high),population_weight=len(part))
        for column in columns:row[column+'_Q4_minus_Q1']=float(high[column].mean()-low[column].mean())
        context.append(row)
context=pd.DataFrame(context);context.to_csv(out/'within_context_UMI_quartile_differences.tsv',sep='\t',index=False)
standardized={key:float(np.average(context[key],weights=context.population_weight)) for key in context if key.endswith('_Q4_minus_Q1')}
crude={column:float(obs.loc[obs.umi_quartile=='Q4_high',column].mean()-obs.loc[obs.umi_quartile=='Q1_low',column].mean()) for column in columns}
write_json(out/'context_provenance.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),n_cells=len(obs),n_donors=obs.patient_id.nunique(),n_samples=obs.sample_id.nunique(),schema_resolution_sha256=sha(out/'metadata_schema_resolution.json'),original_tissue_provenance_sha256=sha(out/'provenance.json'),sources=sources,quartiles='Descriptive full-development canonical n_counts quartiles, not a filter or new feature',within_context_comparison=dict(context='donor×tissue_site×lineage',minimum_cells_per_extreme_quartile=minimum,n_supported_contexts=len(context),weight='Full-development population size of each supported context; fixed common weighting for Q1 and Q4',crude_Q4_minus_Q1=crude,weighted_supported_context_Q4_minus_Q1=standardized),new_training=False,external_data_used=False,limitations=['Conditioning changes composition but does not establish causal explanation','Only contexts with at least5cells in both extreme quartiles contribute to standardized differences; this subset differs from full population','B strata can be tiny and contain one donor; support always accompanies rates','Author Normal tissue label does not establish a separate healthy-donor cohort']))
print('TISSUE_CONTEXT_SUPPLEMENT_COMPLETE',support.to_string(),json.dumps(standardized),flush=True)
