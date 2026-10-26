"""Descriptive development-only QC, mapping, token capacity and shared-error audit."""
import pickle
import anndata as ad
from research_core import *

a=ad.read_h5ad(ASSETS/'data/development_full.h5ad');assert a.n_obs==63036 and not a.obs.patient_id.isin(['HCC02','HCC09']).any()
out=REPORT/'development_diagnostics';out.mkdir(exist_ok=True)
x=a.layers['counts'].tocsr();library=np.asarray(x.sum(1)).ravel();detected=np.diff(x.indptr)
symbols=a.var.gene_symbol.fillna('').astype(str).to_numpy();ensembl=a.var.ensembl_id.fillna('').astype(str).to_numpy()
vocab=json.loads((DATA/'checkpoints/scgpt-whole-human/vocab.json').read_text())
with open(ROOT/'third_party/Geneformer/geneformer/token_dictionary_gc104M.pkl','rb') as f:tokens=pickle.load(f)
gpt=np.isin(symbols,list(vocab));gf=np.isin(ensembl,list(tokens));mito=np.char.startswith(symbols.astype(str),'MT-')
frame=a.obs.copy();frame['library_counts']=library;frame['detected_genes']=detected
frame['mitochondrial_fraction']=np.asarray(x[:,mito].sum(1)).ravel()/np.maximum(library,1)
for name,mask in [('symbol_scgpt',gpt),('ensembl_geneformer',gf)]:
    frame[name+'_count_fraction']=np.asarray(x[:,mask].sum(1)).ravel()/np.maximum(library,1)
    frame[name+'_expressed_genes']=np.asarray((x[:,mask]>0).sum(1)).ravel()
frame['scgpt_exceeds_1199_gene_capacity']=frame.symbol_scgpt_expressed_genes>1199
frame['geneformer_exceeds_4094_gene_capacity']=frame.ensembl_geneformer_expressed_genes>4094
metrics=['library_counts','detected_genes','mitochondrial_fraction','symbol_scgpt_count_fraction','ensembl_geneformer_count_fraction','scgpt_exceeds_1199_gene_capacity','geneformer_exceeds_4094_gene_capacity']
for key in ['patient_id','cell_type_coarse']:
    frame.groupby(key,observed=True)[metrics].agg(['mean','median','min','max']).to_csv(out/f'qc_mapping_token_capacity_by_{key}.tsv',sep='\t')
    frame.groupby(key,observed=True).size().rename('n_cells').to_csv(out/f'support_by_{key}.tsv',sep='\t')
ids=a.obs_names.astype(str);sources=[]
for code,name in [('A00','pca'),('A01','scgpt'),('A02','geneformer'),('A05','geneformer_clcancer'),('A04','scvi')]:
    parts=[]
    for fold in range(4):
        path=REPORT/'runs'/f'{code}_full_fold{fold}_seed42_v2/run.json';run=json.loads(path.read_text());assert run['status']=='executed' and sha(run['prediction_path'])==run['prediction_sha256']
        parts.append(pd.read_csv(run['prediction_path'],sep='\t'));sources.append(dict(run_id=run['run_id'],prediction_sha256=run['prediction_sha256']))
    pred=pd.concat(parts).set_index('cell_id');assert pred.index.is_unique and set(pred.index)==set(ids);pred=pred.loc[ids]
    assert np.array_equal(pred.truth.astype(str),frame.cell_type_coarse.astype(str))
    frame[name+'_correct']=pred.prediction.to_numpy()==pred.truth.to_numpy();frame[name+'_confidence']=pred.confidence.to_numpy()
frame['all_three_correct']=frame[['pca_correct','scgpt_correct','geneformer_correct']].all(axis=1)
frame['all_three_wrong']=~frame[['pca_correct','scgpt_correct','geneformer_correct']].any(axis=1)
error_cols=[name+'_correct' for name in ['pca','scgpt','geneformer','geneformer_clcancer','scvi']]+['all_three_correct','all_three_wrong']
for group in ['patient_id','cell_type_coarse']:
    frame.groupby(group,observed=True)[error_cols].mean().to_csv(out/f'error_overlap_by_{group}.tsv',sep='\t')
for field in ['library_counts','detected_genes','mitochondrial_fraction']:
    frame[field+'_descriptive_quartile']=pd.qcut(frame[field],4,duplicates='drop')
    frame.groupby(field+'_descriptive_quartile',observed=True)[error_cols].agg(['mean','size']).to_csv(out/f'errors_by_{field}.tsv',sep='\t')
context={}
for field in ['stage','virus','site','tissue','platform','doublet_score']:
    if field in frame:
        pd.crosstab(frame.patient_id,frame[field],dropna=False).to_csv(out/f'donor_by_{field}.tsv',sep='\t')
        context[field]='Observed metadata, descriptive only; donor confounding retained'
    else:context[field]='Not present as this explicit column in the canonical development obs; no inferred replacement'
edges=pd.read_csv(ROOT/'manifests/kg_edges_comparison.tsv',sep='\t');sets={p:set(g.head_id) for p,g in edges.groupby('tail_id')};names=sorted(sets);overlaps=[]
for i,left in enumerate(names):
    for right in names[i+1:]:overlaps.append(len(sets[left]&sets[right])/len(sets[left]|sets[right]))
pd.DataFrame([dict(pathway_id=k,n_genes=len(v)) for k,v in sets.items()]).to_csv(out/'legacy_kg_pathway_sizes.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),dataset_sha256=sha(ASSETS/'data/development_full.h5ad'),n_cells=a.n_obs,n_genes=a.n_vars,sources=sources,metadata_availability=context,
    n_scgpt_vocabulary_genes=int(gpt.sum()),n_geneformer_vocabulary_genes=int(gf.sum()),duplicate_nonempty_ensembl_ids=int(pd.Series(ensembl[ensembl!='']).duplicated().sum()),
    token_capacity='Potential overflow from observed vocabulary genes before truncation/sampling, not an observed token inclusion trace. Geneformer duplicate-ID collapse can lower this count. scGPT reserves one CLS slot; Geneformer reserves CLS/EOS.',
    quartiles='Descriptive full-development quartiles only, never a training filter or model-selection threshold',
    legacy_kg=dict(n_pathways=len(sets),n_edges=len(edges),pairwise_jaccard_quantiles=np.quantile(overlaps,[0,.25,.5,.75,.95,1]).tolist(),scope='Legacy comparison cohort; do not treat it as a new blind result or full-development KG ablation'),
    limitations=['Descriptive associations do not identify causes','No inferred doublet labels or platform assignments','No external final predictions or labels used','These diagnostics do not change the preregistered D1/partial confirmation candidates']))
print('DEVELOPMENT_DIAGNOSTICS_COMPLETE',flush=True)
