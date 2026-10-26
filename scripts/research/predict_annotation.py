"""Label-free inference with a fitted annotation bundle; never fits a query statistic."""
import argparse,time
import anndata as ad
import joblib
from scipy import sparse
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--bundle',required=True);p.add_argument('--features');p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);assert not out.exists(),'Prediction destinations are immutable';out.mkdir(parents=True);r=Resources();start=time.monotonic()
a=ad.read_h5ad(args.input);bundle=joblib.load(args.bundle);counts=a.layers['counts'] if 'counts' in a.layers else a.X
counts=sparse.csr_matrix(counts,dtype=np.float32);assert a.obs_names.is_unique and a.var_names.is_unique and np.isfinite(counts.data).all() and (counts.data>=0).all() and np.all(counts.data==np.round(counts.data))
coverage=None
if bundle['model_id']=='pca':
    prep=bundle['preprocessor'];gene_ids=a.var.ensembl_id.fillna('').astype(str).to_numpy() if 'ensembl_id' in a.var else a.var_names.astype(str).to_numpy();assert not pd.Series(gene_ids[gene_ids!='']).duplicated().any()
    lookup={gene:i for i,gene in enumerate(gene_ids) if gene};hvg=prep['hvg_ensembl'];present=[(j,lookup[gene]) for j,gene in enumerate(hvg) if gene in lookup]
    coverage=dict(n_reference_hvg=len(hvg),n_present_hvg=len(present),missing_gene_policy='Raw zero fill before frozen training scaling; original provided query-library sum retained')
    assert present,'No reference HVGs overlap the query Ensembl IDs'
    library=np.asarray(counts.sum(1)).ravel();assert (library>0).all();normalized=counts.multiply((prep['normalization_target']/library)[:,None]).tocsr();normalized.data=np.log1p(normalized.data)
    small=np.zeros((a.n_obs,len(hvg)),dtype=np.float32);columns,source=zip(*present);small[:,columns]=normalized[:,source].toarray();features=prep['pca'].transform(prep['scaler'].transform(small))
else:
    assert args.features,'Supply frozen checkpoint features extracted from this unlabeled query'
    d=np.load(args.features);assert np.array_equal(d['cell_ids'],a.obs_names.astype(str));features=d['X'];meta=json.loads(Path(args.features).with_suffix('.json').read_text())
    assert meta['model_id']==bundle['model_id'] and meta['checkpoint_sha256']==bundle['feature_metadata']['checkpoint_sha256'] and meta['input_data_sha256']==sha(args.input) and meta['feature_sha256']==sha(args.features)
    coverage=meta['coverage']
preprocess=time.monotonic()-start;start=time.monotonic();x=bundle['scaler'].transform(features);head=bundle['classifier'];prediction=head.predict(x);native=head.predict_proba(x);labels=bundle['labels'];prob=np.zeros((a.n_obs,len(labels)))
for j,label in enumerate(head.classes_):prob[:,labels.index(label)]=native[:,j]
frame=pd.DataFrame(dict(cell_id=a.obs_names.astype(str),prediction=prediction,confidence=prob.max(1)))
if 'patient_id' in a.obs:frame['patient_id']=a.obs.patient_id.to_numpy()
frame.to_csv(out/'predictions.tsv',sep='\t',index=False);np.savez_compressed(out/'probabilities.npz',cell_ids=a.obs_names.to_numpy(dtype=str),probabilities=prob,labels=np.array(labels))
res=r.finish();write_json(out/'inference.json',dict(status='executed',task='annotation_inference',regime='inductive',model_id=bundle['model_id'],project_commit=git_commit(),source_sha256=sha(Path(__file__)),input_data_sha256=sha(args.input),bundle_path=args.bundle,bundle_sha256=sha(args.bundle),feature_path=args.features,n_cells=a.n_obs,feature_dimension=features.shape[1],prediction_sha256=sha(out/'predictions.tsv'),probability_sha256=sha(out/'probabilities.npz'),coverage=coverage,
    allowed_query_information='Own raw counts and gene identifiers; cell ID/patient ID copied for output only. No query labels read, no query fit or adaptation',wall_preprocessing_seconds=preprocess,wall_inference_seconds=time.monotonic()-start,wall_training_seconds=0,wall_total_seconds=res.pop('wall_seconds'),**res))
print('ANNOTATION_INFERENCE_COMPLETE',bundle['model_id'],a.n_obs,flush=True)
