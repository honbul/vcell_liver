"""Fit deployable frozen annotation bundles on development donors only; no external access."""
import argparse,time
import anndata as ad
import joblib,yaml
from sklearn.decomposition import PCA
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--model',choices=['pca','scgpt','geneformer','geneformer_clcancer'],required=True);p.add_argument('--run-id',required=True);args=p.parse_args()
c=yaml.safe_load((ROOT/'configs/research/protocol_annotation_full.yaml').read_text());code={'pca':'A00','scgpt':'A01','geneformer':'A02','geneformer_clcancer':'A05'}[args.model]
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True);raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True);r=Resources();start=time.monotonic()
a=ad.read_h5ad(DATA/c['dataset_path']);assert a.n_obs==63036 and not a.obs.patient_id.isin(c['legacy_donors']).any()
training=np.sort(np.random.default_rng(42).choice(a.n_obs,12000,replace=False));assert a.obs.iloc[training].patient_id.nunique()==8
pd.DataFrame({'cell_id':a.obs_names[training]}).to_csv(raw/'training_cell_ids.tsv',sep='\t',index=False)
grid=[];sources=[]
for fold in range(4):
    path=REPORT/'runs'/f'{code}_full_fold{fold}_seed42_v2/run.json';run=json.loads(path.read_text());assert run['status']=='executed'
    grid.extend(run['head_selection']['validation_grid']);sources.append(dict(run_id=run['run_id'],sha256=sha(path)))
scores=[]
for C in c['head_C']:
    for weight in c['head_class_weight']:
        values=[row['validation_macro_f1'] for row in grid if row['C']==C and row['class_weight']==weight];assert len(values)==4
        scores.append(dict(C=C,class_weight=weight,mean_inner_validation_macro_f1=float(np.mean(values))))
selected=max(scores,key=lambda row:row['mean_inner_validation_macro_f1'])
preprocessor=None;feature_metadata=None
if args.model=='pca':
    x=a.layers['counts'][training].astype(np.float32);x=x.multiply((10000/np.maximum(np.asarray(x.sum(1)).ravel(),1))[:,None]).tocsr();x.data=np.log1p(x.data)
    mu=np.asarray(x.mean(0)).ravel();var=np.asarray(x.power(2).mean(0)).ravel()-mu**2;hvg=np.argsort(-var,kind='stable')[:2000];small=x[:,hvg].toarray()
    scale=StandardScaler().fit(small);small=scale.transform(small);pca=PCA(n_components=50,random_state=42).fit(small);features=pca.transform(small)
    preprocessor=dict(hvg_ensembl=a.var.ensembl_id.iloc[hvg].fillna('').astype(str).tolist(),hvg_gene_symbols=a.var.gene_symbol.iloc[hvg].fillna('').astype(str).tolist(),hvg_var_names=a.var_names[hvg].astype(str).tolist(),scaler=scale,pca=pca,normalization_target=10000)
else:
    path=ASSETS/'features'/f'{args.model}_GSE149614_development_full.npz';d=np.load(path);assert np.array_equal(d['cell_ids'],a.obs_names.astype(str));features=d['X'][training];feature_metadata=json.loads(path.with_suffix('.json').read_text());assert sha(path)==feature_metadata['feature_sha256']
preprocessing=time.monotonic()-start;start=time.monotonic();scale=StandardScaler().fit(features);features=scale.transform(features)
head=LogisticRegression(C=selected['C'],class_weight=selected['class_weight'],max_iter=2000,random_state=42).fit(features,a.obs.iloc[training].cell_type_coarse.astype(str));training_time=time.monotonic()-start
bundle=dict(schema_version=1,model_id=args.model,labels=c['labels'],scaler=scale,classifier=head,preprocessor=preprocessor,feature_metadata=feature_metadata,training_manifest_sha256=sha(raw/'training_cell_ids.tsv'),head_selection=selected)
joblib.dump(bundle,raw/'annotation_bundle.joblib');write_json(out/'selection.json',dict(rule='Largest unweighted mean inner-validation macroF1 over four pre-existing development folds; ties retain fixed C/weight grid order',grid=scores,selected=selected,sources=sources,external_data_accessed=False))
res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='annotation_final_fit',regime='inductive',model_id=args.model,project_commit=git_commit(),source_sha256=sha(Path(__file__)),dataset_version=c['dataset_id'],data_sha256=sha(DATA/c['dataset_path']),n_training_cells=len(training),training_donors=sorted(a.obs.iloc[training].patient_id.unique()),training_manifest_sha256=bundle['training_manifest_sha256'],seed=42,head_selection=selected,feature_metadata=feature_metadata,
    trainable_parameters=int(head.coef_.size+head.intercept_.size),feature_dimension=features.shape[1],bundle_path=str(raw/'annotation_bundle.joblib'),bundle_sha256=sha(raw/'annotation_bundle.joblib'),wall_preprocessing_seconds=preprocessing,wall_training_seconds=training_time,wall_inference_seconds=0,wall_total_seconds=res.pop('wall_seconds'),external_data_accessed=False,**res))
print('FINAL_ANNOTATION_FIT',args.model,selected,flush=True)
