"""Nested donor evaluation; PCA refitted per fold; pretrained legacy features only for pilot."""
import argparse,json,sys,time
from pathlib import Path
from research_core import *
import anndata as ad
import yaml
from sampling import sampling_roles
from sklearn.decomposition import PCA

def log_counts(a,target):
    x=a.layers['counts'].astype(np.float32)
    x=x.multiply((target/np.maximum(np.asarray(x.sum(1)).ravel(),1))[:,None]).tocsr();x.data=np.log1p(x.data)
    return x

def fit_pca(x,train,c):
    mu=np.asarray(x[train].mean(0)).ravel();var=np.asarray(x[train].power(2).mean(0)).ravel()-mu**2
    hvg=np.argsort(-var,kind='stable')[:c['n_hvg']]
    small=x[:,hvg].toarray();scale=StandardScaler().fit(small[train]);small=scale.transform(small)
    pca=PCA(n_components=min(c['n_pca'],len(train)-1,len(hvg)),random_state=c['seed']).fit(small[train])
    return pca.transform(small),dict(hvg=hvg,scaler=scale,pca=pca)

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--model',choices=['pca','scgpt','geneformer','scfoundation','geneformer_clcancer','scvi'],required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--run-id',required=True);p.add_argument('--seed',type=int);p.add_argument('--data-variant',choices=['D0_natural','D1_donor_balanced','D2_donor_class_balanced'],default='D0_natural');p.add_argument('--fixed-head-run');args=p.parse_args()
c=yaml.safe_load(Path(args.config).read_text())
if args.seed is not None:c['seed']=args.seed
c['data_variant']=args.data_variant
if args.fixed_head_run:
    baseline=json.loads((REPORT/'runs'/args.fixed_head_run/'run.json').read_text())
    assert baseline['seed']==c['seed'] and baseline['fold']==args.fold
    selected=baseline['head_selection'];c['head_C']=[selected['selected_C']];c['head_class_weight']=[selected['selected_class_weight']]
r=Resources();start=time.monotonic()
assert not (REPORT/'runs'/args.run_id/'run.json').exists(),'Immutable run ID already exists'
a=ad.read_h5ad(DATA/c['dataset_path']);roles=sampling_roles(a.obs,c,args.fold,args.data_variant)
y=a.obs.cell_type_coarse.astype(str).to_numpy();upstream=None;checkpoint=None;feature_metadata=None
if args.model=='pca':
    x=log_counts(a,c['normalization_target']);inner,inner_model=fit_pca(x,roles['inner_training'],c)
    outer=roles['outer_training'];final,preprocessor=fit_pca(x,outer,c)
    feature_source='fold-specific PCA; separate inner and outer preprocessing fits'
else:
    source=c.get('pilot_feature_sources',{}).get(args.model)
    if source:
        d=np.load(DATA/source);lookup={cell:i for i,cell in enumerate(d['cell_ids'])}
        assert all(cell in lookup for cell in a.obs_names)
        inner=final=d['X'][[lookup[cell] for cell in a.obs_names]]
        feature_source=str(DATA/source)
    else:
        source=ASSETS/'features'/f"{args.model}_{c['dataset_id']}.npz"
        d=np.load(source);assert np.array_equal(d['cell_ids'],a.obs_names.to_numpy(dtype=str))
        inner=final=d['X'];feature_source=str(source)
        feature_metadata=json.loads(source.with_suffix('.json').read_text())
        upstream=feature_metadata.get('upstream_commit');checkpoint=feature_metadata.get('checkpoint_sha256')
    preprocessor=None
    if args.model=='scgpt':
        upstream='cebd6fae655b9c585a4807daa3ac31bb764f06b4';checkpoint=sha(DATA/'checkpoints/scgpt-whole-human/best_model.pt')
    elif args.model=='geneformer':
        upstream='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5';checkpoint=sha(DATA/'checkpoints/Geneformer-V2-104M/model.safetensors')
preprocess_seconds=time.monotonic()-start
pred,prob,head,selection,training_seconds,inference_seconds=fit_head(inner,final,y,roles,c,c['seed'])
head['preprocessor']=preprocessor
raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
for role,indices in roles.items():
    pd.DataFrame({'cell_id':a.obs_names[indices]}).to_csv(raw/f'{role}_cell_ids.tsv',sep='\t',index=False)
resources=r.finish()
save_evaluation(a,roles['evaluation'],pred,prob,c,args.run_id,dict(model_id=args.model,fold=args.fold,feature_dimension=final.shape[1],feature_source=feature_source,
    upstream_commit=upstream,base_checkpoint_revision=upstream,checkpoint_sha256=checkpoint,trainable_parameters=int(head['classifier'].coef_.size+head['classifier'].intercept_.size),
    dataset_sha256=sha(DATA/c['dataset_path']),n_inner_training=len(roles['inner_training']),n_inner_validation=len(roles['inner_validation']),n_outer_training=len(roles['inner_training'])+len(roles['inner_validation']),n_evaluation=len(roles['evaluation']),
    training_donors=sorted(a.obs.iloc[np.concatenate([roles['inner_training'],roles['inner_validation']])].patient_id.unique()),
    validation_donors=sorted(a.obs.iloc[roles['inner_validation']].patient_id.unique()),evaluation_donors_or_perturbations=sorted(a.obs.iloc[roles['evaluation']].patient_id.unique()),
    wall_preprocessing_seconds=preprocess_seconds,wall_training_seconds=training_seconds,wall_inference_seconds=inference_seconds,wall_total_seconds=resources.pop('wall_seconds'),
    feature_inference_note='Frozen feature cache reused; head-only time excludes extraction; extraction metadata linked' if args.model!='pca' else 'PCA included in preprocessing time',feature_extraction_metadata=feature_metadata,
    metric_protocol=c['primary_metric'],head_selection=selection,head_hyperparameter_source=args.fixed_head_run,training_manifest_sha256=sha(raw/'outer_training_cell_ids.tsv'),validation_manifest_sha256=sha(raw/'inner_validation_cell_ids.tsv'),failure_or_limitations=['development OOF, not a fresh blind test','legacy HCC02/HCC09 excluded','pretraining overlap unknown'],**resources),head=head)
