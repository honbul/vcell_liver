"""Strict inductive scVI: independent train-only HVG/model fits for inner and outer roles."""
import argparse,time
import anndata as ad
import scanpy as sc
import scvi
import torch,yaml
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/research/protocol_annotation_full.yaml');p.add_argument('--fold',type=int,required=True);p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=100);args=p.parse_args()
c=yaml.safe_load(Path(args.config).read_text());assert not (REPORT/'runs'/args.run_id/'run.json').exists()
scvi.settings.seed=c['seed'];scvi.settings.num_threads=8;torch.set_num_threads(8)
r=Resources();a=ad.read_h5ad(DATA/c['dataset_path']);roles=donor_roles(a.obs,c,args.fold)
root=ASSETS/'runs'/args.run_id;root.mkdir(parents=True,exist_ok=True)
phase={'preprocessing':0.,'training':0.,'inference':0.}

def fit_encode(indices,name):
    start=time.monotonic();training=a[indices].copy();training.X=training.layers['counts'].copy()
    sc.pp.highly_variable_genes(training,n_top_genes=c['n_hvg'],flavor='seurat_v3',layer='counts')
    genes=training.var_names[training.var.highly_variable].tolist();train=training[:,genes].copy()
    # Deliberately no donor batch covariate: evaluation donors have no trained category.
    scvi.model.SCVI.setup_anndata(train,layer='counts')
    model=scvi.model.SCVI(train,n_hidden=128,n_latent=50,n_layers=2,gene_likelihood='nb',dispersion='gene')
    phase['preprocessing']+=time.monotonic()-start;start=time.monotonic()
    model.train(max_epochs=args.epochs,accelerator='gpu',devices=1,train_size=1.,validation_size=0.,early_stopping=False,batch_size=128,enable_progress_bar=False,check_val_every_n_epoch=None)
    phase['training']+=time.monotonic()-start;start=time.monotonic()
    query=a[:,genes].copy();query.X=query.layers['counts'].copy()
    latent=model.get_latent_representation(adata=query,batch_size=256)
    phase['inference']+=time.monotonic()-start
    assert latent.shape==(a.n_obs,50) and np.isfinite(latent).all()
    model.save(root/name,overwrite=False,save_anndata=False)
    write_json(root/f'{name}_training.json',dict(training_cell_ids=a.obs_names[indices].tolist(),genes=genes,epochs=args.epochs))
    return latent

inner=fit_encode(roles['inner_training'],'inner_model')
outer=np.concatenate([roles['inner_training'],roles['inner_validation']]);final=fit_encode(outer,'outer_model')
y=a.obs.cell_type_coarse.astype(str).to_numpy()
pred,prob,head,selection,head_train,head_infer=fit_head(inner,final,y,roles,c,c['seed'])
res=r.finish()
save_evaluation(a,roles['evaluation'],pred,prob,c,args.run_id,dict(model_id='scvi',fold=args.fold,feature_dimension=50,
    feature_source='fold-specific scVI latent mean; no query adaptation',upstream_version=scvi.__version__,checkpoint_sha256=None,
    trainable_parameters='scVI encoder/decoder trained from scratch; see saved architecture',head_selection=selection,
    n_inner_training=len(roles['inner_training']),n_inner_validation=len(roles['inner_validation']),n_outer_training=len(outer),n_evaluation=len(roles['evaluation']),
    training_donors=sorted(a.obs.iloc[outer].patient_id.unique()),validation_donors=sorted(a.obs.iloc[roles['inner_validation']].patient_id.unique()),evaluation_donors_or_perturbations=sorted(a.obs.iloc[roles['evaluation']].patient_id.unique()),
    dataset_sha256=sha(DATA/c['dataset_path']),wall_preprocessing_seconds=phase['preprocessing'],wall_training_seconds=phase['training']+head_train,wall_inference_seconds=phase['inference']+head_infer,wall_total_seconds=res.pop('wall_seconds'),
    input_contract='raw counts, inner/outer train-only 2000 Seurat-v3 HVGs, NB likelihood, 50 latent dimensions, 100 fixed epochs; no donor covariate',
    failure_or_limitations=['development OOF, not a fresh blind test','pretrained-model overlap unknown; scVI is trained only on development training cells'],**res),head=head)
