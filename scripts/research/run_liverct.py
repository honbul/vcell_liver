"""LiverCT native query adaptation and same-adapted-reference SCANVI classifier control."""
import argparse,copy,shutil,sys,time,traceback
from research_core import *
import anndata as ad
import scipy.sparse as sp
import torch,yaml

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/research/protocol_annotation_full.yaml');p.add_argument('--fold',type=int,required=True);p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--resume-from');args=p.parse_args()
c=yaml.safe_load(Path(args.config).read_text());c['evaluation_regime']='query_adaptive_reference_mapping';c['model_variant']='native_reference_classifier';c['data_variant']='external_reference_plus_unlabeled_query'
out=REPORT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=True);assert not (out/'run.json').exists()
raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
upstream=ROOT/'third_party/LiverCT';sys.path.insert(0,str(upstream))
from test_function import feature_alignment,LabelPredict
import scarches as sca
import scvi

torch.set_num_threads(8);scvi.settings.seed=42;np.random.seed(42);r=Resources();start=time.monotonic()
try:
 a=ad.read_h5ad(DATA/c['dataset_path']);roles=donor_roles(a.obs,c,args.fold);te=roles['evaluation']
 b=a[te].copy();symbols=b.var.gene_symbol.astype(str).to_numpy();unique=pd.Index(pd.unique(symbols));ix=unique.get_indexer(symbols)
 mapping=sp.csr_matrix((np.ones(len(ix)),(np.arange(len(ix)),ix)),shape=(len(ix),len(unique)))
 counts=sp.csr_matrix(b.layers['counts'])@mapping
 # Query truth labels are retained only in a, never passed to adaptation.
 query=ad.AnnData(counts.astype(np.float32),obs=pd.DataFrame(index=b.obs_names),var=pd.DataFrame(index=unique))
 reference=DATA/'checkpoints/LiverCT/reference/models';model_dir=raw/'reference_copy'
 assert not model_dir.exists(),'Each query run must start from a fresh original reference'
 shutil.copytree(reference/'scarches_model_new',model_dir/'scarches_model_new');(model_dir/'LiverCT').symlink_to(reference/'LiverCT',target_is_directory=True)
 genes=pd.read_csv(model_dir/'scarches_model_new/var_names.csv',header=None).iloc[:,0]
 coverage=float(genes.isin(query.var_names).mean());test=feature_alignment(query,genes).copy()
 test.obs['batch']='new_batch';test.obs['level1']='Unknown';test.layers['counts']=test.X.copy()
 preprocessing=time.monotonic()-start;start=time.monotonic()
 if args.resume_from:
  resume=ASSETS/'runs'/args.resume_from/'my_surgery_model'
  assert resume.is_dir(),'Saved adapted model is required for recovery'
  model=sca.models.SCANVI.load(str(resume),adata=test,use_gpu=True)
  adaptation=0. # This run performs no new training; prior adaptation is recorded by source run.
 else:
  model=sca.models.SCANVI.load_query_data(test,str(model_dir/'scarches_model_new'),freeze_dropout=True)
  model._unlabeled_indices=np.arange(test.n_obs);model._labeled_indices=[]
  model.train(max_epochs=args.epochs,plan_kwargs=dict(weight_decay=0.),check_val_every_n_epoch=10)
  adaptation=time.monotonic()-start
  model.save(str(raw/'my_surgery_model'),overwrite=False)
 start=time.monotonic();latent=ad.AnnData(model.get_latent_representation());latent.obs_names=test.obs_names
 native=LabelPredict(latent,is_cnt=False,select_feature=False,model_path=str(model_dir/'LiverCT'),model_name='latent_input_classifier_all_adjparam_unbalancedsvm_new',transition_predict=True,disease_predict=True)
 native=native.loc[test.obs_names];native.to_csv(raw/'native_predictions.tsv',sep='\t')
 scanvi=np.asarray(model.predict(adata=test),dtype=str)
 np.save(raw/'scanvi_native_labels.npy',scanvi)
 inference=time.monotonic()-start
 mapping={'B cell':'B','Plasma B cell':'B','Endothelial cell':'Endothelial','Mesenchymal cell':'Fibroblast','Hepatocyte':'Hepatocyte','Myeloid cell':'Myeloid','TNK cell':'T/NK','NK/T cell':'T/NK','Cholangiocyte':'Unknown','Unknown':'Unknown'}
 predicted=native['pred_lv1'].astype(str).map(mapping).fillna('Unknown').to_numpy()
 comparator=pd.Series(scanvi).map(mapping).fillna('Unknown').to_numpy()
 res=r.finish();common=dict(fold=args.fold,upstream_commit='22a982f289feec9938f9967f7f57db5eac582b85',base_checkpoint_revision='Zenodo8239570',checkpoint_sha256=sha(DATA/'checkpoints/LiverCT/models.zip'),dataset_sha256=sha(DATA/c['dataset_path']),n_evaluation=len(te),
  evaluation_donors_or_perturbations=sorted(a.obs.iloc[te].patient_id.unique()),reference_data='21 batches in original reference attribute registry; external reference labels',query_label_access=False,
  allowed_target_information='All counts of this outer evaluation query cohort, unlabeled adaptation; no HCC development training labels used',adaptation_source_run=args.resume_from,training_time_scope='Current run only; recovered adaptation time is unavailable separately in failed source run' if args.resume_from else 'Current 20-epoch query adaptation',reference_gene_coverage=coverage,n_duplicate_symbols_collapsed=len(symbols)-len(unique),epochs=args.epochs,
  wall_preprocessing_seconds=preprocessing,wall_training_seconds=adaptation,wall_inference_seconds=inference,wall_total_seconds=res.pop('wall_seconds'),
  confidence_calibration='Native broad-lineage hard predictions; no calibrated probabilities claimed',
  failure_or_limitations=['Query-adaptive, not strict inductive; separate leaderboard','Mesenchymal mapped to Fibroblast; Cholangiocyte outside fixed six lineages mapped Unknown','Native finer state/disease outputs not scored; no verified fine-state truth','Pretraining overlap unknown; demo Lu2022 overlaps GSE149614'],**res)
 save_evaluation(a,te,predicted,None,c,args.run_id,dict(model_id='LiverCT_native',**common))
 c2=copy.deepcopy(c);c2['model_variant']='same_adapted_reference_SCANVI_head'
 save_evaluation(a,te,comparator,None,c2,args.run_id+'_scanvi_control',dict(model_id='LiverCT_adapted_SCANVI_control',**common))
 write_json(out/'mapping.json',mapping)
 print('LIVERCT_COMPLETE',flush=True)
except Exception as exc:
 write_json(out/'failure.json',dict(status='failed',run_id=args.run_id,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),**r.finish()));raise
