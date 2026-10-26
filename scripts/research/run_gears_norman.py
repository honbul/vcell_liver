"""Pinned official GEARS Norman reproduction; original data/simulation split, separate from HepG2."""
import argparse,random,shutil,time,traceback
from research_core import *
import torch
from gears import PertData,GEARS
from gears.inference import evaluate,compute_metrics

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--seed',type=int,default=1);args=p.parse_args()
out=REPORT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=True);assert not (out/'run.json').exists()
raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
assets=ASSETS/'official/GEARS';cache=ASSETS/'gears_cache/norman_v1';(cache/'norman').mkdir(parents=True,exist_ok=True)
for name in ['gene2go_all.pkl','essential_all_data_pert_genes.pkl']:
 target=cache/name
 if not target.exists():target.symlink_to(assets/name)
for name in ['perturb_processed.h5ad','go.csv']:
 target=cache/'norman'/name
 if not target.exists():target.symlink_to(assets/'norman'/name)
if not (cache/'go_essential_all').exists():
 assert (assets/'go_essential_all').is_dir(),'Recover official GO graph archive before execution'
 (cache/'go_essential_all').symlink_to(assets/'go_essential_all',target_is_directory=True)
random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(8)
r=Resources();start=time.monotonic()
try:
 data=PertData(str(cache));data.load(data_name='norman');data.prepare_split(split='simulation',seed=args.seed);data.get_dataloader(batch_size=32,test_batch_size=128)
 write_json(raw/'split.json',data.set2conditions)
 pd.DataFrame([dict(condition=condition,role=role) for role,conditions in data.set2conditions.items() for condition in conditions]).to_csv(out/'intervention_split.tsv',sep='\t',index=False)
 preprocessing=time.monotonic()-start;start=time.monotonic()
 model=GEARS(data,device='cuda:0',weight_bias_track=False);model.model_initialize(hidden_size=64)
 loading=time.monotonic()-start;start=time.monotonic();model.train(epochs=args.epochs);training=time.monotonic()-start
 model.save_model(str(raw/'model'));start=time.monotonic()
 result=evaluate(data.dataloader['test_loader'],model.best_model,False,'cuda:0');metrics,per=compute_metrics(result);inference=time.monotonic()-start
 np.savez_compressed(raw/'predictions.npz',**result)
 pd.DataFrame(per).T.to_csv(out/'per_intervention.tsv',sep='\t')
 # Preserve data ordering for subsequent matched-control / GO-ridge baselines.
 write_json(raw/'gene_order.json',list(data.adata.var_names))
 res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='measured_genetic_response',regime='official_reproduction',model_id='GEARS',dataset_id='official_norman',
   project_commit=git_commit(),upstream_commit='f374e43e197b295016d80395d7a54ddb81cc6769',seed=args.seed,epochs=args.epochs,hidden_size=64,batch_size=32,test_batch_size=128,
   intervention_type='CRISPRa; single and dual target activation',cell_line='K562',readout='official preprocessed log-normalized expression',split_id=f'official_simulation_seed{args.seed}',
   n_cells=data.adata.n_obs,n_genes=data.adata.n_vars,n_test_cells=len(result['pert_cat']),n_test_interventions=len(per),trainable_parameters=sum(p.numel() for p in model.model.parameters() if p.requires_grad),metrics=metrics,
   data_sha256=sha(assets/'norman/perturb_processed.h5ad'),go_prior_sha256=sha(assets/'gene2go_all.pkl'),prediction_path=str(raw/'predictions.npz'),prediction_sha256=sha(raw/'predictions.npz'),
   wall_preprocessing_seconds=preprocessing,wall_model_initialization_seconds=loading,wall_training_seconds=training,wall_inference_seconds=inference,wall_total_seconds=res.pop('wall_seconds'),
   failure_or_limitations=['Official preprocessed DE dictionaries retained, including evaluation-condition DE ranking; original reproduction only, not strict HepG2 policy','Not a liver or HCC experiment','Random matched control-to-treated pairing is unpaired, not longitudinal single-cell prediction'],**res))
 print('GEARS_NORMAN_COMPLETE',metrics,flush=True)
except Exception as exc:
 write_json(out/'failure.json',dict(status='failed',run_id=args.run_id,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),**r.finish()));raise
