"""Plan-required measured-response RMSE, retrieval and direct-target sensitivity from executed predictions."""
import argparse
import anndata as ad
from research_core import *
from response_metrics import score_response

p=argparse.ArgumentParser();p.add_argument('--task',choices=['norman','hepg2'],required=True);args=p.parse_args()
root=ASSETS/'response/hepg2_v1';base_id='B00_B02_hepg2_unseen_target_seed42_v1' if args.task=='hepg2' else 'B00_B02_norman_simulation_seed1_v2';raw=ASSETS/'runs'/base_id
sources=[]
if args.task=='hepg2':
    obs=np.load(root/'test_aggregate.npz');targets=obs['targets'];truth=obs['truth'];ctrl=obs['control'];n=obs['n_cells'];genes=obs['gene_names'];tv=np.load(root/'trainval_aggregate.npz');ix=tv['split']=='train';training_magnitude=np.linalg.norm(tv['truth'][ix]-tv['control'][ix],axis=1)
else:
    a=ad.read_h5ad(ASSETS/'official/GEARS/norman/perturb_processed.h5ad');split=json.loads((ASSETS/'runs/R01_norman_simulation_seed1_v2/split.json').read_text());conditions=a.obs.condition.astype(str).to_numpy();control=np.asarray(a.X[conditions=='ctrl'].mean(0)).ravel();targets=np.array(sorted(set(split['test'])-{'ctrl'}));truth=np.stack([np.asarray(a.X[conditions==t].mean(0)).ravel() for t in targets]);ctrl=np.broadcast_to(control,truth.shape).copy();n=np.array([(conditions==t).sum() for t in targets]);genes=a.var.gene_name.astype(str).to_numpy();train=np.stack([np.asarray(a.X[conditions==t].mean(0)).ravel()-control for t in sorted(set(split['train'])-{'ctrl'})]);training_magnitude=np.linalg.norm(train,axis=1)
cutoffs=np.quantile(training_magnitude,[1/3,2/3]);models={}
def align(names,pred):
    names=np.asarray(names,dtype=str);assert len(names)==len(set(names)) and set(names)==set(targets)
    lookup={t:i for i,t in enumerate(names)};return pred[[lookup[t] for t in targets]]
for name in ['B00_no_change','B01_train_mean','B02_GO_PCA_ridge']:
    path=raw/f'{name}_predictions.npz';d=np.load(path);models[name]=align(d['targets'],d['prediction']);sources.append(dict(model_id=name,path=str(path),sha256=sha(path)))
gears_id='B03_hepg2_strict_unseen_target_seed42_v1' if args.task=='hepg2' else 'R01_norman_simulation_seed1_v2';runpath=REPORT/'runs'/gears_id/'run.json'
if runpath.exists():
    run=json.loads(runpath.read_text());assert run['status']=='executed';path=Path(run['prediction_path']);assert sha(path)==run['prediction_sha256'];d=np.load(path)
    if args.task=='hepg2':
        assert np.array_equal(d['gene_names'],genes);models['B03_GEARS']=align(d['targets'],d['prediction'])
    else:
        cats=d['pert_cat'];pred=d['pred'];measured=d['truth'];assert set(cats)==set(targets)
        aggregate=np.stack([pred[cats==t].mean(0) for t in targets]);check=np.stack([measured[cats==t].mean(0) for t in targets]);assert np.allclose(check,truth,atol=1e-4,rtol=1e-4),'GEARS and baselines must have the same observed response population';models['B03_GEARS']=aggregate
    sources.append(dict(model_id='B03_GEARS',path=str(path),sha256=sha(path)))
rows=[]
for name,pred in models.items():
    assert pred.shape==truth.shape and np.isfinite(pred).all()
    for row in score_response(targets,truth,ctrl,pred,n,gene_names=genes):
        magnitude=row['observed_mean_delta_l2'];row.update(model_id=name,effect_stratum=['weak','medium','strong'][int(np.searchsorted(cutoffs,magnitude))],cell_support_stratum='<=25' if row['n_cells']<=25 else '26-50' if row['n_cells']<=50 else '>50');rows.append(row)
frame=pd.DataFrame(rows);out=REPORT/f'response_evaluation_{args.task}';out.mkdir(exist_ok=True)
frame.to_csv(out/'per_intervention.tsv',sep='\t',index=False);metrics=['delta_rmse','delta_mae','pearson_delta','top20_response_retrieval_average_precision','top20_response_retrieval_precision_at20','delta_rmse_excluding_direct_target']
frame.groupby('model_id')[metrics].mean().to_csv(out/'condition_macro_metrics.tsv',sep='\t')
frame.groupby(['model_id','effect_stratum'])[metrics].mean().to_csv(out/'by_effect_size.tsv',sep='\t')
frame.groupby(['model_id','cell_support_stratum'])[metrics].mean().to_csv(out/'by_cell_support.tsv',sep='\t')
nulls=frame.groupby('model_id')[['pearson_delta','top20_response_retrieval_precision_at20']].agg(lambda x:int(x.isna().sum()));nulls.to_csv(out/'undefined_metric_counts.tsv',sep='\t')
pivot=frame.pivot(index='target',columns='model_id',values='delta_rmse');rng=np.random.default_rng(42);draw=rng.integers(0,len(pivot),size=(5000,len(pivot)));paired=[]
for name in models:
    delta=(pivot['B00_no_change']-pivot[name]).to_numpy();boot=delta[draw].mean(1);paired.append(dict(model_id=name,delta_RMSE_gain_vs_no_change=float(delta.mean()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),n_interventions=len(delta)))
pd.DataFrame(paired).to_csv(out/'paired_intervention_bootstrap.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),task='measured_genetic_response',dataset=args.task,n_interventions=len(targets),n_cells=int(n.sum()),n_genes=len(genes),model_ids=list(models),sources=sources,
    primary_metric='condition-macro Delta expression RMSE, lower better; subtracting the same matched-control vector makes prediction error identical in expression and Delta space',
    evaluator_only_top20='Top20 absolute observed mean Delta; not statistical FDR DE; never supplied to predictors or model selection',constant_prediction_policy='Delta Pearson undefined; constant ranking precision@20 undefined; AP of constant scores equals reference prevalence',
    effect_size_cutoffs_training_only=cutoffs.tolist(),uncertainty='5000 paired bootstrap samples of target/intervention groups; shared controls and unknown biological replicates limit interpretation; no cell bootstrap',
    failure_or_limitations=['Conditional on this experimental population; does not establish biological replicate reproducibility','Native official Norman reproduction remains separate from strict HepG2','Pre-existing V2 RMSE/retrieval/sensitivity requirements added by an evaluator; fitted models and immutable original runs are unchanged']))
print('MEASURED_RESPONSE_EVALUATION',args.task,frame.groupby('model_id')[metrics].mean().to_string(),flush=True)
