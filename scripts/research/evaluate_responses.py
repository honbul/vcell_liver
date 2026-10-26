"""Plan-required measured-response RMSE, retrieval and direct-target sensitivity from executed predictions."""
import argparse
import anndata as ad
from research_core import *
from response_metrics import score_response

p=argparse.ArgumentParser();p.add_argument('--task',choices=['norman','hepg2','native_common'],required=True);args=p.parse_args()
hepg2=args.task!='norman';native=args.task=='native_common'
root=ASSETS/'response'/('hepg2_foundation_common_v1' if native else 'hepg2_v1');base_id='B00_B02_native_common_seed42_v1' if native else 'B00_B02_hepg2_unseen_target_seed42_v1' if hepg2 else 'B00_B02_norman_simulation_seed1_v2';raw=ASSETS/'runs'/base_id
sources=[]
if hepg2:
    obs=np.load(root/'test_aggregate.npz');targets=obs['targets'];truth=obs['truth'];ctrl=obs['control'];n=obs['n_cells'];genes=obs['gene_names'];tv=np.load(root/'trainval_aggregate.npz');ix=tv['split']=='train';training_magnitude=np.linalg.norm(tv['truth'][ix]-tv['control'][ix],axis=1)
else:
    a=ad.read_h5ad(ASSETS/'official/GEARS/norman/perturb_processed.h5ad');split=json.loads((ASSETS/'runs/R01_norman_simulation_seed1_v2/split.json').read_text());conditions=a.obs.condition.astype(str).to_numpy();control=np.asarray(a.X[conditions=='ctrl'].mean(0)).ravel();targets=np.array(sorted(set(split['test'])-{'ctrl'}));truth=np.stack([np.asarray(a.X[conditions==t].mean(0)).ravel() for t in targets]);ctrl=np.broadcast_to(control,truth.shape).copy();n=np.array([(conditions==t).sum() for t in targets]);genes=a.var.gene_name.astype(str).to_numpy();train=np.stack([np.asarray(a.X[conditions==t].mean(0)).ravel()-control for t in sorted(set(split['train'])-{'ctrl'})]);training_magnitude=np.linalg.norm(train,axis=1)
cutoffs=np.quantile(training_magnitude,[1/3,2/3]);models={}
def align(names,pred):
    names=np.asarray(names,dtype=str);assert len(names)==len(set(names)) and set(names)==set(targets)
    lookup={t:i for i,t in enumerate(names)};return pred[[lookup[t] for t in targets]]
for name in ['B00_no_change','B01_train_mean','B02_GO_PCA_ridge']:
    path=raw/f'{name}_predictions.npz';d=np.load(path);models[name]=align(d['targets'],d['prediction']);sources.append(dict(model_id=name,path=str(path),sha256=sha(path)))
gears_id='B03_GEARS_native_common_seed42_v1' if native else 'B03_hepg2_strict_unseen_target_seed42_v1' if hepg2 else 'R01_norman_simulation_seed1_v2';runpath=REPORT/'runs'/gears_id/'run.json'
if runpath.exists():
    run=json.loads(runpath.read_text());assert run['status']=='executed';path=Path(run['prediction_path']);assert sha(path)==run['prediction_sha256'];d=np.load(path)
    if hepg2:
        assert np.array_equal(d['gene_names'],genes);models['B03_GEARS']=align(d['targets'],d['prediction'])
    else:
        cats=d['pert_cat'];pred=d['pred'];measured=d['truth'];assert set(cats)==set(targets)
        aggregate=np.stack([pred[cats==t].mean(0) for t in targets]);check=np.stack([measured[cats==t].mean(0) for t in targets]);assert np.allclose(check,truth,atol=1e-4,rtol=1e-4),'GEARS and baselines must have the same observed response population';models['B03_GEARS']=aggregate
    sources.append(dict(model_id='B03_GEARS',path=str(path),sha256=sha(path)))
if native:
    # Do not publish a partial pretrained/random comparison as the completed matched experiment.
    assert 'B03_GEARS' in models,'Await the matched GEARS execution'
    for initialization in ['pretrained','random']:
        rid=f'B04_{initialization}_native_common_seed42_v1';runpath=REPORT/'runs'/rid/'run.json';run=json.loads(runpath.read_text());assert run['status']=='executed'
        path=Path(run['prediction_path']);assert sha(path)==run['prediction_sha256'];d=np.load(path);assert np.array_equal(d['gene_names'],genes)
        name=f'B04_scGPT_{initialization}';models[name]=align(d['targets'],d['prediction']);sources.append(dict(model_id=name,path=str(path),sha256=sha(path)))
    full_genes=genes.copy();full_truth=truth.copy();full_ctrl=ctrl.copy();full_models=models.copy()
    primary=obs['primary_mask'].astype(bool);assert primary.sum()==1977
    truth=truth[:,primary];ctrl=ctrl[:,primary];genes=genes[primary];models={name:pred[:,primary] for name,pred in models.items()}
    training_magnitude=np.linalg.norm((tv['truth'][ix]-tv['control'][ix])[:,primary],axis=1);cutoffs=np.quantile(training_magnitude,[1/3,2/3])
guide_info=None
if hepg2:
    guide_path=root/'target_split.tsv';guide_info=pd.read_csv(guide_path,sep='\t').set_index('target');assert guide_info.index.is_unique and set(targets)<=set(guide_info.index) and (guide_info.loc[targets,'role']=='test').all()
rows=[]
for name,pred in models.items():
    assert pred.shape==truth.shape and np.isfinite(pred).all()
    for row in score_response(targets,truth,ctrl,pred,n,gene_names=genes):
        magnitude=row['observed_mean_delta_l2'];row.update(model_id=name,effect_stratum=['weak','medium','strong'][int(np.searchsorted(cutoffs,magnitude))],cell_support_stratum='<=25' if row['n_cells']<=25 else '26-50' if row['n_cells']<=50 else '>50')
        if guide_info is not None:
            row['n_observed_guide_assignments']=int(guide_info.loc[row['target'],'n_guide_assignments']);row['n_technical_batches']=int(guide_info.loc[row['target'],'n_batches']);row['guide_assignment_stratum']='one_assignment' if row['n_observed_guide_assignments']==1 else 'multiple_assignments'
        rows.append(row)
frame=pd.DataFrame(rows);out=REPORT/f'response_evaluation_{args.task}';out.mkdir(exist_ok=True)
frame.to_csv(out/'per_intervention.tsv',sep='\t',index=False);metrics=['delta_rmse','delta_mae','pearson_delta','top20_response_retrieval_average_precision','top20_response_retrieval_precision_at20','delta_rmse_excluding_direct_target','sign_accuracy_observed_top20_delta']
example_index=int(np.argsort(targets.astype(str),kind='stable')[0]);selected_genes=np.argsort(-np.abs(truth[example_index]-ctrl[example_index]),kind='stable')[:20]
example=pd.DataFrame(dict(gene_symbol=genes[selected_genes],control_mean=ctrl[example_index,selected_genes],observed_treated_mean=truth[example_index,selected_genes],observed_mean_delta=(truth[example_index]-ctrl[example_index])[selected_genes]))
for name,pred in models.items():example[name+'_predicted_mean']=pred[example_index,selected_genes];example[name+'_predicted_delta']=(pred[example_index]-ctrl[example_index])[selected_genes]
example.to_csv(out/'example_lexfirst_target.tsv',sep='\t',index=False)
write_json(out/'example_selection.json',dict(target=str(targets[example_index]),n_treated_cells=int(n[example_index]),target_selection='Lexicographically first fixed test target, no performance-based selection',display_genes='Top20 absolute observed mean Delta, evaluator-only; never supplied to any predictor',model_ids=list(models),source_prediction_hashes=sources))
frame.groupby('model_id')[metrics].mean().to_csv(out/'condition_macro_metrics.tsv',sep='\t')
frame.groupby(['model_id','effect_stratum'])[metrics].mean().to_csv(out/'by_effect_size.tsv',sep='\t')
frame.groupby(['model_id','cell_support_stratum'])[metrics].mean().to_csv(out/'by_cell_support.tsv',sep='\t')
if guide_info is not None:
    frame.groupby(['model_id','guide_assignment_stratum'])[metrics].mean().to_csv(out/'by_observed_guide_assignments.tsv',sep='\t')
    frame.groupby(['model_id','guide_assignment_stratum']).size().rename('n_interventions').to_csv(out/'guide_assignment_stratum_support.tsv',sep='\t')
if native:
    all_rows=[]
    for name,pred in full_models.items():all_rows.extend(dict(model_id=name,**row) for row in score_response(targets,full_truth,full_ctrl,pred,n,gene_names=full_genes))
    auxiliary=pd.DataFrame(all_rows);assert (auxiliary.n_direct_targets_in_output==1).all(),'All native intervention targets must be observed in the shared3303-gene panel'
    auxiliary.to_csv(out/'all3303_genes_per_intervention.tsv',sep='\t',index=False);auxiliary.groupby('model_id')[metrics].mean().to_csv(out/'all3303_genes_condition_macro_metrics.tsv',sep='\t')
nulls=frame.groupby('model_id')[['pearson_delta','top20_response_retrieval_precision_at20']].agg(lambda x:int(x.isna().sum()));nulls.to_csv(out/'undefined_metric_counts.tsv',sep='\t')
pivot=frame.pivot(index='target',columns='model_id',values='delta_rmse');rng=np.random.default_rng(42);draw=rng.integers(0,len(pivot),size=(5000,len(pivot)));paired=[]
for name in models:
    delta=(pivot['B00_no_change']-pivot[name]).to_numpy();boot=delta[draw].mean(1);paired.append(dict(model_id=name,delta_RMSE_gain_vs_no_change=float(delta.mean()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),n_interventions=len(delta)))
pd.DataFrame(paired).to_csv(out/'paired_intervention_bootstrap.tsv',sep='\t',index=False)
comparisons=[]
for name in models:
    if name=='B02_GO_PCA_ridge':continue
    delta=(pivot['B02_GO_PCA_ridge']-pivot[name]).to_numpy();samples=delta[draw].mean(1);comparisons.append(dict(reference='B02_GO_PCA_ridge',candidate=name,delta_RMSE_gain=float(delta.mean()),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),n_interventions=len(delta)))
if native:
    delta=(pivot['B04_scGPT_random']-pivot['B04_scGPT_pretrained']).to_numpy();samples=delta[draw].mean(1);comparisons.append(dict(reference='B04_scGPT_random',candidate='B04_scGPT_pretrained',delta_RMSE_gain=float(delta.mean()),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),n_interventions=len(delta)))
pd.DataFrame(comparisons).to_csv(out/'paired_model_comparisons.tsv',sep='\t',index=False)
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),task='measured_genetic_response',dataset=args.task,n_interventions=len(targets),n_cells=int(n.sum()),n_genes=len(genes),model_ids=list(models),sources=sources,
    primary_metric='condition-macro Delta expression RMSE, lower better; subtracting the same matched-control vector makes prediction error identical in expression and Delta space',
    evaluator_only_top20='Top20 absolute observed mean Delta; not statistical FDR DE; never supplied to predictors or model selection',constant_prediction_policy='Delta Pearson undefined; constant ranking precision@20 undefined; AP of constant scores equals reference prevalence',
    effect_size_cutoffs_training_only=cutoffs.tolist(),guide_coverage='Observed distinct sgID_AB assignment strings per target, not independently randomized guides or biological replicates; small multi-assignment subgroups have explicit support counts' if hepg2 else 'Official Norman assignment/replicate coverage unavailable in the released processed benchmark; no invented guide strata',guide_manifest_sha256=sha(guide_path) if hepg2 else None,
    gene_universes=dict(primary=len(genes),all_observed_shared_genes=len(full_genes) if native else len(genes),native_secondary='all3303 observed shared genes, all intervention targets present; same fixed predictions' if native else None),
    baseline_selection_metric_resolved='Native common B02 selects alpha using the1977 primary genes; its immutable original metadata has a generic all-gene string. Executed source, selection grid and primary-mask contract take precedence.' if native else 'Original selected validation metric unchanged',
    uncertainty='5000 paired bootstrap samples of target/intervention groups; shared controls and unknown biological replicates limit interpretation; no cell bootstrap',
    failure_or_limitations=['Conditional on this experimental population; does not establish biological replicate reproducibility','Native official Norman reproduction remains separate from strict HepG2','Pre-existing V2 RMSE/retrieval/sensitivity requirements added by an evaluator; fitted models and immutable original runs are unchanged']))
print('MEASURED_RESPONSE_EVALUATION',args.task,frame.groupby('model_id')[metrics].mean().to_string(),flush=True)
