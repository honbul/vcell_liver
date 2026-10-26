"""Bounded post-selection control-aggregation sensitivity; reuse all original fits and predictions."""
import anndata as ad
from research_core import *
from response_metrics import correlation

out=REPORT/'control_sensitivity';assert not (out/'aggregation_provenance.json').exists();out.mkdir(exist_ok=True);root=ASSETS/'response/hepg2_foundation_common_v1';config=ROOT/'configs/research/control_sensitivity_v3.json'
a=ad.read_h5ad(root/'trainval_controls.h5ad',backed='r');test=ad.read_h5ad(root/'heldout_treated.h5ad',backed='r');d=np.load(root/'test_aggregate.npz');targets=d['targets'];primary=d['primary_mask'].astype(bool);truth=d['truth'][:,primary];full=d['control'][:,primary];assert len(targets)==191 and primary.sum()==1977
ix=np.flatnonzero(a.obs.role.to_numpy()=='control');controls=a.X[ix].toarray()[:,primary];ids=a.obs_names[ix].astype(str);lookup={v:i for i,v in enumerate(ids)};batches=a.obs.iloc[ix].batch.astype(str).to_numpy();bybatch={b:np.flatnonzero(batches==b) for b in set(batches)};means={b:controls[v].mean(0) for b,v in bybatch.items()};groups={str(t):part for t,part in test.obs.groupby('target',observed=True)}
recorded=[];reconstructed=[];support=[]
for i,t in enumerate(targets):
    part=groups[str(t)];recorded.append(controls[[lookup[str(v)] for v in part.control_match_id]].mean(0));weights=part.batch.astype(str).value_counts();reconstructed.append(sum(means[b]*n for b,n in weights.items())/len(part));support.append(dict(target=str(t),n_treated=len(part),n_unique_sampled_controls=part.control_match_id.nunique(),n_batches=part.batch.nunique()))
recorded=np.stack(recorded);reconstructed=np.stack(reconstructed);difference=float(np.max(np.abs(reconstructed-full)));assert difference<1e-5
sources=[];models={}
for name in ['B00_no_change','B01_train_mean','B02_GO_PCA_ridge']:
    path=ASSETS/'runs/B00_B02_native_common_seed42_v1'/f'{name}_predictions.npz';p=np.load(path);assert np.array_equal(p['targets'],targets);models[name]=p['prediction'][:,primary];sources.append(dict(model_id=name,path=str(path),sha256=sha(path)))
for name,rid in [('B03_GEARS','B03_GEARS_native_common_seed42_v1'),('B04_scGPT_pretrained','B04_pretrained_native_common_seed42_v1'),('B04_scGPT_random','B04_random_native_common_seed42_v1')]:
    run=json.loads((REPORT/'runs'/rid/'run.json').read_text());path=Path(run['prediction_path']);assert sha(path)==run['prediction_sha256'];p=np.load(path);assert np.array_equal(p['targets'],targets);models[name]=p['prediction'][:,primary];sources.append(dict(model_id=name,path=str(path),sha256=sha(path)))
def metrics(pred,control):
    return dict(mean_intervention_RMSE=float(np.sqrt(np.mean((truth-pred)**2,axis=1)).mean()),mean_intervention_delta_Pearson=float(pd.Series([correlation(y-c,p-c) for y,p,c in zip(truth,pred,control)]).mean()))
rows=[]
for name,pred in models.items():
    rows.append(dict(model_id=name,comparison='original_predictions_full_control_evaluator',**metrics(pred,full)))
    rows.append(dict(model_id=name,comparison='fixed_predictions_recorded_sampled_control_evaluator',**metrics(pred,recorded)))
    if name.startswith(('B00','B01','B02')):rows.append(dict(model_id=name,comparison='same_fitted_delta_with_recorded_sampled_control_input',**metrics(pred-full+recorded,recorded)))
resampling=[]
for seed in range(1000,1020):
    rng=np.random.default_rng(seed);sampled=[]
    for t in targets:
        part=groups[str(t)];selected=[int(rng.choice(bybatch[b])) for b in part.batch.astype(str)];sampled.append(controls[selected].mean(0))
    sampled=np.stack(sampled)
    for name in ['B00_no_change','B01_train_mean','B02_GO_PCA_ridge']:resampling.append(dict(seed=seed,model_id=name,**metrics(models[name]-full+sampled,sampled)))
pd.DataFrame(rows).to_csv(out/'aggregation_metrics.tsv',sep='\t',index=False);frame=pd.DataFrame(resampling);frame.to_csv(out/'simple_predictor_resampling.tsv',sep='\t',index=False);frame.groupby('model_id')[['mean_intervention_RMSE','mean_intervention_delta_Pearson']].agg(['mean','std','min','max']).to_csv(out/'simple_predictor_resampling_summary.tsv',sep='\t');pd.DataFrame(support).to_csv(out/'target_control_support.tsv',sep='\t',index=False)
np.savez_compressed(ASSETS/'probes/control_sensitivity_aggregation_v3.npz',targets=targets,full_control=full,recorded_sampled_control=recorded)
write_json(out/'aggregation_provenance.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),config_sha256=sha(config),n_test_targets=len(targets),n_test_cells=test.n_obs,n_primary_genes=int(primary.sum()),full_control_reconstruction_max_abs_difference=difference,mean_intervention_control_mean_discrepancy_rmse=float(np.sqrt(np.mean((recorded-full)**2,axis=1)).mean()),sources=sources,selection='All pre-existing test targets; no new model selection or training. Native input intervention is a separate lexicographic-first-target probe.',metric_notes=['Fixed prediction expression RMSE is algebraically invariant to subtracting the same evaluation control from truth and prediction; changing Delta reference can change Pearson and rankings','For simple additive predictors, replacing full control with sampled control is an actual alternative input mean with fitted Delta held fixed','20 seeds measure control-sampling Monte Carlo variability, not biological replicate uncertainty','Original191-target model predictions and scores remain unchanged; top20 is not FDR DEG validation']))
print('CONTROL_AGGREGATION_SENSITIVITY_COMPLETE',pd.DataFrame(rows).to_string(index=False),flush=True)
