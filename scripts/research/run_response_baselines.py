"""Actual measured-response controls: no change, training-intervention mean, GO/PCA ridge."""
import argparse,pickle,time
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from research_core import *
from response_metrics import score_response

p=argparse.ArgumentParser();p.add_argument('--task',choices=['norman','hepg2'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--source-run',default='R01_norman_simulation_seed1_v2');args=p.parse_args()
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True)
raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True);r=Resources();start=time.monotonic()
gene2go=pickle.load(open(ASSETS/'official/GEARS/gene2go_all.pkl','rb'))
if args.task=='norman':
    a=ad.read_h5ad(ASSETS/'official/GEARS/norman/perturb_processed.h5ad')
    split=json.loads((ASSETS/'runs'/args.source_run/'split.json').read_text())
    control=np.asarray(a[a.obs.condition=='ctrl'].X.mean(0)).ravel()
    conditions=a.obs.condition.astype(str).to_numpy()
    def group(role):
        targets=sorted(set(split[role])-{'ctrl'});truth=np.stack([np.asarray(a.X[conditions==target].mean(0)).ravel() for target in targets]);counts=np.array([int((conditions==target).sum()) for target in targets]);ctrl=np.broadcast_to(control,truth.shape).copy()
        return dict(targets=np.array(targets),truth=truth,control=ctrl,n_cells=counts)
    train=group('train');val=group('val');test_loader=lambda:group('test')
    data_provenance=dict(dataset_id='official_norman',regime='official_reproduction_matched_baselines',split_source=args.source_run,split_sha256=sha(ASSETS/'runs'/args.source_run/'split.json'),data_sha256=sha(ASSETS/'official/GEARS/norman/perturb_processed.h5ad'),readout='official preprocessed log-normalized expression',cell_line='K562',intervention_type='CRISPRa')
else:
    root=ASSETS/'response/hepg2_v1';tv=np.load(root/'trainval_aggregate.npz');tr=tv['split']=='train';va=tv['split']=='val'
    keys=['targets','truth','control','n_cells'];train={k:tv[k][tr] for k in keys};val={k:tv[k][va] for k in keys}
    def test_loader():
        d=np.load(root/'test_aggregate.npz');return {k:d[k] for k in keys}
    data_provenance=json.loads((root/'contract.json').read_text())
# GO features depend only on a fixed external prior and declared target identities.
terms=sorted(set.union(*(set(v) for v in gene2go.values())));lookup={term:i for i,term in enumerate(terms)}
def features(targets):
    ri=[];ci=[]
    for i,target in enumerate(targets):
        genes=[g for g in str(target).split('+') if g!='ctrl']
        assert genes and all(g in gene2go for g in genes),f'Missing authentic GO prior for {target}'
        for gene in genes:
            for term in gene2go[gene]:ri.append(i);ci.append(lookup[term])
    return sp.csr_matrix((np.ones(len(ri),dtype=np.float32),(ri,ci)),shape=(len(targets),len(terms)))
xtr=features(train['targets']);xva=features(val['targets']);delta=train['truth']-train['control'];mean_delta=delta.mean(0)
pca=PCA(n_components=min(32,len(delta)-1,delta.shape[1]),random_state=42).fit(delta);z=pca.transform(delta)
preprocessing=time.monotonic()-start;start=time.monotonic();grid=[];best=None
for alpha in [1.,10.,100.]:
    ridge=Ridge(alpha=alpha,solver='lsqr').fit(xtr,z);prediction=val['control']+pca.inverse_transform(ridge.predict(xva));error=float(np.mean((prediction-val['truth'])**2));grid.append(dict(alpha=alpha,validation_mse=error))
    if best is None or error<best[0]:best=(error,alpha,ridge)
training=time.monotonic()-start
# The selection record is written before opening held-out measured responses.
write_json(out/'selection.json',dict(selected_alpha=best[1],grid=grid,response_PCA_components=pca.n_components_,training_interventions=train['targets'].tolist(),validation_interventions=val['targets'].tolist(),test_responses_used_for_selection=False))
import joblib
joblib.dump(dict(pca=pca,ridge=best[2],GO_terms=terms),raw/'go_ridge.joblib')
start=time.monotonic();test=test_loader();predictions={'B00_no_change':test['control'],'B01_train_mean':test['control']+mean_delta,'B02_GO_PCA_ridge':test['control']+pca.inverse_transform(best[2].predict(features(test['targets'])))}
rows=[]
for model,pred in predictions.items():
    rows.extend(dict(model_id=model,**row) for row in score_response(test['targets'],test['truth'],test['control'],pred,test['n_cells']))
    np.savez_compressed(raw/f'{model}_predictions.npz',targets=test['targets'],prediction=pred)
inference=time.monotonic()-start;frame=pd.DataFrame(rows);frame.to_csv(out/'per_intervention.tsv',sep='\t',index=False)
summary=frame.groupby('model_id').mean(numeric_only=True);summary.to_csv(out/'metrics.tsv',sep='\t')
res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='measured_genetic_response',model_id='B00_B01_B02_matched_controls',project_commit=git_commit(),seed=42,
    source_contract=data_provenance,n_train_interventions=len(train['targets']),n_validation_interventions=len(val['targets']),n_test_interventions=len(test['targets']),n_genes=delta.shape[1],
    checkpoint_sha256=None,GO_prior_sha256=sha(ASSETS/'official/GEARS/gene2go_all.pkl'),wall_preprocessing_seconds=preprocessing,wall_training_seconds=training,wall_inference_seconds=inference,wall_total_seconds=res.pop('wall_seconds'),
    selection_metric='validation all-gene intervention-mean MSE; fixed alpha grid; training-only response PCA32; no refit on validation responses',
    failure_or_limitations=['Scores average interventions equally, not cells','Observed top20 absolute mean-delta genes are evaluation-only and were not supplied to the predictor','Zero-change delta Pearson is undefined and recorded blank, never presented as zero predictive correlation','No prospective clinical or efficacy prediction'],**res))
print('RESPONSE_BASELINES_COMPLETE',summary.to_string(),flush=True)
