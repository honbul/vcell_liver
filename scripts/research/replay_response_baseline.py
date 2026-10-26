"""One real heldout intervention: actual saved GO/ridge inference, no fitting or truth-derived input."""
import argparse,pickle,time
import joblib
from scipy import sparse
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--task',choices=['hepg2','native_common'],default='hepg2');p.add_argument('--target');p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);assert not out.exists();out.mkdir(parents=True);started=time.monotonic();native=args.task=='native_common';root=ASSETS/'response'/('hepg2_foundation_common_v1' if native else 'hepg2_v1');rid='B00_B02_native_common_seed42_v1' if native else 'B00_B02_hepg2_unseen_target_seed42_v1';raw=ASSETS/'runs'/rid
d=np.load(root/'test_aggregate.npz');target=args.target or sorted(d['targets'].astype(str))[0];assert target in d['targets'];index=int(np.flatnonzero(d['targets']==target)[0]);model=joblib.load(raw/'go_ridge.joblib');gene2go=pickle.load(open(ASSETS/'official/GEARS/gene2go_all.pkl','rb'));assert target in gene2go
lookup={term:i for i,term in enumerate(model['GO_terms'])};columns=[lookup[term] for term in gene2go[target]];features=sparse.csr_matrix((np.ones(len(columns),dtype=np.float32),([0]*len(columns),columns)),shape=(1,len(lookup)))
# Prediction sees only the declared target's external GO annotations and provided matched controls.
prediction=d['control'][index]+model['pca'].inverse_transform(model['ridge'].predict(features))[0];elapsed=time.monotonic()-started
stored=np.load(raw/'B02_GO_PCA_ridge_predictions.npz');recorded=stored['prediction'][np.flatnonzero(stored['targets']==target)[0]];difference=float(np.abs(prediction-recorded).max());assert difference<=1e-6
truth=d['truth'][index];control=d['control'][index];order=np.argsort(-np.abs(truth-control),kind='stable')[:20]
pd.DataFrame(dict(gene_symbol=d['gene_names'][order],control_mean=control[order],observed_treated_mean=truth[order],predicted_treated_mean=prediction[order],observed_delta=(truth-control)[order],predicted_delta=(prediction-control)[order])).to_csv(out/'example.tsv',sep='\t',index=False)
np.savez_compressed(out/'prediction.npz',target=np.array([target]),gene_names=d['gene_names'],prediction=prediction)
write_json(out/'replay.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),task='measured_genetic_response_inference_replay',dataset=args.task,source_run_id=rid,target=target,n_observed_treated_cells=int(d['n_cells'][index]),n_genes=len(prediction),input='Authentic external GO terms of target plus supplied matched control mean; no treated response used in prediction',truth_usage='After prediction, evaluator-only gene display order and observed comparison; no fitting or selection',model_path=str(raw/'go_ridge.joblib'),model_sha256=sha(raw/'go_ridge.joblib'),GO_prior_sha256=sha(ASSETS/'official/GEARS/gene2go_all.pkl'),stored_prediction_sha256=sha(raw/'B02_GO_PCA_ridge_predictions.npz'),fresh_inference_vs_recorded_prediction_max_abs_difference=difference,wall_loading_and_inference_seconds=elapsed,biological_pairing='Intervention-level means of different cells; no paired-cell or clinical effect claim'))
print('RESPONSE_BASELINE_REPLAY',target,'max_difference',difference,flush=True)
