"""Release exactly the preregistered model batch after completed development and interface checks."""
from datetime import datetime,timezone
import anndata as ad
import yaml
from research_core import *

out=REPORT/'external_final';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists(),'External release is immutable; inspect the original batch before any recovery'
spec_path=ROOT/'configs/research/external_evaluation_spec_v1.yaml';spec=yaml.safe_load(spec_path.read_text());assert spec['status']=='preregistered_before_external_predictions'
confirmation=json.loads((REPORT/'factorial_confirmation/provenance.json').read_text());assert confirmation['status']=='executed' and confirmation['n_executed_runs']==48
probe=json.loads((REPORT/'annotation_interface_probe/annotation_query_interface_v2.json').read_text());assert probe['status']=='executed' and not probe['external_data_accessed'];assert probe['records'][0]['direct_training_preprocessor_probability_max_abs_difference']<=1e-7
models={};manifests={}
for name in spec['fixed_12000_candidates']+spec['fixed_4096_factorial_candidates']:
    factor=name.startswith('DM');version=2 if name=='pca' else 1;run_id=f'F01_{name}_development4096_seed42_v1' if factor else f'F00_{name}_development12000_seed42_v{version}';path=REPORT/'runs'/run_id/'run.json';run=json.loads(path.read_text());assert run['status']=='executed' and not run['external_data_accessed'] and sha(run['bundle_path'])==run['bundle_sha256']
    assert run['n_training_cells']==(4096 if factor else 12000) and sorted(run['training_donors'])==sorted(spec['final_training_donors'])
    if factor:
        check=json.loads((ASSETS/'probes/annotation_query_interface_v2'/name/'inference.json').read_text());assert check['status']=='executed' and check['bundle_sha256']==run['bundle_sha256'] and check['n_cells']==64
    models[name]=dict(run_id=run_id,run_sha256=sha(path),bundle_path=run['bundle_path'],bundle_sha256=run['bundle_sha256'],training_manifest_sha256=run['training_manifest_sha256'],n_training_cells=run['n_training_cells']);manifests[name]=run['training_manifest_sha256']
assert len({manifests[name] for name in spec['fixed_12000_candidates']})==1
assert manifests['DM00']==manifests['DM01'] and manifests['DM10']==manifests['DM11']
root=ASSETS/'external/GSE151530_HCC';contract=json.loads((root/'contract.json').read_text());assert contract['status']=='prepared_unscored' and not contract['model_scores_observed']
source=Path(contract['counts_path']);assert sha(source)==contract['counts_sha256'];a=ad.read_h5ad(source);assert a.n_obs==50023 and a.obs.patient_id.nunique()==25
query=root/'counts_query_v1.h5ad';assert not query.exists();a.obs=a.obs[['cell_id','patient_id','n_counts']].copy();a.X=a.layers['counts'].copy();a.layers.clear();a.write_h5ad(query,compression='gzip')
prediction_root=ASSETS/'external_final_predictions_v1';assert not prediction_root.exists(),'No external predictions may precede this release'
write_json(out/'freeze.json',dict(status='frozen_before_predictions',released_at_utc=datetime.now(timezone.utc).isoformat(),project_commit=git_commit(),source_sha256=sha(Path(__file__)),spec_path=str(spec_path),spec_sha256=sha(spec_path),spec=spec,models=models,
    development_confirmation_sha256=sha(REPORT/'factorial_confirmation/provenance.json'),interface_probe_sha256=sha(REPORT/'annotation_interface_probe/annotation_query_interface_v2.json'),external_metadata_contract_sha256=sha(root/'contract.json'),query_path=str(query),query_sha256=sha(query),truth_path=str(source),truth_sha256=sha(source),prediction_root=str(prediction_root),external_scores_observed_before_release=False))
print('EXTERNAL_BATCH_FROZEN',list(models),flush=True)
