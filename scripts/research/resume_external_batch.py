"""Resume the already frozen external batch after a storage-dtype reader failure; no re-selection."""
import argparse,subprocess,traceback
from datetime import datetime,timezone
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--attempt-id',required=True);args=p.parse_args();assert Path(args.attempt_id).name==args.attempt_id
attempt=REPORT/'external_final/attempts'/args.attempt_id;assert not attempt.exists();attempt.mkdir(parents=True);freeze_path=REPORT/'external_final/freeze.json';freeze=json.loads(freeze_path.read_text());assert freeze['status']=='frozen_before_predictions' and not (REPORT/'external_final/evaluation.json').exists()
assert not list((REPORT/'runs').glob('E00_*_GSE151530_seed42_v1/run.json')),'Inspect any partial scoring stage explicitly before resuming inference'
assert list(freeze['models'])==['pca','scgpt','geneformer','geneformer_clcancer','DM00','DM10','DM01','DM11']
for path,expected in [(freeze['query_path'],freeze['query_sha256']),(freeze['truth_path'],freeze['truth_sha256']),(freeze['spec_path'],freeze['spec_sha256'])]:assert sha(path)==expected
assert sha(REPORT/'factorial_confirmation/provenance.json')==freeze['development_confirmation_sha256'],'Do not rewrite frozen antecedent metadata after release'
for model in freeze['models'].values():assert sha(model['bundle_path'])==model['bundle_sha256'] and sha(REPORT/'runs'/model['run_id']/'run.json')==model['run_sha256']
probe=json.loads((REPORT/'annotation_interface_probe/annotation_categorical64_v1.json').read_text());assert probe['status']=='executed' and probe['categorical_metadata_storage'] and all(record['categorical_vs_original_metadata_probability_max_abs_difference']<=1e-7 for record in probe['records'])
write_json(attempt/'start.json',dict(status='resuming_frozen_batch',started_utc=datetime.now(timezone.utc).isoformat(),project_commit=git_commit(),source_sha256=sha(Path(__file__)),freeze_sha256=sha(freeze_path),change='Storage categorical→object conversion only; all genes/counts/model weights/preprocessing/selection/metrics unchanged',model_or_data_selection_changed=False,external_scores_observed_before_recovery=False,compatibility_probe_sha256=sha(REPORT/'annotation_interface_probe/annotation_categorical64_v1.json')))
python=str(ROOT/'.venvs/vcell-runtime310/bin/python');bootstrap=str(ROOT/'envs/portable310.py');scripts=ROOT/'scripts/research';query=freeze['query_path'];output=Path(freeze['prediction_root']);archived=[]
def run(environment,script,*arguments):subprocess.run([python,bootstrap,'--env',environment,str(scripts/script),*map(str,arguments)],cwd=ROOT,check=True)
def feature(model):
    path=ASSETS/'query_features'/f'{model}_external_final_v1.npz'
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['status']=='executed' and meta['model_id']==model and meta['input_data_sha256']==freeze['query_sha256'] and meta['feature_sha256']==sha(path)
    else:run('scgpt' if model=='scgpt' else 'geneformer','extract_query_features.py','--input',query,'--model',model,'--cache-name','external_final_v1')
    return path
try:
    for name,model in freeze['models'].items():
        destination=output/name
        if (destination/'inference.json').exists():
            meta=json.loads((destination/'inference.json').read_text());assert meta['status']=='executed' and meta['bundle_sha256']==model['bundle_sha256'] and meta['input_data_sha256']==freeze['query_sha256'] and meta['prediction_sha256']==sha(destination/'predictions.tsv') and meta['probability_sha256']==sha(destination/'probabilities.npz');continue
        if destination.exists():
            assert not any(destination.iterdir()),'Inspect nonempty incomplete output; never overwrite a partial prediction silently'
            archive=ASSETS/'external_recovery'/args.attempt_id/(name+'_empty_output');archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();destination.rename(archive);archived.append(dict(source=str(destination),archive=str(archive)))
        extra=[];environment='research'
        if name in ['scgpt','geneformer','geneformer_clcancer']:extra=['--features',feature(name)]
        elif name in ['DM00','DM10']:extra=['--features',feature('geneformer')];environment='geneformer'
        elif name in ['DM01','DM11']:extra=['--token-cache',DATA/'tokenized/query_external_final_v1/cells.dataset'];environment='geneformer'
        run(environment,'predict_annotation.py','--input',query,'--bundle',model['bundle_path'],'--output',destination,*extra)
    run('research','evaluate_external.py')
    write_json(attempt/'completed.json',dict(status='executed',freeze_sha256=sha(freeze_path),evaluation_sha256=sha(REPORT/'external_final/evaluation.json'),archived_empty_directories=archived,model_or_data_selection_changed=False))
except Exception as exc:
    write_json(attempt/'failure.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),freeze_sha256=sha(freeze_path),archived_empty_directories=archived));raise
print('FROZEN_EXTERNAL_BATCH_RECOVERED',flush=True)
