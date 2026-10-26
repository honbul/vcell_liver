"""Record a read-only Slurm/artifact observation separately from the immutable external freeze."""
import argparse
from datetime import datetime,timezone
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);args=p.parse_args();assert args.job_id.isdigit();out=REPORT/'external_final';freeze=json.loads((out/'freeze.json').read_text());command=subprocess.run(['scontrol','show','job',args.job_id],text=True,capture_output=True);state=next((part.split('=',1)[1] for part in command.stdout.split() if part.startswith('JobState=')),None)
records=[]
for name,model in freeze['models'].items():
    path=Path(freeze['prediction_root'])/name/'inference.json';meta=json.loads(path.read_text()) if path.exists() else None
    records.append(dict(candidate=name,status=meta.get('status') if meta else 'no_completed_inference_metadata',n_cells=meta.get('n_cells') if meta else None,inference_metadata_sha256=sha(path) if meta else None,source_fit_run_id=model['run_id']))
evaluation=out/'evaluation.json';record=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),job_id=args.job_id,slurm_state=state,slurm_query_returncode=command.returncode,status='evaluated' if evaluation.exists() else 'inference_in_progress' if state=='RUNNING' else 'inspect_terminal_state',project_commit=git_commit(),freeze_sha256=sha(out/'freeze.json'),completed_inference_candidates=records,final_evaluation_exists=evaluation.exists(),evaluation_sha256=sha(evaluation) if evaluation.exists() else None,scope='Read-only live observation; immutable freeze, model choice, predictions and scores untouched. Missing scontrol history is not proof of failure.')
write_json(out/'current_status.json',record);print(json.dumps(record),flush=True)
