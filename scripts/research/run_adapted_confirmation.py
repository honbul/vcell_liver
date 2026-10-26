"""Run one fixed confirmation shard; completed immutable runs are reused."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True);group.add_argument('--shard',type=int,choices=[0,1]);group.add_argument('--index',type=int);args=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/research'))
from research_core import *
selection=json.loads((ROOT/'configs/research/confirmation_selection_v2.json').read_text());rows=pd.read_csv(ROOT/'manifests/research_phase2/adapted_confirmation_runs_v2.tsv',sep='\t')
chosen=rows[rows.shard==args.shard] if args.shard is not None else rows.iloc[[args.index]]
for row in chosen.itertuples():
 path=REPORT/'runs'/row.run_id
 if (path/'run.json').exists():
  run=json.loads((path/'run.json').read_text());assert run['status']=='executed';continue
 assert not (path/'failure.json').exists(),f'Inspect and explicitly repair failed run {row.run_id} before restarting'
 subprocess.run([sys.executable,str(ROOT/'scripts/research/run_geneformer_adaptation.py'),'--model',selection['selected_model'],'--data-variant',row.data_variant,'--seed',str(row.seed),'--fold',str(row.fold),'--run-id',row.run_id],cwd=ROOT,check=True)
 print('CONFIRMATION_RUN_EXECUTED',row.run_id,flush=True)
