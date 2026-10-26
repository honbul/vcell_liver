"""Record process terminal state and preserve failures without fabricated metrics."""
import argparse,datetime,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
assert cmd
state=dict(name=a.name,command=cmd,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='running')
path=ROOT/'reports/stages'/f'{a.name}.json';path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(state,indent=2));start=time.monotonic()
try:
    proc=subprocess.Popen(cmd,cwd=ROOT)
    state['pid']=proc.pid;path.write_text(json.dumps(state,indent=2))
    code=proc.wait()
except BaseException as exc:
    state.update(status='failed',error=str(exc));path.write_text(json.dumps(state,indent=2));raise
state.update(status='executed' if code==0 else 'failed',exit_code=code,wall_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
sys.exit(code)
