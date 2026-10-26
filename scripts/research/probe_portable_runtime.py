"""Slurm runtime probe, not a model experiment."""
import json,os,socket,sys
import anndata,torch,scgpt
from pathlib import Path
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
root=Path(__file__).resolve().parents[2];job=os.environ['SLURM_JOB_ID'];out=root/'reports/research_phase2/runtime_probes';out.mkdir(exist_ok=True,parents=True)
result=dict(status='verified',host=socket.gethostname(),slurm_job_id=job,python=sys.version,executable=sys.executable,torch=torch.__version__,anndata=anndata.__version__,gpu=torch.cuda.get_device_name(),bf16=torch.cuda.is_bf16_supported(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),weights_and_counts_changed=False)
(out/f'slurm_{job}.json').write_text(json.dumps(result,indent=2));print('PORTABLE_RUNTIME_VERIFIED',result,flush=True)
