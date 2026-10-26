#!/usr/bin/env bash
#SBATCH --job-name=vc_research_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --chdir=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
#SBATCH --output=logs/slurm_probe_%j.log
set -euo pipefail
source envs/activate.sh research
python - <<'PY'
import os,socket,torch
from pathlib import Path
assert Path('/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/research_phase2/response/hepg2_v1/contract.json').exists()
assert torch.cuda.is_available()
print(dict(host=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(),torch=torch.__version__,bf16=torch.cuda.is_bf16_supported()))
PY
