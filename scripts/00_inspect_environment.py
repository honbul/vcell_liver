import argparse, json, os, platform, shutil, subprocess, sys
from pathlib import Path
import psutil
from common import write_json

def command(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return dict(command=args,exit_code=p.returncode,output=p.stdout)

p=argparse.ArgumentParser();p.add_argument('--out',default='reports/environment.json');a=p.parse_args()
import torch
write_json(a.out,dict(hostname=platform.node(),platform=platform.platform(),python=sys.version,
    executable=sys.executable,cpu=platform.processor(),logical_cpu_count=os.cpu_count(),
    ram_bytes=psutil.virtual_memory().total,disk=shutil.disk_usage('/mnt/S1/sdata')._asdict(),
    torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
    gpu=command(['nvidia-smi']),os_release=Path('/etc/os-release').read_text()))
print(Path(a.out).read_text())
