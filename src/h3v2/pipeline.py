"""Schedule only H3-v2 waves, respecting the three-GPU cap and the one-score seal."""
import argparse
import os
from pathlib import Path
from subprocess import check_output

from nextgen_h2.run import read_json
from vc_build.io import publish_json,sha256
from .data import ROOT,SPLITS
from .final_fit import external_operating_point
from .grid import FAMILIES,SEEDS
from .selection import select


def slurm(snap):
    return snap/"experiments/nextgen_h3_v2"


def submit(name, script, arguments, *, deps=(), gpu=False,
           array=None, limit="1-12:00:00", threads=None, memory=None):
    if not all(str(x).isdecimal() for x in deps):
        raise ValueError("Unexpected Slurm dependency")
    n=threads or (8 if gpu else 12)
    reserve=memory or ("32G" if gpu else "52G")
    log=ROOT/"logs"/(f"{name}-%A_%a.log" if array else f"{name}-%j.log")
    command=["sbatch","--parsable","-J",f"ng2-h3-{name}","-p","gpu" if gpu else "cpu",
             "-c",str(n),f"--mem={reserve}","-t",limit,"-o",str(log)]
    if gpu:
        command.extend(["--gres=gpu:a100:1"])
    if array is not None:
        command.append(f"--array={array}")
    if deps:
        command.extend(["--kill-on-invalid-dep=yes","--dependency=afterok:"+":".join(map(str,deps))])
    result=check_output([*command,str(script),*map(str,arguments)],cwd=ROOT,text=True).strip().split(";")[0]
    if not result.isdecimal():
        raise ValueError(f"Unexpected Slurm submission result: {result}")
    with (ROOT/"logs/trajectory.tsv").open("a") as log_tuple:
        print(name,result,"a100" if gpu else "cpu",array or "-",",".join(deps),sep="\t",file=log_tuple)
    return result


def schedule_inner(stages,snap,label):
    planned=read_json(ROOT/"locks"/(label+".json"))["splits"]
    selection=slurm(snap)
    arrays={}
    for backend,families in (("gbdt",("gbdt","hybrid")),("torch",("film","attention","gnn","hybrid"))):
        n=len(families)
        needed=[str(o*3*n+i*n+k) for o,split in enumerate(SPLITS) for i in range(3)
                for k,f in enumerate(families) if any(c["backend"]==backend for c in planned[split][f])]
        if needed:
            concurrent=3 if backend=="torch" else 5
            arrays[backend]=",".join(needed)+f"%{concurrent}"
    jobs=[submit(f"{label}-{backend}",selection/"inner.sbatch",[snap,label,backend],
                 gpu=backend=="torch",array=array) for backend,array in arrays.items()]
    if not jobs:
        raise ValueError("An inner round was selected with no fits")
    return submit(f"{label}-select",selection/"job.sbatch",
                  [snap,"-m","h3v2.pipeline","advance","--snapshot",snap,*stages,label],
                  deps=tuple(jobs),threads=6,memory="18G",limit="04:00:00")


def schedule_final(snap):
    selected=read_json(ROOT/"locks/frozen.json")
    helpers=slurm(snap)
    jobs={"torch":[],"gbdt":[]}
    for partition in (*SPLITS,"external"):
        for family in FAMILIES:
            settings=external_operating_point(selected,family) if partition=="external" else (
                selected["splits"][partition]["families"][family])
            backend=settings["config"]["backend"]
            jobs[backend].extend({"split":partition,"family":family,"seed":seed} for seed in SEEDS)
    lock=ROOT/"locks/final-jobs.json"
    publish_json(lock,jobs)
    ext=submit("external-atlas",helpers/"job.sbatch",[snap,"-m","h3v2.final_fit","external-atlas"],
               threads=8,memory="28G",limit="01:00:00")
    finals=[submit(f"final-{backend}",helpers/"job.sbatch",
                   [snap,"-m","h3v2.pipeline","fit-task","--backend",backend],
                   gpu=backend=="torch",array=f"0-{len(parts)-1}%{3 if backend=='torch' else 5}",deps=(ext,))
            for backend,parts in jobs.items() if parts]
    refs=submit("references",helpers/"job.sbatch",[snap,"-m","h3v2.pipeline","references-task"],
                deps=(ext,),array=f"0-{len(SPLITS)}%4",threads=6,memory="20G",limit="02:00:00")
    before=submit("preflight",helpers/"job.sbatch",[snap,"-m","h3v2.score","--preflight"],
                  deps=tuple([refs,*finals]),threads=8,memory="42G",limit="03:00:00")
    after=submit("score-once",helpers/"job.sbatch",[snap,"-m","h3v2.score"],
                 deps=(before,),threads=18,memory="66G",limit="1-00:00:00")
    package=submit("report",helpers/"job.sbatch",[snap,"-m","h3v2.report"],
                   deps=(after,),threads=6,memory="20G",limit="02:00:00")
    publish_json(ROOT/"locks/terminal_jobs.json",{"kernel_and_reference":refs,"preflight":before,"score":after,"reports":package,
                                                   "frozen_sha256":sha256(ROOT/"locks/frozen.json")})
    return package


def advance(stages,snapshot,*,dispatch_only):
    if not dispatch_only:
        select(stages)
    file=ROOT/"locks"/(stages[-1]+"-selection.json")
    choice=read_json(file)
    if choice["stages"]!=stages:
        raise ValueError("Unexpected trajectory plan")
    stage=stages[-1]
    marker=ROOT/"locks"/f"{stage}-dispatch.json"
    if marker.exists():
        raise FileExistsError("Already submitted the next wave for this phase")
    publish_json(marker,{"input_sha256":sha256(file),"stage":stage})
    successor=schedule_inner(stages,snapshot,choice["next"]) if choice["next"] else schedule_final(snapshot)
    print("Next phase will be handled by",successor,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("advance","dispatch","fit-task","references-task"))
    p.add_argument("stages",nargs="*")
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--backend",choices=("gbdt","torch"))
    args=p.parse_intermixed_args()
    if args.command in ("advance","dispatch"):
        advance(args.stages,args.snapshot,dispatch_only=args.command=="dispatch")
    else:
        from .final_fit import final_candidate,final_references
        index=int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.command=="fit-task":
            work=read_json(ROOT/"locks/final-jobs.json")[args.backend]
            final_candidate(**work[index])
        else:
            final_references((*SPLITS,"external")[index])
