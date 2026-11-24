"""Schedule H3-v2 training with the authorized Slurm caps and the one-score seal."""
import argparse
import os
from pathlib import Path
from subprocess import check_output

from nextgen_h2.run import read_json
from vc_build.io import publish_json,sha256
from .data import ROOT,SPLITS
from .final_fit import external_operating_point
from .grid import FAMILIES,SEEDS,identify,initial
from .selection import select

TRAINING_CONCURRENCY = {"gbdt": 3, "torch": 6}
CPUS_PER_TRAINING_TASK = 8
CPU_NODE = "c5"
GPU_NODE = "g4"


def slurm(snap):
    return snap/"experiments/nextgen_h3_v2"


def submit(name, script, arguments, *, deps=(), gpu=False,
           array=None, limit="1-12:00:00", threads=None, memory=None, dependency="afterok"):
    if not all(str(x).isdecimal() for x in deps):
        raise ValueError("Unexpected Slurm dependency")
    if dependency not in {"afterok", "afterany"}:
        raise ValueError("Unsupported Slurm dependency condition")
    n=threads or CPUS_PER_TRAINING_TASK
    reserve=memory or ("32G" if gpu else "52G")
    log=ROOT/"logs"/(f"{name}-%A_%a.log" if array else f"{name}-%j.log")
    command=["sbatch","--parsable","-J",f"ng2-h3-{name}","-p","gpu" if gpu else "cpu",
             f"--nodelist={GPU_NODE if gpu else CPU_NODE}","-c",str(n),f"--mem={reserve}","-t",limit,"-o",str(log)]
    if gpu:
        command.extend(["--gres=gpu:a100:1"])
    if array is not None:
        command.append(f"--array={array}")
    if deps:
        command.extend(["--kill-on-invalid-dep=yes", f"--dependency={dependency}:" + ":".join(map(str,deps))])
    result=check_output([*command,str(script),*map(str,arguments)],cwd=ROOT,text=True).strip().split(";")[0]
    if not result.isdecimal():
        raise ValueError(f"Unexpected Slurm submission result: {result}")
    with (ROOT/"logs/trajectory.tsv").open("a") as log_tuple:
        print(name,result,"a100" if gpu else "cpu",array or "-",",".join(deps),sep="\t",file=log_tuple)
    return result


def inner_tasks(stage):
    if stage == "r1":
        planned = {part: {family: initial(family) for family in FAMILIES} for part in SPLITS}
    else:
        planned = read_json(ROOT / "locks" / f"{stage}.json")["splits"]
    tasks={}
    for backend,families in (("gbdt",("gbdt","hybrid")),("torch",("film","attention","gnn","hybrid"))):
        n=len(families)
        tasks[backend] = {}
        for outer, split in enumerate(SPLITS):
            for inner in range(3):
                for offset, family in enumerate(families):
                    bucket = ROOT / "fits" / stage / split / f"inner-{inner}" / family
                    receipts = [bucket / identify(cfg) / "receipt.json" for cfg in planned[split][family]
                                if cfg["backend"] == backend]
                    if receipts:
                        tasks[backend][outer * 3 * n + inner * n + offset] = receipts
    return tasks


def schedule_inner(stages, snap, *, retry=False, tasks=None):
    stage = stages[-1]
    planned = inner_tasks(stage) if tasks is None else tasks
    suffix = "-retry" if retry else ""
    helpers = slurm(snap)
    jobs = []
    for backend, indexed in planned.items():
        if indexed:
            array = ",".join(map(str, indexed)) + f"%{TRAINING_CONCURRENCY[backend]}"
            jobs.append(submit(f"{stage}-{backend}{suffix}", helpers / "inner.sbatch", [snap, stage, backend],
                               gpu=backend == "torch", array=array))
    if not jobs:
        raise ValueError("An inner round was selected with no fits")
    arguments = [snap, "-m", "h3v2.pipeline", "settle", "--snapshot", snap, *stages]
    if retry:
        arguments.append("--retried")
    return submit(f"{stage}-select{suffix}", helpers / "job.sbatch", arguments,
                  deps=tuple(jobs), threads=6, memory="18G", limit="04:00:00", dependency="afterany")


def settle_inner(stages, snapshot, *, retried):
    """Run only after this phase's CPU and GPU arrays are terminal."""
    if (ROOT / "score/consumed.json").exists():
        raise ValueError("Inner tuning is closed after the outer release")
    stage = stages[-1]
    missing = {backend: {index: files for index, files in tasks.items() if not all(p.is_file() for p in files)}
               for backend, tasks in inner_tasks(stage).items()}
    if any(missing.values()):
        indices = {backend: list(tasks) for backend, tasks in missing.items() if tasks}
        if retried:
            raise RuntimeError(f"{stage} still has incomplete trials after the registered retry: {indices}")
        marker = ROOT / "locks" / f"{stage}-missing-fit-retry.json"
        publish_json(marker, {"stage":stage, "array_indices":indices})
        retry_job = schedule_inner(stages, snapshot, retry=True, tasks=missing)
        print("Resubmitted incomplete inner array elements", stage, indices, retry_job, flush=True)
        return
    advance(stages, snapshot, dispatch_only=False)


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
    finals={backend:submit(f"final-{backend}",helpers/"job.sbatch",
                   [snap,"-m","h3v2.pipeline","fit-task","--backend",backend],
                   gpu=backend=="torch",array=f"0-{len(parts)-1}%{TRAINING_CONCURRENCY[backend]}",deps=(ext,))
            for backend,parts in jobs.items() if parts}
    reference_deps=(ext,finals["gbdt"]) if "gbdt" in finals else (ext,)
    refs=submit("references",helpers/"job.sbatch",[snap,"-m","h3v2.pipeline","references-task"],
                deps=reference_deps,array=f"0-{len(SPLITS)}%3",threads=6,memory="20G",limit="02:00:00")
    before=submit("preflight",helpers/"job.sbatch",[snap,"-m","h3v2.score","--preflight"],
                  deps=(refs,*finals.values()),threads=8,memory="42G",limit="03:00:00")
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
    successor=schedule_inner([*stages,choice["next"]],snapshot) if choice["next"] else schedule_final(snapshot)
    print("Next phase will be handled by",successor,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("advance","dispatch","settle","fit-task","references-task"))
    p.add_argument("stages",nargs="*")
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--backend",choices=("gbdt","torch"))
    p.add_argument("--retried", action="store_true")
    args=p.parse_intermixed_args()
    if args.command in ("advance","dispatch"):
        advance(args.stages,args.snapshot,dispatch_only=args.command=="dispatch")
    elif args.command=="settle":
        settle_inner(args.stages,args.snapshot,retried=args.retried)
    else:
        from .final_fit import final_candidate,final_references
        index=int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.command=="fit-task":
            work=read_json(ROOT/"locks/final-jobs.json")[args.backend]
            final_candidate(**work[index])
        else:
            final_references((*SPLITS,"external")[index])
