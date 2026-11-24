"""Schedule H3-v2 training with the authorized Slurm caps and the one-score seal."""
import argparse
from getpass import getuser
import os
from pathlib import Path
from subprocess import check_output

from nextgen_h2.run import read_json
from vc_build.io import publish_json,sha256
from .data import ROOT,SPLITS
from .final_fit import external_operating_point
from .grid import FAMILIES,SEEDS,identify,initial
from .selection import NEURAL_FAMILIES, next_boundary, read_neural_expansion, select

CPUS_PER_TRAINING_TASK = 8
GPU_TRAINING_CONCURRENCY = 4
CPU_NODE = "c5"
GPU_NODE = "g3"
GPU_TYPE = "a40"
# Three 8-core tasks and one 6-core task leave two C5 cores for the ledger.
CPU_TRAINING_LANES = (("g4", 16, 8), ("g3", 6, 8), ("c4", 6, 8), ("c5", 3, 8), ("c5", 1, 6))
FINAL_PARTITIONS = (*SPLITS, "external")


def slurm(snap):
    return snap/"experiments/nextgen_h3_v2"


def submit(name, script, arguments, *, deps=(), gpu=False,
           array=None, limit="1-12:00:00", threads=None, memory=None, dependency="afterok", node=None,
           terminal_deps=()):
    if not all(str(x).isdecimal() for x in (*deps, *terminal_deps)):
        raise ValueError("Unexpected Slurm dependency")
    if dependency not in {"afterok", "afterany"}:
        raise ValueError("Unsupported Slurm dependency condition")
    n=threads or CPUS_PER_TRAINING_TASK
    reserve=memory or ("32G" if gpu else "52G")
    host = node or (GPU_NODE if gpu else CPU_NODE)
    partition = "gpu" if gpu or host == "g4" else "cpu"
    log=ROOT/"logs"/(f"{name}-%A_%a.log" if array else f"{name}-%j.log")
    command=["sbatch","--parsable","-J",f"ng2-h3-{name}","-p",partition,
             f"--nodelist={host}","-c",str(n),f"--mem={reserve}","-t",limit,"-o",str(log)]
    if gpu:
        command.append(f"--gres=gpu:{GPU_TYPE}:1")
    if array is not None:
        command.append(f"--array={array}")
    conditions = [rule + ":" + ":".join(map(str, jobs))
                  for rule, jobs in ((dependency, deps), ("afterany", terminal_deps)) if jobs]
    if conditions:
        command.extend(["--kill-on-invalid-dep=yes", "--dependency=" + ",".join(conditions)])
    result=check_output([*command,str(script),*map(str,arguments)],cwd=ROOT,text=True).strip().split(";")[0]
    if not result.isdecimal():
        raise ValueError(f"Unexpected Slurm submission result: {result}")
    with (ROOT/"logs/trajectory.tsv").open("a") as log_tuple:
        print(name,result,GPU_TYPE if gpu else "cpu",array or "-",",".join(map(str, (*deps, *terminal_deps))),sep="\t",file=log_tuple)
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


def submit_training_arrays(name, script, arguments, backend, indexed, *, deps=(), terminal_deps=()):
    indices = sorted(indexed)
    if backend == "torch":
        if not indices:
            return []
        array = ",".join(map(str, indices)) + f"%{GPU_TRAINING_CONCURRENCY}"
        return [submit(name, script, arguments, gpu=True, array=array, deps=deps, terminal_deps=terminal_deps)]
    slots = [lane for seat in range(max(lane[1] for lane in CPU_TRAINING_LANES))
             for lane in CPU_TRAINING_LANES if seat < lane[1]]
    batches = {lane: [] for lane in CPU_TRAINING_LANES}
    for offset, index in enumerate(indices):
        batches[slots[offset % len(slots)]].append(index)
    return [submit(f"{name}-{node}-{threads}core", script, arguments, node=node,
                   array=",".join(map(str, batch)) + f"%{concurrency}", threads=threads, deps=deps)
            for (node, concurrency, threads), batch in batches.items() if batch]


def controller_dependency():
    job_id = os.environ.get("SLURM_JOB_ID")
    return (job_id,) if job_id else ()


def missing_inner(stage):
    return {backend: {index: files for index, files in tasks.items() if not all(p.is_file() for p in files)}
            for backend, tasks in inner_tasks(stage).items()}


def active_early_array(stage):
    route = ROOT / "locks" / f"{stage}-early-neural-jobs.json"
    if not route.is_file():
        return None
    record = read_json(route)
    plan = read_neural_expansion(stage)
    if record["early_plan_sha256"] != sha256(ROOT / "locks" / f"{stage}-early-neural.json"):
        raise ValueError("Early Slurm registration has a different settings lock")
    planned = {outer * 12 + inner * 4 + offset
               for outer, split in enumerate(SPLITS) for inner in range(3)
               for offset, family in enumerate(NEURAL_FAMILIES) if plan["splits"][split][family]}
    job = str(record["early_array"])
    if (planned != set(record["indices"]) or record["gpu_node"] != GPU_NODE
            or record["site_cap"] != GPU_TRAINING_CONCURRENCY or not job.isdecimal()):
        raise ValueError("Early Slurm registration does not match the current GPU schedule")
    queue = check_output(["squeue", "-r", "-h", "-u", getuser(), "-o", "%i|%j|%T|%N"], text=True)
    running = set()
    for row in queue.splitlines():
        identity, name, state, host = row.split("|")
        parent, _, index = identity.partition("_")
        if parent == job:
            if (name != f"ng2-h3-{stage}-early-neural" or host not in {"", GPU_NODE} or not index.isdecimal()
                    or int(index) not in planned or state not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}):
                raise ValueError(f"Unexpected early neural allocation {row}")
            running.add(int(index))
    return (job, planned) if running else None


def schedule_inner(stages, snap, *, retry=False, tasks=None):
    stage = stages[-1]
    planned = missing_inner(stage) if tasks is None else tasks
    suffix = "-retry" if retry else ""
    helpers = slurm(snap)
    early = active_early_array(stage)
    future = active_early_array(next_boundary(stage)) if stage.startswith("boundary-") else None
    jobs = [early[0]] if early else []
    predecessor = controller_dependency()
    for backend, indexed in planned.items():
        drains = ()
        if backend == "torch":
            if early:
                indexed = set(indexed) - early[1]
            drains = tuple(array[0] for array in (early, future) if array)
        jobs.extend(submit_training_arrays(f"{stage}-{backend}{suffix}", helpers / "inner.sbatch",
                                           [snap, stage, backend], backend, indexed, deps=predecessor,
                                           terminal_deps=drains))
    if not jobs:
        return advance(stages, snap, dispatch_only=False)
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
    missing = missing_inner(stage)
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
    jobs={"torch":[],"gbdt":[]}
    for partition in FINAL_PARTITIONS:
        for family in FAMILIES:
            settings=external_operating_point(selected,family) if partition=="external" else (
                selected["splits"][partition]["families"][family])
            backend=settings["config"]["backend"]
            jobs[backend].extend({"split":partition,"family":family,"seed":seed} for seed in SEEDS)
    lock=ROOT/"locks/final-jobs.json"
    publish_json(lock,jobs)
    return schedule_final_attempt(snap, missing_final(jobs), retried=False)


def missing_final(jobs):
    atlas = not (ROOT / "features/external/receipt.json").is_file()
    fits = {backend: [i for i, item in enumerate(group) if not (
        ROOT / "final" / item["split"] / item["family"] / str(item["seed"]) / "receipt.json").is_file()]
            for backend, group in jobs.items()}
    references = [i for i, split in enumerate(FINAL_PARTITIONS) if not (
        ROOT / "final" / split / "references/receipt.json").is_file()]
    return {"atlas": atlas, "fits": fits, "references": references}


def schedule_final_attempt(snap, missing, *, retried):
    script = slurm(snap) / "job.sbatch"
    suffix = "-retry" if retried else ""
    previous = controller_dependency()
    atlas = ()
    if missing["atlas"]:
        atlas = (submit("external-atlas" + suffix, script, [snap, "-m", "h3v2.final_fit", "external-atlas"],
                        threads=8, memory="28G", limit="01:00:00", deps=previous),)
    finals = {backend: submit_training_arrays(f"final-{backend}{suffix}", script,
              [snap, "-m", "h3v2.pipeline", "fit-task", "--backend", backend],
              backend, group, deps=atlas or previous) for backend, group in missing["fits"].items()}
    refs = ()
    if missing["references"]:
        array = ",".join(map(str, missing["references"])) + "%3"
        parent = (*atlas, *finals["gbdt"]) or previous
        refs = (submit("references" + suffix, script, [snap, "-m", "h3v2.pipeline", "references-task"],
                       deps=parent, array=array, threads=6, memory="20G", limit="02:00:00"),)
    watched = (*atlas, *finals["gbdt"], *finals["torch"], *refs)
    if not watched:
        return settle_final(snap, retried=retried)
    arguments = [snap, "-m", "h3v2.pipeline", "settle-final", "--snapshot", snap]
    if retried:
        arguments.append("--retried")
    return submit("final-settle" + suffix, script, arguments, deps=watched, threads=6, memory="18G",
                  limit="01:00:00", dependency="afterany")


def settle_final(snap, *, retried):
    if (ROOT / "score/consumed.json").exists():
        raise ValueError("Final fitting is closed after the outer release")
    missing = missing_final(read_json(ROOT / "locks/final-jobs.json"))
    if missing["atlas"] or any(missing["fits"].values()) or missing["references"]:
        if retried:
            raise RuntimeError(f"Incomplete final artifacts after the registered retry: {missing}")
        publish_json(ROOT / "locks/final-missing-retry.json", missing)
        return schedule_final_attempt(snap, missing, retried=True)
    frozen_sha256 = sha256(ROOT / "locks/frozen.json")
    parents = controller_dependency()
    publish_json(ROOT / "locks/score-dispatch.json", {"settled_by": parents, "frozen_sha256": frozen_sha256})
    helpers=slurm(snap)
    before=submit("preflight",helpers/"job.sbatch",[snap,"-m","h3v2.score","--preflight"],
                  deps=parents,threads=8,memory="42G",limit="03:00:00")
    after=submit("score-once",helpers/"job.sbatch",[snap,"-m","h3v2.score"],
                 deps=(before,),threads=18,memory="66G",limit="1-00:00:00")
    package=submit("report",helpers/"job.sbatch",[snap,"-m","h3v2.report"],
                   deps=(after,),threads=6,memory="20G",limit="02:00:00")
    publish_json(ROOT/"locks/terminal_jobs.json",{"settled_by": parents,"preflight":before,"score":after,"reports":package,
                                                   "frozen_sha256":frozen_sha256})
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
    if successor is not None:
        print("Next phase will be handled by",successor,flush=True)
    return successor


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("advance","dispatch","settle","settle-final","fit-task","references-task"))
    p.add_argument("stages",nargs="*")
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--backend",choices=("gbdt","torch"))
    p.add_argument("--retried", action="store_true")
    args=p.parse_intermixed_args()
    if args.command in ("advance","dispatch"):
        advance(args.stages,args.snapshot,dispatch_only=args.command=="dispatch")
    elif args.command=="settle":
        settle_inner(args.stages,args.snapshot,retried=args.retried)
    elif args.command=="settle-final":
        settle_final(args.snapshot,retried=args.retried)
    else:
        from .final_fit import final_candidate,final_references
        index=int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.command=="fit-task":
            work=read_json(ROOT/"locks/final-jobs.json")[args.backend]
            final_candidate(**work[index])
        else:
            final_references(FINAL_PARTITIONS[index])
