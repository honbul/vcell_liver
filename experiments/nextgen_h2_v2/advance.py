"""Advance the isolated adaptive Round 2/3 workflow through checked Slurm outputs."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


class Queue:
    def __init__(self, root, code):
        self.root, self.code = root, code
        self.study = root / "study"
        self.contracts = root / "contracts"
        self.anchors = root / "run-anchors-01" / "selected.json"
        self.job = code / "experiments/nextgen_h2_v2/job.sbatch"
        self.log = root / "round2-and-3-jobs.tsv"

    def submit(self, stage, module, *args, gpu=False, after=None):
        flag = ["-p", "cpu"]
        if gpu:
            flag = ["-p", "gpu", "--gres=gpu:1", "--exclude=g1,g2", "--array=0-8%3"]
        if after:
            flag += ["--dependency=afterok:" + after]
        name = "ng2-h2-" + stage
        output = self.root / "logs" / ("%x-%A_%a.log" if gpu else "%x-%j.log")
        cmd = ["sbatch", "--parsable", "--chdir", self.root, "-J", name, "-o", output,
               *flag, self.job, self.code, module, *args]
        identifier = subprocess.check_output([str(x) for x in cmd], text=True, timeout=30).strip().split(";")[0]
        contents = [identifier, name, "GPU array max 3" if gpu else "CPU", module, " ".join(map(str, args))]
        with self.log.open("a") as out:
            out.write("\t".join(contents) + "\n")
        print(*contents[:3], flush=True)
        return identifier

    def check(self, job, paths):
        # NFS visibility can trail Slurm's terminal notification on a different host.
        absent_since = None
        while True:
            try:
                p = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T|%R"],
                                   capture_output=True, text=True, timeout=20)
                live = p.stdout.strip() != ""
                if "DependencyNeverSatisfied" in p.stdout:
                    raise RuntimeError(f"Slurm dependency failed for {job}: {p.stdout.strip()}")
                if p.returncode != 0 and "Invalid job id" not in p.stderr:
                    time.sleep(10)
                    continue
            except subprocess.TimeoutExpired:
                time.sleep(10)
                continue
            if live:
                absent_since = None
            elif all(path.is_file() for path in paths):
                print("verified", job, *[x.name for x in paths], flush=True)
                return
            elif absent_since is None:
                absent_since = time.monotonic()
            elif time.monotonic() - absent_since >= 55:
                raise RuntimeError(f"Terminal Slurm job {job} is missing {paths}")
            time.sleep(10)

    def copy(self, source, stage):
        destination = self.contracts / (stage + ".json")
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
        return destination

    def inner(self, phase, contract, name, after=None):
        return self.submit(name, "inner", self.study, self.root, contract, phase, gpu=True, after=after)

    def views(self, initial_job):
        stages = ["view1"]
        result = self.root / "run-views-01"
        self.check(initial_job, [result / "best.json"])
        while not read(result / "best.json")["settled"]:
            phase = f"view{len(stages)+1}"
            contract = self.copy(result / "next.json", phase)
            fit = self.inner(phase, contract, "r2-view-expand")
            stages.append(phase)
            result = self.root / f"run-views-{len(stages):02d}"
            r = self.submit("r2-select", "views", "expand", result, "--work", self.study,
                            "--contracts", self.contracts, "--anchors", self.anchors, "--stages", *stages, after=fit)
            self.check(r, [result / "best.json"])
        controls = self.copy(result / "controls.json", "view_controls")
        coupling = self.inner("view_controls", controls, "r2-controls")
        out = self.root / "run-view-controls"
        gated = self.submit("r2-gates", "views", "gates", out, "--work", self.study,
                            "--contracts", self.contracts, "--anchors", self.anchors, after=coupling)
        self.check(gated, [out / "gates.json", out / "integration.json"])
        integration = self.copy(out / "integration.json", "integration")
        kernel = self.inner("integration", integration, "r2-integration")
        result = self.root / "run-integration"
        picked = self.submit("r2-kernel-select", "views", "integrate", result, "--work", self.study,
                             "--contracts", self.contracts, after=kernel)
        self.check(picked, [result / "kernel_max.json"])
        return result / "kernel_max.json"

    def virtual(self):
        stages = ["virtual1"]
        contract = self.contracts / "virtual1.json"
        registry = self.submit("virtual-reg", "virtual", "register", contract)
        while True:
            phase = stages[-1]
            fit = self.submit("virtual-fit", "virtual", "measure", self.root, "--work", self.study,
                              "--root", self.root, "--manifest", contract, "--stages", phase, gpu=True, after=registry)
            result = self.root / f"run-virtual-{len(stages):02d}"
            reduction = self.submit("virtual-select", "virtual", "vary", result, "--work", self.study,
                                    "--contracts", self.contracts, "--stages", *stages, after=fit)
            self.check(reduction, [result / "winner.json"])
            if read(result / "winner.json")["settled"]:
                break
            stages.append(f"virtual{len(stages)+1}")
            contract = self.copy(result / "next.json", stages[-1])
            registry = None
        control = self.copy(result / "virtual_controls.json", "virtual_controls")
        fit = self.submit("virtual-controls", "virtual", "measure", self.root, "--work", self.study,
                          "--root", self.root, "--manifest", control, "--stages", "virtual_controls", gpu=True)
        selected = self.root / "virtual_gate.json"
        gates = self.submit("virtual-gate", "virtual", "gates", selected, "--work", self.study,
                            "--contracts", self.contracts, after=fit)
        self.check(gates, [selected])
        return selected

    def residual(self, kernel, virtual):
        manifest = self.contracts / "resid_strict.json"
        reg = self.submit("r3-reg", "residual_run", "manifest", manifest, "--kernel", kernel)
        fit = self.submit("r3-nonlinear", "residual_run", "measure", self.root, "--work", self.study,
                          "--root", self.root, "--manifest", manifest, gpu=True, after=reg)
        result = self.root / "run-final-strict"
        chosen = self.submit("r3-select", "residual_run", "decide", result, "--work", self.study,
                             "--contracts", self.contracts, "--kernel", kernel, "--virtual", virtual, after=fit)
        self.check(chosen, [result / "champion.json", result / "receipt.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("code", type=Path)
    parser.add_argument("view_reducer_job")
    a = parser.parse_args()
    runner = Queue(a.root, a.code)
    kernel = runner.views(a.view_reducer_job)
    virtual = runner.virtual()
    runner.residual(kernel, virtual)
