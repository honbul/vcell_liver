"""Continue a signed inner search from its final configuration."""
import argparse
import json
from pathlib import Path

from nextgen_h1_anchor.inputs import json_write
from vc_build.io import sha256

from .data import TASKS
from .search import Search, budget_diagnostics, cache_read, extrapolate, key


def close(task, family, result, r1, out, device):
    source = result / task / family / "result.json"
    payload = json.loads(source.read_text())
    if payload["outer_labels_opened"]:
        raise ValueError("Cannot refine a search that opened held outcomes")
    output = out / task / family
    output.mkdir(parents=True)
    states, policies, _, names, moa = cache_read(r1 / task)
    for mode, record in payload["modes"].items():
        search = Search(states, mode, policies[mode], device)
        search.evaluations = {key(e["config"]): e for e in record["grid"]}
        search.curves = record["learning_curves"]
        initial = key(record["selected"]["config"])
        limits = []
        for i in range(8):
            current = search.winner()["config"]
            proposals, edges = extrapolate(current, [r["config"] for r in search.evaluations.values()], search.dual)
            unseen = {key(c): c for c in proposals if key(c) not in search.evaluations}
            limits.append(dict(iteration=i, incumbent=current, edges=edges, new_trials=len(unseen)))
            if not unseen:
                break
            for candidate in unseen.values():
                search.score(candidate, f"round3-final-boundary-{i+1}")
            print(task, family, mode, "boundary cycle", i, len(unseen), flush=True)
        selected = search.winner()
        remaining, last_edges = extrapolate(selected["config"], [e["config"]
                                              for e in search.evaluations.values()], search.dual)
        checked = [c for c in remaining if key(c) not in search.evaluations]
        record.update(selected=selected, grid=list(search.evaluations.values()), learning_curves=search.curves,
                      final_boundary=last_edges, additional_boundary_search=limits,
                      remaining_numerical_boundary_trials=len({key(c) for c in checked}))
        if key(selected["config"]) != initial:
            record["budget"], record["errors"] = budget_diagnostics(search, selected["config"], moa, names)
    payload["resume_sha256"] = sha256(source)
    json_write(output / "result.json", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("result", "r1", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("families")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task")
    args = parser.parse_args()
    from os import environ
    i = int(environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task = args.task or TASKS[i % len(TASKS)]
    close(task, args.families[i // len(TASKS)], args.result, args.r1, args.out, args.device)
