"""Round 2 independent, controlled and integrated view selection."""
import argparse
import itertools
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from nextgen_h2.run import read_json
from vc_build.io import publish_json

from .kernel import OMICS, View, design
from .search import common_score, expand_axis, item, ranker, score_index, stage_input, write_manifest

WEIGHTS = (0., .05, .10, .25, .50, 1.)


def modality(view):
    return "cell" if view.name in OMICS else "drug"


def support(view):
    return [view.name, view.recipe, modality(view)]


def environment(base, view, control="real"):
    spec = replace(view, control=control)
    if modality(view) == "cell":
        return replace(base, cell_views=(spec,))
    return replace(base, drug_views=(spec,))


def entry(base, v, name=None):
    tag = name or f"{v.name}/{v.recipe}/{v.transform}/{v.dimension}/{v.weight:g}"
    return item(tag, environment(base, v), axis="view", view=asdict(v), support=support(v))


def options():
    morphology = [View("painting", "cache", trans, d) for trans, d in [("raw", 0), ("pca", 32),
                                                                   ("white", 64), ("zca", 64), ("select", 128)]]
    tahoe = [View("tahoe", recipe, trans, d) for recipe in ("mean", "heterogeneity", "dose")
             for trans, d in (("raw", 0), ("white", 32))]
    genetics = [View(name, "default", trans, d) for name, trans, d in
                [("mut", "raw", 0), ("cnv", "white", 128), ("fusion", "white", 64),
                 ("sig", "raw", 0), ("crispr", "white", 128)]]
    return [replace(record, weight=w) for record in [*morphology, *tahoe, *genetics] for w in WEIGHTS]


def first(anchor_file, path):
    anchors = read_json(anchor_file)
    rows = []
    for k in range(3):
        base = design(anchors[str(k)]["design"])
        rows.append([item("R1", base, axis="baseline", all_supports=True)]
                    + [entry(base, view) for view in options()])
    write_manifest(path, rows)


def program_variants(work, contracts, out, anchor_file, stages):
    anchors = read_json(anchor_file)
    picks, all_points, future, controls = {}, [], [], []
    for k in range(3):
        base = design(anchors[str(k)]["design"])
        possible, points = [], []
        for stage in stages:
            ss, _, p = stage_input(work, contracts, stage, k)
            possible.extend(x for x in ss if x["axis"] == "view")
            points.append(p)
        summary = pd.concat(points)
        scores = score_index(summary.drop_duplicates(["support", "axis", "arm"], keep="last"))
        next_phase, check = [], []
        selected = {}
        for name in sorted({x["view"]["name"] for x in possible}):
            candidates = [x for x in possible if x["view"]["name"] == name]
            # Rank across representations on the same all-drug denominator. Then validate on
            # the selected view's exact support against the mask and shuffled controls.
            best = max(candidates, key=lambda x: ranker(scores, x["id"], "all"))
            view = View(**best["view"])
            vals = [x["view"]["weight"] for x in candidates if
                    (x["view"]["recipe"], x["view"]["transform"], x["view"]["dimension"]) ==
                    (view.recipe, view.transform, view.dimension)]
            for weight in expand_axis(vals, view.weight):
                next_phase.append(entry(base, replace(view, weight=weight)))
            selected[name] = best
            for control in ("real", "mask", "shuffle"):
                check.append(item(f"{name}/{control}", environment(base, view, control),
                                  axis=control, support=support(view), view=asdict(view)))
        picks[str(k)] = selected
        all_points.append(summary)
        future.append(next_phase)
        controls.append([item("R1", base, axis="baseline", all_supports=True), *check])
    out.mkdir(exist_ok=False)
    pd.concat(all_points).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "best.json", picks | {"settled": not any(future)})
    if any(future):
        for k, rows in enumerate(future):
            if not rows:
                rows.append(item("incumbent-check", design(anchors[str(k)]["design"]), axis="replay"))
        write_manifest(out / "next.json", future)
    else:
        write_manifest(out / "controls.json", controls, interaction=True)


def gates(work, contracts, out, anchor_file):
    anchors = read_json(anchor_file)
    reports, selected, joint = [], {}, []
    for k in range(3):
        base = design(anchors[str(k)]["design"])
        specs, unit, table = stage_input(work, contracts, "view_controls", k)
        passing = {"drug": [], "cell": []}
        evidence = {}
        for arm in [x for x in specs if x["axis"] == "real"]:
            view = View(**arm["view"])
            name = view.name
            same = "/".join(support(view))
            comparisons = {label: common_score(unit, control, arm["id"], same)
                           for label, control in (("vs_R1", "R1"), ("vs_mask", name + "/mask"),
                                                  ("vs_shuffle", name + "/shuffle"))}
            guard = common_score(unit, "R1", arm["id"], same, "ndcg_at_5")
            clears = all(v["delta"] > 0 and v["lo"] > 0 for v in comparisons.values()) and guard["lo"] >= -.005
            if clears and name != "crispr":
                passing[modality(view)].append(view)
            evidence[name] = {"view": asdict(view), "comparisons": comparisons, "ndcg": guard,
                              "pass": bool(clears), "eligible_for_champion": name != "crispr"}
        candidates = [item("R1", base, axis="baseline")]
        drug = passing["drug"]
        for i in range(1, len(drug) + 1):
            for combination in itertools.combinations(drug, i):
                names = "+".join(v.name for v in combination)
                candidates.append(item("drug/" + names, replace(base, drug_views=combination), axis="drug"))
        cells = passing["cell"]
        # Retain independent omics arms, a pooled arm and the eligible assay combinations.
        cell_sets = [tuple([v]) for v in cells]
        if len(cells) > 1:
            cell_sets.append(tuple(cells))
        for upstream in list(candidates):
            spec = design(upstream["design"])
            for group in cell_sets:
                names = "+".join(v.name for v in group)
                candidates.append(item(upstream["id"] + "/cell/" + names, replace(spec, cell_views=group),
                                       axis="cell", drug_reference=upstream["id"]))
        joint.append(candidates)
        reports.append(table)
        selected[str(k)] = evidence
    out.mkdir(exist_ok=False)
    pd.concat(reports).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "gates.json", selected)
    write_manifest(out / "integration.json", joint, interaction=True)


def integration(work, contracts, out):
    policies, records = {}, []
    for k in range(3):
        ss, unit, table = stage_input(work, contracts, "integration", k)
        scores = score_index(table)

        def eligible(arm, control):
            a = common_score(unit, control, arm["id"], "all")
            nd = common_score(unit, control, arm["id"], "all", "ndcg_at_5")
            return a["lo"] > 0 and nd["lo"] >= -.005

        bas = ss[0]
        drug = max([bas, *[m for m in ss if m["axis"] == "drug" and eligible(m, "R1")]],
                   key=lambda x: ranker(scores, x["id"], "all"))
        cell = max([drug, *[m for m in ss if m["axis"] == "cell"
                            and m["drug_reference"] == drug["id"] and eligible(m, drug["id"])]],
                   key=lambda x: ranker(scores, x["id"], "all"))
        policies[str(k)] = {"R1": bas, "R2a": drug, "R2b": cell}
        records.append(table)
    out.mkdir(exist_ok=False)
    pd.concat(records).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "kernel_max.json", policies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first", "expand", "gates", "integrate"))
    parser.add_argument("target", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--stages", nargs="+")
    a = parser.parse_args()
    if a.stage == "first":
        first(a.anchors, a.target)
    elif a.stage == "expand":
        program_variants(a.work, a.contracts, a.target, a.anchors, a.stages)
    elif a.stage == "gates":
        gates(a.work, a.contracts, a.target, a.anchors)
    else:
        integration(a.work, a.contracts, a.target)
