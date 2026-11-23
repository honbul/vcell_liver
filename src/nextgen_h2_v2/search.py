"""Preregister coordinates and expand any empirically active grid boundary."""
import argparse
import itertools
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from nextgen_h2.metrics import contrast, macro
from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256

from .kernel import CHEM_AXES, SHAPES, Design, View, design

CORE_WEIGHTS = (.25, .5, 1., 2., 4., 8., 16.)
PENALTY = (.5, 1., 2., 4., 8., 16., 32., 64.)
MASK = (0., .05, .15, .30, .60)


def item(key, spec, **meta):
    return {"id": key, "design": asdict(spec)} | meta


def core():
    base = Design()
    return [item(f"ridge/{weight:g}/{penalty:g}",
                 replace(base, lincs=replace(base.lincs, weight=weight), penalty=penalty),
                 axis="ridge", weight=weight, penalty=penalty)
            for weight, penalty in itertools.product(CORE_WEIGHTS, PENALTY)]


def write_manifest(path, rows, *, interaction=False):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_json(path, {str(f): settings for f, settings in enumerate(rows)} | {"interaction": interaction})


def collect(root, stage, manifest, outer):
    out = {"drug": [], "cell": []}
    for i in range(3):
        path = root / stage / f"{outer}-{i}"
        receipt = read_json(path / "receipt.json")
        if (receipt["outer_query_labels_read"], receipt["outer"], receipt["inner"],
            receipt["study_manifest_sha256"]) != (0, outer, i, sha256(manifest)):
            raise ValueError("Changed inner partition or scoring manifest")
        for axis in out:
            blob = path / (axis + ".parquet")
            if sha256(blob) != receipt["files"][blob.name]:
                raise ValueError("Changed inner per-unit result")
            out[axis].append(pd.read_parquet(blob))
    return {name: pd.concat(parts, ignore_index=True) for name, parts in out.items()}


def points(unit, stage, outer):
    rows = []
    for axis, frame in unit.items():
        columns = (["spearman", "pearson", "interaction_spearman", "interaction_pearson"]
                    if axis == "drug" else ["ndcg_at_5", "precision_at_5"])
        for (arm, support), p in frame.groupby(["arm", "support"], sort=True):
            rows.append({"round_stage": stage, "outer_fit": outer, "axis": axis, "arm": arm,
                         "support": support, "fold_units": len(p)}
                        | {name: macro(p, name) for name in columns if name in p})
    return pd.DataFrame(rows)


def score_index(points):
    return points.set_index(["support", "axis", "arm"])


def ranker(scores, candidate, support="lincs"):
    drug = scores.loc[(support, "drug", candidate)]
    cells = scores.loc[(support, "cell", candidate)]
    return (drug.spearman, cells.ndcg_at_5)


def select_core(rows, scores):
    return max(rows, key=lambda x: ranker(scores, x["id"]))


def expand_axis(candidates, choice):
    vals = sorted(set(candidates))
    further = []
    if choice == vals[0] and choice > 0:
        further = [choice / 4, choice / 2]
    if choice == vals[-1]:
        further = [choice * 2, choice * 4]
    return further


def stage_fine(core_selection):
    selected = design(core_selection["design"])
    result = [item("incumbent", selected, axis="incumbent", value=0)]
    for field in CHEM_AXES:
        for multiplier in (.25, .5, 2., 4.):
            value = getattr(selected, field) * multiplier
            result.append(item(f"{field}/{value:g}", replace(selected, **{field: value}), axis=field, value=value))
    for mask in MASK:
        result.append(item(f"mask/{mask:g}", replace(selected, lincs=replace(selected.lincs, mask=mask)),
                           axis="mask", value=mask))
    for shape in SHAPES:
        result.append(item(f"shape/{shape}", replace(selected, lincs=replace(selected.lincs, kernel=shape)),
                           axis="shape", value=shape))
    for method in ("sum", "norm"):
        result.append(item(f"normalization/{method}", replace(selected, normalization=method),
                           axis="normalization", value=method))
    profiles = [("v1", "raw", 0)]
    for axis in ("landmark", "bing", "full"):
        for transform, dimension in (("raw", 0), ("pca", 64), ("pca", 128), ("white", 64), ("white", 128)):
            profiles.append((axis + "_mean", transform, dimension))
        for recipe in ("sd", "iqr", "multitime", "dose"):
            for transform, dimension in (("raw", 0), ("white", 64)):
                profiles.append((axis + "_" + recipe, transform, dimension))
    for recipe, trans, dim in profiles:
        view = replace(selected.lincs, recipe=recipe, transform=trans, dimension=dim)
        k = "/".join(map(str, [recipe, trans, dim]))
        result.append(item(f"expression/{k}", replace(selected, lincs=view), axis="expression", value=k,
                           support=["lincs", recipe, "drug"]))
    return result


def common_score(unit, reference, candidate, support, metric="spearman"):
    frame = unit["drug" if metric in ("spearman", "pearson", "interaction_spearman") else "cell"]
    sub = frame[frame.support.eq(support)]
    delta, lo, hi, clusters = contrast(sub, candidate, reference, metric)
    return {"delta": delta, "lo": lo, "hi": hi, "n_clusters": clusters}

def stage_input(work, contracts, stage, outer):
    source = contracts / (stage + ".json")
    specs = read_json(source)[str(outer)]
    unit = collect(work, stage, source, outer)
    return specs, unit, points(unit, stage, outer)


def core_update(work, contracts, out, stages):
    tables, winners, following = [], [], []
    for outer in range(3):
        specs, frames = [], []
        for stage in stages:
            ss, _u, t = stage_input(work, contracts, stage, outer)
            specs.extend(x for x in ss if x["axis"] == "ridge")
            frames.append(t)
        table = pd.concat(frames, ignore_index=True)
        scores = score_index(table.drop_duplicates(["support", "axis", "arm"], keep="last"))
        winner = select_core(specs, scores)
        winners.append(winner | {"lincs_score": ranker(scores, winner["id"])[0]})
        rows = []
        hit = {(d["weight"], d["penalty"]) for d in specs}
        ws = {a[0] for a in hit}
        ps = {a[1] for a in hit}
        base = design(winner["design"])
        new_w = expand_axis(ws, winner["weight"])
        new_pen = expand_axis(ps, winner["penalty"])
        pairs = ((list(itertools.product(new_w, sorted(ps | set(new_pen))))) +
                 (list(itertools.product(sorted(ws), new_pen))))
        for weight, penalty in pairs:
            if (weight, penalty) in hit:
                continue
            rows.append(item(f"ridge/{weight:g}/{penalty:g}",
                             replace(base, lincs=replace(base.lincs, weight=weight), penalty=penalty),
                             axis="ridge", weight=weight, penalty=penalty))
        following.append(rows)
        tables.append(table)
    out.mkdir(parents=True, exist_ok=False)
    pd.concat(tables).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "winners.json", {str(i): x for i, x in enumerate(winners)} | {
        "settled": not any(following), "stages": stages,
        "manifest_sha256": {stage: sha256(contracts / (stage + ".json")) for stage in stages}})
    if any(following):
        for i, rows in enumerate(following):
            if not rows:
                following[i] = [winners[i] | {"axis": "replay", "id": "incumbent-check"}]
        write_manifest(out / "next.json", following)
    else:
        write_manifest(out / "fine.json", [stage_fine(value) for value in winners])


def refine(work, contracts, out, stage="fine"):
    tables, selections, proposals = [], {}, []
    for outer in range(3):
        specs, _, summary = stage_input(work, contracts, stage, outer)
        scores = score_index(summary)
        incumbent = next(x for x in specs if x["axis"] == "incumbent")
        per_axis = {}
        for key in (*CHEM_AXES, "mask", "shape", "normalization", "expression"):
            candidates = [x for x in specs if x["axis"] == key] + [incumbent]
            selected = max(candidates, key=lambda x: ranker(scores, x["id"]))
            per_axis[key] = selected
        selected = max(specs, key=lambda x: ranker(scores, x["id"]))
        specs_winner = design(selected["design"])
        changes = {name: getattr(design(per_axis[name]["design"]), name) for name in (*CHEM_AXES, "normalization")}
        representation = design(per_axis["expression"]["design"]).lincs
        flags = replace(representation, mask=design(per_axis["mask"]["design"]).lincs.mask,
                        kernel=design(per_axis["shape"]["design"]).lincs.kernel)
        merged = replace(specs_winner, lincs=flags, **changes)
        bases = [("best", specs_winner), ("coordinate_join", merged)]
        result = [item(label, model, axis="incumbent", value=label) for label, model in bases]
        for label, model in bases:
            for w in sorted(set(CORE_WEIGHTS) | {model.lincs.weight / 2, model.lincs.weight * 2}):
                result.append(item(f"{label}/lincs/{w:g}", replace(model, lincs=replace(model.lincs, weight=w)),
                                   axis="lincs_weight", value=w))
            for p in sorted(set(PENALTY) | {model.penalty / 2, model.penalty * 2}):
                result.append(item(f"{label}/lambda/{p:g}", replace(model, penalty=p), axis="penalty", value=p))
            for axis in CHEM_AXES:
                x = getattr(model, axis)
                vals = {0., x / 4, x / 2, x, x * 2, x * 4}
                for v in sorted(vals):
                    result.append(item(f"{label}/{axis}/{v:g}", replace(model, **{axis: v}), axis=axis, value=v))
            for m in sorted(set(MASK) | {model.lincs.mask * 2, model.lincs.mask * 4}):
                result.append(item(f"{label}/presence/{m:g}", replace(model, lincs=replace(model.lincs, mask=m)),
                                   axis="mask", value=m))
        proposals.append(result)
        selections[str(outer)] = {"selected": selected, "axes": per_axis}
        tables.append(summary)
    out.mkdir(parents=True, exist_ok=False)
    pd.concat(tables).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "selected.json", selections)
    write_manifest(out / "polish.json", proposals)


def anchors(spec):
    result = [item("anchor/chem", spec, axis="intercept", value="chem")]
    for weight in (.01, .10, 1., 10.):
        result.append(item(f"anchor/krr/{weight:g}", replace(spec, intercept="krr", anchor_penalty=weight),
                           axis="intercept", value="krr"))
    for neighbors in (3, 5, 10, 20):
        result.append(item(f"anchor/neighbors/{neighbors}", replace(spec, intercept="neighbors", anchor_penalty=neighbors),
                           axis="intercept", value="neighbors"))
    return result


def polish(work, contracts, out, stages):
    tables, following, anchor_fits, chosen = [], [], [], {}
    directions = ("lincs_weight", "penalty", *CHEM_AXES, "mask")
    for outer in range(3):
        configs, tables_here = [], []
        for stage in stages:
            spec, _, t = stage_input(work, contracts, stage, outer)
            configs.extend(spec)
            tables_here.append(t)
        table = pd.concat(tables_here, ignore_index=True)
        scores = score_index(table.drop_duplicates(["support", "axis", "arm"], keep="last"))
        best = max(configs, key=lambda x: ranker(scores, x["id"]))
        base = design(best["design"])
        rows = []
        for axis in directions:
            vals = [x["value"] for x in configs if x["axis"] == axis]
            if axis in ("lincs_weight", "mask"):
                attr = "weight" if axis == "lincs_weight" else "mask"
                weight = getattr(base.lincs, attr)
                make = lambda x: replace(base, lincs=replace(base.lincs, **{attr: x}))
            else:
                weight = getattr(base, axis)
                make = lambda x: replace(base, **{axis: x})
            for v in expand_axis([*vals, weight], weight):
                rows.append(item(f"expand{len(stages)}/{axis}/{v:g}", make(v), axis=axis, value=v))
        following.append(rows)
        anchor_fits.append(anchors(base))
        chosen[str(outer)] = best | {"lincs_score": ranker(scores, best["id"])[0]}
        tables.append(table)
    out.mkdir(parents=True, exist_ok=False)
    pd.concat(tables).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "selected.json", chosen | {"settled": not any(following), "stages": stages})
    if any(following):
        for i, rows in enumerate(following):
            if not rows:
                following[i] = [chosen[str(i)] | {"axis": "replay", "id": f"replay{len(stages)}"}]
        write_manifest(out / "next.json", following)
    else:
        write_manifest(out / "anchors.json", anchor_fits)


def select_anchors(work, contracts, out, stages):
    rows, final, control, following = [], {}, [], []
    for outer in range(3):
        spec, tables = [], []
        for stage in stages:
            options, _, table = stage_input(work, contracts, stage, outer)
            spec.extend(x for x in options if x["axis"] == "intercept")
            tables.append(table)
        table = pd.concat(tables, ignore_index=True)
        scores = score_index(table.drop_duplicates(["support", "axis", "arm"], keep="last"))
        best = max(spec, key=lambda x: ranker(scores, x["id"], "all")[1])
        base = design(best["design"])
        final[str(outer)] = best | {"all_ndcg": ranker(scores, best["id"], "all")[1]}
        nxt = []
        if base.intercept == "krr":
            vals = [x["design"]["anchor_penalty"] for x in spec if x["value"] == "krr"]
            for ridge in expand_axis(vals, base.anchor_penalty):
                nxt.append(item(f"anchor/krr/{ridge:g}", replace(base, anchor_penalty=ridge),
                                axis="intercept", value="krr"))
        following.append(nxt)
        ctrl = [item("R1", base, axis="control")]
        for kind in ("mask", "shuffle"):
            ctrl.append(item("LINCS/" + kind, replace(base, lincs=replace(base.lincs, control=kind)), axis="control"))
        control.append(ctrl)
        rows.append(table)
    out.mkdir(parents=True, exist_ok=False)
    pd.concat(rows).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "selected.json", final | {"settled": not any(following)})
    if any(following):
        for i, options in enumerate(following):
            if not options:
                options.append(final[str(i)] | {"id": "incumbent-check", "axis": "replay"})
        write_manifest(out / "next.json", following)
    else:
        write_manifest(out / "r1controls.json", control, interaction=True)


def r1gate(work, contracts, out):
    summaries, results = [], {}
    for outer in range(3):
        spec, unit, points = stage_input(work, contracts, "r1controls", outer)
        del spec
        reports = {}
        for support in ("lincs", "all"):
            reports[support] = {
                arm: {metric: common_score(unit, arm, "R1", support, metric) for metric in
                      ("spearman", "interaction_spearman", "ndcg_at_5")}
                for arm in ("H2-v1", "LINCS/mask", "LINCS/shuffle")}
        results[str(outer)] = reports
        summaries.append(points)
    out.mkdir(exist_ok=False)
    pd.concat(summaries).to_csv(out / "points.tsv", sep="\t", index=False, na_rep="NA")
    publish_json(out / "paired.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("core", "core_update", "refine", "polish", "anchors", "r1gate"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--contracts", type=Path)
    parser.add_argument("--stages", nargs="+")
    a = parser.parse_args()
    if a.stage == "core":
        write_manifest(a.out, [core()] * 3)
    elif a.stage == "core_update":
        core_update(a.work, a.contracts, a.out, a.stages)
    elif a.stage == "refine":
        refine(a.work, a.contracts, a.out)
    elif a.stage == "polish":
        polish(a.work, a.contracts, a.out, a.stages)
    elif a.stage == "anchors":
        select_anchors(a.work, a.contracts, a.out, a.stages or ["anchors"])
    else:
        r1gate(a.work, a.contracts, a.out)
