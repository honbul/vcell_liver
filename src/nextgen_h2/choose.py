"""Select policies from inner data only and write immutable decision receipts."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from vc_build.io import publish_json, sha256

from .metrics import contrast, macro
from .model import Fusion, Representation, WEIGHTS
from .run import baseline_options, base_key, read_json, representations

R1_WEIGHT = .25


def collected(root, stage, outer):
    parts = {"drug": [], "cell": []}
    for inner in range(3):
        target = root / stage / f"{outer}-{inner}"
        proof = read_json(target / "receipt.json")
        if (proof["outer"], proof["inner"], proof["outer_response_rows_loaded"]) != (outer, inner, 0):
            raise ValueError("Unexpected inner forecast registration")
        for axis in parts:
            path = target / f"{axis}.parquet"
            if sha256(path) != proof[f"{axis}_sha256"]:
                raise ValueError(f"Changed inner metric artifact {path}")
            parts[axis].append(pd.read_parquet(path))
    return {axis: pd.concat(items, ignore_index=True) for axis, items in parts.items()}


def scored(table, arm, support, metric="spearman"):
    return macro(table[table.arm.eq(arm) & table.support.eq(support)], metric)


def diff(table, proposal, control, support, metric="spearman"):
    use = table[table.support.eq(support)]
    point, lower, upper, clusters = contrast(use, proposal, control, metric)
    return {"delta": point, "lo": lower, "hi": upper, "clusters": clusters}


def h2a(mask, shuffle):
    return bool(mask["delta"] >= .02 and mask["lo"] > 0 and shuffle["lo"] > 0)


def alternative_weight(table, view, name):
    fractions = ("0", "0.25", "0.5")

    def key(weight):
        scores = [scored(table, f"{view}/{x}/{name}/{weight:g}", f"{view}/{x}/all") for x in fractions]
        return sum(scores) / 3

    weight = max(WEIGHTS, key=key)
    return weight, key(weight)


def candidate(configs, spec, **extra):
    return {"configs": [asdict(cfg) for cfg in configs], "spec": asdict(spec)} | extra


def settings_key(item):
    return tuple(item["configs"][i]["view"] for i in range(len(item["configs"])))


def select_fusion(frames, registered):
    drug, cell = frames["drug"], frames["cell"]
    all_selected = {}
    for view, entry in registered.items():
        if not entry["pass"]:
            continue
        config = Representation(**entry["config"])
        score = {name: alternative_weight(drug, view, name)
                 for name in ("concat_zero", "concat_mean", "psd", "norm", "gated")}
        naive = max(("concat_zero", "concat_mean"), key=lambda name: score[name][1])
        naive_model = candidate([config], Fusion(naive, score[naive][0]))
        name0 = f"{view}/0/{naive}/{score[naive][0]:g}"
        options = {}
        for name in ("psd", "norm", "gated"):
            weight, stress_gain = score[name]
            tag = f"{view}/0/{name}/{weight:g}"
            rho = diff(drug, tag, name0, f"{view}/0/all")
            ndcg = diff(cell, tag, name0, f"{view}/0/all", "ndcg_at_5")
            passes = ((rho["delta"] >= .02 or rho["lo"] > 0) and ndcg["lo"] >= -.01)
            options[name] = {"rho": rho, "ndcg": ndcg, "pass": bool(passes), "stress_macro": stress_gain}
        challenger = max(options, key=lambda name: options[name]["stress_macro"])
        chosen = max((name for name in options if options[name]["pass"]),
                     key=lambda name: options[name]["stress_macro"], default=naive)
        all_selected[view] = {
            "champion": candidate([config], Fusion(chosen, score[chosen][0]), naive=naive_model),
            "challenger": candidate([config], Fusion(challenger, score[challenger][0])),
            "naive": naive_model, "candidates": options, "stress_macro": score[chosen][1],
            "natural_rho": scored(drug, f"{view}/0/{chosen}/{score[chosen][0]:g}", f"{view}/0/all"),
            "weights_and_stress_macro": score,
        }
    best = max(all_selected, key=lambda view: all_selected[view]["stress_macro"], default=None)
    champion = None if best is None else all_selected[best]["champion"]
    dual = None
    if len(all_selected) == 2:
        dual_rho = scored(drug, "dual/norm/.25", "dual/all")
        dual = {"natural_rho": dual_rho, "best_single_natural_rho": all_selected[best]["natural_rho"]}
        if dual_rho >= dual["best_single_natural_rho"] + .02:
            configs = [Representation(**registered[view]["config"]) for view in sorted(all_selected)]
            champion = candidate(configs, Fusion("norm"), naive=all_selected[best]["naive"])
            dual["selected"] = True
        else:
            dual["selected"] = False
    lincs = Representation(**registered["lincs"]["config"])
    path_config = replace(lincs, transform="pathway", dimension=44)
    path = None
    if not all_selected:
        path_mask = diff(drug, "path_diag/C4", "path_diag/C2", "lincs_path_observed")
        path_shuffle = diff(drug, "path_diag/C4", "path_diag/C3", "lincs_path_observed")
        path = {"mask": path_mask, "permutation": path_shuffle, "pass": h2a(path_mask, path_shuffle)}
        if path["pass"]:
            champion = candidate([path_config], Fusion())
    return {"champion": champion, "path_config": asdict(path_config), "views": all_selected, "best_view": best,
            "dual": dual, "path_diagnostic": path}


def select_matching(drug, plan):
    support = "match_observed"
    scores = {}
    for strength in (.05, .20):
        arm = f"match/direct/{strength:g}"
        geom = diff(drug, arm, "match/geometry", support)
        mask = diff(drug, arm, "match/C2", support)
        shuffled = diff(drug, arm, f"match/shuffled/{strength:g}", support)
        scores[strength] = {"geometry": geom, "mask": mask, "shuffle": shuffled,
                           "pass": geom["delta"] > 0 and geom["lo"] > 0 and h2a(mask, shuffled),
                           "all_rho": scored(drug, arm, "match_all")}
    best = max((x for x in scores if scores[x]["pass"]), key=lambda x: scores[x]["all_rho"], default=None)
    champ = plan["champion"]
    if best is not None:
        if champ is None or settings_key(champ) == ("painting",):
            # A second view cannot join an individually supported view unless both clear Round 1.
            # Painting selection is retained unless a like-for-like comparison is available.
            if champ is None:
                champ = candidate([Representation(**plan["path_config"])], Fusion(match=best))
            else:
                best = None
        else:
            champ = champ | {"spec": champ["spec"] | {"match": best}}
    return champ, {"strengths": scores, "selected_strength": best}


def external_policy(policies):
    def signature(item):
        c = item["champion"]
        return "structure" if not c else "+".join(settings_key(c)) + "/" + c["spec"]["method"]

    votes = [signature(item) for item in policies]
    selected = max(range(3), key=lambda outer: (votes.count(votes[outer]), -outer))
    return policies[selected] | {"external_inner_policy_from_outer_fit": selected,
                                 "external_votes": votes}


def choose(root, stage):
    names = {"r1base": "base.json", "r1": "r1_select.json", "r1controls": "r1_gate.json",
             "r2": "r2_select.json", "r3": "champion.json"}
    output = root / names[stage]
    if output.exists():
        raise FileExistsError(output)
    result, audit = {}, {}
    point_rows = []
    for outer in range(3):
        frames = collected(root, stage, outer)
        drug, cell = frames["drug"], frames["cell"]
        for axis, all_rows, valid in (("drug", drug, ("spearman", "pearson")),
                                      ("cell", cell, ("ndcg_at_5", "precision_at_5"))):
            for (support, arm), rows in all_rows.groupby(["support", "arm"], sort=True):
                point_rows.append({"outer": outer, "stage": stage, "axis": axis, "arm": arm, "support": support,
                                   "units": int(len(rows)), **{metric: macro(rows, metric) for metric in valid}})
        if stage == "r1base":
            bases = baseline_options()
            selected = max(bases, key=lambda base: scored(drug, base_key(base), "all"))
            result[str(outer)] = asdict(selected)
        elif stage == "r1":
            arms = list(representations())
            result[str(outer)] = {}
            for view in ("painting", "lincs"):
                support = view + "_common"
                candidates = [item for item in arms if item.view == view]

                def effect(cfg):
                    return (scored(drug, "C4/" + cfg.key, support)
                            - scored(drug, f"C2/{view}/{cfg.recipe}", support))
                config = max(candidates, key=effect)
                result[str(outer)][view] = asdict(config)
        elif stage == "r1controls":
            configs = read_json(root / "r1_select.json")[str(outer)]
            result[str(outer)] = {}
            for view, config in configs.items():
                proof = {}
                for suffix in ("observed", "all"):
                    support = view + "_" + suffix
                    proof[suffix] = {f"C4_minus_{ref}": diff(drug, view + "/C4", view + "/" + ref, support)
                                     for ref in ("C0", "C1", "C2", "C3")}
                selected = proof["observed"]
                result[str(outer)][view] = {"pass": h2a(selected["C4_minus_C2"], selected["C4_minus_C3"]),
                                            "config": config, "contrasts": proof}
        elif stage == "r2":
            result[str(outer)] = select_fusion(frames, read_json(root / "r1_gate.json")[str(outer)])
        elif stage == "r3":
            plan = read_json(root / "r2_select.json")[str(outer)]
            champ, matching = select_matching(drug, plan)
            result[str(outer)] = {"base": read_json(root / "base.json")[str(outer)],
                "r1": read_json(root / "r1_gate.json")[str(outer)], "r2": plan, "r3": matching,
                "champion": champ}
        audit[str(outer)] = [sha256(root / stage / f"{outer}-{inner}/receipt.json") for inner in range(3)]
    if stage == "r3":
        result["3"] = external_policy([result[str(fold)] for fold in range(3)])
    pd.DataFrame(point_rows).to_csv(root / f"{stage}_points.tsv", sep="\t", index=False)
    publish_json(output, result)
    publish_json(root / f"{stage}_decision_receipt.json", {
        "stage": stage, "selection_sha256": sha256(output), "inner_receipts": audit,
        "outer_outcomes_opened": False, "points_sha256": sha256(root / f"{stage}_points.tsv")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("r1base", "r1", "r1controls", "r2", "r3"))
    args = parser.parse_args()
    choose(args.root, args.stage)
