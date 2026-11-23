"""Publish small aggregate inner-CV evidence and cost provenance."""
import argparse
import json
from pathlib import Path
import posixpath
from zipfile import ZipFile

import numpy as np
import pandas as pd

from nextgen_h1_anchor.inputs import LICOB_LABEL, json_write
from vc_build.io import sha256
from .data import STUDY, TASKS, V1
from .freeze import FAMILIES, load_results
from .search import key


def inner(r1, result, kernel, final, out):
    out.mkdir()
    anchors, grid, budget, boundary, errors, validation, winners = [], [], [], [], [], [], []
    for task in TASKS:
        directory = r1 / task
        a_receipt = json.loads((directory / "round1.json").read_text())
        path = directory / "anchors.tsv"
        if sha256(path) != a_receipt["table_sha256"]:
            raise ValueError("Anchor policy receipt failed")
        anchors.append(pd.read_csv(path, sep="\t"))
        entries, bytes_by_family = load_results(task, result, kernel, final)
        for family, record in entries.items():
            for mode, r in record["modes"].items():
                tags = dict(task=task, family=family, modality=mode)
                top = r["selected"]
                for e in r["grid"]:
                    grid.append(dict(**tags, phase=e["phase"], objective=e["objective"], ndcg=e["ndcg_at_5"],
                                     rho=e["rho"], recipe=key(e["config"]), winner=e == top,
                                     fold_n="|".join(str(x["n"]) for x in e["folds"])))
                winners.append(dict(**tags, policy=r["policy"], round2=r["round2"]["objective"], objective=top["objective"],
                                    ndcg=top["ndcg_at_5"], rho=top["rho"], recipe=key(top["config"]),
                                    initial_gap=r["train_validation_gap"], remaining=r.get("remaining_numerical_boundary_trials", -1),
                                    receipt_sha256=bytes_by_family[family]))
                budget.extend(dict(**tags, **b) for b in r["budget"])
                errors.extend(dict(task=task, family=family, **b) for b in r["errors"])
                for s in r["saturation"]:
                    boundary.extend(dict(**tags, phase="expansion", cycle=s["iteration"], **e) for e in s["edges"])
                for s in r.get("additional_boundary_search", []):
                    boundary.extend(dict(**tags, phase="closure", cycle=s["iteration"], **e) for e in s["edges"])
                for e in r.get("final_boundary", []):
                    boundary.append(dict(**tags, phase="final", cycle=-1, **e))
                curves = r["learning_curves"].get(key(top["config"]), [])
                for fold, curve in enumerate(curves):
                    if family == "C":
                        validation.extend(dict(**tags, fold=fold, **row) for row in curve)
                    else:
                        for head, scores in curve.items():
                            metric = next(iter(scores["train"]))
                            validation.extend(dict(**tags, fold=fold, head=head, metric=metric, epoch=i+1,
                                                   train=x, val=y)
                                              for i, (x, y) in enumerate(zip(scores["train"][metric], scores["inner_val"][metric])))
    files = {"H1V2_INNER_ANCHORS.tsv":pd.concat(anchors,ignore_index=True), "H1V2_INNER_GRID.tsv":pd.DataFrame(grid),
             "H1V2_INNER_WINNERS.tsv":pd.DataFrame(winners), "H1V2_INNER_BUDGET.tsv":pd.DataFrame(budget),
             "H1V2_INNER_BOUNDARIES.tsv":pd.DataFrame(boundary), "H1V2_INNER_ERRORS.tsv":pd.DataFrame(errors),
             "H1V2_LEARNING_CURVES.tsv":pd.DataFrame(validation)}
    for name, data in files.items():
        data.to_csv(out / name, index=False, sep="\t", float_format="%.9g", na_rep="NA")
    json_write(out / "inner_receipt.json", {"tasks": len(TASKS), "families": list(FAMILIES),
               "outer_labels_opened": False, "outputs": {k: sha256(out / k) for k in files},
               "grid_rows": len(grid), "anchor_rows": sum(map(len, anchors)),
               "selected_candidates": len(winners)})


def workbook_layout(path):
    from xml.etree import ElementTree
    with ZipFile(path) as workbook:
        manifest = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        entries = {s.attrib["name"]: next(v for k, v in s.attrib.items() if k.endswith("}id"))
                   for s in manifest.iter() if s.tag.endswith("}sheet")}
        relations = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        paths = {r.attrib["Id"]: r.attrib["Target"] for r in relations}
        strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            strings = ["".join(t.text or "" for t in item.iter() if t.tag.endswith("}t"))
                       for item in ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))]
        titles = {}
        for sheet, relation in entries.items():
            if not sheet.startswith(("S5.", "S6.", "S7.", "S8.", "S9.", "S10.")):
                continue
            part = paths[relation]
            member = part.lstrip("/") if part.startswith("/") else posixpath.join("xl", part)
            with workbook.open(member) as stream:
                for _, element in ElementTree.iterparse(stream, events=("end",)):
                    if not element.tag.endswith("}row"):
                        continue
                    for cell in element:
                        value = next((x.text for x in cell.iter() if x.tag.endswith(("}v", "}t"))), None)
                        kind = cell.attrib.get("t")
                        if value and kind in {"s", "inlineStr", "str"}:
                            titles[sheet] = strings[int(value)] if kind == "s" else value
                            break
                    break
    return list(entries), titles


def cost(out):
    register = V1 / "support.json"
    expected = json.loads((STUDY / "receipt.json").read_text())["support_sha256"]
    actual = sha256(LICOB_LABEL)
    if sha256(register) != expected or json.loads(register.read_text())["file_sha256"][str(LICOB_LABEL)] != actual:
        raise ValueError("LICOB cost analysis requires the signed response workbook")
    sheets, titles = workbook_layout(LICOB_LABEL)
    options = []
    for k in (0, 1, 3, 5, 8, 12):
        for concentration in (11, 8, 7, 6):
            wells = 2 * k * concentration + 12
            options.append(dict(anchors=k, concentrations=concentration, wells=wells, one_plate=wells <= 96))
    prior = V1 / "summary-v1/inner_freeze_summary.json"
    json_write(out, {"workbook_sheets": sheets, "sheet_data_read": False,
                     "source": str(LICOB_LABEL), "source_sha256": actual, "registry_sha256": expected,
                     "response_sheet_titles": titles, "scope": "registered LICOB response workbook; title rows only",
                     "assumptions": {"replicates_per_dose": 2, "control_wells": 12},
                     "dose_specific_responses_registered": False, "plans": options,
                     "sensitivity": "iid additive anchor noise 0.10/0.25/0.50 inner-fit SD; distinct from empirical concentration downsampling",
                     "reference_counts": {"LICOB": 38, "LIMORE": 50, "Tier4_local": 0},
                     "v1_summary_sha256": sha256(prior)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name, fields in (("inner", ("r1", "result", "kernel", "final", "out")), ("cost", ("out",))):
        cli = commands.add_parser(name)
        for name in fields:
            cli.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    stage = args.pop("stage")
    {"inner": inner, "cost": cost}[stage](**args)
