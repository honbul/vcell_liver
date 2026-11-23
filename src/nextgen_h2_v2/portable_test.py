"""Exercise the packaged CLI on an exact parent outside the fit catalogue."""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nextgen_h2.assets import KEY, canonical_chemistry
from nextgen_h2.run import read_json, response, subfold
from vc_build.io import publish_json, sha256

from .kernel import Engine, context, design
from .portable import create_kit


def test(work, out):
    # The child entrypoint independently activates its socket audit guard.
    dest = work / "forecasts" / "3"
    path = dest / "portable.pkl"
    frozen = read_json(dest / "receipt.json")
    digest = sha256(path)
    if digest != frozen["files"][path.name]:
        raise ValueError("The offline CLI package changed")
    with path.open("rb") as handle:
        model = pickle.load(handle)
    smiles = next(x for x in ("CCO", "OCCO", "CC(C)CO") if canonical_chemistry(x)[1] not in model.known_drugs)
    names = list(model.rna.index[:4])
    out.mkdir(exist_ok=False, parents=True)
    base = [sys.executable, "-m", "nextgen_h2_v2.portable", str(path), "--sha256", digest,
            "--smiles", smiles, "--cell-id", *names]
    env = os.environ | {"PYTHONDONTWRITEBYTECODE": "1"}
    structural = json.loads(subprocess.check_output(base, text=True, env=env))
    signature = np.random.default_rng(29).normal(0, .25, size=len(model.projector.columns)).astype(np.float32)
    feature = out / "synthetic_lincs.tsv"
    pd.DataFrame({"feature": model.projector.columns, "value": signature}).to_csv(
        feature, sep="\t", index=False)
    assisted = json.loads(subprocess.check_output([*base, "--lincs-tsv", str(feature)], text=True, env=env))
    _, quiet = model.predict(smiles, names)
    _, with_view = model.predict(smiles, names, model.signature(feature))
    structure_error = np.max(np.abs(quiet - np.asarray(list(structural["predictions"].values()))))
    view_error = np.max(np.abs(with_view - np.asarray(list(assisted["predictions"].values()))))
    if (max(structure_error, view_error) >= 1e-6
            or structural["mode"] != "Structure-only" or assisted["mode"] != "Assay-assisted"):
        raise ValueError("CLI did not reproduce the portable predictor")
    publish_json(out / "receipt.json", {"package_sha256": digest, "novel_parent": structural["parent"],
                "fit_parent_member": False, "query_cells": len(names), "LINCS_recipe": model.assay.params.lincs.recipe,
                "synthetic_signature": True, "signature_sha256": sha256(feature),
                "structural_max_abs_diff": float(structure_error), "assisted_max_abs_diff": float(view_error),
                "explicit_cli_fallback": "no LINCS TSV invokes the registered structure-only kernel"})


def inner_parity(root, selection, out):
    device = torch.device("cpu")
    data, assets, reg, _, _ = context(root, device)
    checks = []
    approved = read_json(selection)
    for role in (0, 1):
        cfg = design(approved[str(role)]["design"])
        fit, query = subfold(reg, role, 0)
        train, held = response(fit, True), response(query, False)
        engine = Engine(data, assets, train, held, reg)
        kit = create_kit(engine, cfg, 59187)
        registered = engine.view_data(cfg.lincs)
        with_lincs = sorted(set(engine.query_drugs) & set(registered.index))
        no_lincs = sorted(set(engine.query_drugs) - set(registered.index))
        examples = [(with_lincs[0], registered.loc[[with_lincs[0]]].to_numpy(np.float32)),
                    (no_lincs[0], None)]
        cache = engine.solve(cfg, 59187).cpu().numpy()
        for key, values in examples:
            ident = str(reg.loc[key, "smiles"])
            group = list(engine.query_cells[:8])
            resolved, portable = kit.predict(ident, group, values)
            if resolved != key:
                raise ValueError("Portable chemistry changed fitted identities")
            rx = engine.query_cells.get_indexer(group)
            dx = engine.query_drugs.get_loc(key)
            expected = cache[rx, dx]
            diff = float(np.abs(portable - expected).max())
            # Single-column BLAS and a full frozen batch differ in float32 accumulation.
            # This agreement check is separate from the exact same-path package replay.
            tolerance = 5e-5 * max(1., float(np.max(np.abs(expected))))
            if diff >= tolerance:
                raise AssertionError(f"Portable cross-kernel parity failure in role {role}: {diff}")
            checks.append({"outer_fit": role, "branch": "assisted" if values is not None else "structure",
                           "parity_max_abs_diff": diff, "cross_shape_tolerance": tolerance,
                           "query_response_columns_read": []})
    publish_json(out, {"checks": checks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--inner-selection", type=Path)
    a = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    if a.inner_selection:
        inner_parity(a.work, a.inner_selection, a.out)
    else:
        test(a.work, a.out)
