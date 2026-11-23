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

from nextgen_h2.assets import canonical_chemistry
from nextgen_h2.run import read_json
from vc_build.io import publish_json, sha256


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    test(a.work, a.out)
