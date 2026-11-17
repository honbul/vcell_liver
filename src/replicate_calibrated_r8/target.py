"""Guarded access to the HCC targets. Nothing here runs before the freeze manifest is committed."""
import json, os, subprocess
import numpy as np
import pandas as pd

from .common import MAN, B_MAN, TARGETS, sha256

FREEZE = os.path.join(MAN, "freeze_manifest.json")


def assert_frozen():
    """The freeze manifest must exist, be committed in HEAD unchanged, and name a code commit that is HEAD or an ancestor of it."""
    assert os.environ.get("R8_FINAL_EVAL") == "1", "HCC treated profiles are only for hcc_eval.py"
    assert os.path.exists(FREEZE), "freeze manifest missing: HCC evaluation stays sealed"
    r = subprocess.run(["git", "show", f"HEAD:{FREEZE}"], capture_output=True)
    assert r.returncode == 0, "freeze manifest is not committed"
    assert r.stdout == open(FREEZE, "rb").read(), "freeze manifest differs from the committed version"
    fz = json.loads(r.stdout)
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", fz["code_commit"], "HEAD"])
    assert anc.returncode == 0, "frozen code commit is not an ancestor of HEAD"
    return fz


def load_target(target):
    """Plate rows of one HCC target with X0 and X1 (study-B arrays, hash-checked). Sealed until frozen."""
    assert target in TARGETS
    fz = assert_frozen()
    d = os.environ["TB_PROC"]
    man = json.load(open(os.path.join(B_MAN, "build_manifest.json")))
    for f in ("rows.tsv", "x0_all.npy", "x1_target.npy"):
        assert sha256(os.path.join(d, f)) == man["files"][f]["sha256"]
    rows = pd.read_csv(os.path.join(d, "rows.tsv"), sep="\t")
    n_src = man["n_source_rows"]
    t = rows[(rows.role == "target") & (rows.cell == target)]
    assert len(t) and (t.row_id >= n_src).all()
    x0 = np.asarray(np.load(os.path.join(d, "x0_all.npy"), mmap_mode="r")[t.row_id.to_numpy()])
    x1 = np.asarray(np.load(os.path.join(d, "x1_target.npy"), mmap_mode="r")[t.row_id.to_numpy() - n_src])
    return t.reset_index(drop=True), x0, x1, fz
