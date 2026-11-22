"""Freeze D3 control ablations before scoring on the balanced D1 outcome roles."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.metrics import Contrast
from src.r19_plan.d1_inputs import access_reference
from src.r19_plan.d12_workflow import score_truth
from src.r19_plan.d3_predict import ARMS, code_fingerprints
from src.r19_plan.dev_selection import checked_array
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_score import METRICS, checked_rows, paired_effects, parent_summaries, squared_rows
from src.r19_plan.i_stage import plan_receipt
from src.r19_plan.i_workflow import registered_axes
from src.r19_plan.r18_predict import model_rows
from src.r19_plan.sl_components import exact_pools
from src.r19_plan.sl_predict import dose_map

FILES = {"rows.tsv", "landmarks.tsv", "forecasts.npz", "selection.json"}
PAIRS = [("D3b", "D3a"), ("D3c", "D3a"), *((name, name+"_parent_shuffle") for name in ("D3a", "D3b", "D3c"))]


def verify_slice(args, key, rows, genes):
    root = args.forecasts / key
    receipt = read_json(root / "receipt.json")
    if (receipt["slice"] != key or receipt["protocol_sha256"] != sha256(args.protocol)
            or receipt["access_sha256"] != sha256(args.access_reference)
            or receipt["control_sha256"] != sha256(args.control_reference)
            or receipt["code_sha256"] != code_fingerprints() or set(receipt["files"]) != FILES):
        raise ValueError("The D3 forecast uses an unexpected registered input, role or code")
    for name, digest in receipt["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"A D3 artifact changed since writing: {key}/{name}")
    if ((root / "rows.tsv").read_bytes() != rows.to_csv(sep="\t", index=False).encode()
            or (root / "landmarks.tsv").read_bytes() != genes.to_csv(sep="\t", index=False).encode()):
        raise ValueError("D3 changed the established held wells or gene coordinate")
    with np.load(root / "forecasts.npz", allow_pickle=False) as array:
        if set(array.files) != set(ARMS):
            raise ValueError("Unexpected D3 model and shuffle inventory")
        out = {arm: checked_array(array[arm], (len(rows), len(genes)), arm) for arm in ARMS}
    return out, sha256(root / "receipt.json")


def score_arrays(rows, truth, arrays, contrast):
    energy = squared_rows(truth)
    errors = {arm: squared_rows(truth, data) for arm, data in arrays.items()}
    paired, diagnostic = [], []
    for scope in ("ALL", *sorted(rows.fold.unique())):
        active = np.ones(len(rows), dtype=bool) if scope == "ALL" else rows.fold.eq(scope).to_numpy()
        for arm, baseline in PAIRS:
            inputs = parent_summaries(rows.loc[active], truth.shape[1], energy[active], errors[arm][active],
                                      errors[baseline][active])
            points, bounds, counts = paired_effects(*inputs)
            low_rel, high_rel, _ = relative_interval(inputs[0])
            for metric, (a, b), (lo, hi), n in zip(METRICS, points, bounds, counts, strict=True):
                paired.append(dict(scope=scope, arm=arm, reference=baseline, metric=metric, arm_value=a,
                    reference_value=b, difference=a-b, low99=lo, high99=hi, parents=len(inputs[0]),
                    rows=int(active.sum()), valid_draws=int(n), relative_gain=1-a/b if metric=="rmse" and b>0 else np.nan,
                    relative_low99=low_rel if metric=="rmse" else np.nan,
                    relative_high99=high_rel if metric=="rmse" else np.nan))
    for arm, vector in arrays.items():
        comparable = (vector != 0) & (truth != 0)
        correct = (vector > 0) == (truth > 0)
        counts = comparable.sum(axis=1)
        per_row = np.divide((correct & comparable).sum(axis=1), counts, out=np.full(len(rows), np.nan), where=counts > 0)
        sign = rows.assign(sign=per_row).groupby("parent").sign.mean().mean()
        diagnostic.append(dict(arm=arm, contrast_rmse=contrast.point(vector, truth), support_rows=len(contrast.active),
            delta_sign_agreement=sign, sign_valid_rows=int((counts > 0).sum())))
    return paired, diagnostic


def run(args):
    if args.action == "score" and (args.out.exists() or sha256(args.freeze) != args.expected_sha256):
        raise ValueError("D3 requires a fresh score output and the independent freeze SHA-256")
    balanced = access_reference(args.access_reference, "f0-A")
    prior, original_digest = plan_receipt(args.original_plan, args.original_reference, "f0-A")
    if (balanced["source_access_sha256"] != original_digest
            or sha256(args.access / "receipt.json") != sha256(args.access_reference)
            or balanced["balanced_sha256"] != sha256(args.control_reference)):
        raise ValueError("D3 must inherit its registered D1 treatment and role lock")
    raw, queries, genes, fit = registered_axes(args.original_plan, args.original_reference, args.cache)
    receipts, parts = {}, {arm: [] for arm in ARMS}
    for (fold, role), q in queries.items():
        key = f"f{fold}-{role}"
        values, receipts[key] = verify_slice(args, key, q, genes)
        if args.action == "score":
            for arm in ARMS:
                parts[arm].append(values[arm])
    stamp = dict(protocol_sha256=sha256(args.protocol), access_sha256=sha256(args.access_reference),
                 controls_sha256=sha256(args.control_reference), dose_sha256=sha256(args.dose_reference),
                 slices=receipts, predictor=code_fingerprints(), code_sha256=sha256(__file__))
    if args.action == "freeze":
        write_json(args.freeze, stamp)
        return sha256(args.freeze)
    if stamp != read_json(args.freeze):
        raise ValueError("A D3 prediction or protocol changed after the freeze")
    rows = checked_rows(pd.concat(queries.values(), ignore_index=True), fit)
    truth = score_truth(args, prior, balanced, raw, genes)
    models = exact_pools(model_rows(rows), dose_map(args.instances, args.dose_reference, rows.inst_id))
    models["plate_id"] += ":" + rows.fold.astype(str) + ":" + rows.role
    contrast = Contrast(models, minimum=2)
    paired, diagnostics = score_arrays(rows, truth, {arm: np.concatenate(parts[arm]) for arm in ARMS}, contrast)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".d3-score-") as temp:
        stage = Path(temp)
        for name, data in (("paired", paired), ("diagnostics", diagnostics)):
            pd.DataFrame(data).to_csv(stage / f"{name}.tsv", sep="\t", index=False, na_rep="NA")
        write_json(stage / "receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
                   files={p.name: sha256(p) for p in sorted(stage.iterdir())}))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score"))
    for name in ("access", "access-reference", "control-reference", "dose-reference", "original-plan", "original-reference",
                 "cache", "protocol", "forecasts", "freeze"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("vehicles", "instances", "out"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "score" and any(value is None for value in (args.out, args.vehicles, args.instances, args.expected_sha256)):
        parser.error("score requires --out, --vehicles, --instances, and --expected-sha256")
    print(run(args))
