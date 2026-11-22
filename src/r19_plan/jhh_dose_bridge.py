"""Test whether a matched high-dose donor improves unseen lower-dose JHH-5 prediction."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_lincs_24h_access import CELL, CONTEXT, VIEW, register
from src.r19_plan.hcc_lincs_24h_support import evaluate_support, supported_controls
from src.r19_plan.hcc_lincs_24h_transfer import context, effect, group_means
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import checked_array
from src.r19_plan.jhh5_donor_attribution import wrong_parent
from src.r19_plan.jhh_low_dose_transfer import finish, load_plan
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_state import state_forecast
from src.r19_plan.t4_workflow import axes, scalar
from src.r19_plan.x_inputs import check_unique


DOSES = ("1.11111", "10")
MATCH = ["parent", "pert_id", *CONTEXT]
MAIN, LOW, HIGH, STATE, WRONG = "two_dose", "low_scalar", "high_scalar", "two_token_State", "wrong_high"
ARMS = (MAIN, LOW, HIGH, STATE, WRONG)
PAIRS = tuple((MAIN, other) for other in (*ARMS[1:], "B0")) + ((LOW, "B0"), (HIGH, "B0"), (STATE, "B0"))


def audited(directory, receipt):
    if sha256(directory / "receipt.json") != sha256(receipt):
        raise ValueError("The dose-support receipt differs from the pinned audit")
    contents = read_json(receipt)
    if not contents["metadata_only"] or contents["expression_read"] or not contents["registered_control_block"]:
        raise ValueError("Expected a metadata-only dose inventory with R18 exclusion")
    return {name: read_pinned(directory, name + ".tsv", contents["files"][name + ".tsv"])
            for name in ("targets", "donors", "controls")}


def access(args):
    low, high = audited(args.audit, args.audit_pin), audited(args.high_audit, args.high_pin)
    targets = low["targets"].loc[lambda r: r.cell_id.eq(CELL) & r.actual_dose_uM.eq(DOSES[0])].copy()
    targets["b1_seen"] = False
    old_targets = high["targets"].loc[lambda r: r.cell_id.eq(CELL)]
    if set(targets.inst_id) & set(old_targets.inst_id):
        raise ValueError("A low-dose target treatment ID was already scored at high dose")
    low_donors, high_donors = [data["donors"].loc[lambda r: r.view.eq(VIEW) & r.actual_dose_uM.eq(dose)
        & r.parent.isin(targets.parent) & ~r.rna_plate.isin(targets.rna_plate)] for data, dose in zip((low, high), DOSES, strict=True)]
    shared = low_donors[MATCH].drop_duplicates().merge(high_donors[MATCH].drop_duplicates(), on=MATCH, validate="one_to_one")
    source = pd.concat([rows.merge(shared, on=MATCH, validate="many_to_one")
                        for rows in (low_donors, high_donors)], ignore_index=True)
    control_fields = [col for col in high["controls"] if col in low["controls"]]
    common_controls = low["controls"][control_fields].merge(high["controls"][control_fields], on=control_fields, validate="one_to_one")
    donors, donor_controls = supported_controls(source, common_controls)
    counts = donors.groupby("parent").agg(cells=("cell_id", "nunique"), plates=("rna_plate", "nunique"))
    carry = counts.index[counts.cells.ge(8) & counts.plates.ge(8)]
    targets = targets.loc[targets.parent.isin(carry)]
    donors = donors.loc[donors.parent.isin(carry)]
    donor_controls = donor_controls.merge(donors[CONTEXT].drop_duplicates(), on=CONTEXT, validate="many_to_one")
    target_controls = low["controls"][control_fields].merge(targets[CONTEXT].drop_duplicates(), on=CONTEXT, validate="many_to_one")
    controls = pd.concat([donor_controls, target_controls], ignore_index=True)
    check_unique(pd.concat([targets, donors, controls], ignore_index=True))
    if not donors.groupby(MATCH).actual_dose_uM.nunique().eq(2).all():
        raise ValueError("A common donor context lost one of its declared concentrations")
    targets["fold"] = targets.fold.astype(int)
    stats, splits, _ = evaluate_support(targets, donors, CELL, VIEW)
    support = next(row for row in stats if row["scope"] == "donor_covered")
    if not support["eligible"] or min((x["score_parents"] for x in splits), default=0) < 3:
        raise ValueError(f"The registered bridge support did not pass: {support}")
    details = dict(high_audit_sha256=sha256(args.high_pin), bridge_support={
        **{k: int(support[k]) for k in ("fold0", "fold1", "fold2", "fit_parent_min")},
        "target_controls": len(target_controls), "shared_prior_target_controls": int(target_controls.inst_id.isin(high["controls"].inst_id).sum()),
        "shared_prior_target_parents": int(targets.loc[targets.parent.isin(old_targets.parent)].parent.nunique()),
        "shared_prior_target_plates": int(targets.loc[targets.rna_plate.isin(old_targets.rna_plate)].rna_plate.nunique()),
        "shared_donor_entity_contexts": len(donors[MATCH].drop_duplicates())})
    targets["fold"] = targets.fold.astype(str)
    for split in splits:
        split["fold"] = str(split["fold"])
    register(args, targets, donors, controls, pd.DataFrame(splits), details)


def donor(args, ref, genes, cols, stamp):
    role = args.role
    tables = {name: read_pinned(args.access / "donor-plan" / role, name, digest)
              for name, digest in ref["donors"][role].items()}
    rows = tables["source.tsv"]
    if set(rows.actual_dose_uM) != set(DOSES) or not rows.view.eq(VIEW).all() or len(rows) != ref["donor_wells"]:
        raise ValueError("Dose-bridge donor axis differs from the access plan")
    values = effect(rows, tables["controls.tsv"], ref, genes, cols, role)
    averaged = []
    parent_axis = None
    for dose in DOSES:
        keep = rows.actual_dose_uM.eq(dose)
        entity, v = group_means(values[keep], rows.loc[keep], ["parent", "cell_id", "rna_plate", "pert_id"])
        plates, v = group_means(v, entity, ["parent", "cell_id", "rna_plate"])
        cells, v = group_means(v, plates, ["parent", "cell_id"])
        parents, v = group_means(v, cells, ["parent"])
        axis = pd.Index(parents.parent)
        if parent_axis is not None and not axis.equals(parent_axis):
            raise ValueError("Low- and high-dose parent axes differ")
        parent_axis = axis
        averaged.append(v)
    if len(parent_axis) != ref["target_parents"]:
        raise ValueError("A target parent lacks a matched dose pair")
    output = args.run / "donors" / role
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".bridge-donor-") as temporary:
        root = Path(temporary)
        payload = root / "donors.npz"
        np.savez_compressed(payload, parents=parent_axis.to_numpy(dtype=str), values=np.stack(averaged, axis=1))
        write_json(root / "receipt.json", dict(stamp, role=role, data_sha256=sha256(payload)))
        root.rename(output)


def paired_source_fit(x, y):
    design = x.transpose(0, 2, 1).reshape(-1, 2)
    target = y.reshape(-1)
    # Two coefficients suffice. QR reduces the bounded fit to the exact two-dimensional problem.
    q, r = np.linalg.qr(design, mode="reduced")
    result = lsq_linear(r, q.T @ target, bounds=(0, 2))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Bounded dose-bridge fit did not converge")
    return result.x


def predict(args, ref, genes, cols, stamp):
    from src.goal_tracks.t2_state import UPSTREAM, load_network

    role = args.group[-1]
    root = args.run / "donors" / role
    donor_file = root / "donors.npz"
    receipt = read_json(root / "receipt.json")
    if receipt["role"] != role or any(receipt[k] != v for k, v in stamp.items()) or sha256(donor_file) != receipt["data_sha256"]:
        raise ValueError("Dose-bridge donor predictions do not match the access lineage")
    with np.load(donor_file, allow_pickle=False) as artifact:
        parents = pd.Index(artifact["parents"])
        vector = checked_array(artifact["values"], (ref["target_parents"], 2, len(genes)), "matched doses")
    network = load_network(args.state)
    output = args.run / "forecasts" / args.group
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".bridge-forecast-") as temporary:
        staged = Path(temporary)
        files, fits = {}, []
        for key, entry in sorted(ref["slices"].items()):
            if not key.startswith(args.group + "-"):
                continue
            fold, _, _, _, plate = axes(key)
            plan = load_plan(args, "predict", key, entry["predict"])
            fit, control, query = (plan[name] for name in ("fit.tsv", "fit_controls.tsv", "query.tsv"))
            if (set(fit.parent) & set(query.parent) or set(fit.scaffold) & set(query.scaffold)
                    or fit.fold.eq(str(fold)).any() or fit.rna_plate.eq(plate).any() or not query.fold.eq(str(fold)).all()
                    or not query.rna_plate.eq(plate).all() or "level3_row_index" in query):
                raise ValueError("A query dose-bridge treatment intersects the fit")
            labels, y = group_means(effect(fit, control, ref, genes, cols, role), fit, ["parent"])
            permuted = wrong_parent(query, fold, plate)
            i, j, wrong = (parents.get_indexer(names) for names in (labels.parent, query.parent, permuted))
            if any((i < 0).any() for i in (i, j, wrong)) or (j == wrong).any():
                raise ValueError("Missing donor or invalid wrong-parent pairing")
            x, q = vector[i], vector[j]
            low = scalar(labels, x[:, 0], y)
            high = scalar(labels, x[:, 1], y)
            gains = paired_source_fit(x, y)
            forecast = {MAIN: np.einsum("d,ndg->ng", gains, q), LOW: low*q[:, 0], HIGH: high*q[:, 1],
                        STATE: state_forecast(network, (labels, x, y), q),
                        WRONG: gains[0]*q[:, 0] + gains[1]*vector[wrong, 1]}
            file = staged / (key + ".npz")
            np.savez_compressed(file, **{arm: checked_array(v, (len(query), len(genes)), arm).astype("float32")
                                        for arm, v in forecast.items()})
            files[key] = sha256(file)
            fits.append(dict(slice=key, low_scalar=low, high_scalar=high, low_joint=float(gains[0]),
                             high_joint=float(gains[1]), fit_parents=len(labels), query_wells=len(query)))
        if not files:
            raise ValueError("The forecast group has no registered splits")
        write_json(staged / "receipt.json", dict(stamp, group=args.group, files=files, fits=fits,
            donor_sha256=sha256(root / "receipt.json"), state_upstream=UPSTREAM,
            state_model_sha256=sha256(args.state / "src/state/tx/models/state_transition.py")))
        staged.rename(output)


def score_pairs(rows, truth, outcomes, cell, scope):
    return [dict(x, cell=cell) for x in compare(rows, truth, outcomes, PAIRS, scope)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("access", "donor", "predict", "freeze", "score"))
    for key in ("access", "access-pin", "protocol", "run", "audit", "audit-pin", "high-audit", "high-pin", "cmap-pin", "landmarks", "out", "state"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--role", choices=tuple("ABC"))
    parser.add_argument("--group")
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.phase == "access":
        access(args)
    else:
        ref, genes, cols, stamp = context(args)
        stamp.update(bridge_code_sha256=sha256(__file__), high_audit_sha256=ref["high_audit_sha256"])
        if args.phase in ("donor", "predict"):
            (donor if args.phase == "donor" else predict)(args, ref, genes, cols, stamp)
        else:
            finish(args, ref, genes, cols, stamp, arms=ARMS, score_pairs=score_pairs, panel_count=0)
