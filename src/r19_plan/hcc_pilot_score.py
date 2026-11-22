"""Validate, freeze and separately score the independent adult HCC pilot."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_pilot_layout import LINES, ROLES, verify

DRAWS = 10000
COORD = ["plate_slot", "well"]
BASE = "B0"
REPEAT = "repeat_x0.25"
RESERVED_ARMS = {BASE, REPEAT, "best_VC"}


def table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def aligned_layout(path, receipt, realized):
    if sha256(path) != read_json(receipt)["files"][path.name]:
        raise ValueError("The approved pilot layout changed")
    planned = pd.read_csv(path, sep="\t")
    verify(planned)
    return bind_realized(planned, realized, dose=5)


def bind_realized(planned, realized, dose):
    if len(realized) != len(planned) or realized.duplicated(COORD).any():
        raise ValueError("One realized record is required for each planned physical well")
    fields = COORD + ["specimen_id", "plate_id", "culture_id", "canonical_parent", "actual_uM",
                      "actual_hours", "qc_status", "qc_reason"]
    physical = planned.merge(realized[fields], on=COORD, how="left", validate="one_to_one")
    if physical[fields].isna().any().any() or physical[["plate_id", "culture_id"]].eq("").any().any():
        raise ValueError("The realized manifest does not identify every planned plate and culture")
    if physical.duplicated(["plate_id", "well"]).any():
        raise ValueError("Two records resolve to one physical plate and well")
    for registered, real in (("plate_slot", "plate_id"), ("culture_block_slot", "culture_id")):
        keys = physical[[registered, real]].drop_duplicates()
        if keys[registered].duplicated().any() or keys[real].duplicated().any():
            raise ValueError(f"Reused or fragmented actual {real}; biological and physical units must each be unique")
    active = physical.qc_status.eq("PASS")
    if (not physical.qc_status.isin(("PASS", "FAIL")).all() or physical.loc[~active, "qc_reason"].eq("").any()
            or physical.loc[active, "specimen_id"].eq("").any()
            or physical.loc[physical.specimen_id.ne(""), "specimen_id"].duplicated().any()):
        raise ValueError("Ambiguous specimen identity or missing failure provenance")
    treated = physical.kind.eq("treated")
    ids = physical.loc[treated, ["parent_slot", "canonical_parent"]].drop_duplicates()
    if (ids.eq("").any().any() or ids.parent_slot.duplicated().any() or ids.canonical_parent.duplicated().any()
            or physical.loc[~treated, "canonical_parent"].ne("").any()
            or not pd.to_numeric(physical.actual_hours, errors="raise").eq(24).all()
            or not pd.to_numeric(physical.loc[treated, "actual_uM"], errors="raise").eq(dose).all()
            or physical.loc[~treated, "actual_uM"].ne("").any()):
        raise ValueError("A real treatment has the wrong identity, exposure or vehicle status")
    return physical


def validate_config(cfg):
    arms = cfg["arms"]
    comparators = cfg["existing_vc"]
    genes = cfg["genes"]
    minimum = cfg["min_vehicles_per_role"]
    if (not isinstance(arms, list) or not all(isinstance(arm, str) and arm for arm in arms)
            or len(set(arms)) != len(arms) or set(arms) & RESERVED_ARMS
            or not comparators or len(set(comparators)) != len(comparators)
            or not set(comparators) < set(arms) or cfg["candidate"] not in set(arms) - set(comparators)
            or not genes or len(set(genes)) != len(genes) or "" in genes
            or not isinstance(minimum, int) or isinstance(minimum, bool) or not 1 <= minimum <= 4
            or cfg["information"] not in ("A", "B") or not cfg["qc_protocol_sha256"]
            or not isinstance(cfg["synthetic"], bool)
            or (cfg["information"] == "B" and not cfg.get("donor_access_sha256"))):
        raise ValueError("Invalid frozen comparator, information contract, QC or gene axis")


def availability(physical, minimum):
    passed = physical.qc_status.eq("PASS")
    accepted = physical.loc[passed]
    controls = accepted.loc[accepted.kind.eq("vehicle")]
    count = controls.groupby(["plate_id", "vehicle_role"]).size().unstack(fill_value=0)
    count = count.reindex(index=physical.plate_id.unique(), columns=ROLES, fill_value=0)
    approved = set(count.index[count.ge(minimum).all(axis=1)])
    targets = accepted.loc[accepted.kind.eq("treated") & accepted.plate_id.isin(approved)].reset_index(drop=True)
    if set(targets.cell) != set(LINES):
        raise ValueError("At least one eligible treatment per planned cell line is required")
    paired = targets.groupby(["cell", "canonical_parent"]).culture_id.transform("nunique").ge(2)
    return accepted, controls, targets, paired.to_numpy(), count


def validate_declared_access(allowed, ledger, arms):
    catalog = allowed.set_index("specimen_id")
    if (ledger.duplicated(["arm", "rotation", "specimen_id"]).any()
            or not set(ledger.arm) <= set(arms) or not ledger.rotation.isin(ROLES).all()
            or not ledger.specimen_id.isin(catalog.index).all()
            or not ledger.usage.isin(("fit_vehicle", "select_vehicle")).all()):
        raise ValueError("Declared forecast access includes a forbidden specimen or role")
    used = ledger.specimen_id.map(catalog.vehicle_role)
    fit = ledger.rotation
    select = fit.map(dict(zip(ROLES, ROLES[1:]+ROLES[:1], strict=True)))
    expected = fit.where(ledger.usage.eq("fit_vehicle"), select)
    if not used.eq(expected).all():
        raise ValueError("A score vehicle or wrong-role vehicle entered the declared prediction access")


def read_forecast(path, cfg, targets):
    with np.load(path, allow_pickle=False) as data:
        if (set(data.files) != {"specimen_id", "genes", "rotations", *cfg["arms"]}
                or not np.array_equal(data["specimen_id"], targets.specimen_id)
                or not np.array_equal(data["genes"], cfg["genes"]) or not np.array_equal(data["rotations"], ROLES)):
            raise ValueError("Frozen forecast identifiers, genes or roles changed")
        predictions = {arm: data[arm] for arm in cfg["arms"]}
    shape = (len(ROLES), len(targets), len(cfg["genes"]))
    if any(a.shape != shape or a.dtype.kind != "f" or not np.isfinite(a).all() for a in predictions.values()):
        raise ValueError("Invalid frozen prediction values or shape")
    return predictions


def prepare(args):
    cfg = read_json(args.config)
    validate_config(cfg)
    observed = aligned_layout(args.layout, args.layout_receipt, table(args.realized))
    accepted, controls, targets, paired, control_counts = availability(observed, cfg["min_vehicles_per_role"])
    validate_declared_access(controls, table(args.access), cfg["arms"])
    predictions = read_forecast(args.forecast, cfg, targets)
    fields = ("layout", "layout_receipt", "realized", "access", "config", "forecast", "protocol")
    proof = {f"{name}_sha256": sha256(getattr(args, name)) for name in fields}
    proof.update(code_sha256=sha256(__file__), expression_read=False, synthetic=cfg["synthetic"],
                 forecast_specimens=len(targets), score_rotations=list(ROLES), information=cfg["information"],
                 arms=cfg["arms"], existing_vc=cfg["existing_vc"], candidate=cfg["candidate"])
    return cfg, observed, accepted, controls, targets, paired, control_counts, predictions, proof


def measured(path, cfg, accepted, controls, targets):
    with np.load(path, allow_pickle=False) as data:
        if (set(data.files) != {"values", "specimen_id", "genes"}
                or not np.array_equal(data["specimen_id"], accepted.specimen_id)
                or not np.array_equal(data["genes"], cfg["genes"])):
            raise ValueError("Score data do not match the frozen specimen and gene axes")
        matrix = data["values"]
    if matrix.shape != (len(accepted), len(cfg["genes"])) or matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
        raise ValueError("Invalid measured well expression matrix")
    return vehicle_contrasts(matrix, accepted, controls, targets)


def vehicle_contrasts(matrix, accepted, controls, targets):
    index = pd.Index(accepted.specimen_id)
    y = matrix[index.get_indexer(targets.specimen_id)].astype(np.float64)
    vehicle = pd.DataFrame(matrix[index.get_indexer(controls.specimen_id)]).groupby(
        [controls.plate_id.to_numpy(), controls.vehicle_role.to_numpy()]).mean()
    out = []
    for i in range(len(ROLES)):
        active = ROLES[(i+2)%len(ROLES)]
        keys = pd.MultiIndex.from_arrays([targets.plate_id, np.full(len(targets), active)])
        scores = vehicle.reindex(keys)
        if scores.isna().any().any():
            raise ValueError("A physical score vehicle role is missing")
        out.append(y - scores.to_numpy())
    return np.stack(out)


def repeat_reference(targets, truth):
    vectors = np.zeros_like(truth)
    for idx in targets.groupby(["cell", "canonical_parent"], sort=False).indices.values():
        if len(idx) < 2:
            continue
        same = targets.iloc[idx]
        if not all(same[field].is_unique for field in ("plate_id", "culture_id", "well")):
            raise ValueError("Repeat donor shares a culture, physical plate or position")
        for i in idx:
            vectors[:, i] = .25 * truth[:, idx[idx != i]].mean(axis=1)
    return vectors


def bootstrap(rows, errors, energy, draws, crossed):
    """Apply independent weights to parent and culture, with roles already averaged per well."""
    pid, parents = pd.factorize(rows.canonical_parent, sort=True)
    bid, cultures = pd.factorize(rows.culture_id, sort=True)
    parent_sum = np.eye(len(parents))[pid]
    eps = np.finfo(float).tiny
    rng = np.random.default_rng(191096)
    rmse, r2 = [np.empty((draws, len(errors))) for _ in range(2)]
    well_rmse = np.sqrt(errors).mean(axis=1).T
    well_sse = errors.mean(axis=1).T

    def moments(p, b):
        if not crossed:
            b = np.ones_like(b)
        repeat = b[:, bid]
        well = p[:, pid]*repeat
        per_parent = repeat @ parent_sum
        macro = well / np.maximum(per_parent[:, pid], eps)
        v_rmse = (macro @ well_rmse)/macro.sum(axis=1, keepdims=True)
        denom = well @ energy
        v_r2 = np.divide(well @ well_sse, denom[:, None], out=np.full_like(v_rmse, np.nan), where=denom[:, None]>0)
        return v_rmse, 1-v_r2

    point = moments(np.ones((1, len(parents))), np.ones((1, len(cultures))))
    for start in range(0, draws, 256):
        stop = min(draws, start+256)
        count = stop-start
        rmse[start:stop], r2[start:stop] = moments(rng.exponential(size=(count, len(parents))),
                                                    rng.exponential(size=(count, len(cultures))))
    return (point[0][0], point[1][0]), (rmse, r2)


def fit_scores(targets, truth, paired, predictions, cfg, draws=DRAWS):
    arms = (BASE, *cfg["arms"], REPEAT)
    vectors = {BASE: np.zeros_like(truth), **predictions, REPEAT: repeat_reference(targets, truth)}
    errors = np.stack([np.square(truth - vectors[name]).mean(axis=2) for name in arms])
    points, paired_rows = [], []
    for cell in LINES:
        line = targets.cell.eq(cell).to_numpy()
        for scope, keep in (("all_eligible", line), ("different_culture_repeat", line & paired)):
            if not keep.any():
                continue
            rows = targets.loc[keep]
            count = dict(cell=cell, scope=scope, n_parents=int(rows.canonical_parent.nunique()), wells=len(rows),
                         cultures=int(rows.culture_id.nunique()), roles=len(ROLES))
            retained = slice(None, -1) if scope == "all_eligible" else slice(None)
            names = arms[retained]
            for label, cross in (("parent", False), ("parent_and_culture", True)):
                estimate, bootstrap_draw = bootstrap(rows, errors[retained][:, :, keep], errors[0][:, keep].mean(axis=0), draws, cross)
                for metric, x, sample in zip(("RMSE", "pooled_R2"), estimate, bootstrap_draw, strict=True):
                    lows, highs = np.quantile(sample, [.005, .995], axis=0)
                    for i, name in enumerate(names):
                        points.append(dict(count, resampling=label, metric=metric, arm=name, estimate=x[i],
                                           low99=lows[i], high99=highs[i]))
                    base = names.index(BASE)
                    candidate = names.index(cfg["candidate"])
                    subset = [names.index(name) for name in cfg["existing_vc"]]
                    rmse = metric == "RMSE"
                    choose = np.argmin if rmse else np.argmax
                    candidates = sample[:, subset]
                    best = subset[int(choose(x[subset]))]
                    api = np.min if rmse else np.max
                    comparisons = [(cfg["candidate"], candidate, BASE, base, sample[:, base]),
                                   (cfg["candidate"], candidate, "best_VC", best, api(candidates, axis=1))]
                    if REPEAT in names:
                        rep = names.index(REPEAT)
                        comparisons.append((REPEAT, rep, BASE, base, sample[:, base]))
                    for arm, i, ref, j, series in comparisons:
                        lo, hi = np.quantile(sample[:, i]-series, [.005,.995])
                        paired_rows.append(dict(count, resampling=label, metric=metric, arm=arm, reference=ref,
                                                best_at_point=names[j], difference=x[i]-x[j], low99=lo, high99=hi))
    return pd.DataFrame(points), pd.DataFrame(paired_rows)


def lock(args):
    result = prepare(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, result[-1])


def score(args):
    cfg, physical, accepted, controls, targets, paired, counts, predictions, proof = prepare(args)
    if sha256(args.freeze) != args.expected_sha256 or read_json(args.freeze) != proof:
        raise ValueError("Score must use a verified immutable forecast freeze before reading measurements")
    truth = measured(args.measurements, cfg, accepted, controls, targets)
    points, paired_rows = fit_scores(targets, truth, paired, predictions, cfg)
    support = physical.groupby(["cell", "plate_id"], as_index=False).agg(registered=("specimen_id", "size"),
                   qc_pass=("qc_status", lambda x: x.eq("PASS").sum())).set_index("plate_id").join(counts)
    support["scored_treatments"] = targets.groupby("plate_id").size().reindex(support.index, fill_value=0)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp)
        for name, frame in (("points", points), ("paired", paired_rows), ("support", support.reset_index())):
            frame.to_csv(stage/(name+".tsv"), sep="\t", index=False, na_rep="NA")
        write_json(stage/"receipt.json", dict(proof, expression_read=True, bootstrap_draws=DRAWS,
            freeze_sha256=args.expected_sha256, measurements_sha256=sha256(args.measurements),
            files={p.name:sha256(p) for p in stage.iterdir()}))
        if args.output.exists():
            raise FileExistsError(args.output)
        stage.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lock", "score"))
    for name in ("layout","layout-receipt","realized","access","config","forecast","protocol","output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.phase == "score" and not all((args.measurements, args.freeze, args.expected_sha256)):
        parser.error("score requires measured data and an explicit forecast freeze and SHA-256")
    (lock if args.phase=="lock" else score)(args)


if __name__ == "__main__":
    main()
