"""Predict and separately score location-residual RNA transfer in the HepaRG development cohort."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.linalg import svd

from src.goal_tracks.heparg import DATASETS, PRIMARY, active_rows, read_source, response, verify
from src.goal_tracks.heparg_design import enrich, released_design
from src.goal_tracks.heparg_models import location_candidates, strata
from src.p123.p3_run import T3_FREEZE
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.heparg_position_repeat import CHEMISTRY, LOCATION
from src.r19_plan.i_operational import relative_interval
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.i_score import paired_effects, parent_summaries, squared_rows
from src.r19_plan.r18_input_bundle import extract_matrix, indices
from src.r19_plan.t4_heparg_support import DATASETS as CACHE_NAMES
from src.r19_plan.t4_workflow import deltas
from src.r19_plan.x_inputs import check_unique, control_axis

ROLES = "ABC"
ARMS = ("baseline", "RNA_rank8", "parent_shuffle")
LABEL = "ge10uM"


def bound(args):
    audit = args.work / "audit"
    path = audit / "receipt.json"
    info = read_json(path)
    if (not info["metadata_only"] or info["expression_read"] or info["t3_freeze"] != T3_FREEZE
            or sha256(args.t3/"freeze.json") != T3_FREEZE
            or sha256(args.protocol) != info["protocol_sha256"]
            or sha256(args.matrix) != info["matrix_reference_sha256"]):
        raise ValueError("Unregistered source RNA access")
    return info, dict(audit_sha256=sha256(path), protocol_sha256=sha256(args.protocol),
                      code_sha256=sha256(__file__), source_t3=T3_FREEZE)


def write_product(path, proof, predicted, rows=None):
    with TemporaryDirectory(dir=path.parent) as temp:
        stage = Path(temp)
        np.savez_compressed(stage / "values.npz", **predicted)
        if rows is not None:
            rows.to_csv(stage/"rows.tsv", sep="\t", index=False)
        write_json(stage/"receipt.json", dict(proof, files={p.name: sha256(p) for p in stage.iterdir()}))
        if path.exists():
            raise FileExistsError(path)
        stage.rename(path)


def input_profiles(args, info, proof):
    if not info["eligible"]:
        raise ValueError("No study satisfies the registered donor support floor")
    audit = args.work/"audit"
    files = info["files"]
    rows, vehicle, axis = [read_pinned(audit, name, files[name]) for name in ("donors.tsv", "vehicle.tsv", "genes.tsv")]
    combined = pd.concat((rows, vehicle), ignore_index=True)
    check_unique(combined)
    record = read_json(args.matrix)["matrix_records"]["combined"]
    raw = extract_matrix(record, combined, indices(axis.level3_gene_index), axis.entrez_id.astype(int).to_numpy())
    result = {}
    for role in ROLES:
        use = vehicle.role.eq(role)
        controls = vehicle.loc[use]
        control_axis(controls, rows, role)
        response_each = deltas(rows, raw[:len(rows)], controls, raw[len(rows):][use.to_numpy()])
        by_plate = pd.DataFrame(response_each).groupby([rows.parent, rows.cell_id, rows.rna_plate]).mean()
        profile = by_plate.groupby(level=(0, 1)).mean().groupby(level=0).mean()
        result[role] = profile.to_numpy(dtype=np.float32)
        if not np.isfinite(result[role]).all():
            raise ValueError("A donor profile is nonfinite")
    write_product(args.work/"profiles", proof, dict(result, parents=profile.index.to_numpy(dtype=str)))


def validate_product(path, proof):
    receipt = read_json(path/"receipt.json")
    if any(receipt[k] != v for k,v in proof.items()):
        raise ValueError(f"Prediction lineage changed for {path.name}")
    verify(path, receipt["files"])


def original(args, info, fold):
    t3 = args.t3
    frozen = read_json(t3/"freeze.json")
    key = f"{args.dataset}-{fold}"
    origin = t3/"predict"/key
    receipt = read_json(origin/"receipt.json")
    if info["t3_freeze"] != T3_FREEZE or sha256(origin/"receipt.json") != frozen["part_receipts"][f"{key}/receipt.json"]:
        raise ValueError("T3 fold receipt changed")
    verify(origin, {name: receipt["files"][name] for name in ("keys.npz", "settings.json", LABEL+".npz")})
    with np.load(origin/"keys.npz", allow_pickle=False) as data:
        return origin, data["index"], pd.Index(data["genes"]), sha256(origin/"receipt.json")


def fit_source(args):
    """Read target metadata and a read-only matrix mapping; only fitted row indices get materialized."""
    folder = args.cache/"data"/CACHE_NAMES[args.dataset]
    stamp = folder/"receipt.json"
    if sha256(stamp) != read_json(args.t3/"freeze.json")["inputs"][args.dataset]:
        raise ValueError("Target fit metadata no longer match the frozen dataset")
    files = read_json(stamp)["files"]
    rows, genes = [read_pinned(folder, name, files[name]) for name in ("rows.tsv", "genes.tsv")]
    rows = rows.astype({"outer_fold": int, "dose_rank": int, "dose_uM": float})
    rows = enrich(rows, released_design(args.index, args.dataset))
    delta = np.load(folder/"delta.npy", mmap_mode="r", allow_pickle=False)
    if delta.shape != (len(rows), len(genes)):
        raise ValueError("Unexpected target fit axes")
    return folder, rows, pd.Index(genes.ensembl_id), delta, None


def fit_target(source, outer, panel, origin):
    settings = read_json(origin/"settings.json")[LABEL]
    a = settings["B_context"]["gain"]
    b = settings[LOCATION]["gain"]
    coordinates = {"position": source[1].position.to_numpy()}
    parts = []
    for hold in range(3):
        if hold == outer:
            continue
        fit, = set(range(3)) - {hold, outer}
        training = active_rows(source, [fit])
        validation = active_rows(source, [hold])
        truth = response(source, training, panel)
        for name, estimate, _ in location_candidates(training, truth, validation, coordinates,
                                                      training.index.to_numpy(), validation.index.to_numpy()):
            if name == "B_context":
                context = estimate
            if name == LOCATION:
                residual = response(source, validation, panel) - a*context - b*estimate
                break
        keep = strata(validation)[LABEL]
        parts.append(pd.DataFrame(residual[keep]).groupby(validation.parent_key.to_numpy()[keep]).mean())
    return pd.concat(parts, verify_integrity=True).sort_index()


def permute(x, rng):
    ordering = rng.permutation(len(x))
    output = x.copy()
    if len(ordering) > 1:
        output[ordering] = x[np.roll(ordering, 1)]
    return output


def map_residual(train, observed, query):
    mean = train.mean(axis=0)
    _, singular, axes = svd(train-mean, full_matrices=False)
    if len(singular) < 8 or singular[7] <= 1e-8:
        raise ValueError("The fit-only source matrix lacks the registered eight PCA axes")
    axes = axes[:8].T
    scores = (train-mean) @ axes
    coef = np.linalg.solve(scores.T @ scores + np.eye(8), scores.T @ observed)
    return (query-mean) @ axes @ coef


def within_support(rows, values, cover):
    output = np.zeros_like(values)
    for ix in rows.groupby(["plate", "dose_rank"], sort=False).indices.values():
        active = ix[cover[ix]]
        counts = rows.iloc[active].parent_key.value_counts()
        if len(counts) < 2:
            continue
        weight = 1 / rows.iloc[active].parent_key.map(counts).to_numpy()
        output[active] = values[active] - np.average(values[active], axis=0, weights=weight)
    return output


def predict(args, info, proof, fold):
    source = fit_source(args)
    origin, keys, panel, t3_receipt = original(args, info, fold)
    query = active_rows(source, [fold])
    if not np.array_equal(query.index, keys):
        raise ValueError("Target query differs from the frozen T3 row axis")
    keep = strata(query)[LABEL]
    query = query.loc[keep]
    observed = fit_target(source, fold, panel, origin)
    if set(observed.index) & set(query.parent_key):
        raise ValueError("Held target parents entered the RNA map")
    root = args.work/"audit"
    roster = read_pinned(root, "parent_roster.tsv", info["files"]["parent_roster.tsv"])
    roster = roster.loc[roster.dataset.eq(args.dataset)].set_index("parent_key", verify_integrity=True)
    roster = roster.assign(outer_fold=roster.outer_fold.astype(int), support=roster.support.eq("True"))
    if set(roster.loc[roster.outer_fold.eq(fold)].index) != set(query.parent_key):
        raise ValueError("Admitted parent scope differs from the frozen T3 query")
    train_ids = np.intersect1d(observed.index, roster.index[roster.support])
    query_ids = np.intersect1d(query.parent_key, roster.index[roster.support])
    cover = query.parent_key.isin(query_ids).to_numpy()
    if min(len(train_ids), len(query_ids)) < 20 or set(train_ids) & set(query_ids):
        raise ValueError("Insufficient eligible independent parents")
    with np.load(origin/(LABEL+".npz"), allow_pickle=False) as data:
        if not np.array_equal(np.flatnonzero(keep), data["row"]):
            raise ValueError("The high-dose prediction row axis changed")
        anchor = data[LOCATION]+data[CHEMISTRY]
    result = {"baseline": anchor}
    path = args.work/"profiles"
    validate_product(path, proof)
    with np.load(path/"values.npz", allow_pickle=False) as profiles:
        source_index = pd.Index(profiles["parents"])
        positions = [source_index.get_indexer(roster.loc[k].parent) for k in (train_ids, query_ids)]
        if any((idx < 0).any() for idx in positions):
            raise ValueError("An admitted donor lacks a profile")
        for role in ROLES:
            rng = np.random.default_rng(1915+fold)
            fit_data, held = [profiles[role][idx] for idx in positions]
            for name, train, test in (("RNA_rank8", fit_data, held),
                                      ("parent_shuffle", permute(fit_data, rng), permute(held, rng))):
                estimate = map_residual(train, observed.loc[train_ids].to_numpy(), test)
                expanded = pd.DataFrame(estimate, index=query_ids).reindex(query.parent_key).fillna(0).to_numpy()
                result[f"{name}_{role}"] = anchor + within_support(query, expanded, cover)
    if not all(v.shape==anchor.shape and np.isfinite(v).all() and np.array_equal(v[~cover], anchor[~cover])
               for v in result.values()):
        raise ValueError("The prediction has invalid rows or a nonzero RNA fallback")
    directory = args.work/args.dataset
    directory.mkdir(exist_ok=True)
    rows = query[["parent_key", "library"]].reset_index(drop=True).assign(support=cover)
    write_product(directory/f"f{fold}", dict(proof, t3_receipt=t3_receipt, fold=fold, dataset=args.dataset), result, rows)


def freeze_body(args, info, proof):
    hashes = {}
    for fold in range(3):
        _, _, _, receipt = original(args, info, fold)
        path = args.work/args.dataset/f"f{fold}"
        validate_product(path, dict(proof, t3_receipt=receipt, fold=fold, dataset=args.dataset))
        hashes[f"f{fold}"] = sha256(path/"receipt.json")
    return dict(proof, dataset=args.dataset, parts=hashes)


def score(args, info, proof):
    root = args.work/args.dataset
    freeze = root/"freeze.json"
    if sha256(freeze) != args.expected_sha256 or read_json(freeze) != freeze_body(args, info, proof):
        raise ValueError("HepaRG held labels require a verified prior prediction freeze")
    source = read_source(args, args.dataset)
    blocks = {}
    for fold in range(3):
        _, keys, panel, _ = original(args, info, fold)
        query = active_rows(source, [fold])
        if not np.array_equal(keys, query.index):
            raise ValueError("Unexpected score row axis")
        mask = strata(query)[LABEL]
        truth = response(source, query, panel)[mask]
        chosen = query.loc[mask].reset_index(drop=True)
        path = root/f"f{fold}"
        rows = pd.read_csv(path/"rows.tsv", sep="\t")
        if not chosen[["parent_key", "library"]].equals(rows[["parent_key", "library"]]):
            raise ValueError("The frozen prediction and score ledgers differ")
        with np.load(path/"values.npz", allow_pickle=False) as data:
            base = squared_rows(truth, data["baseline"])
            errors = {arm: np.stack([squared_rows(truth, data[f"{arm}_{role}"]) for role in ROLES]) for arm in ARMS[1:]}
        for name, keep in (("full_fallback", np.ones(len(rows), dtype=bool)), ("donor_supported", rows.support.to_numpy())):
            ledger = pd.concat([rows.loc[keep, ["parent_key"]]] * len(ROLES), ignore_index=True).rename(columns={"parent_key":"parent"})
            denominator = np.tile(base[keep], len(ROLES))
            for ref, error in (("baseline", denominator), ("parent_shuffle", errors["parent_shuffle"][:, keep].ravel())):
                moment = parent_summaries(ledger, len(panel), denominator, errors["RNA_rank8"][:, keep].ravel(), error)
                blocks.setdefault((name,ref), []).append((moment, ledger.parent.nunique(), int(keep.sum())))
    results = []
    for (scope, ref), values in blocks.items():
        moments, n, wells = zip(*values, strict=True)
        rmse, sse, energy = (np.concatenate(bit) for bit in zip(*moments, strict=True))
        points, intervals, _ = paired_effects(rmse,sse,energy)
        lo, hi, _ = relative_interval(rmse)
        primary = args.dataset==PRIMARY and scope=="donor_supported"
        results.append(dict(dataset=args.dataset, scope=scope, arm="RNA_rank8", reference=ref, primary=primary,
                            passes=bool(primary and lo>0 and intervals[1,0]>0), parents=sum(n), wells=sum(wells),
                            rmse=points[0,0], reference_rmse=points[0,1], relative_rmse_gain=1-points[0,0]/points[0,1],
                            relative_rmse_low99=lo, relative_rmse_high99=hi, delta_residual_R2=points[1,0]-points[1,1],
                            residual_R2_low99=intervals[1,0], residual_R2_high99=intervals[1,1]))
    with TemporaryDirectory(dir=root) as temp:
        path = Path(temp)
        pd.DataFrame(results).to_csv(path/"paired.tsv", sep="\t", index=False)
        write_json(path/"receipt.json", dict(proof, freeze_sha256=args.expected_sha256,
                                            files={"paired.tsv":sha256(path/"paired.tsv")}))
        if (root/"score").exists():
            raise FileExistsError(root/"score")
        path.rename(root/"score")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("source", "predict", "lock", "score"))
    cli.add_argument("--dataset", choices=DATASETS)
    cli.add_argument("--fold", type=int, choices=range(3))
    cli.add_argument("--expected-sha256")
    for name in ("work", "t3", "cache", "index", "matrix", "protocol"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    info, proof = bound(args)
    if args.stage == "source":
        input_profiles(args, info, proof)
    elif args.dataset not in info["eligible"]:
        print(f"{args.dataset} lacks registered coverage")
    elif args.stage == "predict":
        predict(args, info, proof, args.fold)
    elif args.stage == "lock":
        freeze = args.work/args.dataset/"freeze.json"
        if freeze.exists():
            raise FileExistsError(freeze)
        write_json(freeze, freeze_body(args, info, proof))
    else:
        score(args, info, proof)
