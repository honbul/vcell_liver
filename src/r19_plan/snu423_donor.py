"""Predict, freeze, and independently score the matched SNU-423 donor experiment."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_GROUPS, WELL, array, eligibility, slice_name, verify_one
from src.goal_tracks.t1_io import align_pool, score_truth
from src.goal_tracks.t1_predict import PositionBank
from src.goal_tracks.t1_score import axis_draw, registry
from src.p123.p123_io import load_prediction
from src.p4.common import History, WellProjection, common_args, finish_args
from src.p4.score import binding, load_slice
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.objectives import center, row_products
from src.r19_plan.hcc_scaffold import CASES, DOSE, LINE_ID, MINIMUM, scaffold_draw
from src.r19_plan.i_preflight import read_pinned
from src.r19_plan.r18_input_bundle import extract_matrix, indices
from src.r19_plan.t4_workflow import deltas
from src.r19_plan.x_inputs import check_unique, control_axis


STATE = "State-Morgan-well-adaptation"
STATE_ARMS = (STATE, "fair-" + STATE)
EQUAL_RNA = tuple(name + "-scalar" for name in STATE_ARMS)
CANDIDATE = "donor_rank4"
ARMS = ("B0", "P4-sibling", "position", "donor_scalar", CANDIDATE, "donor_rank4_shuffle", *STATE_ARMS, *EQUAL_RNA)
PAIRS = [(CANDIDATE, name) for name in ("B0", "P4-sibling", "position", "donor_scalar", "donor_rank4_shuffle",
                                        "best-State-plus-RNA", *EQUAL_RNA)]
PAIRS += [("donor_scalar", name) for name in ("B0", "position", "best-State-plus-RNA")]


def contracts(args):
    args.dose = DOSE
    args = finish_args(args)
    p4_path = args.archive / "p4/data/d5/freeze.json"
    local_audit = args.archive / "r19_plan/data/snu_donor/expanded/receipt.json"
    if sha256(p4_path) != sha256(args.out/"d5/freeze.json") or sha256(local_audit) != sha256(args.audit/"receipt.json"):
        raise ValueError("P4 forecast or expanded audit differs from the committed receipt")
    audit, frozen, stage = map(read_json, (local_audit, p4_path, args.stage/"receipt.json"))
    coverage = read_pinned(args.audit, "parent_support.tsv", audit["files"]["parent_support.tsv"])
    admitted = coverage.loc[coverage.measured_5_to_20uM.eq("True")]
    if (audit["expression_read"] or not audit["metadata_only"] or sha256(args.matrix) != audit["matrix_reference_sha256"]
            or coverage[GROUP].duplicated().any() or admitted.fold.value_counts().sort_index().tolist() != [34, 31, 35]
            or set(coverage[GROUP].astype(int)) != set(registry(args.stage))):
        raise ValueError("The nonreserved donor admission or frozen split changed")
    gene_root = args.archive / "r15_lincs/data/external/A/A"
    gene_reference = read_json(gene_root / "receipt.json")["files"]["genes.tsv"]
    if sha256(gene_root/"genes.tsv") != gene_reference:
        raise ValueError("P4 common gene record differs from the published R15 subset")
    genes = table(gene_root/"genes.tsv")
    reference = dict(code_sha256=sha256(__file__), protocol_sha256=sha256(args.design),
                     audit_sha256=sha256(local_audit), p4_sha256=sha256(p4_path), genes_sha256=gene_reference)
    return audit, frozen, stage, coverage, admitted, genes, reference


def donors(args, audit, admitted, genes, role):
    registration = [read_pinned(args.audit, name, audit["files"][name]) for name in
                    ("admitted_donors.tsv", "admitted_vehicles.tsv", "gene_indices.tsv")]
    source, controls, axis = registration
    distance = source.actual_dose_uM.astype(float) - 5.
    minimum = distance.groupby([source.parent, source.cell_id]).transform("min")
    source = source.loc[distance.eq(minimum)].reset_index(drop=True)
    controls = controls.loc[controls.role.eq(role) & controls.rna_plate.isin(source.rna_plate)]
    control_axis(controls, source, role)
    combined = pd.concat((source, controls), ignore_index=True)
    check_unique(combined)
    if set(source.parent) != set(admitted.parent) or not axis.gene_id.astype(int).equals(genes.token_id_lincs):
        raise ValueError("The input RNA subset changed its canonical parent or gene axis")
    record = read_json(args.matrix)["matrix_records"]["combined"]
    raw = extract_matrix(record, combined, indices(axis.level3_gene_index), genes.token_id_lincs.to_numpy())
    response = deltas(source, raw[:len(source)], controls, raw[len(source):]) * np.log(2.)
    frame = pd.DataFrame(response)
    frame = frame.groupby([source.parent, source.cell_id, source.rna_plate]).mean()
    profile = frame.groupby(level=(0, 1)).mean().groupby(level=0).mean()
    return profile.reindex(admitted.parent).set_axis(admitted[GROUP].astype(int))


def target_fit(args, stage, genes, fold, role):
    folder = args.stage/f"f{fold}"
    for name in ("tokens.npy", "fit_rows.tsv", "fit.npy", f"{role}_rows.tsv", f"{role}.npy"):
        verify_one(args.stage, stage, folder/name)
    tokens = pd.Index(array(folder/"tokens.npy"))
    columns = tokens.get_indexer(genes.token_id_tahoe)
    if (columns < 0).any():
        raise ValueError("The common Tahoe gene axis changed")
    register = table(folder/"fit_rows.tsv").assign(matrix_row=lambda frame: np.arange(len(frame)))
    fit = eligibility(register, MINIMUM, DOSE)
    controls = table(folder/f"{role}_rows.tsv")
    controls_i = align_pool(fit, controls, ["plate_id", LINE])
    if not np.array_equal(fit[f"control_{role}"], controls.iloc[controls_i][WELL]):
        raise ValueError("Wrong physical DMSO role for target adaptation")
    treated = array(folder/"fit.npy")[np.ix_(fit.matrix_row, columns)]
    vehicle = array(folder/f"{role}.npy")[np.ix_(controls_i, columns)]
    projection = WellProjection(fit)
    keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
    return projection.rows.loc[keep].reset_index(drop=True), projection.centered(treated-vehicle)[keep]


def expand(rows, profiles):
    return profiles.reindex(rows[GROUP]).fillna(0.).to_numpy()


def shuffle(values, random):
    order = random.permutation(len(values))
    shuffled = values.copy()
    shuffled.iloc[order] = values.iloc[np.roll(order, 1)].to_numpy()
    return shuffled


def gain(x, y, weights):
    mass = weights @ row_products(x, x)
    return float(np.clip(weights @ row_products(x, y) / mass, 0., 2.)) if mass > 0 else 0.


def ridge4(train, query, fit_profile, query_profile, residual, metric, weight):
    x = fit_profile.to_numpy()
    mean = x.mean(axis=0)
    _, singular, right = np.linalg.svd(x-mean, full_matrices=False)
    if len(singular) < 4 or singular[3] <= 1e-8:
        raise ValueError("A fit partition lacks four independent donor coordinates")
    axes = right[:4].T
    train_score = expand(train, pd.DataFrame((x-mean) @ axes, index=fit_profile.index))
    query_score = expand(query, pd.DataFrame((query_profile.to_numpy()-mean) @ axes, index=query_profile.index))
    z = center(metric, train_score)
    weights = z.T * weight
    return query_score @ np.linalg.solve(weights @ z + np.eye(4), weights @ residual)


def predict(args, contract):
    audit, frozen, stage, _, admitted, genes, stamp = contract
    fold, role = args.fold, args.role
    history = History(args, DOSE)
    key = slice_name(DOSE, MINIMUM, fold, role)
    if any(frozen[k] != value for k, value in binding(args, history).items()):
        raise ValueError("The registered P4 baseline binding changed")
    rows, panel, forecasts, _, proof = load_slice(args, history, MINIMUM, fold, role)
    if proof != frozen["tasks"][key]:
        raise ValueError("The P4 comparator forecast does not match its freeze")
    cols = pd.Index(panel).get_indexer(genes.token_id_tahoe)
    if (cols < 0).any():
        raise ValueError("The common panel does not index the P4 prediction")
    projection = WellProjection(rows)
    keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
    query = projection.rows.loc[keep].reset_index(drop=True)
    train, fit = target_fit(args, stage, genes, fold, role)
    if set(train[GROUP]) & set(query[GROUP]) or set(train[WELL]) & set(query[WELL]):
        raise ValueError("A held parent or treated well entered SNU-423 fitting")
    profiles = donors(args, audit, admitted, genes, role)
    train_profile = profiles.loc[profiles.index.isin(train[GROUP])]
    query_profile = profiles.loc[profiles.index.isin(query[GROUP])]
    if len(query_profile) != (34, 31, 35)[fold] or len(train_profile) != 100-len(query_profile):
        raise ValueError("The donor arms and the physical forecast have different coverage")

    metric = Contrast(train, MIN_GROUPS)
    weight = group_weights(train)
    bank = PositionBank(train, fit)
    fit_pos, _ = bank.predict(train, exclude_parent=True)
    query_pos, _ = bank.predict(query)
    alpha = gain(center(metric, fit_pos), center(metric, fit), weight)
    anchor = alpha*query_pos
    residual = center(metric, fit-alpha*fit_pos)
    beta = gain(center(metric, expand(train, train_profile)), residual, weight)
    scalar = beta*expand(query, query_profile)
    operator = ridge4(train, query, train_profile, query_profile, residual, metric, weight)
    random = np.random.default_rng(23017+fold)
    fake = ridge4(train, query, shuffle(train_profile, random), shuffle(query_profile, random), residual, metric, weight)
    result = {name: projection.centered(forecasts[name][:, cols])[keep] for name in ("B0", "P4-sibling", *STATE_ARMS)}
    result.update(position=anchor, donor_scalar=anchor+scalar, donor_rank4=anchor+operator, donor_rank4_shuffle=anchor+fake)
    result.update({out: result[name]+scalar for out, name in zip(EQUAL_RNA, STATE_ARMS, strict=True)})
    if set(result) != set(ARMS) or any(y.shape != (len(query), len(genes)) or not np.isfinite(y).all() for y in result.values()):
        raise ValueError("Invalid SNU-423 forecast axes or values")
    path = args.run / "forecasts" / key
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path/"forecasts.npz", **result)
    query[KEY+[GROUP]].to_csv(path/"rows.tsv", sep="\t", index=False)
    write_json(path/"receipt.json", dict(stamp, fold=fold, role=role, position_gain=alpha, scalar_gain=beta,
               fit_parents=int(train[GROUP].nunique()), fit_donors=len(train_profile), query_donors=len(query_profile),
               p4_receipts=proof, files={p.name: sha256(p) for p in path.iterdir()}))


def freeze(args, contract):
    _, frozen, _, _, _, genes, stamp = contract
    p4, receipts, selections, cases = {}, {}, [], []
    for fold, role in CASES:
        key = slice_name(DOSE, MINIMUM, fold, role)
        path = args.run/"forecasts"/key
        rec = read_json(path/"receipt.json")
        if (any(rec[k] != v for k, v in stamp.items()) or (rec["fold"], rec["role"]) != (fold, role)
                or rec["p4_receipts"] != frozen["tasks"][key] or set(rec["files"]) != {"rows.tsv", "forecasts.npz"}
                or any(sha256(path/name) != h for name, h in rec["files"].items())):
            raise ValueError("The SNU-423 forecast changed before or after the freeze")
        original = args.out/"predictions/chemical"/key
        if sha256(original/"receipt.json") != frozen["tasks"][key]["chemical"]:
            raise ValueError("P4 evaluation metadata changed")
        _, rows, panel, _ = load_prediction(original, args.stage, args.protocol, DOSE, MINIMUM, fold, role)
        projection = WellProjection(rows)
        keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
        query = projection.rows.loc[keep].reset_index(drop=True)
        if not query[KEY+[GROUP]].equals(table(path/"rows.tsv")):
            raise ValueError("Donor and P4 physical scoring coordinates differ")
        with np.load(path/"forecasts.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(ARMS):
                raise ValueError("A registered SNU-423 arm changed")
            forecast = {a: saved[a] for a in ARMS}
        if any(y.shape != (len(query), len(genes)) or not np.isfinite(y).all() for y in forecast.values()):
            raise ValueError("The frozen forecast lost its finite gene matrix shape")
        p4[key] = query[GROUP].unique()
        cases.append((fold, role, rows, panel, projection, keep, query, forecast))
        receipts[key] = sha256(path/"receipt.json")
        selections.append({k: rec[k] for k in ("fold", "role", "position_gain", "scalar_gain", "fit_parents", "fit_donors", "query_donors")})
    if len(set().union(*(set(groups) for groups in p4.values()))) != 203:
        raise ValueError("The forecast does not cover the full development cohort")
    return dict(stamp, slices=receipts), cases, pd.DataFrame(selections)


def scored_cases(args, cases, genes):
    results = []
    for fold, role, rows, panel, projection, keep, query, forecasts in cases:
        cols = pd.Index(panel).get_indexer(genes.token_id_tahoe)
        if (cols < 0).any():
            raise ValueError("The scorer lacks the fixed gene axis")
        truth = projection.centered(score_truth(args.stage, DOSE, MINIMUM, fold, role, rows, panel)[:, cols])[keep]
        metric = Contrast(query, MIN_GROUPS)
        weight = group_weights(query)
        summary = query[[GROUP, WELL]].copy().assign(weight=weight)
        for arm, values in forecasts.items():
            error = center(metric, values-truth)
            squared = row_products(error, error)
            summary[arm+"_rmse"] = np.sqrt(squared)*weight
            summary[arm+"_sse"] = squared
        results.append(summary)
    return pd.concat(results, ignore_index=True)


def comparisons(errors, coverage, blocks):
    parents = errors.drop(columns=WELL).groupby(GROUP).sum().join(coverage.set_index(GROUP), validate="one_to_one")
    measured = parents.measured_5_to_20uM.eq("True")
    scopes = {"all": parents.index.notna(), "measured": measured, "measured_tanimoto_below_0.7": measured & parents.min_tanimoto.lt(.7)}
    groups, parent_draw = axis_draw(parents.index.to_list())
    all_draws = {"parent": parent_draw, "scaffold": scaffold_draw(groups, blocks)}
    points, paired = [], []
    for scope, keep in scopes.items():
        rows = parents.loc[keep]
        meta = dict(scope=scope, parents=len(rows), physical_wells=errors.loc[errors[GROUP].isin(rows.index), WELL].nunique())
        for suffix, den in (("_rmse", "weight"), ("_sse", "B0_sse")):
            smaller = suffix == "_rmse"
            denominator = rows[den].to_numpy()
            values = rows[[a+suffix for a in ARMS]].to_numpy()
            point = values.sum(0)/denominator.sum()
            if not smaller:
                point = 1.-point
            by_arm = dict(zip(ARMS, point, strict=True))
            measure = "contrast_rmse" if smaller else "r2_pooled"
            points.extend(dict(meta, arm=arm, metric=measure, score=x) for arm,x in by_arm.items())
            for unit, weights in all_draws.items():
                draw = weights[:, groups.get_indexer(rows.index)]
                boot = (draw @ values)/(draw @ denominator)[:, None]
                if not smaller:
                    boot = 1.-boot
                boot = dict(zip(ARMS, boot.T, strict=True))
                best = (min if smaller else max)(EQUAL_RNA, key=by_arm.get)
                for arm, baseline in PAIRS:
                    use = best if baseline == "best-State-plus-RNA" else baseline
                    reference = ((np.minimum if smaller else np.maximum)(*(boot[a] for a in EQUAL_RNA))
                                 if baseline == "best-State-plus-RNA" else boot[use])
                    low, high = np.quantile(boot[arm]-reference, (.005, .995))
                    paired.append(dict(meta, arm=arm, reference=baseline, best_at_point=use, metric=measure, unit=unit,
                        difference=by_arm[arm]-by_arm[use], low99=low, high99=high))
    return pd.DataFrame(points), pd.DataFrame(paired)


def score(args, contract, stamp, cases, settings):
    path = args.run / "freeze.json"
    if sha256(path) != args.expected_sha256 or read_json(path) != stamp:
        raise ValueError("Independent scoring requires the original SNU-423 freeze")
    _, _, _, coverage, _, genes, _ = contract
    coverage = coverage.assign(**{GROUP: coverage[GROUP].astype(int), "min_tanimoto": coverage.min_tanimoto.astype(float)})
    blocks = table(args.archive/"r12_multiaxis/group_scaffolds.tsv").set_index(GROUP).scaffold_component
    points, pairs = comparisons(scored_cases(args, cases, genes), coverage, blocks)
    with TemporaryDirectory(dir=args.run, prefix=".snu-score-") as temporary:
        output = Path(temporary)
        for name, frame in (("points.tsv",points), ("paired.tsv",pairs), ("settings.tsv",settings)):
            frame.to_csv(output/name, sep="\t", index=False)
        write_json(output/"receipt.json", dict(stamp, freeze_sha256=args.expected_sha256,
                                              files={file.name: sha256(file) for file in output.iterdir()}))
        output.rename(args.run/"score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("phase", choices=("predict", "freeze", "score"))
    for name in ("audit", "matrix", "design", "run"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(3))
    parser.add_argument("--role", choices="AB")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    contract = contracts(args)
    if args.phase == "predict":
        predict(args, contract)
        return
    stamp, cases, settings = freeze(args, contract)
    if args.phase == "freeze":
        write_json(args.run/"freeze.json", stamp)
    else:
        score(args, contract, stamp, cases, settings)


if __name__ == "__main__":
    main()
