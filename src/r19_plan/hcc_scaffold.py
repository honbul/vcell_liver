"""Score the frozen SNU-423 P4 predictions by fit-only chemical support."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.goal_tracks.t1_data import MIN_GROUPS, WELL, eligibility, slice_name, verify_one
from src.goal_tracks.t1_io import score_truth
from src.goal_tracks.t1_score import DRAW_COUNT, STRATUM_LINES, registry
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.p4.common import History, WellProjection, common_args, finish_args
from src.p4.score import binding, comparisons, load_slice, reference_names
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, LINE, group_weights, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.mechanism import bit_tanimoto
from src.r13_optimization.objectives import center, row_products


DOSE, MINIMUM = 5., 20
TARGET = "SNU-423"
LINE_ID, = STRATUM_LINES[TARGET]
ARM = "P4-sibling"
REFS = ("B0", "best-VC")
CI = (.005, .995)
CASES = [(fold, role) for fold in range(3) for role in "AB"]
BANDS = ("below_0.3", "0.3_to_below_0.5", "0.5_to_below_0.7", "0.7_or_more")


def contract(args):
    args.dose = DOSE
    args = finish_args(args)
    previous = args.archive / "p4/data/d5"
    frozen_path = previous / "freeze.json"
    original_score = previous / "scored/receipt.json"
    reference = read_json(original_score)
    paired = previous / "scored/paired.tsv"
    if (sha256(args.out/"d5/freeze.json") != sha256(frozen_path)
            or sha256(frozen_path) != reference["freeze_sha256"]
            or sha256(paired) != reference["files"]["paired.tsv"]):
        raise ValueError("The original P4 forecast and score differ from the published copies")
    frozen = read_json(frozen_path)
    stage = read_json(args.stage/"receipt.json")
    if sha256(args.stage/"receipt.json") != frozen["stage_sha256"]:
        raise ValueError("The P4 fit manifest does not match the original forecast")
    feature_pin = read_json(args.archive/"r11_chemical_probe/manifest/feature_receipt.json")
    if sha256(args.features) != feature_pin["outputs"]["features.npz"]:
        raise ValueError("The structure fingerprint cache changed")
    scaffold_path = args.archive/"r12_multiaxis/group_scaffolds.tsv"
    blocks = table(scaffold_path).set_index(GROUP, verify_integrity=True).scaffold_component
    if len(blocks) != 203 or blocks.nunique() != 181 or set(blocks.index) != set(registry(args.stage)):
        raise ValueError("The R12 structure components are not aligned to the P4 canonical parent axis")
    inputs = dict(protocol_sha256=sha256(args.design), code_sha256=sha256(Path(__file__)),
                  original_freeze_sha256=sha256(frozen_path), original_score_sha256=sha256(original_score),
                  scaffold_sha256=sha256(scaffold_path), features_sha256=sha256(args.features))
    original = table(paired)
    wanted = original.loc[original.stratum.eq(TARGET) & original.arm.eq(ARM) & original.minimum_cells.eq(MINIMUM)]
    return inputs, frozen, stage, blocks, wanted


def neighbors(train, query, entities, fingerprints, blocks):
    fit = train.sort_values([GROUP, ENTITY], kind="stable").drop_duplicates(GROUP).set_index(GROUP)
    available = sorted(set(train.loc[train[LINE].eq(LINE_ID), GROUP]))
    targets = query.loc[query[LINE].eq(LINE_ID), [GROUP, ENTITY]].drop_duplicates().sort_values([GROUP, ENTITY])
    if set(available) & set(targets[GROUP]) or not available:
        raise ValueError("The structure lookup crosses the original held parent fold")

    def fp(molecules):
        positions = entities.get_indexer(molecules)
        if (positions < 0).any():
            raise ValueError("An admitted chemical entity lacks its frozen ECFP")
        return fingerprints[positions].astype(np.float64)

    similar = bit_tanimoto(fp(targets[ENTITY]), fp(fit.loc[available, ENTITY]))
    tops = similar.max(axis=1)
    closest = np.isclose(similar, tops[:, None], rtol=0, atol=1e-10)
    fit_components = set(blocks.loc[available])
    return targets.assign(max_tanimoto=tops, nearest_fit_parents=["|".join(map(str, np.asarray(available)[r])) for r in closest],
                          scaffold_component=targets[GROUP].map(blocks),
                          fit_scaffold_support=targets[GROUP].map(blocks).isin(fit_components))


def summarize_neighbors(rows):
    result = rows.groupby([GROUP, "fold", "scaffold_component", "fit_scaffold_support"], as_index=False).agg(
        min_tanimoto=("max_tanimoto", "min"), max_tanimoto=("max_tanimoto", "max"),
        entities=(ENTITY, "nunique"), all_entities_pass_AB=("passes_both_roles", "all"))
    if result[GROUP].duplicated().any() or len(result) != 203:
        raise ValueError("Expected the full set of 203 pair-disjoint evaluation parents")
    result["band"] = pd.cut(result.min_tanimoto, [-np.inf, .3, .5, .7, np.inf], labels=BANDS, right=False)
    return result


def audit(args, inputs, frozen, stage, blocks):
    cases = []
    with np.load(args.features, allow_pickle=False) as features:
        entity, fingerprints = pd.Index(features["entity"]), features["E"]
    if not entity.is_unique or not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Frozen ECFP is not a unique binary entity axis")
    for fold in range(3):
        root = args.stage/f"f{fold}"
        fit, query = (root/(name+"_rows.tsv") for name in ("fit", "score"))
        for path in (fit, query):
            verify_one(args.stage, stage, path)
        train, held = [WellProjection(eligibility(table(path), MINIMUM, DOSE)).rows for path in (fit, query)]
        axes = neighbors(train, held, entity, fingerprints, blocks).assign(fold=fold)
        for role in "AB":
            key = slice_name(DOSE, MINIMUM, fold, role)
            candidate = args.out / "predictions/chemical" / key / "receipt.json"
            if sha256(candidate) != frozen["tasks"][key]["chemical"]:
                raise ValueError("P4 inherited nearest-neighbor threshold changed")
            threshold = read_json(candidate)["selection"]["inherited"]["threshold"]
            axes[f"threshold_{role}"] = threshold
        axes["passes_both_roles"] = axes.max_tanimoto.ge(axes[["threshold_A", "threshold_B"]].max(axis=1))
        cases.append(axes)
    support = pd.concat(cases, ignore_index=True)
    output(args.product, dict(inputs, phase="structure_only", expression_read=False),
           {"entities.tsv": support, "parent_support.tsv": summarize_neighbors(support)})


def output(path, proof, artifacts):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".hcc-scaffold-") as tmp:
        stage = Path(tmp)
        for name, frame in artifacts.items():
            frame.to_csv(stage/name, sep="\t", index=False)
        write_json(stage/"receipt.json", dict(proof, files={p.name: sha256(p) for p in sorted(stage.iterdir())}))
        stage.rename(path)


def scaffold_draw(groups, blocks):
    codes, labels = pd.factorize(blocks.loc[groups], sort=True)
    values = np.random.default_rng(191705).exponential(size=(DRAW_COUNT, len(labels)))
    return values[:, codes]


def fixed_errors(rows, truth, forecasts, refs):
    metric = Contrast(rows, MIN_GROUPS)
    weight = group_weights(rows)
    base = rows[[GROUP, WELL, ENTITY]].copy().assign(weight=weight)
    for name in (ARM, *refs):
        error = center(metric, forecasts[name] - truth)
        raw = row_products(error, error)
        base["rmse_"+name] = weight * np.sqrt(raw)
        base["sse_"+name] = raw * truth.shape[1]
    return base


def scores_by_support(errors, support, refs, groups, block_draw):
    parents = errors.drop(columns=[WELL, ENTITY]).groupby(GROUP).sum().join(support.set_index(GROUP), validate="one_to_one")
    counts = errors.groupby(GROUP)[WELL].nunique().rename("physical_wells")
    parents = parents.join(counts)
    subtotal = parents["sse_B0"] - parents["sse_" + ARM]
    scopes = {key: parents.band.eq(key) for key in BANDS}
    scopes.update(all_fixed=parents.min_tanimoto.notna(), at_least_0_5=parents.min_tanimoto.ge(.5),
                  below_0_7=parents.min_tanimoto.lt(.7), seen_component=parents.fit_scaffold_support,
                  unseen_component=~parents.fit_scaffold_support)
    records = []
    for name, mask in scopes.items():
        selected = parents.loc[mask]
        if selected.empty:
            continue
        axis = block_draw[:, groups.get_indexer(selected.index)]
        row = dict(scope=name, parents=len(selected), physical_wells=int(selected.physical_wells.sum()),
                   scaffold_components=selected.scaffold_component.nunique(), eligible=len(selected) >= 20,
                   all_entities_pass_AB=int(selected.all_entities_pass_AB.sum()),
                   positive_sse_share=subtotal.loc[mask].clip(lower=0).sum()/subtotal.clip(lower=0).sum(),
                   net_sse_share=subtotal.loc[mask].sum()/subtotal.sum())
        for ref in refs:
            for metric, prefix, denominator, factor in (("contrast_rmse", "rmse_", "weight", -1),
                                                         ("r2_pooled", "sse_", "sse_B0", 1)):
                gain = factor * (selected[prefix+ref] - selected[prefix+ARM])
                mass = selected[denominator]
                # RMSE uses candidate minus reference. R² uses reference error minus candidate error.
                estimate = gain.sum()/mass.sum()
                lo, hi = np.quantile((axis @ gain.to_numpy())/(axis @ mass.to_numpy()), CI)
                records.append(dict(row, arm=ARM, reference=ref, metric=metric, difference=estimate, low99=lo, high99=hi))
    return parents.reset_index(), pd.DataFrame(records)


def score(args, inputs, frozen, blocks, original):
    proof = read_json(args.structure/"receipt.json")
    if sha256(args.structure/"receipt.json") != args.expected_sha256 or any(proof[key] != h for key, h in inputs.items()):
        raise ValueError("Structure bands were not frozen with the original study assets")
    for name, h in proof["files"].items():
        if sha256(args.structure/name) != h:
            raise ValueError("The frozen structure support changed")
    support = table(args.structure/"parent_support.tsv")
    keep_refs = tuple(dict.fromkeys(original.best_at_point))
    history = History(args, DOSE)
    if any(frozen[k] != v for k, v in binding(args, history).items()):
        raise ValueError("The P4 forecast code or registered input lineage changed")
    arms = {*reference_names(frozen["arms"], history), "B0", ARM}
    groups = registry(args.stage)
    parent_book, scaffold_book = [StudyBook(groups, DOSE, MINIMUM, "development") for _ in range(2)]
    scaffold_book.draw = scaffold_draw(scaffold_book.index, blocks)

    # Verify arrays and freeze receipts for all slices before looking up any scoring truth.
    cases = []
    for fold, role in CASES:
        rows, panel, forecasts, _, evidence = load_slice(args, history, MINIMUM, fold, role)
        if evidence != frozen["tasks"][slice_name(DOSE, MINIMUM, fold, role)] or sorted(forecasts) != frozen["arms"]:
            raise ValueError("One frozen P4 prediction or source array is inconsistent")
        cases.append((fold, role, rows, panel, {arm: forecasts[arm] for arm in sorted(arms)}))
    errors = []
    for fold, role, rows, panel, forecasts in cases:
        original_truth = score_truth(args.stage, DOSE, MINIMUM, fold, role, rows, panel)
        projection = WellProjection(rows)
        selected = projection.rows[LINE].eq(LINE_ID).to_numpy()
        line = projection.rows.loc[selected].reset_index(drop=True)
        if set(line[GROUP]) != set(support.loc[support.fold.eq(fold), GROUP]):
            raise ValueError("The scorer and the metadata audit disagree on a held canonical parent")
        truth = projection.centered(original_truth)[selected]
        pred = {name: projection.centered(value)[selected] for name, value in forecasts.items()}
        for book in (parent_book, scaffold_book):
            book.add(line, truth, pred)
        errors.append(fixed_errors(line, truth, pred, keep_refs))

    results = []
    keys = ["reference", "metric"]
    fields = ["difference", "ci99_low", "ci99_high"]
    old = original.set_index(keys, verify_integrity=True).loc[:, fields]
    for unit, book in (("parent_original", parent_book), ("scaffold_components", scaffold_book)):
        points, boot = book.results()
        points = points.loc[points.stratum.eq(TARGET)]
        paired = comparisons(points, boot, arms - {"B0", ARM})
        paired = paired.loc[paired.arm.eq(ARM)].set_index(keys, verify_integrity=True).loc[old.index].reset_index()
        check = fields if unit == "parent_original" else ["difference"]
        if not np.allclose(paired[check].to_numpy(), old.loc[:, check].to_numpy(), atol=1e-10, rtol=0):
            raise ValueError(f"{unit} altered the published P4 point or original interval")
        results.append(paired.assign(resampling_unit=unit))
    parents, strata = scores_by_support(pd.concat(errors, ignore_index=True), support, keep_refs,
                                        scaffold_book.index, scaffold_book.draw)
    output(args.product, dict(inputs, structure_freeze_sha256=args.expected_sha256, verified_original_p4=True),
           {"paired.tsv": pd.concat(results), "by_parent.tsv": parents, "strata.tsv": strata})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("phase", choices=("audit", "score"))
    for name in ("design", "features", "product"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    inputs, frozen, stage, blocks, original = contract(args)
    if args.phase == "audit":
        audit(args, inputs, frozen, stage, blocks)
    else:
        score(args, inputs, frozen, blocks, original)


if __name__ == "__main__":
    main()
