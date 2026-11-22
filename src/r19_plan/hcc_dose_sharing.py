"""Fit and score a three-dose chemical kernel on the exposed SNU-423 development wells."""

import argparse
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.goal_tracks.t1_data import DOSES, MIN_GROUPS, WELL, array, eligibility, slice_name, verify_one
from src.goal_tracks.t1_io import align_pool, score_truth
from src.goal_tracks.t1_score import registry
from src.p123.p1_score import StudyBook
from src.p4.common import History, WellProjection, common_args, finish_args
from src.p4.score import binding, comparisons, load_slice, reference_names
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE, table
from src.r11_chemical_probe.metrics import Contrast
from src.r13_optimization.mechanism import bit_tanimoto
from src.r13_optimization.objectives import center
from src.r19_plan.hcc_scaffold import CASES, DOSE, LINE_ID, MINIMUM, TARGET, scaffold_draw

BASE = "ECFP-5only"
JOINT = "ECFP-dose-joint"
CHEMBERTA = "ChemBERTa-dose-joint"
SHUFFLE = "ECFP-dose-shuffled"
SIBLING = "P4-sibling"
ARMS = (BASE, JOINT, CHEMBERTA, SHUFFLE)


def contracts(args):
    args.dose = DOSE
    finish_args(args)
    original = args.archive / "p4/data/d5/freeze.json"
    past_score = args.archive / "p4/data/d5/scored/receipt.json"
    original_hash = sha256(original)
    if original_hash != sha256(args.out/"d5/freeze.json") or original_hash != read_json(past_score)["freeze_sha256"]:
        raise ValueError("Published P4 forecasts or score lineage changed")
    frozen, stage = read_json(original), read_json(args.stage/"receipt.json")
    features = read_json(args.archive/"r11_chemical_probe/manifest/feature_receipt.json")["outputs"]["features.npz"]
    if sha256(args.stage/"receipt.json") != frozen["stage_sha256"] or sha256(args.features) != features:
        raise ValueError("The study uses a changed staged cohort or chemistry feature bank")
    structure_path = args.archive/"r19_plan/data/hcc_scaffold/structure"
    support = structure_path/"parent_support.tsv"
    if sha256(support) != read_json(structure_path/"receipt.json")["files"][support.name]:
        raise ValueError("The reported chemical-support bands changed")
    blocks = args.archive/"r12_multiaxis/group_scaffolds.tsv"
    proof = dict(protocol_sha256=sha256(args.design), code_sha256=sha256(__file__),
                 p4_freeze_sha256=original_hash, stage_sha256=frozen["stage_sha256"],
                 features_sha256=features, scaffold_sha256=sha256(blocks), support_sha256=sha256(support))
    return frozen, stage, proof, blocks, support


def axes(args, stage, fold):
    folder = args.stage/f"f{fold}"
    for name in ("tokens.npy", "fit_rows.tsv", "score_rows.tsv", "panels.json"):
        verify_one(args.stage, stage, folder/name)
    panel = read_json(folder/"panels.json")[slice_name(DOSE, MINIMUM, fold, "panel")+"_-1"]
    tokens = pd.Index(array(folder/"tokens.npy"))
    cols = tokens.get_indexer(panel)
    if len(cols) != 2000 or len(set(cols)) != len(cols) or (cols<0).any():
        raise ValueError("The P4 fit-only 2,000 gene panel is unavailable")
    fit = table(folder/"fit_rows.tsv")
    score = table(folder/"score_rows.tsv")
    wanted = set(registry(args.stage))
    if (set(fit[GROUP]) & set(score[GROUP]) or set(fit[WELL]) & set(score[WELL])
            or not set(pd.concat((fit,score))[GROUP]) == wanted
            or pd.concat((fit,score)).plate_id.eq("plate13").any()):
        raise ValueError("Parent, physical well or plate is outside the original development split")
    raw_query = eligibility(score, MINIMUM, DOSE)
    projection = WellProjection(raw_query)
    line = projection.rows.loc[projection.rows[LINE].eq(LINE_ID)].reset_index(drop=True)
    return folder, fit, raw_query, line, np.asarray(panel), cols


def mean_fit(args, stage, folder, fit, cols, role):
    for name in ("fit.npy", f"{role}.npy", f"{role}_rows.tsv"):
        verify_one(args.stage, stage, folder/name)
    t = array(folder/"fit.npy")
    c = array(folder/f"{role}.npy")
    vehicles = table(folder/f"{role}_rows.tsv")
    measured = []
    fit = fit.assign(_matrix_row=np.arange(len(fit)))
    for value in DOSES:
        selected = eligibility(fit, MINIMUM, value)
        matching = align_pool(selected, vehicles, ["plate_id", LINE])
        if not np.array_equal(selected[f"control_{role}"], vehicles.iloc[matching][WELL]):
            raise ValueError("Fit did not subtract the originally assigned physical DMSO wells")
        delta = t[np.ix_(selected._matrix_row, cols)] - c[np.ix_(matching, cols)]
        projection = WellProjection(selected)
        line = projection.rows[LINE].eq(LINE_ID).to_numpy()
        rows = projection.rows.loc[line].reset_index(drop=True)
        z = center(Contrast(rows, MIN_GROUPS), projection.centered(delta)[line])
        measured.append((rows, z))
    high = measured[-1][0].sort_values([GROUP, ENTITY]).drop_duplicates(GROUP).set_index(GROUP)[ENTITY]
    profiles = []
    for dose, (rows, z) in enumerate(measured):
        use = rows[ENTITY].eq(rows[GROUP].map(high)).to_numpy()
        values = pd.DataFrame(z[use]).groupby(rows.loc[use, GROUP].to_numpy()).mean()
        values.index = pd.MultiIndex.from_arrays([values.index, np.full(len(values), dose)], names=[GROUP, "dose"])
        profiles.append(values)
    return high, pd.concat(profiles, verify_integrity=True).sort_index()


def dose_covariance(profiles):
    names = profiles.index.get_level_values("dose")
    levels = [profiles.loc[names==dose].droplevel("dose") for dose in range(len(DOSES))]
    overlap = levels[0].index.intersection(levels[1].index).intersection(levels[2].index)
    if len(overlap)<20:
        return np.eye(len(DOSES)), len(overlap)
    matrices = [v.loc[overlap].to_numpy() for v in levels]
    flattened = np.stack([(v-v.mean(axis=0)).ravel() for v in matrices])
    norms = np.linalg.norm(flattened, axis=1)
    if np.any(norms==0):
        return np.eye(len(DOSES)), len(overlap)
    scaled = flattened/norms[:,None]
    return .5 * (scaled @ scaled.T + np.eye(len(DOSES))), len(overlap)


def product_ridge(gram, cross, train_dose, covariance, labels):
    product = gram * covariance[np.ix_(train_dose, train_dose)]
    product.flat[::len(product)+1] += 1.0
    projection = cross * covariance[-1, train_dose]
    return projection @ cho_solve(cho_factor(product), labels)


def cosine_kernel(a, b):
    def unit(x):
        length = np.linalg.norm(x, axis=1, keepdims=True)
        return np.divide(x, length, out=np.zeros_like(x), where=length>0)
    return unit(a) @ unit(b).T


def mapped_arm(kernel, labels, codes, cov, q, source_fp, duplicate):
    reference = source_fp[duplicate]
    return product_ridge(kernel(reference, reference), kernel(q, reference), codes, cov, labels)


def forecasts_from_profiles(features, high, query, profiles, fold):
    covariance, common = dose_covariance(profiles)
    parent_codes = high.index.get_indexer(profiles.index.get_level_values(GROUP))
    doses = profiles.index.get_level_values("dose").to_numpy()
    if (parent_codes<0).any() or set(high.index) & set(query[GROUP]):
        raise ValueError("A fitted structure was missing or entered the held query")
    structure = pd.Index(features["entity"])
    index = structure.get_indexer(high)
    target = structure.get_indexer(query[ENTITY])
    if not structure.is_unique or (index<0).any() or (target<0).any():
        raise ValueError("A matched chemical entity lacks its frozen feature")
    data = profiles.to_numpy()
    together = {}
    selected = doses==len(DOSES)-1
    e = features["E"].astype(np.float64)
    ridge = partial(mapped_arm, bit_tanimoto, q=e[target], source_fp=e[index])
    together[BASE] = ridge(data[selected], doses[selected], np.eye(len(DOSES)), duplicate=parent_codes[selected])
    together[JOINT] = ridge(data, doses, covariance, duplicate=parent_codes)
    rng = np.random.default_rng(42319+fold)
    shuffled = data.copy()
    for dose in range(len(DOSES)-1):
        ix = np.flatnonzero(doses==dose)
        ix = rng.permutation(ix)
        shuffled[ix] = data[np.roll(ix, 1)]
    together[SHUFFLE] = ridge(shuffled, doses, covariance, duplicate=parent_codes)
    chem = features["C"].astype(np.float64)
    together[CHEMBERTA] = mapped_arm(cosine_kernel, data, doses, covariance, chem[target], chem[index], parent_codes)
    return together, covariance, common, np.bincount(doses, minlength=len(DOSES)).tolist()


def predict(args, contract):
    _, stage, proof, _, _ = contract
    fold, role = args.fold, args.role
    folder, fit, _, query, panel, cols = axes(args, stage, fold)
    high, profiles = mean_fit(args, stage, folder, fit, cols, role)
    with np.load(args.features, allow_pickle=False) as features:
        forecasts, cov, complete, n = forecasts_from_profiles(features, high, query, profiles, fold)
    if any(y.shape != (len(query),len(panel)) or not np.isfinite(y).all() for y in forecasts.values()):
        raise ValueError("A registered dose model did not produce a finite aligned array")
    path = args.work / f"f{fold}-{role}"
    with TemporaryDirectory(dir=args.work) as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage/"forecasts.npz", panel=panel, **forecasts)
        query[KEY+[GROUP]].to_csv(stage/"rows.tsv", sep="\t", index=False)
        write_json(stage/"receipt.json", dict(proof, fold=fold, role=role, high_training_parents=len(high),
                   complete_training_parents=complete, dose_training_parents=n, training_dose_covariance=cov.tolist(),
                   files={p.name:sha256(p) for p in stage.iterdir()}))
        if path.exists():
            raise FileExistsError(path)
        stage.rename(path)


def freeze(args, contract):
    _, stage, proof, _, _ = contract
    files, settings = {}, []
    for fold, role in CASES:
        path = args.work/f"f{fold}-{role}"
        receipt = read_json(path/"receipt.json")
        if (any(receipt[k]!=v for k,v in proof.items()) or (receipt["fold"],receipt["role"])!=(fold,role)
                or any(sha256(path/name)!=h for name,h in receipt["files"].items())):
            raise ValueError("An immutable dose prediction or fit receipt changed")
        _, _, _, query, panel, _ = axes(args, stage, fold)
        if not query[KEY+[GROUP]].equals(table(path/"rows.tsv")):
            raise ValueError("The model did not use the full P4 SNU-423 held axis")
        with np.load(path/"forecasts.npz", allow_pickle=False) as z:
            if (set(z.files)!={*ARMS,"panel"} or not np.array_equal(panel,z["panel"])
                    or any(z[k].shape!=(len(query),len(panel)) or not np.isfinite(z[k]).all() for k in ARMS)):
                raise ValueError("A frozen dose prediction is missing its expected axes")
        files[path.name] = sha256(path/"receipt.json")
        settings.append({k:receipt[k] for k in ("fold","role","complete_training_parents",
                                               "dose_training_parents","training_dose_covariance")})
    return dict(proof, parts=files), pd.DataFrame(settings)


def newly_paired(points, draws, benchmark):
    ret = comparisons(points, draws, benchmark)
    ret = ret.loc[ret.arm.isin(ARMS)].copy()
    index = points.set_index("arm")
    explicit = []
    for ref in (BASE, CHEMBERTA, SHUFFLE, SIBLING):
        for metric in ("contrast_rmse", "r2_pooled"):
            dist = draws[TARGET, JOINT, metric] - draws[TARGET, ref, metric]
            lo, hi = np.quantile(dist, [.005, .995])
            explicit.append(dict(dose_uM=DOSE, minimum_cells=MINIMUM, stratum=TARGET, arm=JOINT,
                                 reference=ref, best_at_point=ref, metric=metric,
                                 difference=index.at[JOINT, metric] - index.at[ref, metric],
                                 ci99_low=lo, ci99_high=hi, passes=bool(hi < 0 if metric=="contrast_rmse" else lo > 0)))
    ret = pd.concat((ret, pd.DataFrame(explicit)), ignore_index=True)
    ret["primary"] = ret.arm.eq(JOINT) & ret.reference.isin({BASE,CHEMBERTA,SHUFFLE,"best-VC"})
    return ret


def score(args, contract, manifest):
    frozen, _, _, block_path, support_path = contract
    new, settings = freeze(args, contract)
    if sha256(manifest)!=args.expected_sha256 or read_json(manifest)!=new:
        raise ValueError("Score requires the verified forecast freeze")
    history = History(args, DOSE)
    if any(frozen[k]!=value for k,value in binding(args,history).items()):
        raise ValueError("Published P4 scoring provenance differs")
    groups = registry(args.stage)
    blocks = table(block_path).set_index(GROUP, verify_integrity=True).scaffold_component
    support = table(support_path).set_index(GROUP, verify_integrity=True).min_tanimoto
    if len(blocks)!=203 or blocks.nunique()!=181 or set(groups)!=set(blocks.index) or set(groups)!=set(support.index):
        raise ValueError("The original structural support axes changed")
    parent, scaffold, low = (StudyBook(groups, DOSE, MINIMUM, "development") for _ in range(3))
    scaffold.draw = scaffold_draw(scaffold.index, blocks)
    low.draw = scaffold.draw
    benchmark = reference_names(frozen["arms"],history)
    retained = {*benchmark,"B0",SIBLING}
    for fold, role in CASES:
        rows, panel, frozen_preds, _, evidence = load_slice(args, history, MINIMUM, fold, role)
        key = slice_name(DOSE, MINIMUM, fold, role)
        if evidence!=frozen["tasks"][key] or sorted(frozen_preds)!=frozen["arms"]:
            raise ValueError("A historical P4 comparator changed")
        truth = score_truth(args.stage, DOSE, MINIMUM, fold, role, rows, panel)
        projection = WellProjection(rows)
        keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
        line = projection.rows.loc[keep].reset_index(drop=True)
        y = projection.centered(truth)[keep]
        predictions = {name:projection.centered(frozen_preds[name])[keep] for name in retained}
        del frozen_preds
        path = args.work/f"f{fold}-{role}"
        if not line[KEY+[GROUP]].equals(table(path/"rows.tsv")):
            raise ValueError("New models are not aligned to the historical score rows")
        with np.load(path/"forecasts.npz", allow_pickle=False) as z:
            if not np.array_equal(z["panel"], panel):
                raise ValueError("The new models disagree on the original gene axis")
            predictions.update({name:z[name] for name in ARMS})
        parent.add(line,y,predictions)
        scaffold.add(line,y,predictions)
        # Apply previously fixed structure support after all predictions and truths are projected.
        distant = line[GROUP].map(support).lt(.7)
        subset = line.loc[distant].assign(_matrix_row=lambda rows: rows.index)
        eligible = eligibility(subset, 0)._matrix_row.to_numpy(dtype=int)
        if len(eligible):
            low.add(line.iloc[eligible],y[eligible],{arm:p[eligible] for arm,p in predictions.items()})
        print(f"Scored {fold} {role}", flush=True)
    output = {"points":[], "paired":[]}
    for scope, book in (("parent",parent),("scaffold",scaffold),("below_0.7_scaffold",low)):
        values, samples = book.results()
        values = values.loc[values.stratum.eq(TARGET)]
        pairs = newly_paired(values,samples,benchmark)
        pairs["primary"] &= scope=="parent"
        output["paired"].append(pairs.assign(scope=scope))
        output["points"].append(values.assign(scope=scope))
    path = args.work/"score"
    with TemporaryDirectory(dir=args.work) as tmp:
        stage = Path(tmp)
        for name, parts in output.items():
            pd.concat(parts, ignore_index=True).to_csv(stage/(name+".tsv"), sep="\t", index=False)
        settings.to_csv(stage/"settings.tsv", sep="\t", index=False)
        write_json(stage/"receipt.json", dict(new, freeze_sha256=args.expected_sha256,
                   files={p.name:sha256(p) for p in stage.iterdir()}))
        if path.exists():
            raise FileExistsError(path)
        stage.rename(path)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    common_args(cli)
    cli.add_argument("phase", choices=("predict","lock","score"))
    for name in ("design", "features", "work"):
        cli.add_argument(f"--{name}",type=Path,required=True)
    cli.add_argument("--fold",type=int,choices=range(3))
    cli.add_argument("--role",choices="AB")
    cli.add_argument("--expected-sha256")
    args = cli.parse_args()
    contract = contracts(args)
    manifest = args.work/"freeze.json"
    if args.phase=="predict":
        predict(args,contract)
    elif args.phase=="lock":
        if manifest.exists():
            raise FileExistsError(manifest)
        write_json(manifest,freeze(args,contract)[0])
    else:
        score(args,contract,manifest)
