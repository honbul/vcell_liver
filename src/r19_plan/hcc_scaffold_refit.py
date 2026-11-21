"""Refit Tahoe chemistry after blocking scaffold components from the fitting set."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from src.goal_tracks.t1_data import HIGH_PLATES, MIN_GROUPS, WELL, eligibility, verify_one
from src.goal_tracks.t1_io import align_pool
from src.goal_tracks.t1_score import registry
from src.p123.p1_score import ENDPOINTS, StudyBook
from src.p4.common import WellProjection
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import ENTITY, GROUP, KEY, LINE, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r13_optimization.mechanism import bit_tanimoto
from src.r13_optimization.objectives import center
from src.r19_plan.hcc_dose_sharing import cosine_kernel
from src.r19_plan.hcc_scaffold import CASES, CI, DOSE, LINE_ID, MINIMUM, TARGET, scaffold_draw


MODES = ("full", "scaffold", "same_count")
CHEMBERTA = "scaffold_chemb"
ARMS = ("B0", CHEMBERTA, *(f"{mode}_{head}" for mode in MODES for head in ("ecfp", "copy")))
PAIRS = (
    ("scaffold_ecfp", CHEMBERTA, True), ("scaffold_ecfp", "B0", True),
    ("scaffold_ecfp", "same_count_ecfp", False), ("scaffold_ecfp", "full_ecfp", False),
    ("scaffold_copy", "same_count_copy", False), ("scaffold_copy", "full_copy", False),
    ("scaffold_copy", "B0", False), ("full_copy", "B0", False),
)
TAHOE_GENE_SHA = "08d72bafdfdf08289736d490e0533a4fcdaac76257a656ef10dcea34ad207e76"


def require_hash(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"Input digest differs: {path}")
    return path


def find_inputs(args):
    archive = args.archive
    fold_source = args.base / "vcell_liver_goal_tracks/v1/t1-cache"
    p4 = archive / "p4/data/d5"
    original = require_hash(p4 / "freeze.json", read_json(p4 / "scored/receipt.json")["freeze_sha256"])
    stage_path = require_hash(fold_source / "receipt.json", read_json(original)["stage_sha256"])
    stage = read_json(stage_path)
    verify_one(fold_source, stage, fold_source / "split.json")
    groups = registry(fold_source)
    structure = archive / "r12_multiaxis/group_scaffolds.tsv"
    blocks = table(structure).set_index(GROUP, verify_integrity=True).scaffold_component
    if set(groups) != set(blocks.index) or len(groups) != 203 or blocks.nunique() != 181:
        raise ValueError("The original P4 parent and scaffold axes changed")
    assets = args.base / "vcell_liver_r11_chemical_probe/assets"
    feature_sha = read_json(archive / "r11_chemical_probe/manifest/feature_receipt.json")["outputs"]["features.npz"]
    feature_path = require_hash(assets / "features/features.npz", feature_sha)
    proof = dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
                 source_sha256=sha256(stage_path), p4_freeze_sha256=sha256(original),
                 scaffold_sha256=sha256(structure), features_sha256=feature_sha)
    return fold_source, stage, assets, feature_path, blocks, proof


def static_genes(args, assets):
    lincs_asset = args.base / "vcell_liver_r15_lincs/v1/model-inputs/metadata/genes.tsv"
    lincs_sha = read_json(args.archive / "r15_lincs/data/model-inputs/features/receipt.json")["metadata_sha256"]
    lincs = pd.read_csv(require_hash(lincs_asset, lincs_sha), sep="\t")
    tahoe = pd.read_csv(require_hash(assets / "metadata/genes.tsv", TAHOE_GENE_SHA), sep="\t")
    keys = "ensembl_id"
    usable = [frame.loc[frame[keys].notna() & ~frame[keys].duplicated(keep=False), [keys, "token_id"]]
              for frame in (lincs, tahoe)]
    joined = usable[0].merge(usable[1], on=keys, validate="one_to_one", suffixes=("_lincs", "_tahoe"))
    genes = joined.sort_values("token_id_lincs").reset_index(drop=True)
    if len(lincs) != 978 or genes.empty or not genes.token_id_tahoe.ge(3).all():
        raise ValueError("The static landmark map is unavailable")
    return genes, lincs_sha


def archive_rows(source, stamp, kind, fold):
    path = source / f"f{fold}/{kind}_rows.tsv"
    verify_one(source, stamp, path)
    rows = table(path)
    if rows.quarantined.any() or rows.plate_id.eq("plate13").any():
        raise ValueError(f"Protected study rows in {path}")
    return eligibility(rows, MINIMUM, DOSE)


def prep(args, contract):
    source, source_stamp, assets, _, blocks, proof = contract
    auxiliary = args.base / "vcell_liver_r12_multiaxis/assets/doses/source"
    aux_pin = require_hash(auxiliary / "receipt.json", source_stamp["auxiliary_sha256"])
    require_hash(args.archive / "r12_multiaxis/asset_receipts/09.json", sha256(aux_pin))
    aux_receipt = read_json(aux_pin)
    for kind in ("treated", "A", "B"):
        name = f"{kind}_rows.tsv"
        original = table(require_hash(auxiliary / name, aux_receipt["files"][name]))
        split = "train" if kind == "treated" else "control"
        if original.quarantined.any() or original.plate_id.eq("plate13").any() or not original.split.eq(split).all():
            raise ValueError(f"Nondevelopment rows in the source bundle for {kind}")
    genes, lincs_sha = static_genes(args, assets)
    tokens = genes.token_id_tahoe.to_numpy(dtype=int)
    treated, values = load_data(auxiliary, "treated", panel=tokens, mask=lambda rows: (
        rows.dose_uM.eq(DOSE) & rows.plate_id.isin(HIGH_PLATES) & rows[GROUP].isin(blocks.index)
        & rows.n_cells.ge(MINIMUM)).to_numpy())
    controls = {role: load_data(auxiliary, role, panel=tokens,
                                mask=lambda r: r.plate_id.isin(HIGH_PLATES).to_numpy()) for role in "AB"}
    with TemporaryDirectory(dir=args.run, prefix=".scaffold-inputs-") as temp:
        root = Path(temp)
        visible = root / "visible"
        score = root / "score"
        visible.mkdir()
        score.mkdir()
        genes.to_csv(visible / "genes.tsv", sep="\t", index=False)
        for fold in range(3):
            fit, held = [archive_rows(source, source_stamp, kind, fold) for kind in ("fit", "score")]
            if set(fit[GROUP]) & set(held[GROUP]) or set(fit[WELL]) & set(held[WELL]):
                raise ValueError("P4 fit and held physical rows overlap")
            for kind, rows in (("fit", fit), ("held", held)):
                rows.to_csv(visible / f"f{fold}-{kind}.tsv", sep="\t", index=False)
                linked = align(rows, treated)
                # The raw table allows blank group fields outside this subset, so pandas
                # stores the same numeric identity as a float rather than an integer.
                if not np.array_equal(rows[GROUP], treated.iloc[linked][GROUP]):
                    raise ValueError("Raw and P4 treatment identities disagree")
                t = values[linked].toarray()
                for role, (control_rows, control_values) in controls.items():
                    ci = align_pool(rows, control_rows, ["plate_id", LINE])
                    if not np.array_equal(rows[f"control_{role}"], control_rows.iloc[ci][WELL]):
                        raise ValueError("The fresh projection does not share the P4 physical DMSO roles")
                    dest = visible if kind == "fit" else score
                    np.save(dest / f"f{fold}-{role}.npy", t - control_values[ci].toarray(), allow_pickle=False)
        for name, stage in (("visible", visible), ("score", score)):
            write_json(stage / "receipt.json", dict(
                proof, kind=name, source_counts_sha256=sha256(aux_pin), gene_source_sha256=lincs_sha,
                genes=len(genes), files={p.name: sha256(p) for p in stage.iterdir()}))
        for name in ("visible", "score"):
            if (args.run / name).exists():
                raise FileExistsError(args.run / name)
        visible.rename(args.run / "visible")
        score.rename(args.run / "score")


def checked_receipt(root, proof, names):
    stamp = read_json(root / "receipt.json")
    if any(stamp[key] != expected for key, expected in proof.items()):
        raise ValueError(f"Staged input or code changed: {root}")
    for name in names:
        require_hash(root / name, stamp["files"][name])
    return stamp


def projected_training(rows, delta):
    selected = eligibility(rows.assign(_matrix_index=np.arange(len(rows))), MINIMUM, DOSE)
    indices = selected.pop("_matrix_index").to_numpy(dtype=int)
    projection = WellProjection(selected)
    keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
    line = projection.rows.loc[keep].reset_index(drop=True)
    return line, center(Contrast(line, MIN_GROUPS), projection.centered(delta[indices])[keep])


def cut_training(fit, held, blocks, fold):
    base = WellProjection(fit).rows
    eligible = np.sort(base.loc[base[LINE].eq(LINE_ID), GROUP].unique())
    held_components = set(held[GROUP].map(blocks))
    same = fit[GROUP].map(blocks).isin(held_components)
    n = len(set(eligible) & set(fit.loc[same, GROUP]))
    ordinary_delete = np.random.default_rng(4232026 + fold).choice(eligible, size=n, replace=False)
    masks = dict(full=np.ones(len(fit), dtype=bool), scaffold=~same.to_numpy(),
                 same_count=~fit[GROUP].isin(ordinary_delete).to_numpy())
    if set(fit.loc[masks["scaffold"], GROUP].map(blocks)) & held_components:
        raise ValueError("An excluded scaffold entered fitting")
    return masks, dict(scaffold=sorted(set(fit.loc[same, GROUP])), same_count=sorted(ordinary_delete.tolist()), full=[])


def training_profiles(rows, labels):
    leaders = rows.sort_values([GROUP, ENTITY]).drop_duplicates(GROUP).set_index(GROUP)[ENTITY]
    keep = rows[ENTITY].eq(rows[GROUP].map(leaders)).to_numpy()
    averages = pd.DataFrame(labels[keep]).groupby(rows.loc[keep, GROUP].to_numpy()).mean()
    return leaders.loc[averages.index], averages.to_numpy()


def ridge(train, query, labels, kernel):
    gram = kernel(train, train)
    gram.flat[::len(gram) + 1] += 1
    return kernel(query, train) @ cho_solve(cho_factor(gram), labels)


def predict(args, contract):
    _, _, _, feature_path, blocks, proof = contract
    fold, role = args.fold, args.role
    visible = args.run / "visible"
    fit_name, held_name, assay = f"f{fold}-fit.tsv", f"f{fold}-held.tsv", f"f{fold}-{role}.npy"
    metadata = checked_receipt(visible, proof, (fit_name, held_name, assay, "genes.tsv"))
    fit, held = table(visible / fit_name), table(visible / held_name)
    query = WellProjection(held).rows.loc[lambda r: r[LINE].eq(LINE_ID)].reset_index(drop=True)
    if set(fit[GROUP]) & set(query[GROUP]):
        raise ValueError("A query parent entered fitting")
    delta = np.load(visible / assay, allow_pickle=False)
    if delta.shape != (len(fit), metadata["genes"]) or not np.isfinite(delta).all():
        raise ValueError("Invalid fit expression axis")
    masks, deleted = cut_training(fit, query, blocks, fold)
    predictions = {"B0": np.zeros((len(query), metadata["genes"]))}
    support = []
    with np.load(feature_path, allow_pickle=False) as features:
        mapping = pd.Index(features["entity"])
        ecfp, cls = [features[key].astype(np.float64) for key in ("E", "C")]
    qi = mapping.get_indexer(query[ENTITY])
    for mode, mask in masks.items():
        rows, label = projected_training(fit.loc[mask].reset_index(drop=True), delta[mask])
        entity, prototypes = training_profiles(rows, label)
        fi = mapping.get_indexer(entity)
        if (fi < 0).any() or (qi < 0).any() or len(entity) < 20:
            raise ValueError("Insufficient registered fit chemistry")
        if mode == "scaffold" and set(entity.index.map(blocks)) & set(query[GROUP].map(blocks)):
            raise ValueError("An excluded scaffold entered the actual prototype pool")
        similarity = bit_tanimoto(ecfp[qi], ecfp[fi])
        nearest = similarity.argmax(axis=1)
        maxima = similarity.max(axis=1)
        predictions[f"{mode}_copy"] = prototypes[nearest] * (maxima >= .7)[:, None]
        predictions[f"{mode}_ecfp"] = ridge(ecfp[fi], ecfp[qi], prototypes, bit_tanimoto)
        if mode == "scaffold":
            predictions[CHEMBERTA] = ridge(cls[fi], cls[qi], prototypes, cosine_kernel)
        by_parent = query[[GROUP]].assign(maximum=maxima).groupby(GROUP).maximum.min()
        support.append(dict(mode=mode, fit_parents=len(entity), fit_wells=rows[WELL].nunique(),
                            query_parents=len(by_parent), median_query_tanimoto=float(by_parent.median()),
                            query_at_least_0_7=int(by_parent.ge(.7).sum()), removed_groups=deleted[mode],
                            shared_scaffold_parents=int(query[GROUP][query[GROUP].map(blocks).isin(entity.index.map(blocks))].nunique())))
    if set(predictions) != set(ARMS) or any(
            p.shape != predictions["B0"].shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Invalid prospective forecast array")
    dest = args.run / f"f{fold}-{role}"
    with TemporaryDirectory(dir=args.run) as scratch:
        part = Path(scratch)
        query[KEY + [GROUP]].to_csv(part / "rows.tsv", sep="\t", index=False)
        np.savez_compressed(part / "forecasts.npz", **predictions)
        write_json(part / "receipt.json", dict(
            proof, fold=fold, role=role, fit_input_sha256=sha256(visible / "receipt.json"),
            shape=list(predictions["B0"].shape), support=support,
            files={p.name: sha256(p) for p in part.iterdir()}))
        if dest.exists():
            raise FileExistsError(dest)
        part.rename(dest)


def freeze(args, proof):
    visible = args.run / "visible"
    fit_sha = sha256(visible / "receipt.json")
    cases = {}
    for fold, role in CASES:
        key = f"f{fold}-{role}"
        root = args.run / key
        stamp = checked_receipt(root, proof, ("rows.tsv", "forecasts.npz"))
        if (stamp["fold"], stamp["role"], stamp["fit_input_sha256"]) != (fold, role, fit_sha):
            raise ValueError("A prediction slice changed file identity")
        manifest = checked_receipt(visible, proof, (f"f{fold}-held.tsv",))
        projection = WellProjection(table(visible / f"f{fold}-held.tsv"))
        query = projection.rows.loc[projection.rows[LINE].eq(LINE_ID), KEY + [GROUP]].reset_index(drop=True)
        if not query.equals(table(root / "rows.tsv")) or stamp["shape"] != [len(query), manifest["genes"]]:
            raise ValueError("The forecast is bound to a different query or gene axis")
        with np.load(root / "forecasts.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(ARMS) or any(
                    list(arrays[key].shape) != stamp["shape"] or not np.isfinite(arrays[key]).all() for key in ARMS):
                raise ValueError("Invalid frozen arrays")
        cases[key] = sha256(root / "receipt.json")
    return dict(proof, fit_input_sha256=fit_sha, cases=cases)


def paired(points, draws, scope):
    rows = []
    p = points.set_index("arm")
    for candidate, ref, primary in PAIRS:
        for metric in ENDPOINTS:
            effect = float(p.at[candidate, metric] - p.at[ref, metric])
            lo, hi = np.quantile(draws[TARGET, candidate, metric] - draws[TARGET, ref, metric], CI)
            good = hi < 0 if metric == "contrast_rmse" else lo > 0
            relative = -100 * effect / p.at[ref, metric] if metric == "contrast_rmse" else 100 * effect
            rows.append(dict(scope=scope, arm=candidate, reference=ref, metric=metric, difference=effect,
                             low99=lo, high99=hi, improvement_percent_or_pp=relative, positive_ci=good,
                             practical=relative >= .5, primary=primary and scope == "scaffold"))
    return pd.DataFrame(rows)


def score(args, contract):
    _, _, _, _, blocks, proof = contract
    manifest = require_hash(args.run / "freeze.json", args.expected_sha256)
    frozen = read_json(manifest)
    if frozen != freeze(args, proof):
        raise ValueError("Prediction freeze changed")
    visible, target = (args.run / name for name in ("visible", "score"))
    assay = checked_receipt(target, proof, [f"f{fold}-{role}.npy" for fold, role in CASES])
    checked_receipt(visible, proof, [f"f{fold}-held.tsv" for fold in range(3)])
    books = {kind: StudyBook(sorted(blocks.index), DOSE, MINIMUM, "scaffold_refit")
             for kind in ("parent", "scaffold")}
    books["scaffold"].draw = scaffold_draw(books["scaffold"].index, blocks)
    settings = []
    for fold, role in CASES:
        other = "B" if role == "A" else "A"
        path = args.run / f"f{fold}-{role}"
        held = table(visible / f"f{fold}-held.tsv")
        projection = WellProjection(held)
        keep = projection.rows[LINE].eq(LINE_ID).to_numpy()
        query = projection.rows.loc[keep].reset_index(drop=True)
        if not query[KEY + [GROUP]].equals(table(path / "rows.tsv")):
            raise ValueError("New forecasts do not align to the frozen score axis")
        truth = projection.centered(np.load(target / f"f{fold}-{other}.npy", allow_pickle=False))[keep]
        with np.load(path / "forecasts.npz", allow_pickle=False) as arrays:
            forecasts = {arm: arrays[arm] for arm in ARMS}
        for book in books.values():
            book.add(query, truth, forecasts)
        for record in read_json(path / "receipt.json")["support"]:
            settings.append(dict(fold=fold, role=role, **record))
    points, effects = [], []
    for kind, book in books.items():
        summary, draws = book.results()
        selected = summary.loc[summary.stratum.eq(TARGET)].copy().assign(scope=kind)
        # StudyBook reports total energy on its historical 2,000-column panel.
        selected[["sse", "sst"]] *= assay["genes"] / 2000
        if selected.parent_groups.ne(203).any():
            raise ValueError("The original 203 held SNU-423 parents were not all retained for scoring")
        points.append(selected)
        effects.append(paired(selected, draws, kind))
    dest = args.run / "scored"
    with TemporaryDirectory(dir=args.run) as scratch:
        out = Path(scratch)
        pd.concat(points, ignore_index=True).to_csv(out / "points.tsv", sep="\t", index=False)
        pd.concat(effects, ignore_index=True).to_csv(out / "paired.tsv", sep="\t", index=False)
        pd.DataFrame(settings).to_csv(out / "support.tsv", sep="\t", index=False)
        write_json(out / "receipt.json", dict(
            proof, freeze_sha256=sha256(manifest), score_input_sha256=sha256(target / "receipt.json"),
            files={p.name: sha256(p) for p in out.iterdir()}))
        if dest.exists():
            raise FileExistsError(dest)
        out.rename(dest)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prep", "predict", "freeze", "score"))
    for name in ("run", "base", "archive", "protocol"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--fold", type=int, choices=range(3))
    cli.add_argument("--role", choices=("A", "B"))
    cli.add_argument("--expected-sha256")
    args = cli.parse_args()
    contract = find_inputs(args)
    if args.action == "freeze":
        write_json(args.run / "freeze.json", freeze(args, contract[-1]))
    else:
        {"prep": prep, "predict": predict, "score": score}[args.action](args, contract)
