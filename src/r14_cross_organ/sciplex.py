"""Prespecified Sci-Plex3 candidates with separate prediction, freeze and scoring."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from src.moe_drug_v1.prepare import canonical_parent
from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r13_optimization.mechanism import annotation_similarity, bit_tanimoto, tokens
from src.tahoe_position.followup_c import ALPHAS, SEED, balanced_weights, chemistry, ridge, score_fold, summary_metrics
from src.tahoe_position.sciplex_replicate import DRAWS, LINES, paired_dose, plate_profiles


PROTOCOL_SHA256 = "09774371b5965afbee825e37f955756e90b84700985605a01d1f6c6230c7f63c"
TASKS = ((10000, "rep1"), (10000, "rep2"), (1000, "rep1"), (1000, "rep2"))
MODELS = ("mixed_kernel_search", "annotation_neighbors", "smooth_rmse", "shared_ridge")
BASELINES = ("B0", "replicate_quarter", "replicate_full")
SPECS = {
    "mixed_kernel_search": [*(("ridge", a) for a in ALPHAS), ("knn", 3), ("knn", 8)],
    "annotation_neighbors": [("annotation", 3), ("annotation", 8)],
    "smooth_rmse": [("smooth", 20), ("smooth", 50)],
    "shared_ridge": [("shared", a) for a in ALPHAS],
}


def key_for(smiles):
    if not isinstance(smiles, str) or not smiles:
        return ""
    parent = canonical_parent(smiles)
    mol = Chem.MolFromSmiles(parent[1]) if parent else None
    return Chem.MolToInchiKey(mol).split("-")[0] if mol else ""


def compound_annotations(path, connectivity):
    rows = pd.read_parquet(path, columns=["canonical_smiles", "targets", "moa-fine"])
    rows["connectivity"] = rows.canonical_smiles.map(key_for)
    ann = []
    for key, part in rows.loc[rows.connectivity.ne("")].groupby("connectivity"):
        record = dict(connectivity=key)
        for col in ("targets", "moa-fine"):
            terms = {tuple(sorted(t)) for _, row in part.iterrows() if (t := tokens(row, col))}
            record[col] = ",".join(next(iter(terms))) if len(terms) == 1 else ""
        ann.append(record)
    data = pd.DataFrame(ann, columns=["connectivity", "targets", "moa-fine"]).set_index("connectivity").reindex(connectivity)
    matrix = np.maximum(*(annotation_similarity(data, data, column) for column in ("targets", "moa-fine")))
    return matrix, np.diag(matrix) > 0


def features(wells, compounds, ecfp, annotation_path):
    valid = compounds.loc[compounds.status.eq("valid")].sort_values("fingerprint")
    raw = wells.loc[~wells.vehicle].groupby("drug").SMILES.first()
    mac = []
    for drug in valid.index:
        parent = canonical_parent(raw[drug])
        fingerprint = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(parent[1]))
        x = np.zeros(fingerprint.GetNumBits(), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fingerprint, x)
        mac.append(x)
    mac = np.asarray(mac)
    annotation, annotated = compound_annotations(annotation_path, valid.connectivity)
    chemical = 0.7 * bit_tanimoto(ecfp, ecfp) + 0.3 * bit_tanimoto(mac, mac)
    return chemical, np.concatenate((ecfp, mac), axis=1), annotation, annotated


def read_study(args, dose):
    wells = pd.read_csv(args.cache / "wells.tsv", sep="\t", dtype={"drug": str}).set_index("Combo", verify_integrity=True)
    genes = pd.read_csv(args.cache / "genes_human.tsv", sep="\t")
    if set(wells.time_point) != {24} or set(wells.line) != set(LINES):
        raise ValueError("Sci-Plex3 cache differs from the C time/cell-line cohort")
    history = read_json(args.c / "receipt.json")
    if sha256(args.c / "split.tsv") != history["split_sha256"]:
        raise ValueError("Historical C split receipt differs from the split")
    compounds, universe, ecfp = chemistry(wells)
    expected = pd.read_csv(args.c / "split.tsv", sep="\t", keep_default_na=False).set_index("drug")
    pd.testing.assert_frame_equal(compounds, expected)
    units = {}
    for line in LINES:
        cultures = wells.loc[wells.line.eq(line)]
        values, valid, panel = plate_profiles(args.cache, line, cultures, len(genes))
        gene_ids = genes.id.iloc[panel].astype(str).to_numpy()
        panel_hash = hashlib.sha256("\n".join(gene_ids).encode()).hexdigest()
        if history["panel_sha256"][line] != panel_hash:
            raise ValueError(f"{line} panel differs from C")
        drugs, one, two = paired_dose(line, dose, cultures, values, valid, {drug: drug for drug in compounds.index})
        cohort = compounds.loc[drugs].copy()
        matched = cohort.status.eq("valid").to_numpy()
        cohort, one, two = cohort.loc[matched], one[matched], two[matched]
        for rep in ("rep1", "rep2"):
            actual_well = cultures.loc[(~cultures.vehicle) & cultures.replicate.eq(rep) & cultures.dose.eq(dose)]
            ids = pd.Series(actual_well.index, index=actual_well.drug)
            cohort[f"well_{rep}"] = ids.loc[cohort.index].to_numpy()
        units[line] = (cohort, one, two, gene_ids)
    return units, compounds, universe, ecfp


def centered(values, groups):
    return values - np.average(values, axis=0, weights=balanced_weights(groups))


def average_error(actual, prediction, groups):
    residual = centered(actual - prediction, groups)
    return np.average(np.sqrt(np.mean(residual * residual, axis=1)), weights=balanced_weights(groups))


def neighbors(kernel, values, groups, fit, query, k):
    sim = np.maximum(kernel[np.ix_(query, fit)], 0)
    if sim.shape[1] > k:
        cutoff = np.partition(sim, -k, axis=1)[:, -k, None]
        sim[sim < cutoff] = 0
    weight = sim**2 * balanced_weights(groups[fit])
    totals = weight.sum(axis=1, keepdims=True)
    weight = np.divide(weight, totals, out=np.zeros_like(weight), where=totals > 0)
    return weight @ centered(values[fit], groups[fit])


def smooth_model(features, values, groups, fit, query, epochs):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    x = torch.as_tensor(features[fit], dtype=torch.float32)
    y = torch.as_tensor(centered(values[fit], groups[fit]), dtype=torch.float32)
    held = torch.as_tensor(features[query], dtype=torch.float32)
    weight = torch.as_tensor(balanced_weights(groups[fit]), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(128, y.shape[1]))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    result = {}
    for step in range(1, max(epochs) + 1):
        net.train()
        for ix in torch.randperm(len(x)).split(64):
            optimizer.zero_grad(set_to_none=True)
            log_error = net(x[ix]) - y[ix]
            loss = (log_error.square().mean(dim=1) + 1e-6).sqrt()
            ((loss * weight[ix]).sum() / weight[ix].sum()).backward()
            optimizer.step()
        if step in epochs:
            net.eval()
            with torch.inference_mode():
                result[step] = net(held).numpy().astype(np.float64)
    return result


class Source:
    def __init__(self, units, direction, feature_blocks):
        self.units = {
            line: (cohort, one if direction == "rep1" else two, genes)
            for line, (cohort, one, two, genes) in units.items()
        }
        self.kernel, self.chemical, self.annotation, self.annotated = feature_blocks
        self.genes = np.array(sorted(set.intersection(*(set(x[2]) for x in self.units.values()))))

    def shared(self, line, fit_folds, target_idx, query, own, alpha):
        x, groups, indices = [], [], []
        for frame, observed, genes in self.units.values():
            fit = frame.fold.isin(fit_folds).to_numpy()
            x.append(centered(observed[fit][:, pd.Index(genes).get_indexer(self.genes)], frame.connectivity.to_numpy()[fit]))
            groups.extend(frame.connectivity.loc[fit])
            indices.extend(frame.fingerprint.loc[fit])
        indices = np.r_[indices, target_idx[query]]
        values = np.vstack(x)
        n, m = len(values), len(query)
        pad = np.zeros((m, values.shape[1]), dtype=np.float64)
        shared = ridge(self.kernel[np.ix_(indices, indices)], np.vstack([values, pad]),
                       np.array(groups + [""] * m), np.arange(n), np.arange(n, n + m), alpha)
        cols = pd.Index(self.units[line][2]).get_indexer(self.genes)
        own[:, cols] = 0.5 * (own[:, cols] + shared)
        return own

    def run_one(self, line, fit_folds, output_fold, specs):
        frame, values, _ = self.units[line]
        fit = np.flatnonzero(frame.fold.isin(fit_folds))
        query = np.flatnonzero(frame.fold.eq(output_fold))
        groups, idx = frame.connectivity.to_numpy(), frame.fingerprint.to_numpy()
        if set(groups[fit]) & set(groups[query]):
            raise ValueError("Query parent in model training")
        kernel = self.kernel[np.ix_(idx, idx)]
        ann = self.annotation[np.ix_(idx, idx)]
        smooth_steps = [p for method, p in specs if method == "smooth"]
        smooth = smooth_model(self.chemical[idx], values, groups, fit, query, smooth_steps) if smooth_steps else {}
        out = {}
        for method, p in specs:
            if method in ("ridge", "shared"):
                value = ridge(kernel, values, groups, fit, query, p)
                if method == "shared":
                    value = self.shared(line, fit_folds, idx, query, value, p)
            elif method in ("knn", "annotation"):
                value = neighbors(ann if method == "annotation" else kernel, values, groups, fit, query, p)
            else:
                value = smooth[p]
            out[method, p] = value
        return query, out

    def validate(self, line, outer):
        frame, values, _ = self.units[line]
        all_specs = list(dict.fromkeys(spec for specs in SPECS.values() for spec in specs))
        errors = {spec: [] for spec in all_specs}
        fit_folds = {0, 1, 2} - {outer}
        for inner in sorted(fit_folds):
            query, fit = self.run_one(line, fit_folds - {inner}, inner, all_specs)
            for spec, predicted in fit.items():
                mask = self.annotated[frame.fingerprint.to_numpy()[query]] if spec[0] == "annotation" else np.ones(len(query), bool)
                if mask.any():
                    errors[spec].append(average_error(values[query][mask], predicted[mask], frame.connectivity.to_numpy()[query][mask]))
        return {
            arm: min(specs, key=lambda spec: np.mean(errors[spec]) if errors[spec] else np.inf)
            for arm, specs in SPECS.items()
        }


def predict(args):
    dose, source_rep = TASKS[args.task]
    units, compounds, _, ecfp = read_study(args, dose)
    feature_blocks = features(
        pd.read_csv(args.cache / "wells.tsv", sep="\t", dtype={"drug": str}), compounds, ecfp, args.annotation,
    )
    src = Source(units, source_rep, feature_blocks)
    task = args.output / "predictions" / str(args.task)
    task.mkdir(parents=True, exist_ok=False)
    selected = []
    for line in LINES:
        frame, _, _ = src.units[line]
        predictions = {arm: np.empty((len(frame), 2000), dtype=np.float32) for arm in MODELS}
        for outer in range(3):
            specs = src.validate(line, outer)
            query, fit = src.run_one(line, {0, 1, 2} - {outer}, outer, list(specs.values()))
            for arm, spec in specs.items():
                predictions[arm][query] = fit[spec]
                selected.append(dict(line=line, fold=outer, arm=arm, method=spec[0], parameter=spec[1]))
            print(f"Frozen inputs fitted for {line}, fold {outer}, dose {dose}, {source_rep}", flush=True)
        frame = frame.reset_index()
        frame["annotated"] = src.annotated[frame.fingerprint]
        frame.to_csv(task / f"{line}.tsv", sep="\t", index=False)
        np.savez(task / f"{line}.npz", **predictions)
    pd.DataFrame(selected).to_csv(task / "selected.tsv", sep="\t", index=False)
    write_json(task / "receipt.json", dict(
        task=args.task, dose_nM=dose, source_rep=source_rep, shared_genes=src.genes.tolist(),
        code_sha256=sha256(__file__), protocol_sha256=PROTOCOL_SHA256,
        split_sha256=sha256(args.c / "split.tsv"), annotation_sha256=sha256(args.annotation),
        files={path.name: sha256(path) for path in task.iterdir()},
    ))


def files_in(folder):
    return {path.name: sha256(path) for path in sorted(folder.iterdir()) if path.is_file()}


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Changed frozen file: {path}")


def freeze(args):
    predictions = args.output / "predictions"
    files = {}
    for task, (dose, rep) in enumerate(TASKS):
        folder = predictions / str(task)
        actual = files_in(folder)
        receipt = read_json(folder / "receipt.json")
        if receipt["dose_nM"] != dose or receipt["source_rep"] != rep or receipt["task"] != task:
            raise ValueError(f"Unexpected prediction task {task}")
        verify(Path(__file__), receipt["code_sha256"])
        verify(args.c / "split.tsv", receipt["split_sha256"])
        verify(args.annotation, receipt["annotation_sha256"])
        if receipt["protocol_sha256"] != PROTOCOL_SHA256:
            raise ValueError(f"Prediction used a different protocol: {task}")
        declared = receipt["files"]
        if {k: v for k, v in actual.items() if k != "receipt.json"} != declared:
            raise ValueError(f"Prediction artifacts changed in {task}")
        files.update({f"predictions/{task}/{name}": digest for name, digest in actual.items()})
    prior = read_json(args.c / "receipt.json")
    for name, digest in prior["cache_sha256"].items():
        verify(args.cache / name, digest)
    write_json(args.output / "freeze.json", dict(protocol_sha256=PROTOCOL_SHA256, files=files,
               c_receipt_sha256=sha256(args.c / "receipt.json"), code_sha256=sha256(__file__),
               doses_nM=[10000, 1000], arms=list(MODELS)))


def score_line(cohort, source, target, arrays, draws, universe, subset):
    codes = universe.get_indexer(cohort.connectivity)
    total = {name: np.zeros(len(draws)) for name in (*BASELINES, *arrays)}
    mass = np.zeros(len(draws))
    for outer in range(3):
        query = np.flatnonzero(cohort.fold.eq(outer).to_numpy() & subset)
        if not len(query):
            continue
        summed, denominator = score_fold(target[query], source[query],
                                        {arm: values[query] for arm, values in arrays.items()}, codes[query], draws)
        mass += denominator
        for name in total:
            total[name] += summed[name]
    if not subset.any():
        return None
    return {name: value / mass for name, value in total.items()}


def average_metrics(pieces):
    return {arm: np.mean([part[arm] for part in pieces], axis=0) for arm in pieces[0]}


def score(args):
    freeze_path = args.output / "freeze.json"
    frozen = read_json(freeze_path)
    if frozen["protocol_sha256"] != PROTOCOL_SHA256 or tuple(frozen["arms"]) != MODELS:
        raise ValueError("Unexpected frozen protocol or candidate set")
    verify(args.c / "receipt.json", frozen["c_receipt_sha256"])
    verify(Path(__file__), frozen["code_sha256"])
    for name, digest in frozen["files"].items():
        verify(args.output / name, digest)
    prior = read_json(args.c / "receipt.json")
    for name, digest in prior["cache_sha256"].items():
        verify(args.cache / name, digest)
    results, sizes = {}, {}
    for dose in (10000, 1000):
        units, _, universe, _ = read_study(args, dose)
        draws = np.vstack((np.ones(len(universe)), np.random.default_rng(SEED).exponential(size=(DRAWS, len(universe)))))
        for task, (task_dose, source_rep) in enumerate(TASKS):
            if task_dose != dose:
                continue
            folder = args.output / "predictions" / str(task)
            for line, (frame, one, two, _) in units.items():
                stored = pd.read_csv(folder / f"{line}.tsv", sep="\t", dtype={"drug": str}).set_index("drug")
                pd.testing.assert_frame_equal(frame, stored.drop(columns="annotated"))
                source, target = (one, two) if source_rep == "rep1" else (two, one)
                with np.load(folder / f"{line}.npz", allow_pickle=False) as stream:
                    predictions = {arm: stream[arm].astype(np.float64) for arm in MODELS}
                if any(arr.shape != target.shape or not np.isfinite(arr).all() for arr in predictions.values()):
                    raise ValueError(f"Invalid forecast shape or values: {line} {dose} {source_rep}")
                for label, mask in (("all", np.ones(len(frame), bool)), ("annotated_only", stored.annotated.to_numpy())):
                    current = score_line(frame, source, target, predictions, draws, universe, mask)
                    if current is None:
                        continue
                    results[dose, label, line, source_rep] = current
                    sizes[dose, label, line] = (int(mask.sum()), set(frame.connectivity[mask]))
    table = []
    for dose in (10000, 1000):
        for label in ("all", "annotated_only"):
            for lines in ((line,) for line in LINES):
                if (dose, label, lines[0]) not in sizes:
                    continue
                line = lines[0]
                n, groups = sizes[dose, label, line]
                for rep in ("rep1", "rep2"):
                    out = summary_metrics(results[dose, label, line, rep], n, len(groups), line, rep, dose)
                    table.extend(dict(**row, subset=label, gate=None) for row in out)
            keys = [k for k in results if k[:2] == (dose, label)]
            if keys:
                merged = average_metrics([results[k] for k in keys])
                coverage = [value for key, value in sizes.items() if key[:2] == (dose, label)]
                n = sum(x[0] for x in coverage)
                groups = len(set.union(*(x[1] for x in coverage)))
                out = summary_metrics(merged, n, groups, "equal_line_mean", "equal_direction", dose)
                table.extend(dict(**row, subset=label, gate=(row["reduction_low"] > 0 if
                                  dose == 10000 and label == "all" and row["arm"] in MODELS else None))
                             for row in out)
    destination = args.output / "scored"
    destination.mkdir(exist_ok=False)
    pd.DataFrame(table).drop(columns="positive_10uM_gate").to_csv(destination / "scores.tsv", sep="\t", index=False)
    write_json(destination / "receipt.json", dict(protocol_sha256=PROTOCOL_SHA256, freeze_sha256=sha256(freeze_path),
               files=files_in(destination), bootstrap=dict(draws=DRAWS, seed=SEED, parent_weights="exponential")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["predict", "freeze", "score"])
    parser.add_argument("--task", choices=range(len(TASKS)), type=int)
    for name in ("cache", "c", "annotation", "protocol", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    verify(args.protocol, PROTOCOL_SHA256)
    {"predict": predict, "freeze": freeze, "score": score}[args.phase](args)


if __name__ == "__main__":
    main()
