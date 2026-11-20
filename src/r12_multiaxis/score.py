"""Freeze R12 inference, then score paired roles with compound-weighted pool re-centering."""

import argparse
from functools import cached_property
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, ENTITY, GROUP, KEY, LINE, group_weights, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align
from src.r6_followup_v1.metrics import cond_rmse, grams

from .stage import load_staged


FORMAL = {
    "Shift-B_abs": ("absolute", {"Shift": 1, "B_abs": -1}),
    "Context-B_contrast": ("contrast", {"Context": 1, "B_contrast": -1}),
    "Structure-B_contrast": ("contrast", {"Structure": 1, "B_contrast": -1}),
    "Loss-B_contrast": ("contrast", {"Loss": 1, "B_contrast": -1}),
    "Integrated-B_contrast": ("contrast", {"Integrated": 1, "B_contrast": -1}),
}
ATTRIBUTION = {
    "Shift_neural-Shift_pool": ("absolute", {"Shift_neural": 1, "Shift_pool": -1}),
    "Shift-Shift_unadjusted": ("absolute", {"Shift": 1, "Shift_unadjusted": -1}),
    "Shift-Shift_unadjusted_contrast": ("contrast", {"Shift": 1, "Shift_unadjusted": -1}),
    "Context-Context_matched": ("contrast", {"Context": 1, "Context_matched": -1}),
    "Structure-Structure_matched": ("contrast", {"Structure": 1, "Structure_matched": -1}),
    "Loss-Loss_matched": ("contrast", {"Loss": 1, "Loss_matched": -1}),
    "J1_interaction": ("contrast", {"J1_FiLM_L2": 1, "L2": -1, "A5": -1, "A4": 1}),
    "J1_FiLM_L2-B_contrast": ("contrast", {"J1_FiLM_L2": 1, "B_contrast": -1}),
    "J2_raw-single": ("contrast", {"J2_raw": 1, "single": -1}),
    "J2_calibrated-single": ("contrast", {"J2_calibrated": 1, "single": -1}),
    "J2_raw-B_contrast": ("contrast", {"J2_raw": 1, "B_contrast": -1}),
    "J2_calibrated-B_contrast": ("contrast", {"J2_calibrated": 1, "B_contrast": -1}),
}
PAIR_ARMS = [
    ("C1", "A1"), ("C2", "A1"), ("C3", "C3_rank0"), ("C4", "C4_mean"), ("C7", "C7_mean"),
    ("C8", "C8_mean"), ("C9", "A4"), ("C9", "C9_id"),
    ("A2", "A1"), ("A3", "A1"), ("A5", "A4"), ("A6", "A4"),
    ("L1", "A4"), ("L2", "A4"), ("L3", "A4"), ("L4", "A4"), ("L5", "A4"), ("L6", "L5"),
    ("L6", "A4"), ("J1_FiLM_L2", "L2"), ("J1_FiLM_L2", "A5"),
    ("E1_128", "E1_32"), ("E2_128", "E2_32"), ("E2_128", "E1_128"), ("E3", "E2_32"),
    ("E3", "E1_32"), ("E4_32", "A4"), ("E4_64", "A4"), ("D1", "D1_match"),
]
ATTRIBUTION.update({f"{a}-{b}": ("contrast", {a: 1, b: -1}) for a, b in PAIR_ARMS})


def freeze(args, cfg):
    if args.freeze.exists():
        raise ValueError("This evaluation already has a frozen manifest")
    stage = args.assets / args.stage_dir
    split = args.assets / args.split_dir
    shapes = {}
    files = {}
    code = {str(p): sha256(p) for p in Path("src/r12_multiaxis").glob("*.py")}
    for k in range(3):
        outer = args.predictions / f"outer{k}"
        for role in ("A", "B"):
            folder = outer / role
            receipt = read_json(folder / "receipt.json")
            selection = read_json(folder / "selection.json")
            if receipt["role"] != role or receipt["outer"] != k or receipt["config_sha256"] != sha256(args.config):
                raise ValueError("Prediction receipt has wrong identity")
            if receipt["partition_sha256"] != sha256(split / "split_manifest.tsv"):
                raise ValueError("Prediction receipt has wrong split")
            for name, expected in receipt["files"].items():
                if expected != sha256(folder / name):
                    raise ValueError(f"Inference artifact changed: {folder / name}")
            for filename, expected in receipt["code"].items():
                if filename not in code or code[filename] != expected:
                    raise ValueError(f"Pipeline source changed after inference: {filename}")
            index = read_json(folder / "prediction_index.json")
            if not all(set(comparison) <= set(index) for _, comparison in FORMAL.values()):
                raise ValueError("Formal effects require missing predictions")
            paths = table(folder / "rows.tsv")
            files[f"outer{k}/{role}/receipt.json"] = sha256(folder / "receipt.json")
            shapes[f"outer{k}/{role}"] = {
                "rows": len(paths), "groups": paths[GROUP].nunique(),
                "selection_sha256": sha256(folder / "selection.json"),
                "integrated": selection["joint"]["selected"],
            }
        row_a = table(outer / "A/rows.tsv")
        row_b = table(outer / "B/rows.tsv")
        if not row_a[KEY].equals(row_b[KEY]) or row_a[CONTROL].eq(row_b[CONTROL]).any():
            raise ValueError("Role rows differ or feature wells overlap")
        if not np.array_equal(np.load(outer / "A/panel.npy"), np.load(outer / "B/panel.npy")):
            raise ValueError("Role neutral panels are inconsistent")
    write_json(args.freeze, dict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), path=cfg["path"],
        exposure="protocol-frozen internal follow-up", code=code, config_sha256=sha256(args.config),
        partition_sha256=sha256(split / "split_manifest.tsv"), staged_receipt_sha256=sha256(stage / "receipt.json"),
        predictions=str(args.predictions.resolve()), predictions_receipts=files,
        shapes=shapes, official=FORMAL, repeats=cfg["evaluation"]["replicates"],
        declaration="Freeze created before opening the staged outer scoring expression arrays.",
    ))


def novelty(rows, fit, features):
    with np.load(features) as f:
        entity = pd.Index(f["entity"])
        fingerprints = f["E"].astype(np.float32)
    train = fingerprints[entity.get_indexer(fit[ENTITY].unique())]
    order = entity.get_indexer(rows[ENTITY].unique())
    if (order < 0).any():
        raise ValueError("Missing ECFP on scoring manifest")
    query = fingerprints[order]
    intersection = query @ train.T
    unions = query.sum(axis=1)[:, None] + train.sum(axis=1) - intersection
    by_entity = pd.Series((intersection / np.maximum(unions, 1)).max(axis=1), index=entity[order])
    return rows[ENTITY].map(by_entity).to_numpy()


def measure(rows, c, pred, observed, own, absolute):
    _, _, contrast, delta = c.points(pred, observed)
    r, top, pool_n, zero = c.retrieval(pred, observed)
    pred_center = c.points(pred, np.zeros_like(pred))[2]
    observed_center = c.points(observed, np.zeros_like(observed))[2]
    return dict(contrast=contrast, absolute=cond_rmse(own + pred, absolute), delta=delta, pds=r,
                top1=top, retrieval_candidates=pool_n, zero_norm=zero.astype(float),
                predicted_contrast_rms=pred_center, observed_contrast_rms=observed_center)


class FoldScore:
    def __init__(self, args, cfg, fold, role, truth_rows, truth_counts, group_ids, draws):
        cfg_eval = cfg["evaluation"]
        r11 = Path(cfg["r11_assets"])
        self.folder = args.predictions / f"outer{fold}" / role
        self.rows = table(self.folder / "rows.tsv")
        self.rows["outer_fold"] = fold
        self.role = role
        self.fold = fold
        self.groups = pd.Index(group_ids).get_indexer(self.rows[GROUP])
        self.draws = draws
        self.panel = np.load(self.folder / "panel.npy")
        self.measured = truth_counts[align(self.rows, truth_rows)][:, self.panel].toarray()
        opposite = "B" if role == "A" else "A"
        ctrl_rows, ctrl = load_data(r11 / "source", role, self.panel)
        score_rows, score_ctrl = load_data(r11 / "source", opposite, self.panel)
        own_pos = align(self.rows, ctrl_rows, ["plate_id", LINE])
        score_pos = align(self.rows, score_rows, ["plate_id", LINE])
        if not np.array_equal(self.rows[CONTROL], ctrl_rows.iloc[own_pos][KEY[0]]):
            raise ValueError("Frozen feature control does not match role")
        if np.any(ctrl_rows.iloc[own_pos][KEY[0]].to_numpy() == score_rows.iloc[score_pos][KEY[0]].to_numpy()):
            raise ValueError("Observed delta uses the feature control as score control")
        self.own = ctrl[own_pos].toarray()
        self.observed = self.measured - score_ctrl[score_pos].toarray()
        self.index = read_json(self.folder / "prediction_index.json")
        self.selection = read_json(self.folder / "selection.json")
        self.index["Shift_unadjusted"] = self.index[self.selection["shift"]["Shift"]["base"]["arm"]]
        self.contrast = Contrast(self.rows, cfg["minimum_groups"])
        if len(self.contrast.active) != len(self.rows):
            raise ValueError("Scoring manifest includes ineligible pools")
        fit_rows = table(args.assets / args.stage_dir / f"outer{fold}" / "fit-rows.tsv")
        similarity = novelty(self.rows, fit_rows, r11 / "features/features.npz")
        np.save(args.out / f"outer{fold}-{role}-novelty.npy", similarity)
        band_low, band_high = cfg_eval["novelty_bands"]
        self.scopes = {"all": np.ones(len(self.rows), dtype=bool)}
        self.scopes.update({plate: self.rows.plate_id.eq(plate).to_numpy() for plate in cfg["plates"]})
        self.scopes.update({f"omit_{plate}": self.rows.plate_id.ne(plate).to_numpy() for plate in cfg["plates"]})
        self.scopes.update({"HepG2/C3A": self.rows[LINE].eq("CVCL_1098").to_numpy(),
                            "SNU-423": self.rows[LINE].eq("CVCL_0366").to_numpy(),
                            "Tanimoto<0.25": similarity < band_low,
                            "0.25<=Tanimoto<0.5": (similarity >= band_low) & (similarity < band_high),
                            "Tanimoto>=0.5": similarity >= band_high})
        self.weights = {name: self.scope_weights(mask) for name, mask in self.scopes.items()}
        self.arm_values = {}
        self.arm_boot = {}
        np.savez(args.out / f"outer{fold}-{role}-truth.npz", absolute=self.measured,
                 own_control=self.own, score_delta=self.observed, panel=self.panel)

    def scope_weights(self, mask):
        result = np.zeros(len(mask))
        if mask.any():
            result[mask] = group_weights(self.rows.loc[mask])
        return result

    def seed_metrics(self, relative):
        if relative not in self.arm_values:
            pred = np.load(self.folder / relative)
            self.arm_values[relative] = measure(self.rows, self.contrast, pred, self.observed, self.own, self.measured)
        return self.arm_values[relative]

    def points(self, arm):
        each = [self.seed_metrics(file) for file in self.index[arm].values()]
        return {m: np.mean([run[m] for run in each], axis=0) for m in each[0]}

    @cached_property
    def scope_names(self):
        return list(self.scopes)

    @cached_property
    def group_totals(self):
        return np.stack([np.bincount(self.groups, weights=self.weights[s], minlength=self.draws.shape[1])
                         for s in self.scope_names], axis=1)

    def bootstrap(self, arm, metric, scopes):
        return np.mean([self.seed_bootstrap(file, metric, scopes) for file in self.index[arm].values()], axis=0)

    def seed_bootstrap(self, file, metric, scopes):
        cache_key = (file, metric, tuple(scopes))
        if cache_key not in self.arm_boot:
            weights = np.column_stack([self.weights[scope] for scope in scopes])
            if metric == "absolute":
                error = self.seed_metrics(file)["absolute"]
                contribution = np.stack([
                    np.bincount(self.groups, weights=error * w, minlength=self.draws.shape[1])
                    for w in weights.T], axis=1)
                values = self.draws @ contribution
            else:
                pred = np.load(self.folder / file)
                kernels = grams(pred.astype(np.float64) - self.observed, self.contrast.pools)
                values = np.zeros((len(self.draws), len(scopes)))
                for kernel, idx, center in zip(kernels, self.contrast.pools, self.contrast.centers):
                    subset_weights = weights[idx]
                    active = np.flatnonzero(subset_weights.any(axis=0))
                    if not len(active):
                        continue
                    group_draws = self.draws[:, self.groups[idx]]
                    probabilities = group_draws * center
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    k = probabilities @ kernel
                    var = np.diag(kernel) - 2 * k + (k * probabilities).sum(axis=1, keepdims=True)
                    error = np.sqrt(np.maximum(0, var)) * group_draws
                    values[:, active] += error @ subset_weights[:, active]
            self.arm_boot[cache_key] = values
        return self.arm_boot[cache_key]


def combine_panels(parts, effects, out):
    all_scopes = list(parts[0].scopes)
    arm_sets = [set(p.index) for p in parts]
    common_arms = set.intersection(*arm_sets)
    rows, detail, statistics = [], [], {}
    for arm in sorted(common_arms):
        for scope, mask in parts[0].scopes.items():
            sums = {}
            total = 0
            groups = set()
            wells = set()
            n_rows = 0
            for part in parts:
                mask = part.scopes[scope]
                weight = part.weights[scope]
                total += weight.sum()
                groups.update(part.rows.loc[mask, GROUP])
                wells.update(part.rows.loc[mask, KEY[0]])
                n_rows += int(mask.sum()) if part.role == "A" else 0
                for name, vals in part.points(arm).items():
                    sums[name] = sums.get(name, 0.0) + weight @ vals
            if total == 0:
                continue
            row = dict(arm=arm, scope=scope, rows=n_rows, groups=len(groups), wells=len(wells),
                       **{name: value / total for name, value in sums.items()})
            rows.append(row)
            statistics[(arm, scope)] = row
    for part in parts:
        for arm in sorted(part.index):
            pts = part.points(arm)
            detail.append(dict(outer=part.fold, role=part.role, arm=arm, seeds=len(part.index[arm]),
                               rows=len(part.rows), groups=part.rows[GROUP].nunique(),
                               **{metric: part.weights["all"] @ v / part.weights["all"].sum()
                                  for metric, v in pts.items()}))

    pd.DataFrame(rows).to_csv(out / "arm_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(detail).to_csv(out / "arm_metrics_by_role_fold.tsv", sep="\t", index=False)
    sims = np.stack([sum(part.group_totals[:, j] for part in parts) for j in range(len(all_scopes))], axis=1)
    denominator = parts[0].draws @ sims
    stats = []
    intervals = {}
    for label, (metric, arms) in effects.items():
        if not set(arms) <= common_arms:
            continue
        formal = label in FORMAL
        scopes = all_scopes if formal else ["all"]
        # Reuse the all-scope bootstrap calculated for each formal or matched candidate.
        index = [all_scopes.index(x) for x in scopes]
        num = sum(coef * sum(p.bootstrap(arm, metric, all_scopes) for p in parts)[:, index]
                  for arm, coef in arms.items())
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled = num / denominator[:, index]
        for j, scope in enumerate(scopes):
            if not all((arm, scope) in statistics for arm in arms):
                continue
            label_confidence = 0.99 if formal and scope == "all" else 0.95
            q = (1 - label_confidence) * 50
            lower, upper = np.percentile(sampled[:, j], [q, 100 - q])
            score = sum(coef * statistics[(arm, scope)][metric] for arm, coef in arms.items())
            control = next((statistics[(a, scope)][metric] for a, c in arms.items() if c == -1), None)
            stats.append(dict(contrast=label, scope=scope, metric=metric, confidence=label_confidence,
                              ri=parts[0].draws.shape[0], unit="compound_group", effect=score,
                              lower=lower, upper=upper, relative_percent=100 * score / control if control else None,
                              groups=statistics[(next(iter(arms)), scope)]["groups"],
                              classification="improves" if upper < 0 else "worse" if lower > 0 else "inconclusive"))
            intervals[f"{label}|{scope}"] = sampled[:, j]
        print(f"Scored {label} {metric} ({len(scopes)} scopes)", flush=True)
    pd.DataFrame(stats).to_csv(out / "paired_effects.tsv", sep="\t", index=False)
    np.savez(out / "bootstrap_effects.npz", **intervals)
    mains = pd.DataFrame(rows)
    mains[mains.arm.str.startswith(("C1", "C5_", "C6_")) | mains.arm.eq("A1")].to_csv(
        out / "context_diagnostics.tsv", sep="\t", index=False)
    joint = ["A4", "A5", "L2", "J1_FiLM_L2"]
    mains[mains.arm.isin(joint) & mains.scope.eq("all")].to_csv(out / "loss_structure_interaction.tsv",
                                                              sep="\t", index=False)


def final_score(args, cfg):
    frozen = read_json(args.freeze)
    if frozen["config_sha256"] != sha256(args.config) or Path(frozen["predictions"]) != args.predictions.resolve():
        raise ValueError("Frozen plan mismatch")
    for name, expected in frozen["predictions_receipts"].items():
        if sha256(args.predictions / name) != expected:
            raise ValueError("Inference receipt changed after freeze")
    stage = args.assets / args.stage_dir
    receipt = read_json(stage / "receipt.json")
    if frozen["staged_receipt_sha256"] != sha256(stage / "receipt.json"):
        raise ValueError("Content index for the score rows changed")
    for name, expected in receipt["files"].items():
        if "/score-" in name and sha256(stage / name) != expected:
            raise ValueError("Staged scoring array changed")
    group_ids = sorted({g for k in range(3) for g in table(args.predictions / f"outer{k}/A/rows.tsv")[GROUP]})
    draws = np.random.default_rng(cfg["evaluation"]["seed"]).exponential(
        size=(cfg["evaluation"]["replicates"], len(group_ids)))
    draws /= draws.mean(axis=1, keepdims=True)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "bootstrap_weights.npz", draws=draws, groups=np.asarray(group_ids))
    parts = []
    for k in range(3):
        truth, matrix = load_staged(stage / f"outer{k}", "score", receipt["genes"])
        for role in ("A", "B"):
            parts.append(FoldScore(args, cfg, k, role, truth, matrix, group_ids, draws))
    effects = {**FORMAL, **ATTRIBUTION}
    for label in parts[0].index:
        if label.startswith(("C5_", "C6_")):
            effects[f"{label}-A1"] = ("contrast", {label: 1, "A1": -1})
    for family in ("S1", "S2", "S3"):
        for base in ("B2", "B3", "B4", "B5", "A4"):
            name = f"{family}_{base}"
            effects[f"{name}-S_base_{base}"] = ("absolute", {name: 1, f"S_base_{base}": -1})
    combine_panels(parts, effects, args.out)
    write_json(args.out / "receipt.json", dict(
        freeze_sha256=sha256(args.freeze), bootstrap_draws=len(draws), groups=len(group_ids),
        files={p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--split-dir", default="partition")
    parser.add_argument("--stage-dir", default="staged")
    parser.add_argument("command", choices=["freeze", "score"])
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "freeze":
        freeze(args, cfg)
    else:
        final_score(args, cfg)


if __name__ == "__main__":
    main()
