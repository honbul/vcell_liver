"""Score checksum-verified predictions on held combination assays."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from .context import verified
from .splits import OUTCOMES, record


SYNERGY_CUTOFF = 10.
EFFICACY_GAIN_CUTOFF = 10.
FOLDS = {"lco_hcc": 1, "lpo": 5, "ldo": 5}


def metrics(g, outcome):
    y, p = g[outcome].to_numpy(), g["prediction"].to_numpy()
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 3:
        return None
    rho = (float(spearmanr(y, p).statistic) if np.ptp(p) and np.ptp(y) else np.nan)
    return {"n": len(y), "spearman": rho,
            "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def cell_performance(rows, endpoint):
    result = []
    for name, cell in rows.groupby("cell_group"):
        fit = metrics(cell, endpoint)
        if fit is not None:
            result.append({"cell_group": name, **fit})
    return pd.DataFrame(result)


def macro_interval(cell, rng, draws):
    if cell.empty:
        return {}
    n = len(cell)
    result = {"lineages": n}
    for name in ("spearman", "mae", "rmse"):
        val = cell[name].dropna().to_numpy()
        if not len(val):
            continue
        projected = rng.choice(val, size=(draws, len(val)), replace=True).mean(axis=1)
        low, high = np.quantile(projected, [.025, .975])
        result.update({f"cell_macro_{name}": val.mean(), f"cell_macro_{name}_low": low, f"cell_macro_{name}_high": high})
    return result


def profiles(frozen):
    for source_group, group in frozen.groupby("source_family"):
        for column in OUTCOMES:
            if column == "added_inhibition" and source_group == "DrugComb":
                continue
            frame = group[["cell_group", column + "_true", column + "_pred"]]
            yield source_group, column, frame.rename(columns={column + "_true": column, column + "_pred": "prediction"})


def joint_score(group):
    subset = group[["bliss_true", "added_inhibition_true", "bliss_pred", "added_inhibition_pred"]].dropna()
    if not len(subset):
        return None
    positive = ((subset.bliss_true >= SYNERGY_CUTOFF)
                & (subset.added_inhibition_true >= EFFICACY_GAIN_CUTOFF)).to_numpy()
    if not positive.any():
        return None
    severity = np.minimum(subset.bliss_pred - SYNERGY_CUTOFF,
                          subset.added_inhibition_pred - EFFICACY_GAIN_CUTOFF)
    rank = severity.sort_values(ascending=False, kind="stable").index[:20]
    aligned = pd.Series(positive, index=subset.index)
    return {"paired_measurements": len(subset), "joint_positives": int(positive.sum()),
            "prevalence": positive.mean(), "auprc": average_precision_score(positive, severity),
            "precision_top20": aligned.loc[rank].mean()}


def panel_scores(panel, outputs, axis):
    contract = json.loads((panel / "receipt.json").read_text())
    registered = json.loads((outputs / "receipt.json").read_text())
    if contract["axis"] != axis or registered["panel"] != record(panel / "receipt.json"):
        raise ValueError(f"Unexpected frozen panel identity {panel}")
    for path in registered["files"]:
        verified(outputs, path)
    keys = pd.read_parquet(verified(panel, "query.parquet"))
    truth = pd.read_parquet(verified(panel, "labels.parquet"))
    forecasts = pd.read_parquet(verified(outputs, "predictions.parquet"))
    candidates = set(registered["model_selection"])
    if set(forecasts.model) != candidates | {"mean", "pair_mean", "target_dependency_sum", "mono_bliss_independent"}:
        raise ValueError(f"Unexpected frozen comparison models: {panel}")
    if forecasts.duplicated(["id", "model"]).any() or not truth.id.is_unique or set(truth.id) != set(keys.id):
        raise ValueError(f"Repeated or unregistered assay scores: {panel}")
    supports = {name: set(g.id) for name, g in forecasts.groupby("model", sort=False)}
    n = registered["role_counts"]["query"]
    if not supports:
        raise ValueError(f"Missing frozen predictions: {panel}")
    selected = next(iter(supports.values()))
    if len(selected) != n or any(other != selected for other in supports.values()) or (selected - set(keys.id)):
        raise ValueError(f"Inconsistent shared forecast support across model families: {panel}")
    combo = keys.merge(truth, on="id", validate="one_to_one").merge(
        forecasts, on="id", validate="one_to_many", suffixes=("_true", "_pred")
    )
    combo["source_family"] = combo.source.str.split(":").str[0]
    combo["fold"] = panel.name
    return combo, {
        "panel": record(panel / "receipt.json"), "frozen": record(outputs / "receipt.json"),
        "labels": record(panel / "labels.parquet"), "scored": n, "registered": len(keys),
    }


def markdown(data, fields):
    result = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in data[fields].itertuples(index=False, name=None):
        cells = [("—" if pd.isna(v) else f"{v:.4f}") if isinstance(v, (float, np.floating)) else str(v) for v in row]
        result.append("| " + " | ".join(cells) + " |")
    return "\n".join(result)


def run(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    rng = np.random.default_rng(a.seed)
    scores, per_line, joint_rows, ledger = [], [], [], {}
    for axis in a.axes:
        pieces = []
        for k in range(FOLDS[axis]):
            name = f"{axis}-{k}"
            table, audit = panel_scores(a.panels / name, a.predictions / name, axis)
            ledger[name] = audit
            pieces.append(table)
        collected = pd.concat(pieces, ignore_index=True)
        if collected.duplicated(["id", "model"]).any():
            raise ValueError(f"Repeated HCC assay across outer folds of {axis}")
        for model, group in collected.groupby("model"):
            for source, endpoint, merged in profiles(group):
                result = metrics(merged, endpoint)
                if result is None:
                    continue
                frame = cell_performance(merged, endpoint)
                if not frame.empty:
                    per_line.append(frame.assign(axis=axis, model=model, endpoint=endpoint, source=source))
                scores.append({"axis": axis, "model": model, "source": source, "endpoint": endpoint,
                               **result, **macro_interval(frame, rng, a.bootstrap)})
            joint = joint_score(group[group.source_family.eq("Sanger")])
            if joint:
                joint_rows.append({"axis": axis, "model": model, **joint})
    metrics_table = pd.DataFrame(scores)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as tmp:
        stage = Path(tmp)
        metrics_table.to_csv(stage / "external_metrics.tsv", sep="\t", index=False)
        pd.concat(per_line, ignore_index=True).to_csv(stage / "per_lineage.tsv", sep="\t", index=False)
        joint = pd.DataFrame(joint_rows)
        joint.to_csv(stage / "joint_efficacy_synergy.tsv", sep="\t", index=False)
        sections = [
            "# Track 8 — held adult HCC drug-combination assays",
            "Every model was selected using its inner blocked validation. Frozen predictions were "
            "verified against held assay IDs and outcome hashes in a separate command.",
            "The Sanger scorecard includes measured Bliss, independently reconstructed ZIP and "
            "maximum growth inhibition gained over the strongest measured single agent. CSS is "
            "an efficacy endpoint of DrugComb. Where a protocol has no held HCC assay, the score is missing.",
        ]
        columns = ["source", "endpoint", "model", "n", "lineages", "spearman", "rmse",
                   "cell_macro_spearman", "cell_macro_spearman_low", "cell_macro_spearman_high"]
        for axis, group in metrics_table.groupby("axis", sort=False):
            sections.extend([f"## {axis}", markdown(group.reindex(columns=columns), columns)])
        if len(joint):
            sections.extend([
                "## Joint efficacy and synergy",
                "Positives required measured Bliss and added maximum inhibition both at least 10 percentage points.",
                markdown(joint, ["axis", "model", "joint_positives", "paired_measurements", "auprc", "precision_top20"]),
            ])
        (stage / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "held_panels": ledger, "bootstrap": a.bootstrap, "seed": a.seed,
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("panels", "predictions", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--axes", nargs="+", choices=tuple(FOLDS), default=tuple(FOLDS))
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=4192)
    a = p.parse_args()
    if a.bootstrap < 100:
        p.error("At least 100 lineage bootstrap draws are required")
    run(a)


if __name__ == "__main__":
    main()
