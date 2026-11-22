"""Publish external HCC response, exploratory fractions and graph hypotheses."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .cohorts import record
from .predict import SOURCE_COHORTS, EXTERNAL_COHORTS
from .reference import checked
from .score import METRICS, markdown


def source_cross_validation(frozen, registered, registry, cohort_root):
    source = pd.read_parquet(checked(frozen, "source_loco.parquet", registered))
    tables = [pd.read_parquet(checked(cohort_root, f"sources/{name}/labels.parquet", registry))
              for name in SOURCE_COHORTS]
    labels = pd.concat(tables, ignore_index=True)
    joined = source.merge(labels, on="id", validate="many_to_one")
    if len(joined) != len(source):
        raise ValueError("Some source validation predictions have no registered endpoint")
    rows = []
    for (model, cohort), subset in joined.groupby(["model", "cohort"]):
        scored = subset[subset.score.notna()]
        if scored.outcome.nunique() != 2:
            continue
        rows.append({"model": model, "cohort": cohort, "n": len(scored),
                     "auroc": roc_auc_score(scored.outcome, scored.score),
                     "auprc": average_precision_score(scored.outcome, scored.score)})
    return pd.DataFrame(rows)


def generate(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    paths = {}
    summaries, scores, deltas, fractions = [], [], [], []
    registry_path = a.cohorts / "receipt.json"
    registry = json.loads(registry_path.read_text())
    for name in EXTERNAL_COHORTS:
        packets = a.results / "predictions" / name
        observed = a.results / "scores" / name
        pre, score = (json.loads((p / "receipt.json").read_text()) for p in (packets, observed))
        if (score["predictions"] != record(packets / "receipt.json") or
                pre["inputs"]["registry"] != record(registry_path)):
            raise ValueError(f"Combined report receipts disagree for {name}")
        spec = pre["targets"][name]
        for file in spec["files"]:
            checked(packets / name, file, spec)
        paths[name] = {"predictions": record(packets / "receipt.json"),
                       "scores": record(observed / "receipt.json"), "quality": score["cohorts"][name]}
        fractions.append(pd.read_csv(checked(observed, "fraction_summary.tsv", {"files": score["outputs"]}), sep="\t"))
        if score["cohorts"][name]["scored_response_patients"]:
            source = source_cross_validation(packets / name, spec, registry, a.cohorts)
            summaries.append(source.assign(target_projection=name))
            for filename, table in (("external_metrics.tsv", scores), ("paired_deltas.tsv", deltas)):
                table.append(pd.read_csv(checked(observed, filename, {"files": score["outputs"]}), sep="\t"))
        if name == "GSE202069":
            effects = pd.read_csv(checked(observed, "counterfactuals.tsv", {"files": score["outputs"]}), sep="\t")
    performance, comparisons = pd.concat(scores, ignore_index=True), pd.concat(deltas, ignore_index=True)
    source_validation = pd.concat(summaries, ignore_index=True)
    tcga_path = a.results / "tcga-exploratory"
    tcga_receipt = json.loads((tcga_path / "receipt.json").read_text())
    if record(tcga_path / "expression_receipt.json") != tcga_receipt["frozen_expression_scores"]:
        raise ValueError("TCGA exploratory expression receipt changed")
    checked(tcga_path, "expression_scores.parquet", tcga_receipt)
    exploratory = pd.read_csv(checked(tcga_path, "immune_subtypes.tsv", tcga_receipt), sep="\t")
    paths["TCGA-LIHC"] = {"scores": record(tcga_path / "receipt.json"),
                         "matched_subtypes": tcga_receipt["n_published_subtypes"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        for filename, table in (("external_metrics", performance), ("paired_deltas", comparisons),
                                ("counterfactuals", effects), ("fraction_summary", pd.concat(fractions)),
                                ("non_hcc_loco", source_validation), ("tcga_immune_subtypes", exploratory)):
            table.to_csv(stage / f"{filename}.tsv", sep="\t", index=False)
        sections = ["# Track 7 — HCC anti-PD-1 external benchmark"]
        for cohort in ("GSE202069", "GSE140901"):
            d = performance.loc[performance.cohort.eq(cohort)].copy()
            for metric in METRICS:
                d[metric] = [f"{v:.3f} ({lo:.3f}–{hi:.3f})" for v, lo, hi in
                             d[[metric, metric + "_low", metric + "_high"]].to_numpy()]
            sections += [f"## {cohort}", markdown(d, ["model", "participants", "responders", *METRICS])]
        primary = comparisons[comparisons.cohort.eq("GSE202069")]
        sections.extend([
            "## Paired primary-cohort differences",
            "Differences compare the TME model to each baseline and ablation on the same held patients. "
            "The column bounds are percentile bootstrap intervals over patients.",
            markdown(primary, ["comparison", "auroc", "auroc_low", "auroc_high", "auprc", "auprc_low", "auprc_high"]),
            "## Simulated TME attenuation",
            "The table ranks hypothetical graph interventions by mean probability change among measured "
            "nonresponders. The score uses frozen predictions, and the labels define the subgroup at scoring. "
            "Changes quantify graph sensitivity. Finn et al. tested atezolizumab plus bevacizumab in a distinct "
            "randomized population. [IMbrave150](https://doi.org/10.1056/NEJMoa1915745)",
        ])
        ranked = effects[effects.group.eq("observed_nonresponders") & np.isclose(effects.attenuation, .90)]
        ranked = ranked.sort_values("delta_probability", ascending=False).copy()
        for source, label in (("delta_probability", "change in percentage points"),
                              ("delta_low", "95% lower bound"), ("delta_high", "95% upper bound")):
            ranked[label] = ranked[source].map(lambda value: f"{100 * value:+.5f}")
        sections.append(markdown(ranked, ["axis", "participants", "change in percentage points",
                                          "95% lower bound", "95% upper bound"]))
        count = paths["GSE141198"]["quality"]["exploratory_profiles"]
        sections.extend(["## Exploratory HCC expression", f"GSE141198 supplies {count} RNA profiles. "
                         "The pooled fraction quantiles appear in the companion table. A matched objective "
                         "anti-PD-1 response endpoint is unavailable for this cohort."])
        brief = exploratory[exploratory.feature.isin(["cd8", "malignant", "ifng6"])]
        sections.extend([
            "## TCGA-LIHC exploratory immune subtypes",
            f"Primary-tumor STAR expression was verified for {tcga_receipt['n_primary_tumors']} donors. "
            f"The author subtype manifest matched {tcga_receipt['n_published_subtypes']} by exact sample barcode. "
            "These distributions do not measure ICI response. [PanImmune study](https://doi.org/10.1016/j.immuni.2018.03.023)",
            markdown(brief, ["subtype", "patients", "feature", "median", "q25", "q75"]),
        ])
        (stage / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "targets": paths, "cohorts": record(registry_path),
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "cohorts", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    generate(p.parse_args())


if __name__ == "__main__":
    main()
