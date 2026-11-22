"""Publish verified donor-held transfer, paired contrasts and QC diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .cohorts import file_record
from .singlecell_inputs import files_receipt
from .source_panels import frozen


def markdown(data, columns):
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in data[columns].itertuples(index=False, name=None):
        result.append("| " + " | ".join("—" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float)
                                        else str(v) for v in row) + " |")
    return "\n".join(result)


def flatten(rows):
    values = []
    for row in rows:
        values.append({**{k: v for k, v in row.items() if k != "ci95"},
                       "ci_low": row["ci95"][0], "ci_high": row["ci95"][1]})
    return pd.DataFrame(values)


def report(a):
    result = a.result
    evidence, receipts = {}, {}
    for name in ("transfer-predictions", "transfer-scores", "figures"):
        path = result / name / "receipt.json"
        evidence[name], receipts[name] = json.loads(path.read_text()), file_record(path)
    scores, figs, pred = (evidence[k] for k in ("transfer-scores", "figures", "transfer-predictions"))
    if not (scores["predictions"] == figs["predictions"] == receipts["transfer-predictions"]
            and figs["scoring"] == receipts["transfer-scores"]):
        raise ValueError("Figures, scores and frozen forecast receipts disagree")
    for name in pred["files"]:
        frozen(result / "transfer-predictions", name, pred)
    file = result / "transfer-scores"
    reported = json.loads(frozen(file, "metrics.json", scores).read_text())
    transfer = pd.read_csv(frozen(file, "transfer.tsv", scores), sep="\t")
    rows = transfer[transfer.within_drug_spearman.notna()]
    summary = rows.groupby(["query", "depth", "model"], dropna=False, as_index=False).agg(
        lineage_min=("lineages", "min"), drugs_scored=("drug", "nunique"),
        within_drug_spearman=("within_drug_spearman", "mean"),
    )
    organ = flatten(reported["organoid_donor_summary"])
    observational = flatten(reported["observational_paired_contrasts"])
    diagnostics = flatten(reported["partial_correlation_summary"])
    primary = figs["model"]
    selected = diagnostics[diagnostics.model.eq(primary)]
    comparative = diagnostics[diagnostics.kind.eq("technical")].copy()
    comparative["magnitude"] = comparative["mean"].abs()
    worst = comparative.sort_values("magnitude", ascending=False, kind="stable").drop_duplicates(["model", "cohort"])
    fields = ["kind", "model", "cohort", "variable", "donors", "mean", "ci_low", "ci_high", "sign_flip_p"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        stage = Path(d)
        for key in ("transfer.tsv", "organoid_pairs.tsv", "donor_partial_correlations.tsv", "sample_fractions.tsv"):
            shutil.copyfile(frozen(file, key, scores), stage / key)
        for key in ("malignant_resistance_umap.png", "displayed_transcriptional_states.tsv"):
            shutil.copyfile(frozen(result / "figures", key, figs), stage / key)
        for name, content in (("transfer_macro", summary), ("paired_progression", observational),
                              ("organoid_summary", organ), ("partial_correlation_summary", diagnostics)):
            content.to_csv(stage / f"{name}.tsv", sep="\t", index=False)
        sections = [
            "# Track 5 — patient donor-held resistance forecasts",
            f"After donor-blocked model fitting, the independent scorer examined "
            f"{reported['held_out_candidate_cells']:,} malignant candidate cells from "
            f"{reported['held_out_candidate_donors']} donors. The plot model was chosen "
            "on source validation by its mean within-drug Spearman across folds.",
            "## Independent held cell-line pharmacology",
            "The table averages lineage-level within-drug Spearman over measured drugs. "
            "RNA subsampling tests preservation of a measured cell-line pharmacology signal. "
            "The exact Tahoe lineage is available in the full table as a single measured expression anchor.",
            markdown(summary, ["query", "depth", "model", "lineage_min", "drugs_scored", "within_drug_spearman"]),
            "## Paired sorafenib organoids",
            "Each donor and class first averages technical replicates. The difference is the predicted "
            "resistance of organoids annotated resistant minus sensitive. The interval and randomization "
            "test use the four held-out donor pairs.",
            markdown(organ, ["model", "donors", "concordant_donors", "mean", "ci_low", "ci_high", "sign_flip_p"]),
            "## Site and time contrasts",
            "Differences are within-patient changes in the fraction above a source-calibrated "
            "high-resistance threshold. The NCI time series follows ICI exposure. Sorafenib "
            "response is unmeasured in the patient biopsies.",
            markdown(observational, ["contrast", "model", "donors", "mean", "ci_low", "ci_high", "sign_flip_p"]),
            f"## Technical and biological diagnostics for {primary}",
            markdown(selected, fields),
            "## Largest absolute technical partial correlation by model and cohort",
            markdown(worst, fields),
            "## Malignant cell UMAP and resistance fraction",
            "The display uses the measured scGPT coordinates in one densely observed donor per cohort. "
            "Transcriptional states come from expression-space clustering. No genomic clonality assay "
            "was performed on these states.",
            "![Malignant cell UMAP and predicted resistance distributions](malignant_resistance_umap.png)",
        ]
        (stage / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "inputs": receipts, "displayed_model": primary, "files": files_receipt(stage),
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    report(p.parse_args())


if __name__ == "__main__":
    main()
