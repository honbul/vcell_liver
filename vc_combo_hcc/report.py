"""Join independently scored HCC assays and the frozen virtual screen."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .context import verified
from .score import markdown
from .splits import record


def report(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    observed = a.results / "scores" / "all"
    screen = a.results / "screen"
    forecast = a.results / "predictions" / "lco_hcc-0" / "receipt.json"
    score_receipt = json.loads((observed / "receipt.json").read_text())
    screening_receipt = json.loads((screen / "receipt.json").read_text())
    if (screening_receipt["predictions"] != record(forecast)
            or screening_receipt["predictions"] != score_receipt["held_panels"]["lco_hcc-0"]["frozen"]):
        raise ValueError("Screen and benchmark did not share the frozen liver-cold predictor")
    original = verified(observed, "RESULTS.md")
    candidate_path = verified(screen, "top20.tsv")
    selected = pd.read_csv(candidate_path, sep="\t")
    data = pd.read_csv(verified(observed, "external_metrics.tsv"), sep="\t")
    if len(selected) != screening_receipt["top_rows"] or len(selected) > 20:
        raise ValueError("Virtual hypothesis table does not match the frozen ranking")
    columns = ["anchor", "candidate", "dili_concern", "organoid_donors", "bliss", "added_inhibition",
               "added_inhibition_donor_low", "added_inhibition_donor_high", "added_inhibition_limore"]
    sections = [
        original.read_text().rstrip(),
        "## Virtual resistance-reversal hypotheses",
        f"The source-validation selector chose {screening_receipt['model']}. "
        f"Of {screening_receipt['tested_candidates']} drug pairs screened in four paired organoid donors, "
        f"{screening_receipt['n_eligible']} met both positive prediction thresholds and "
        f"{len(selected)} are listed. Only independent FDA No or Less concern partners enter the roster. "
        "The 95% donor interval describes variation of HCC forecasts. Full rank files also include "
        "the absolute source-validation error radius and the paired resistant–sensitive contrast. "
        "GSE182593 has no measured combination intervention for these pairs.",
        markdown(selected, columns) if len(selected) else "No pair met the prespecified forecast criteria.",
        "## Coverage",
    ]
    for label, group in data.groupby("endpoint"):
        parts = ", ".join(sorted(group.source.unique()))
        sections.append(f"The held HCC {label} endpoint has scoreable observations from {parts}.")
    css_sources = set(data.loc[data.endpoint.eq("css"), "source"])
    if "Sanger" in css_sources:
        sections.append("The Sanger CSS rows use a reconstruction from observed dose matrices.")
    if "DrugComb" not in css_sources:
        sections.append("The DrugComb native CSS subset has zero eligible adult HCC labels after oncology filtering.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as handle:
        stage = Path(handle)
        for filename in ("external_metrics.tsv", "per_lineage.tsv", "paired_lineage_gains.tsv", "joint_efficacy_synergy.tsv"):
            shutil.copyfile(verified(observed, filename), stage / filename)
        shutil.copyfile(candidate_path, stage / "top20.tsv")
        (stage / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "scores": record(observed / "receipt.json"), "screen": record(screen / "receipt.json"),
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    report(p.parse_args())


if __name__ == "__main__":
    main()
