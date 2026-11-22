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


def track2_reference(root, selected):
    """Verify the completed Track 2 held-scaffold forecasts and match exact structures."""
    splits = root / "splits-v1"
    score_path = root / "full-v2-score" / "receipt.json"
    split_path = splits / "receipt.json"
    score = json.loads(score_path.read_text())
    split = json.loads(split_path.read_text())
    provenance = {"score": record(score_path), "split": record(split_path), "folds": {}}
    if provenance["split"]["sha256"] != score["split_receipt"]:
        raise ValueError("The Track 2 DILI score and source splits disagree")
    rows = []
    for fold in sorted(k for k in score["forecast_freezes"] if k.startswith("murcko-")):
        directory = root / "full-v2-forecasts" / fold
        freeze = directory / "freeze.json"
        query = splits / fold / "query.parquet"
        forecast = directory / "forecast.parquet"
        manifest = json.loads(freeze.read_text())
        fingerprints = {"freeze": record(freeze), "query": record(query), "forecast": record(forecast)}
        if (fingerprints["freeze"]["sha256"] != score["forecast_freezes"][fold]
                or fingerprints["query"]["sha256"] != split["files"][f"{fold}/query.parquet"]
                or fingerprints["forecast"]["sha256"] != manifest["forecast_sha256"]
                or manifest["split_receipt"] != score["split_receipt"]):
            raise ValueError(f"Track 2 DILI model evidence changed for {fold}")
        provenance["folds"][fold] = fingerprints
        keys = pd.read_parquet(query, columns=["inchikey"])
        predictions = pd.read_parquet(forecast)
        predictions = predictions[predictions["mode"].eq("structure") & predictions.method.eq("lightgbm")]
        if not predictions.index.is_unique or set(predictions.index) != set(keys.index):
            raise ValueError(f"Track 2 DILI frozen structure support differs for {fold}")
        rows.append(keys.join(predictions[["probability", "probability_lo90", "probability_hi90", "fold"]],
                              how="inner", validate="one_to_one"))
    if len(rows) != 5:
        raise ValueError("Expected the five completed Track 2 scaffold folds")
    crosswalk = pd.concat(rows, verify_integrity=True).rename(columns={"inchikey": "molecule"})
    crosswalk = crosswalk[~crosswalk.molecule.duplicated(keep=False)]
    columns = ["anchor", "molecule", "candidate", "dili_concern"]
    table = selected[columns].merge(crosswalk.reset_index(), on="molecule", how="left",
                                     sort=False, validate="many_to_one")
    return table, provenance


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
    crosscheck, dili_provenance = None, None
    if a.track2_dili is not None:
        crosscheck, dili_provenance = track2_reference(a.track2_dili, selected)
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
    ]
    if crosscheck is not None:
        supported = int(crosscheck.probability.notna().sum())
        sections.extend([
            "## Track 2 liver reference-label comparison",
            f"{supported} of {len(selected)} hypotheses had a unique exact stereochemical match to an independent "
            "Track 2 Murcko-held structure LightGBM forecast. The interval describes calibration-fit uncertainty for "
            "the probability of an FDA Most or Less concern label for the partner molecule. The combination ranks "
            "use the observed FDA categories and their frozen efficacy forecasts.",
        ])
    sections.append("## Coverage")
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
        if crosscheck is not None:
            crosscheck.to_csv(stage / "track2_dili_reference.tsv", sep="\t", index=False, na_rep="NA")
        (stage / "RESULTS.md").write_text("\n\n".join(sections) + "\n")
        (stage / "receipt.json").write_text(json.dumps({
            "scores": record(observed / "receipt.json"), "screen": record(screen / "receipt.json"),
            "track2_dili": dili_provenance,
            "files": {p.name: record(p) for p in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + "\n")
        stage.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--track2-dili", type=Path, help="Completed Track 2 G1 directory for read-only ancillary checks")
    report(p.parse_args())


if __name__ == "__main__":
    main()
