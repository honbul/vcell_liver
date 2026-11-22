"""Score one sealed TKI forecast against the published concentration endpoints."""

import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .evaluate import join_observations, paired_gain, score_across_cells
from .external import KEY
from .io import publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .tki_gr50 import ARTICLE, COMPONENTS, PANEL, registration
from .v0plus_score import crossed_patient_drug_ci
from .yang_hcc_external import require_hash

PRIMARY = "growth_adjusted"
CONTROLS = ("structured", "growth_shuffled", "drug_ridge")
COMPARISONS = tuple((PRIMARY, method) for method in CONTROLS)


def extract_table(article, values, query):
    document = subprocess.check_output(["pdftotext", "-layout", str(article), "-"], text=True)
    caption = "Table 2 GR Values for Potency and Efficacy"
    if document.count(caption) != 1:
        raise ValueError("Expected one published Table 2")
    section = document.split(caption)[1].split("Notes: This table", 1)[0]
    header = next(line.split() for line in section.splitlines() if "RoS" in line)
    if header != [*PANEL, "RoS"]:
        raise ValueError("Unexpected published drug axis")
    adult = set(query.published_cell)
    entries = []
    for words in map(str.split, section.splitlines()):
        if not words or words[0] not in adult:
            continue
        if len(words) != 14 or not set(words[3:13:3]) <= {"0", "1"}:
            raise ValueError("Unexpected published response table layout")
        for i, drug in enumerate(PANEL):
            a, b = words[1 + 3*i:3 + 3*i]
            entries.append((words[0], drug, a, b.replace("−", "-")))
    fields = ("published_cell", "source_compound_label", "gr50_uM", "grmax")
    canonical = pd.DataFrame(entries, columns=fields).to_csv(sep="\t", index=False, lineterminator="\n")
    if len(entries) != len(query):
        raise ValueError("Published response table has incomplete adult support")
    if values.exists():
        if values.read_text() != canonical:
            raise ValueError("Existing extraction differs from verified article")
    else:
        with values.open("x") as stream:
            stream.write(canonical)


def labels(query, values):
    info = pd.read_csv(values, sep="\t", dtype=str, keep_default_na=False)
    identity = ["published_cell", "source_compound_label"]
    expected = {"gr50_uM", "grmax", *identity}
    if set(info) != expected or len(info) != len(query):
        raise ValueError("Expected exactly the adult portion of published Table 2")
    table = query[[*KEY, *identity, "low_uM", "high_uM"]].merge(info, on=identity, validate="one_to_one")
    if len(table) != len(query):
        raise ValueError("Missing published response identity")
    mark = table.gr50_uM.str.replace(r"\s+", "", regex=True)
    number = pd.to_numeric(mark.str.strip("><≤≥"), errors="raise").to_numpy(dtype=float)
    low, high = table.low_uM.to_numpy(), table.high_uM.to_numpy()
    upper = mark.str.startswith((">", "≥")).to_numpy() | (number > high)
    lower = mark.str.startswith(("<", "≤")).to_numpy() | (number < low)
    if (upper & lower).any() or not (np.isfinite(number) & (number > 0)).all():
        raise ValueError("Invalid concentration bounds")
    clipped = np.where(upper, high, np.where(lower, low, number))
    grmax = pd.to_numeric(table.grmax, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(grmax).all():
        raise ValueError("Expected finite Table 2 GRmax")
    responses = {"GR50": table[KEY].assign(endpoint_value=-np.log10(clipped)),
                 "GRmax": table[KEY].assign(endpoint_value=-grmax)}
    counts = (table.assign(left_censored=lower, right_censored=upper, at_upper_bound=clipped == high)
              .groupby("source_compound_label").agg(
                  measured=("grmax", "size"), left_censored=("left_censored", "sum"),
                  right_censored=("right_censored", "sum"), at_upper_bound=("at_upper_bound", "sum"),
              ).to_dict("index"))
    return responses, counts


def limore_labels(query, path, digest):
    require_hash(path, digest)
    measurements = pd.read_parquet(path, columns=["cell_id", "compound_id", "endpoint_value"])
    matched = query[KEY].merge(measurements, on=["cell_id", "compound_id"], validate="one_to_one")
    if len(matched) != len(query) or not np.isfinite(matched.endpoint_value).all():
        raise ValueError("LIMORE must have all four matched TKI values")
    return matched


def aggregate(forecast, observations, track):
    data = join_observations(forecast, observations)
    grouped = data.groupby(["lineage_group", "compound_id", "method"], as_index=False)[
        ["endpoint_value", "predicted_value"]].mean()
    anchors = data[data.method.eq("drug_mean")].groupby("compound_id").predicted_value
    if not anchors.nunique().eq(1).all():
        raise ValueError("Drug mean has accidental cell dependence")
    is_anchor = grouped.method.eq("drug_mean")
    if is_anchor.any():
        grouped.loc[is_anchor, "predicted_value"] = grouped.loc[is_anchor, "compound_id"].map(anchors.first())
    return grouped.assign(cell_id=grouped.lineage_group, track=track, fold=0,
                          row_id=grouped.lineage_group + "|" + grouped.compound_id)


def ranking_by_donor(data):
    out = []
    for (cell, method), chunk in data.groupby(["lineage_group", "method"]):
        target, prediction = chunk.endpoint_value.to_numpy(), chunk.predicted_value.to_numpy()
        if len(target) != 4:
            raise ValueError("Per-donor ranking requires the complete panel")
        truth, model = rankdata(target), rankdata(prediction)
        correlation = (0. if np.ptp(model) == 0 else float(np.corrcoef(truth, model)[0, 1])) if np.ptp(truth) else None
        top_target, top_prediction = target == target.max(), prediction == prediction.max()
        top_1 = float((top_target & top_prediction).sum() / (top_target.sum() * top_prediction.sum()))
        out.append({"cell_id": cell, "lineage_group": cell, "track": chunk.track.iloc[0], "fold": 0,
                    "method": method, "spearman": correlation, "top_1": top_1})
    return pd.DataFrame(out)


def clean_records(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def describe(data, *, ranked):
    per_drug, _ = score_across_cells(data)
    macro = (per_drug.groupby("method")[["spearman", "pearson"]].mean().reset_index())
    info = {"within_drug": clean_records(macro), "by_drug": clean_records(per_drug)}
    if ranked:
        donor = ranking_by_donor(data)
        info["within_donor"] = clean_records(donor.groupby("method")[["spearman", "top_1"]].mean().reset_index())
        info["paired_drug_ci95"] = {base: crossed_patient_drug_ci(data, PRIMARY, base) for base in CONTROLS
                                    if PRIMARY in set(data.method)}
        ranking = {}
        if PRIMARY in set(data.method):
            for candidate, base in COMPARISONS:
                ranking[base] = {metric: paired_gain(donor, candidate, base, metric) for metric in ("spearman", "top_1")}
        info["paired_donor_ci95"] = ranking
    return info


def score(args):
    doc, expected = registration(args.registration)
    rows, frozen = verify_forecast(args.forecast)
    proof = frozen["provenance"]
    if (proof["registration_sha256"], proof["article_sha256"], set(proof["component_sha256"])) != (
            expected, ARTICLE, set(COMPONENTS)):
        raise ValueError("Unexpected joint GR50 freeze")
    query = pd.read_parquet(args.registration / "query.parquet")
    shared_support(rows, query, [item for values in COMPONENTS.values() for item in values])
    require_hash(args.article, ARTICLE)
    extract_table(args.article, args.values, query)
    outcomes, bounds = labels(query, args.values)
    outcomes["LIMORE_AA"] = limore_labels(query, args.limore / "rows.parquet", doc["limore_response_sha256"])
    donor = {endpoint: aggregate(rows, gold, endpoint) for endpoint, gold in outcomes.items()}
    summaries = {name: describe(data, ranked=(name != "GRmax")) for name, data in donor.items()}
    aa = outcomes["LIMORE_AA"]
    cross = aa[KEY].assign(predicted_value=aa.endpoint_value, method="observed_LIMORE_AA")
    agreement = {
        endpoint: describe(aggregate(cross, outcomes[endpoint], endpoint), ranked=(endpoint == "GR50"))
        for endpoint in ("GR50", "GRmax")
    }
    primary = summaries["GR50"]
    gate = {}
    for base in CONTROLS:
        intervals = primary["paired_drug_ci95"][base]
        rank = primary["paired_donor_ci95"][base]["spearman"]
        gate[base] = bool(all(intervals.get(metric, [-1])[0] > 0 for metric in ("spearman", "pearson"))
                          and rank is not None and rank["ci95"][0] > -.005)
    publish_json(args.out, {
        "cohort": doc["cohort"], "registration_sha256": expected, "article_sha256": ARTICLE,
        "frozen_sha256": frozen["sha256"], "table2_extraction_sha256": sha256(args.values),
        "limore_response_sha256": doc["limore_response_sha256"], "adult_lineages": doc["adult_lineages"],
        "censoring": bounds, "summaries": summaries, "observed_endpoint_agreement": agreement,
        "gate": gate, "growth_supported": all(gate.values()),
        "drugs": query.drop_duplicates("compound_id").set_index("compound_id").source_compound_label.to_dict(),
    })
