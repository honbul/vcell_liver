"""Recover the intact NODE RNA prefix and score previously unsupported LICOB donors."""

import argparse
import json
import mmap
from pathlib import Path
import re
import struct
from zipfile import ZIP_DEFLATED, ZipFile
from zlib import decompressobj

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.reader.strings import read_string_table
from scipy.stats import rankdata

from . import measured_drivers
from .evaluate import join_observations
from .expanded_panel_cv import RECEIPTS
from .external import KEY, _validate
from .function_data import new_output
from .hcc_3d_study_transfer import COHORT_HASHES
from .io import freeze_table, publish_json, sha256, verify_forecast
from .limore_corrected import shared_support
from .measured_driver_transfer import REFERENCES, expected_components, fit_reference, structured_models
from .v0plus_cv import molecular_inputs, training_inputs
from .v0plus_licob import (ASSET_HASHES, adult_donor_outcomes, checked_registration, paired_axis,
                          sample_metadata)
from .v0plus_licob_audit import LICOB_RNA
from .v0plus_score import axis_metrics, compound_strata, crossed_patient_drug_ci
from .yang_hcc_external import SETTINGS, require_hash

TRACK = "LICOB_NODE_PREFIX"
GENOMICS_HASH = "b98c98faef9dff243bcabc4ff10b7781340d00d8f1c9631df3a048690e14281e"
HOLE_START = 41943040
PRIMARY = "structured"
MODELS = {*REFERENCES, PRIMARY, "target_expression", "measured_coding", "genotype_permutation", "ridge_structured_average"}
CONTROLS = ("drug_mean", "drug_ridge", "cellhit")


def recover(path):
    """Decode only present compressed bytes and require contiguous, complete XML rows."""
    with ZipFile(path) as archive, path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as source:
        info = archive.getinfo(LICOB_RNA)
        start = info.header_offset
        if source[start:start + 4] != b"PK\x03\x04" or info.compress_type != ZIP_DEFLATED:
            raise ValueError("Unexpected local header for the raw DEFLATE stream")
        name_size, extra_size = struct.unpack_from("<HH", source, start + 26)
        start += 30 + name_size + extra_size
        hole = re.compile(rb"\0{1048576,}").search(source, start)
        if hole is None or hole.start() != HOLE_START or HOLE_START >= start + info.compress_size:
            raise ValueError("NODE zero range differs from the verified public file")
        decoder = decompressobj(-15)
        prefix = decoder.decompress(source[start:HOLE_START])
        if decoder.eof:
            raise ValueError("The intact prefix must end within the registered RNA worksheet")
        with archive.open("xl/sharedStrings.xml") as handle:
            strings = read_string_table(handle)
    xml_rows = list(re.finditer(rb'<row r="(\d+)"[^>]*>(.*?)</row>', prefix))
    if len(xml_rows) != 14801 or any(int(row[1]) != n for n, row in enumerate(xml_rows, 1)):
        raise ValueError("The complete RNA prefix rows differ from audit")
    read_cells = re.compile(rb'<c r="([A-Z]+)[0-9]+"([^>]*)>\s*<v>([^<]+)</v>\s*</c>')
    header = read_cells.findall(xml_rows[1][2])
    names = [strings[int(value)] for _, attributes, value in header[1:] if b't="s"' in attributes]
    columns = [key for key, _, _ in header]
    if len(names) != 65 or len(set(names)) != len(names) or len(columns) != 66:
        raise ValueError("Unexpected NODE RNA column identities")
    table = {}
    for row in xml_rows[2:]:
        entries = read_cells.findall(row[2])
        if [key for key, _, _ in entries] != columns or b't="s"' not in entries[0][1]:
            raise ValueError("Complete NODE RNA rows must match the column axes")
        gene = strings[int(entries[0][2])]
        if gene in table:
            raise ValueError("Ambiguous NODE RNA gene column")
        table[gene] = [float(value) for _, _, value in entries[1:]]
    profile = pd.DataFrame.from_dict(table, orient="index", columns=names)
    if not np.isfinite(profile.to_numpy()).all():
        raise ValueError("Recovered expression contains nonfinite values")
    return profile, {"decompressed_bytes_before_hole": len(prefix), "hole_start": HOLE_START,
                     "whole_member_crc_verified": False, "complete_genes": len(profile), "profile_columns": len(names)}


def register(args):
    for name, asset in (("node_archive", args.node), ("rna", args.author), ("supplement", args.supplement)):
        require_hash(asset, ASSET_HASHES[name])
    require_hash(args.licob / "receipt.json", COHORT_HASHES["LICOB"])
    checked_registration(args.licob)
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    train, source, _ = training_inputs(args.core)
    profile, recovery = recover(args.node)
    outside = pd.read_csv(args.author, sep="\t", index_col=0)
    genes = sorted(set(source.columns) & set(profile.index))
    intersection = sorted(set(genes) & set(outside.index))
    shared = sorted(set(profile.columns) & set(outside.columns))
    a, b = [table.loc[intersection, shared] for table in (profile, outside)]
    correlations = [np.corrcoef(rankdata(x), rankdata(y))[0, 1] for x, y in zip(a.to_numpy().T, b.to_numpy().T)]
    lower, median = [float(value) for value in np.quantile(correlations, [.05, .5])]
    if (len(genes), len(intersection), len(shared)) != (7123, 7123, 50) or lower < .90 or median < .95:
        raise ValueError("NODE recovery fails the RNA-only identity benchmarks")
    with args.supplement.open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True)
        _, positions = paired_axis(workbook)
        demographics = sample_metadata(workbook)
        workbook.close()
    prior = pd.read_parquet(args.licob / "query.parquet", columns=["cell_id"])
    donors = set(demographics.loc[sorted(set(prior.cell_id)), "donor_id"])
    present = sorted(set(profile.columns) & set(positions) & set(demographics.index))
    patients = demographics.loc[present, ["adult_age_confirmed", "donor_id"]].copy().rename_axis("cell_id")
    patients = patients[patients.adult_age_confirmed].assign(new_donor=lambda d: ~d.donor_id.isin(donors))
    novel = patients[patients.new_donor]
    if (len(patients), patients.donor_id.nunique(), len(novel), novel.donor_id.nunique()) != (43, 38, 9, 9):
        raise ValueError("The recovered donor-disjoint support differs from the registered cohort")
    chemistry = pd.read_parquet(args.licob / "chemistry.parquet").dropna(subset=["isomeric_smiles"])
    query = patients.reset_index().merge(chemistry, how="cross").assign(
        lineage_group=lambda q: q.cell_id, response_row=lambda q: q.cell_id.map(positions),
        row_id=lambda q: TRACK + "|" + q.cell_id + "|" + q.compound_id,
    )
    _validate(train, query)
    coding, _, = measured_drivers.licob_coding(args.node, patients.index.tolist())
    if query.compound_id.nunique() != 53 or len(query) != 2279 or coding.isna().any(axis=None):
        raise ValueError("Missing registered chemistry or coding support in the recovered LICOB panel")
    with new_output(args.out) as temp:
        files = {"query.parquet": query, "rna.parquet": profile.loc[genes, patients.index].T.rename_axis("cell_id"),
                 "coding.parquet": coding}
        hashes = {}
        for name, table in files.items():
            destination = temp / name
            table.to_parquet(destination, index=name != "query.parquet")
            hashes[name] = sha256(destination)
        publish_json(temp / "receipt.json", {"study": TRACK, "artifacts": hashes, "genes": len(genes),
            "node_sha256": ASSET_HASHES["node_archive"], "author_sha256": ASSET_HASHES["rna"],
            "supplement_sha256": ASSET_HASHES["supplement"], "core_sha256": RECEIPTS["core"],
            "previous_cohort_sha256": COHORT_HASHES["LICOB"], "RNA_recovery": recovery,
            "identity": {"shared_models": len(shared), "shared_genes": len(intersection),
                         "within_sample_spearman_median": median, "within_sample_spearman_p05": lower},
            "support": {"models": 43, "donors": 38, "previously_unscored_donors": 9, "drugs": 53},
        })


def registration(path):
    receipt = path / "receipt.json"
    document = json.loads(receipt.read_text())
    if (document["study"], document["core_sha256"], document["previous_cohort_sha256"]) != (
            TRACK, RECEIPTS["core"], COHORT_HASHES["LICOB"]):
        raise ValueError("Unexpected recovered-cohort manifest")
    for filename, digest in document["artifacts"].items():
        require_hash(path / filename, digest)
    return sha256(receipt)


def predict(args):
    receipt = registration(args.input)
    require_hash(args.settings, SETTINGS)
    require_hash(args.core / "receipt.json", RECEIPTS["core"])
    train, source, _ = training_inputs(args.core)
    query = pd.read_parquet(args.input / "query.parquet")
    held = pd.read_parquet(args.input / "rna.parquet").sort_index()
    fit = source.loc[sorted(train.cell_id.unique()), held.columns]
    edges, targets, _ = molecular_inputs(args.priors, args.hallmark, args.targets)
    details = {}
    if args.component == "structured":
        from .v0plus_neural import load_auxiliary

        auxiliary, aux = load_auxiliary(args.auxiliary)
        features, feature_hash = measured_drivers.read(args.genomics)
        if feature_hash != GENOMICS_HASH:
            raise ValueError("Unexpected source-only DNA registration")
        coding = pd.read_parquet(args.input / "coding.parquet")
        groups = query.assign(lineage_group=query.donor_id)
        blocks = {"coding": (features["source_coding"], coding)}
        prediction, details = structured_models(train, query, fit, held, edges, targets, auxiliary, blocks, groups)
        details |= {"genomics_sha256": feature_hash, "auxiliary": aux}
    else:
        prediction, query = fit_reference(args, train, query, fit, held, edges, targets)
    shared_support(prediction, query, prediction.method.unique())
    freeze_table(prediction.assign(track=TRACK, fold=0), args.out, provenance={"registration_sha256": receipt,
        "core_sha256": RECEIPTS["core"], "settings_sha256": SETTINGS, "component": args.component, **details})


def merge(args):
    receipt = registration(args.input)
    query = pd.read_parquet(args.input / "query.parquet")
    parts, hashes = [], {}
    for arm in expected_components("LICOB"):
        part, seal = verify_forecast(args.parts / f"{arm}.parquet")
        info = seal["provenance"]
        if (info["registration_sha256"], info["core_sha256"], info["settings_sha256"], info["component"]) != (
                receipt, RECEIPTS["core"], SETTINGS, arm):
            raise ValueError("Recovered-cohort prediction identity differs from the contract")
        parts.append(part[[*KEY, "predicted_value", "method", "track", "fold"]])
        hashes[arm] = seal["sha256"]
    joint = pd.concat(parts, ignore_index=True)
    blend = joint[joint.method.isin([PRIMARY, "drug_ridge"])].groupby([*KEY, "track", "fold"], as_index=False)
    joint = pd.concat([joint, blend.predicted_value.mean().assign(method="ridge_structured_average")])
    shared_support(joint, query, MODELS)
    freeze_table(joint, args.out, provenance={"registration_sha256": receipt, "component_hashes": hashes,
                 "labels_sha256": ASSET_HASHES["supplement"], "RNA_axis": "NODE complete-row prefix"})


def metrics(frame, query):
    pairs = [(PRIMARY, name) for name in (*CONTROLS, "v0plus_inductive", "codeae_inductive", "velodrome_inductive")]
    pairs += [("measured_coding", "target_expression"), ("measured_coding", "genotype_permutation"),
              ("ridge_structured_average", "drug_ridge"), ("ridge_structured_average", "cellhit")]
    full, _, _ = axis_metrics(frame, pairs)
    cis = [{"candidate": a, "baseline": b, "ci95": crossed_patient_drug_ci(frame, a, b)} for a, b in pairs]
    within_drug = {row["baseline"]: row["ci95"] for row in cis if row["candidate"] == PRIMARY}
    within_cell = {(row["baseline"], row["metric"]): row["ci95"] for row in full["paired_contrasts"]
                   if row["candidate"] == PRIMARY}
    comparisons = {name: all(within_drug[name][metric][0] > 0 for metric in ("spearman", "pearson"))
                   and all(within_cell[name, metric][0] > 0 for metric in ("spearman", "ndcg_at_5"))
                   for name in CONTROLS}
    strata = {}
    for name, ids in compound_strata(query).items():
        if ids:
            values = axis_metrics(frame[frame.compound_id.isin(ids)], ())[0]
            if len(ids) < 6:
                for row in values["scores"]:
                    row.pop("macro_ndcg_at_5", None)
                    row.pop("macro_top5_overlap", None)
            strata[name] = {"drugs": len(ids), "metrics": values}
    return {"overall": full, "across_cell_ci95": cis, "strata": strata, "donors": frame.lineage_group.nunique(),
            "observations": frame.row_id.nunique(), "locked_comparisons": comparisons, "pass": all(comparisons.values())}


def score(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt = registration(args.input)
    forecast, seal = verify_forecast(args.forecast)
    info = seal["provenance"]
    if info["registration_sha256"] != receipt or info["labels_sha256"] != ASSET_HASHES["supplement"]:
        raise ValueError("Recovered LICOB score requires the locked forecast")
    require_hash(args.supplement, ASSET_HASHES["supplement"])
    query = pd.read_parquet(args.input / "query.parquet")
    shared_support(forecast, query, MODELS)
    with args.supplement.open("rb") as stream:
        book = load_workbook(stream, read_only=True, data_only=True)
        rows = list(book["S6. AUC"].values)
        book.close()
    labels = []
    for item in query.itertuples(index=False):
        raw = rows[item.response_row - 1]
        value = raw[item.response_column]
        if raw[0] != item.cell_id or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("Registered recovered LICOB response has a missing or different identity")
        labels.append(-float(value))
    patients = query[["cell_id", "donor_id", "adult_age_confirmed", "new_donor"]].drop_duplicates()
    joined = join_observations(forecast, query.assign(endpoint_value=labels)).merge(
        patients, on="cell_id", validate="many_to_one",
    )
    scopes = {"new_donors_primary": joined[joined.new_donor], "previous_donors_technical": joined[~joined.new_donor],
              "all_donors_technical": joined}
    publish_json(args.out, {"stage": TRACK, "forecast_sha256": seal["sha256"], "registration_sha256": receipt,
        "labels_sha256": ASSET_HASHES["supplement"], "primary": PRIMARY,
        "scopes": {name: metrics(adult_donor_outcomes(frame), query) for name, frame in scopes.items()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="stage", required=True)
    fields = {"register": ("core", "node", "author", "supplement", "licob"),
              "predict": ("input", "core", "genomics", "priors", "hallmark", "targets", "settings", "upstream", "auxiliary"),
              "merge": ("input", "parts"), "score": ("input", "forecast", "supplement")}
    for stage, args in fields.items():
        sub = modes.add_parser(stage)
        for name in (*args, "out"):
            sub.add_argument("--" + name, type=Path, required=True)
        if stage == "predict":
            sub.add_argument("--component", required=True, choices=expected_components("LICOB"))
            sub.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    {"register": register, "predict": predict, "merge": merge, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
