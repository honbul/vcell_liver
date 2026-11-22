"""Frozen sorafenib PDX transfer and cross-publication vinorelbine agreement."""

import argparse
import gzip
from io import BytesIO
from itertools import combinations
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .io import freeze_table, publish_json, sha256, verify_forecast
from .rna_format_replay import GENE_ID, RELEASE_SHA, exon_lengths


SORAFENIB = "MLDQJTXFUGDVEO-UHFFFAOYSA-N"
RNA_SHA = "932879da37c985a5be359686d213131a42ed576b50197802fc242f197453fde6"
GTF_SHA = "81359865bab048361cbeaf91c9441aac4c700b63978e6691d579d78bf2322d30"
ASSETS = {
    "infi_2019.xml": "94fa31b1efcbc6c71d7eff664e184f58044f2a02860de5f1e5cfc187448bce3b",
    "evero.xml": "8931a8be108f6d12b9930771547da36179ce2a3aecd0228be96b8cea0afc4965",
    "fgf401_supp.pdf": "bf760a1d6faee5ce72d47d5e81acdfd37e0aaa2f01c2d1a4bddb6292b40b289f",
    "soraf_vino_supp.zip": "c21fc9b0ef7af0997bbd2398be0b9c3d5eb1732c262974a9186e01469b2c6c6f",
}
NORMALIZE = str.maketrans({"‐": "-", "–": "-", "−": "-"})
DONORS = {"HCC29-0909A": "HCC01-0708", "HCC26-0808B": "HCC26-0808A"}
MODELS = ("drug_mean", "drug_ridge", "structured")


def matched_lengths(gtf, genes):
    by_stable = {}
    with gzip.open(gtf, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 8)
            if fields[2] != "gene":
                continue
            versioned = GENE_ID.search(fields[-1]).group(1)
            if re.fullmatch(r"ENSG\d+\.\d+", versioned):
                stable = versioned.split(".")[0]
                by_stable.setdefault(stable, []).append(versioned)
    supported = {key: ids[0] for key, ids in by_stable.items() if key in genes and len(ids) == 1}
    lengths = exon_lengths(gtf, set(supported.values()))
    return {key: lengths[name] for key, name in supported.items() if name in lengths}


def predict(args):
    for path, expected in ((args.rna, RNA_SHA), (args.gtf, GTF_SHA), (args.package / "release.json", RELEASE_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"Frozen prediction input changed: {path}")
    args.run.mkdir(exist_ok=False)
    rna = pd.read_csv(args.rna, dtype={"gene_id": str, "gene_symbol": str})
    if len(rna) != 38576 or rna.shape[1] != 19 or not rna.gene_id.is_unique:
        raise ValueError("Unexpected PDX RNA axes")
    rna = rna[rna.gene_symbol.notna() & ~rna.gene_symbol.duplicated(keep=False)].copy()
    known = matched_lengths(args.gtf, set(rna.gene_id))
    rna["length_bp"] = rna.gene_id.map(known).astype("Int64")
    counts = args.run / "counts.tsv"
    lengths = args.run / "lengths.tsv"
    panel = args.run / "compounds.tsv"
    rna.set_index("gene_symbol").drop(columns=["gene_id", "length_bp"]).T.to_csv(
        counts, sep="\t", index_label="sample_id",
    )
    rna[["gene_symbol", "length_bp"]].dropna().rename(columns={"gene_symbol": "gene"}).to_csv(
        lengths, sep="\t", index=False,
    )
    panel.write_text(f"compound_id\n{SORAFENIB}\n")
    parts, files = [], {}
    for name in ("cpm", "length_tpm"):
        for model in MODELS:
            dest = args.run / f"{name}-{model}.tsv"
            command = [sys.executable, "-m", "vc_build.v0plus_cli", "predict", "--package", str(args.package),
                       "--rna", str(counts), "--scale", "bulk_counts", "--compounds", str(panel),
                       "--model", model, "--out", str(dest)]
            if name == "length_tpm":
                command.extend(["--gene-lengths", str(lengths)])
            subprocess.run(command, check=True)
            predicted = pd.read_csv(dest, sep="\t")
            if (len(predicted) != 17 or predicted.sample_id.nunique() != 17
                    or not predicted.compound_id.eq(SORAFENIB).all()):
                raise ValueError("Unexpected prediction axes")
            parts.append(predicted[["sample_id", "sensitivity_score", "status", "rna_gene_coverage",
                                    "selected_gene_coverage"]].rename(
                columns={"sample_id": "row_id", "sensitivity_score": "predicted_value"},
            ).assign(track=name, method=model, fold=0))
            files[dest.name] = sha256(dest)
    freeze_table(pd.concat(parts, ignore_index=True), args.run / "forecast.parquet", provenance={
        "rna_sha256": RNA_SHA, "release_sha256": RELEASE_SHA, "gtf_sha256": GTF_SHA,
        "counts_sha256": sha256(counts), "length_sha256": sha256(lengths), "prediction_sha256": files,
        "unique_symbol_count": len(rna), "symbols_with_lengths": len(known),
    })


def table_rows(path, table_id):
    table = ET.parse(path).find(f".//table-wrap[@id='{table_id}']")
    return [["".join(cell.itertext()).strip() for cell in row] for row in table.findall(".//tbody/tr")]


def metric_row(model, drug, value, control=None, treatment=None):
    model = model.translate(NORMALIZE).replace("HC13-0109", "HCC13-0109")
    return dict(model_id=model, drug=drug, tc=value, control=control, treatment=treatment)


def pdf_as_text(data):
    result = subprocess.run(["pdftotext", "-layout", "-", "-"], input=data, capture_output=True, check=True)
    return result.stdout.decode().translate(NORMALIZE)


def response_sources(root):
    for name, expected in ASSETS.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Response source changed: {name}")
    hep, evero, nature, sora = [], [], [], []
    for model, vehicle, inf, vino, *_ in table_rows(root / "infi_2019.xml", "hep30481-tbl-0001"):
        if not re.fullmatch(r"HCC?\d{2}-\d{4}[A-Z]?", model.translate(NORMALIZE)):
            continue
        control, fgfr, tub = (float(value.split("±")[0]) for value in (vehicle, inf, vino))
        hep.extend([metric_row(model, drug, trt / control, control, trt)
                    for drug, trt in (("infigratinib", fgfr), ("vinorelbine", tub))])
    for model, vehicle, mtor, vino, _, sir, _ in table_rows(root / "evero.xml", "ijms-25-00017-t001"):
        if float(vehicle) != 1:
            raise ValueError("Published T/C needs a unit control")
        evero.extend(metric_row(model, drug, float(value)) for drug, value in
                     (("everolimus", mtor), ("vinorelbine", vino), ("sirolimus", sir)) if value)
    block = pdf_as_text((root / "fgf401_supp.pdf").read_bytes()).split("Supplementary Table 3.", 1)[1].split("\f")[0]
    f = r"\d+\.\d+"
    for model, ctrl, fgf, vino in re.findall(
        rf"^\s*(HCC[\d-]+[A-Z]?)\s+({f})\s*±\s+({f})\s*(?:±)?\s+({f})",
        block, re.M,
    ):
        control = float(ctrl)
        nature.extend(metric_row(model, drug, float(value) / control, control, float(value))
                      for drug, value in (("roblitinib", fgf), ("vinorelbine", vino)))
    with ZipFile(BytesIO((root / "soraf_vino_supp.zip").read_bytes())) as zip_file:
        text = pdf_as_text(zip_file.read("ijms-2813614-supplementary.pdf"))
    block = text.split("Supplementary Table S1.", 1)[1].split("Supplementary Table S2.", 1)[0]
    for model, sorafenib, vino in re.findall(rf"^\s*(HCC[\d-]+[A-Z]?)\s+1\s+({f})\s+({f})\s+", block, re.M):
        sora.extend((metric_row(model, "sorafenib", float(sorafenib)),
                     metric_row(model, "vinorelbine", float(vino))))
    frames = {}
    for name, rows, expected in (("2019", hep, 36), ("2020", nature, 22), ("2023", evero, 33), ("2024", sora, 44)):
        frame = pd.DataFrame(rows)
        if len(frame) != expected or frame.duplicated(["model_id", "drug"]).any() or not frame.tc.ge(0).all():
            raise ValueError(f"Unexpected source table join: {name} {len(frame)} vs {expected}")
        frames[name] = frame
    return frames


def donor_axis(frame):
    return frame.assign(donor=frame.model_id.replace(DONORS)).groupby("donor").mean(numeric_only=True)


def correlation(left, right, kind):
    if kind == "spearman":
        left, right = rankdata(left, axis=-1), rankdata(right, axis=-1)
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide((left * right).sum(axis=-1), norm, out=np.zeros_like(norm), where=norm > 1e-12)


def agreements(frame, candidate, reference, comparison=None):
    a, b = (frame[name].to_numpy(dtype=float) for name in (candidate, reference))
    draws = np.random.default_rng(72127).integers(len(frame), size=(4000, len(frame)))
    outputs = {}
    for name in ("spearman", "pearson"):
        point, boot = correlation(a, b, name), correlation(a[draws], b[draws], name)
        if comparison is not None:
            other = frame[comparison].to_numpy(dtype=float)
            point -= correlation(other, b, name)
            boot -= correlation(other[draws], b[draws], name)
        outputs[name] = {"estimate": float(point), "ci95": np.quantile(boot, [.025, .975]).tolist()}
    return outputs


def repeat_agreement(left, right, rna_ids):
    matched = left.merge(right, on="model_id", suffixes=("_left", "_right"), validate="one_to_one")
    repeated = np.isclose(matched.tc_left, matched.tc_right, rtol=0, atol=1e-8)
    weight_columns = ["control_left", "control_right", "treatment_left", "treatment_right"]
    mass = (matched[weight_columns].notna().all(axis=1)
            & matched.control_left.eq(matched.control_right)
            & matched.treatment_left.eq(matched.treatment_right))
    result = {}
    for name, mask in (("all", np.ones(len(matched), dtype=bool)), ("with_rna", matched.model_id.isin(rna_ids)),
                       ("nonidentical", ~repeated)):
        data = donor_axis(matched.loc[mask, ["model_id", "tc_left", "tc_right"]])
        result[name] = {"models": int(mask.sum()), "donors": len(data), "identical_tc": int((mask & repeated).sum()),
                        "identical_reported_arm_weights": int((mask & mass).sum())}
        if len(data) >= 5:
            result[name].update(agreements(data, "tc_left", "tc_right"))
            point = [float(correlation(data.drop(index).tc_left.to_numpy(), data.drop(index).tc_right.to_numpy(),
                                       "spearman")) for index in data.index]
            result[name]["leave_one_donor_out_spearman"] = [min(point), max(point)]
    return result


def score(args):
    forecasts, seal = verify_forecast(args.run / "forecast.parquet")
    frames = response_sources(args.sources)
    measured = frames["2024"].query("drug == 'sorafenib'").assign(response=lambda x: -x.tc)
    rna_ids = set(forecasts.row_id)
    transfer = {}
    for name, part in forecasts.groupby("track"):
        supported = part[part.status.eq("predicted")].pivot(index="row_id", columns="method", values="predicted_value")
        joined = measured.merge(supported, left_on="model_id", right_index=True, validate="one_to_one")
        data = donor_axis(joined[["model_id", "response", *MODELS]])
        transfer[name] = {"rows": len(joined), "donors": len(data), "predicted_model_pairs": 17 * len(MODELS) -
                          int(part.status.ne("predicted").sum()), "rna_coverage_min": float(part.rna_gene_coverage.min()),
                          "selected_coverage_min": float(part.selected_gene_coverage.min())}
        if len(data) >= 5:
            transfer[name].update({
                "scores": {model: agreements(data, model, "response") for model in MODELS},
                "contrasts": {model: agreements(data, "structured", "response", model) for model in MODELS[:-1]},
            })
    repeat = {f"{a}:{b}": repeat_agreement(frames[a].query("drug == 'vinorelbine'"),
                                          frames[b].query("drug == 'vinorelbine'"), rna_ids)
              for a, b in combinations(frames, 2)}
    primary = repeat["2023:2024"]["all"]
    publish_json(args.run / "score.json", {
        "forecast_sha256": seal["sha256"], "response_sha256": ASSETS, "sorafenib": transfer,
        "sorafenib_gate": "contrasts" in transfer["cpm"] and all(
            value["ci95"][0] > 0 for record in transfer["cpm"]["contrasts"].values() for value in record.values()),
        "vinorelbine_repetition": repeat, "vinorelbine_gate": primary["donors"] >= 8 and all(
            primary[kind]["ci95"][0] > 0 for kind in ("spearman", "pearson")),
        "support": {year: {drug: {"all_models": len(group), "rna_models": int(group.model_id.isin(rna_ids).sum()),
                                   "rna_donors": len(donor_axis(group[group.model_id.isin(rna_ids)]))}
                           for drug, group in frame.groupby("drug")} for year, frame in frames.items()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    for action, fn, fields in (("predict", predict, ("rna", "gtf", "package")), ("score", score, ("sources",))):
        command = tasks.add_parser(action)
        command.add_argument("--run", type=Path, required=True)
        for field in fields:
            command.add_argument("--" + field, type=Path, required=True)
        command.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
