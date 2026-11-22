"""Compare matched STAR counts and published TPM through the frozen offline CLI."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .io import publish_json, sha256
from .v0plus_cli import read_expression


INPUTS_SHA = "73d6e6ffd78f4d07f8922e238c37cf9f2c007fad86a3851f6935503519191b77"
RELEASE_SHA = "a9e6582ae6c9a87cb92075d73ae1dc2deb3919a0876a01897237d597c2a7c754"
GENE_ID = re.compile(r'\bgene_id "([^"]+)"')


def exon_lengths(gtf, gene_ids):
    intervals = defaultdict(lambda: defaultdict(list))
    with gzip.open(gtf, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            contig, _, feature, start, end, _, _, _, attributes = line.split("\t", 8)
            if feature != "exon":
                continue
            gene = GENE_ID.search(attributes).group(1)
            if gene in gene_ids:
                intervals[gene][contig].append((int(start), int(end)))
    sizes = {}
    for gene, contigs in intervals.items():
        length = 0
        for exons in contigs.values():
            previous_end = 0
            for start, end in sorted(exons):
                length += max(0, end - max(previous_end, start - 1))
                previous_end = max(previous_end, end)
        sizes[gene] = length
    return sizes


def prepare(inputs, output):
    manifest_path = inputs / "inputs.json"
    if sha256(manifest_path) != INPUTS_SHA:
        raise ValueError("Input manifest changed")
    manifest = json.loads(manifest_path.read_text())
    records = manifest["files"]
    if len(records) != 16 or len({record["case"] for record in records}) != 16:
        raise ValueError("Expected 16 independent primary-tumor samples")
    for record in [*records, manifest["gtf"]]:
        path = inputs / record["file"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Public file changed: {path}")
    axis = ["gene_id", "gene_name", "gene_type"]
    counts, tpm = {}, {}
    for record in records:
        table = pd.read_csv(inputs / record["file"], sep="\t", comment="#",
                            usecols=[*axis, "unstranded", "tpm_unstranded"])
        table = table[table.gene_id.str.startswith("ENSG")].reset_index(drop=True)
        if not counts:
            genes = table[axis]
            if not genes.gene_id.is_unique:
                raise ValueError("GDC gene IDs are ambiguous")
            unique_symbols = genes.gene_name.notna() & ~genes.gene_name.duplicated(keep=False)
            symbols = genes.loc[unique_symbols, "gene_name"]
        if not table[axis].equals(genes):
            raise ValueError("Sample gene axis changed")
        counts[record["sample"]] = table.loc[unique_symbols, "unstranded"].to_numpy()
        tpm[record["sample"]] = table.loc[unique_symbols, "tpm_unstranded"].to_numpy()
    length = exon_lengths(inputs / manifest["gtf"]["file"], set(genes.loc[unique_symbols, "gene_id"]))
    lengths = genes.loc[unique_symbols, ["gene_id", "gene_name"]].rename(columns={"gene_name": "gene"})
    lengths["length_bp"] = lengths.gene_id.map(length).astype("Int64")
    lengths[["gene", "length_bp"]].dropna().to_csv(output / "gene-lengths.tsv", sep="\t", index=False)
    for name, values in (("counts", counts), ("tpm", tpm)):
        pd.DataFrame.from_dict(values, orient="index", columns=symbols).to_csv(
            output / f"{name}.tsv", sep="\t", index_label="sample_id",
        )
    return {
        "gdc_gene_rows": len(genes), "unique_gene_symbols": len(symbols),
        "lengths_available": len(length), "sample_count": len(records),
        "gtf_sha256": manifest["gtf"]["sha256"],
    }


def rank_agreement(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[finite], b[finite]).statistic)


def predict(package, output, genes):
    formats = {
        "published_tpm": ("tpm.tsv", "bulk_tpm", None),
        "counts_cpm": ("counts.tsv", "bulk_counts", None),
        "counts_length_tpm": ("counts.tsv", "bulk_counts", output / "gene-lengths.tsv"),
    }
    scores, rna, receipts = {}, {}, {}
    for name, (filename, scale, lengths) in formats.items():
        inp = output / filename
        dest = output / f"{name}.tsv"
        command = [sys.executable, "-m", "vc_build.v0plus_cli", "predict", "--package", str(package),
                   "--rna", str(inp), "--scale", scale, "--out", str(dest)]
        if lengths is not None:
            command.extend(("--gene-lengths", str(lengths)))
        subprocess.run(command, check=True)
        result = pd.read_csv(dest, sep="\t")
        if len(result) != 16 * 68 or not result.status.eq("predicted").all():
            raise ValueError(f"Incomplete replay: {name}")
        if lengths is not None and (
            not result.gene_lengths_sha256.eq(sha256(lengths)).all()
            or not result.rna_normalization.eq("counts_per_base_to_tpm").all()
        ):
            raise ValueError("Length-normalization provenance is missing")
        scores[name] = result.pivot(index="sample_id", columns="compound_id", values="sensitivity_score")
        if scores[name].shape != (16, 68) or not np.isfinite(scores[name].to_numpy()).all():
            raise ValueError(f"Incomplete prediction axis: {name}")
        rna[name] = read_expression(inp, scale, None, lengths).reindex(index=scores[name].index, columns=genes)
        receipts[name] = {
            "output_sha256": sha256(dest), "input_sha256": sha256(inp), "predicted": len(result),
            "min_rna_coverage": float(result.rna_gene_coverage.min()),
            "min_selected_coverage": float(result.selected_gene_coverage.min()),
        }
    return scores, rna, receipts


def compare(scores, rna):
    reference, reference_rna = scores["published_tpm"], rna["published_tpm"]
    reference_values, reference_gene_values = reference.to_numpy(), reference_rna.to_numpy()
    summary, donor_mae = {}, {}
    reference_top5 = np.argsort(-reference_values, axis=1, kind="stable")[:, :5]
    for name in ("counts_cpm", "counts_length_tpm"):
        score = scores[name]
        if not score.index.equals(reference.index) or not score.columns.equals(reference.columns):
            raise ValueError(f"Mismatched replay axis: {name}")
        values, gene_values = score.to_numpy(), rna[name].to_numpy()
        gene_rank = [rank_agreement(a, b) for a, b in zip(gene_values, reference_gene_values)]
        drug_rank = [rank_agreement(a, b) for a, b in zip(values, reference_values)]
        donor_rank = [rank_agreement(a, b) for a, b in zip(values.T, reference_values.T)]
        top5 = np.argsort(-values, axis=1, kind="stable")[:, :5]
        overlap = [len(set(a) & set(b)) / 5 for a, b in zip(top5, reference_top5)]
        donor_mae[name] = np.abs(values - reference_values).mean(axis=1)
        summary[name] = {
            "median_model_gene_rank_spearman": float(np.median(gene_rank)),
            "median_within_tumor_drug_rank_spearman": float(np.median(drug_rank)),
            "median_within_drug_tumor_rank_spearman": float(np.median(donor_rank)),
            "mean_top5_overlap": float(np.mean(overlap)),
            "mean_absolute_score_error": float(donor_mae[name].mean()),
            "max_absolute_score_error": float(np.abs(values - reference_values).max()),
        }
    difference = donor_mae["counts_length_tpm"] - donor_mae["counts_cpm"]
    boot_index = np.random.default_rng(46112).integers(len(difference), size=(4000, len(difference)))
    interval = np.quantile(difference[boot_index].mean(axis=1), [.025, .975]).tolist()
    corrected = summary["counts_length_tpm"]
    checks = {
        "gene_rank_gt_0.99": corrected["median_model_gene_rank_spearman"] > .99,
        "within_tumor_rank_gt_0.99": corrected["median_within_tumor_drug_rank_spearman"] > .99,
        "within_drug_rank_gt_0.95": corrected["median_within_drug_tumor_rank_spearman"] > .95,
        "score_mae_improves_upper95_lt_0": interval[1] < 0.,
    }
    return summary, {"mae_difference": float(difference.mean()), "paired_donor_bootstrap_95": interval,
                     "checks": checks, "pass": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "package", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    release = args.package / "release.json"
    if sha256(release) != RELEASE_SHA:
        raise ValueError("Release package changed")
    record = json.loads(release.read_text())["models"]["structured"]
    checkpoint = args.package / record["path"]
    if sha256(checkpoint) != record["sha256"]:
        raise ValueError("Checkpoint changed")
    features = joblib.load(checkpoint).features
    args.output.mkdir(exist_ok=False)
    audit = prepare(args.inputs, args.output)
    scores, rna, outputs = predict(args.package, args.output, features.genes)
    agreement, decision = compare(scores, rna)
    publish_json(args.output / "receipt.json", {
        "inputs_manifest_sha256": INPUTS_SHA, "release_sha256": RELEASE_SHA,
        "checkpoint_sha256": record["sha256"], "gene_lengths_sha256": sha256(args.output / "gene-lengths.tsv"),
        "audit": audit, "outputs": outputs, "agreement": agreement, "decision": decision,
    })


if __name__ == "__main__":
    main()
