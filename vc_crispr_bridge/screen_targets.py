"""Extend the verified LIMORE target registry with public pan-cancer annotations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tempfile

import pandas as pd

from .drug_targets import NON_PROTEIN_DISCLAIMS, expand_token
from .gene_features import unambiguous_symbol_map
from .perturbseq import record
from .predict import checked, write_json


SOURCES = ("CTRPv2", "GDSC2", "PRISM20Q2")


def public_pan(root, hgnc, native):
    receipt = json.loads((root / "receipt.json").read_text())
    path = root / "train" / "rows.parquet"
    if record(path) != receipt["artifacts"]["train/rows.parquet"]:
        raise ValueError("Pan-cancer screen target claims differ from the completed source release")
    columns = ["compound_id", "source_id", "native_compound_id"]
    mapping = pd.read_parquet(path, columns=columns).dropna().drop_duplicates()
    if mapping.duplicated(["source_id", "native_compound_id"]).any():
        raise ValueError("The source registrations disagree on exact pan-cancer structures")
    claims = []
    for source, file, sep, identifier, target in native:
        table = pd.read_csv(file, sep=sep, usecols=[identifier, target], dtype=str)
        table = table.dropna().drop_duplicates().rename(columns={identifier: "native_compound_id", target: "claim"})
        table["source_id"] = source
        claims.append(table)
    reported = pd.concat(claims, ignore_index=True)
    mapped = mapping.merge(reported, on=["source_id", "native_compound_id"], validate="one_to_many")
    raw = mapped[["compound_id", "source_id", "claim"]].drop_duplicates()
    if not set(SOURCES).issubset(raw.source_id):
        raise ValueError("Missing publicly annotated compounds from at least one mapped screen")
    symbols = unambiguous_symbol_map(hgnc)
    gene_map = dict(zip(symbols.str.upper(), symbols.index))
    converted, missing = [], Counter()
    for compound, source, claim in raw.itertuples(index=False, name=None):
        for token in re.split(r"[,;|]+", claim):
            term = token.strip().upper()
            if not term or term in NON_PROTEIN_DISCLAIMS:
                continue
            resolved = [symbol for symbol in expand_token(term, gene_map) if symbol in gene_map]
            if not resolved:
                missing[(source, term)] += 1
            for symbol in resolved:
                converted.append((compound, gene_map[symbol], source, term, "assay_annotation", symbol))
    targets = pd.DataFrame(converted, columns=["drug_id", "gene", "source", "claim", "action", "symbol"])
    gaps = [{"source": source, "claim": claim, "records": count} for (source, claim), count in sorted(missing.items())]
    return targets.drop_duplicates().sort_values(list(targets.columns)), gaps


def prepare(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    base_path = a.limore_targets / "receipt.json"
    base_manifest = json.loads(base_path.read_text())
    old = pd.read_csv(checked(a.limore_targets, "targets.tsv", base_manifest), sep="\t", dtype=str)
    sources = (
        ("GDSC2", a.gdsc, ",", "DRUG_ID", "PUTATIVE_TARGET"),
        ("CTRPv2", a.ctrp, "\t", "master_cpd_id", "gene_symbol_of_protein_target"),
        ("PRISM20Q2", a.prism, ",", "broad_id", "target"),
    )
    screen_targets, unmatched = public_pan(a.pan_sources, a.hgnc, sources)
    eligible = set(pd.read_csv(a.limore_compounds, sep="\t", usecols=["compound_id"]).compound_id)
    additions = screen_targets[screen_targets.drug_id.isin(eligible)]
    merged = pd.concat([old, additions], ignore_index=True).drop_duplicates().sort_values(["drug_id", "gene", "source"])
    if any(additions[additions.source.eq(src)].empty for src in SOURCES):
        raise ValueError("There is no exact compound overlap to at least one public source screen")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        out = Path(d)
        merged.to_csv(out / "targets.tsv", sep="\t", index=False)
        screen_targets.to_csv(out / "pan_targets.tsv", sep="\t", index=False)
        write_json(out / "unmapped_public_screen_claims.json", unmatched)
        write_json(out / "receipt.json", {
            "inputs": {"baseline_limore_registry": record(base_path), "hgnc": record(a.hgnc),
                       "pan_cancer": record(a.pan_sources / "receipt.json"), "limore_registry": record(a.limore_compounds),
                       "native_screens": {name: record(path) for name, path, *_ in sources}},
            "unique_targets": merged.gene.nunique(), "mapped_compounds": merged.drug_id.nunique(),
            "pan_cancer": {source: {
                "mapped_compounds": group.drug_id.nunique(), "genes": group.gene.nunique(),
                "exact_matches_to_limore": additions.loc[additions.source.eq(source), "drug_id"].nunique(),
            } for source, group in screen_targets.groupby("source")},
            "holdout_semantics": "Exact matching structures carry all registered gene annotations from "
                                 "LIMORE, ChEMBL37 and all three non-liver source screens.",
            "files": {p.name: record(p) for p in sorted(out.iterdir())},
        })
        out.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("limore-targets", "limore-compounds", "pan-sources", "hgnc", "gdsc", "ctrp", "prism", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
