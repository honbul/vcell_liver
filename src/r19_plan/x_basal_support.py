"""Audit a distinct, existing basal RNA source for the matched LINCS X2 contexts."""

import argparse
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json

IDENTITY = ["ModelID", "CellLineName", "StrippedCellLineName", "RRID", "CatalogNumber", "OncotreePrimaryDisease"]
PROFILE = ["ModelID", "SequencingID", "ModelConditionID", "IsDefaultEntryForModel"]


def normalize_names(values):
    return values.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def yes(values):
    return values.str.casefold().isin(["true", "1", "yes"])


def resolve_contexts(cells, source, models, profiles):
    names = normalize_names(models.StrippedCellLineName)
    aliases = normalize_names(models.CellLineName)
    resolved = []
    origin = source.set_index("cell_id", verify_integrity=True)
    for cell in sorted(cells):
        key = re.sub(r"[^A-Z0-9]", "", cell.upper())
        matches = models.loc[names.eq(key) | aliases.eq(key)]
        catalog = str(origin.provider_catalog_id.get(cell, ""))
        matched = matches.loc[normalize_names(matches.CatalogNumber).eq(re.sub(r"[^A-Z0-9]", "", catalog.upper()))]
        if catalog not in ("", "-666") and len(matched):
            matches = matched
        match = matches.iloc[0] if len(matches) == 1 else None
        frame = profiles.loc[profiles.ModelID.eq(match.ModelID)] if match is not None else profiles.iloc[:0]
        conflicting_catalog = bool(match is not None and catalog not in ("", "-666")
                                   and match.CatalogNumber not in ("", "-666") and matched.empty)
        support = not conflicting_catalog and len(frame) == 1 and bool(frame.SequencingID.iloc[0])
        row = dict(cell=cell, model_candidates=len(matches), profile_candidates=len(frame), eligible=support,
                   lincs_subtype=origin.subtype.get(cell, ""), lincs_catalog=catalog,
                   catalog_conflict=conflicting_catalog)
        if match is not None:
            row.update(model_id=match.ModelID, rrid=match.RRID, depmap_disease=match.OncotreePrimaryDisease,
                       depmap_catalog=match.CatalogNumber)
        if support:
            row.update(sequence_id=frame.SequencingID.iloc[0], condition_id=frame.ModelConditionID.iloc[0])
        resolved.append(row)
    return pd.DataFrame(resolved)


def eligible_groups(groups, mapping):
    cells = set(mapping.loc[mapping.eligible, "cell"])
    result = groups.assign(basal_supported=groups.contexts.map(lambda ids: set(ids.split("|")) <= cells))
    minimum = result[["fold0", "fold1", "fold2"]].min(axis=1)
    return (result.assign(_minimum=minimum)
            .sort_values(["parents", "_minimum", "contexts"], ascending=[False, False, True])
            .drop(columns="_minimum").reset_index(drop=True))


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    registered = read_json(args.x / "receipt.json")
    candidates = args.x / "loco_candidates.tsv"
    if (not registered["metadata_only"] or registered["expression_read"]
            or sha256(candidates) != registered["files"][candidates.name]):
        raise ValueError("The original X cohort eligibility changed")
    groups = pd.read_csv(candidates, sep="\t")
    cells = set(groups.loc[groups.eligible, "contexts"].str.split("|").explode())
    models = args.depmap / "Model.csv"
    profiles = args.depmap / "OmicsProfiles.csv"
    matrix = args.depmap / "OmicsExpressionTPMLogp1HumanAllGenes.csv"
    profile_metadata = pd.read_csv(profiles, dtype=str, keep_default_na=False, usecols=[*PROFILE, "DataType", "Stranded"])
    profile_metadata = profile_metadata.loc[profile_metadata.DataType.eq("rna") & ~yes(profile_metadata.Stranded)
                                           & yes(profile_metadata.IsDefaultEntryForModel), PROFILE].drop_duplicates()
    # The CSV has profile identifiers and numeric features. Only identifier columns are materialized.
    matrix_ids = pd.read_csv(matrix, dtype=str, keep_default_na=False, usecols=PROFILE)
    chosen = matrix_ids.loc[yes(matrix_ids.IsDefaultEntryForModel)].merge(
        profile_metadata, on=PROFILE[:3], how="inner", validate="one_to_one")
    model_metadata = pd.read_csv(models, dtype=str, keep_default_na=False, usecols=IDENTITY)
    lincs_cells = pd.read_csv(args.lincs_cells, sep="\t", dtype=str, keep_default_na=False)
    mapping = resolve_contexts(cells, lincs_cells, model_metadata, chosen)
    options = eligible_groups(groups, mapping)
    axes = pd.read_csv(matrix, nrows=0).columns
    entrez = [match[1] for name in axes if (match := re.search(r"\(([0-9]+)\)$", name))]
    landmark = pd.read_csv(args.landmarks, sep="\t", dtype={"pr_gene_id": str})
    counts = pd.Series(entrez).value_counts()
    common = set(counts.loc[counts.eq(1)].index) & set(landmark.pr_gene_id)
    best = options.loc[options.eligible & options.basal_supported]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent, prefix=".x-basal-support-") as tmp:
        stage = Path(tmp)
        mapping.to_csv(stage / "context_profiles.tsv", sep="\t", index=False, na_rep="NA")
        options.to_csv(stage / "loco_candidates.tsv", sep="\t", index=False)
        receipts = {str(path): sha256(path) for path in (models, profiles, matrix, args.lincs_cells, args.landmarks)}
        write_json(stage / "receipt.json", dict(
            metadata_only=True, expression_values_loaded=False, source_release=args.depmap.name,
            matched_landmarks=len(common), l1000_landmarks=len(landmark), upstream_sha256=receipts,
            context_support_sha256=sha256(args.x / "receipt.json"),
            selected_contexts=[] if best.empty else best.contexts.iloc[0].split("|"),
            common_parents=0 if best.empty else int(best.parents.iloc[0]), code_sha256=sha256(__file__),
            files={p.name: sha256(p) for p in sorted(stage.iterdir())},
        ))
        stage.rename(args.out)
    return sha256(args.out / "receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("x", "depmap", "lincs-cells", "landmarks", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(run(parser.parse_args()))
