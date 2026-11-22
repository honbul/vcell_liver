"""Extract DepMap Chronos and HCC driver annotations for source supervision."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .context_features import COPY, MUTATION
from .io import publish_json, sha256
from .v0plus_cv import training_inputs


def gene_columns(path, genes):
    matches = {}
    for column in pd.read_csv(path, nrows=0).columns:
        match = re.fullmatch(r"(.+) \(\d+\)", column)
        if match and match[1] in genes:
            if match[1] in matches.values():
                raise ValueError(f"Multiple DepMap columns for {match[1]}")
            matches[column] = match[1]
    return matches


def default_profiles(path, cells, genes):
    wanted = gene_columns(path, genes)
    frame = pd.read_csv(path, usecols=["ModelID", "IsDefaultEntryForModel", *wanted])
    present = frame.ModelID.isin(cells) & frame.IsDefaultEntryForModel.eq("Yes")
    output = frame.loc[present, ["ModelID", *wanted]].set_index("ModelID").rename(columns=wanted)
    if not output.index.is_unique:
        raise ValueError(f"Duplicate default DepMap profiles in {path.name}")
    return output.astype(np.float32)


def prepare(depmap, cells, targets):
    files = {name: depmap / filename for name, filename in {
        "chronos": "CRISPRGeneEffect.csv", "copy": "OmicsCNGeneWGS.csv",
        "mutations": "OmicsSomaticMutationsMatrixHotspot.csv",
    }.items()}
    genes = set(MUTATION) | set(COPY) | set().union(*targets.values())
    gene_map = gene_columns(files["chronos"], genes)
    chronos = (pd.read_csv(files["chronos"], usecols=["Unnamed: 0", *gene_map])
               .set_index("Unnamed: 0").rename(columns=gene_map))
    if not chronos.index.is_unique:
        raise ValueError("Duplicate DepMap Chronos model IDs")
    chronos = chronos.reindex(cells).astype(np.float32).sort_index(axis=1)
    copy = default_profiles(files["copy"], cells, set(COPY)).reindex(cells)
    mutation = default_profiles(files["mutations"], cells, set(MUTATION)).reindex(cells)
    drivers = pd.concat([mutation.add_prefix("hotspot_"), copy.add_prefix("copy_")], axis=1).sort_index(axis=1)
    if (drivers.filter(like="copy_") < 0).any().any():
        raise ValueError("Expected nonnegative linear gene copy number relative to ploidy")
    drivers.loc[:, drivers.columns.str.startswith("copy_")] = np.log1p(drivers.filter(like="copy_"))
    for result in (chronos, drivers):
        result.index.name = "cell_id"
    sources = {name: {"file": file.name, "sha256": sha256(file), "bytes": file.stat().st_size}
               for name, file in files.items()}
    return chronos, drivers, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--depmap", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fit, _, _ = training_inputs(args.bundle)
    cells = sorted(fit.cell_id.unique())
    targets = json.loads(args.targets.read_text())
    chronos, drivers, sources = prepare(args.depmap, cells, targets)
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {"chronos": args.out / "chronos.parquet", "drivers": args.out / "drivers.parquet"}
    chronos.to_parquet(paths["chronos"])
    drivers.to_parquet(paths["drivers"])
    publish_json(args.out / "receipt.json", {
        "data_release": "DepMap Public 25Q3", "inputs": sources, "target_file_sha256": sha256(args.targets),
        "fit_model_count": len(cells), "endpoints": {
            key: {"features": table.shape[1], "models_with_any": int(table.notna().any(axis=1).sum()),
                  "observed": int(table.notna().sum().sum()), "sha256": sha256(paths[key])}
            for key, table in {"chronos": chronos, "drivers": drivers}.items()
        },
    })


if __name__ == "__main__":
    main()
