"""Check the sample-specific Parse barcode against within-plate sample order."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.r11_chemical_probe.common import sha256, write_json


def audit(root, out):
    sample_path = root / "metadata/sample_metadata.parquet"
    obs_path = root / "metadata/obs_metadata.parquet"
    samples = pq.read_table(sample_path, columns=["sample", "plate"]).to_pandas()
    if samples["sample"].duplicated().any():
        raise ValueError("Duplicate Tahoe sample IDs")
    samples["number"] = samples["sample"].str.extract(r"^smp_(\d+)$")[0].astype(int)
    samples["sample_rank"] = samples.groupby("plate").number.rank(method="first").astype(int)

    support = Counter()
    for batch in pq.ParquetFile(obs_path).iter_batches(batch_size=1_000_000, columns=["sample", "BARCODE"]):
        first = pc.list_element(pc.split_pattern(batch.column(1), "_", max_splits=1), 0)
        pairs = pa.table({"sample": batch.column(0), "first_barcode": first})
        for row in pairs.group_by(["sample", "first_barcode"]).aggregate([("first_barcode", "count")]).to_pylist():
            support[row["sample"], int(row["first_barcode"])] += row["first_barcode_count"]
    records = pd.DataFrame([(sample, pos, n) for (sample, pos), n in support.items()],
                           columns=["sample", "barcode_position", "assigned_cells"])
    if set(records["sample"]) != set(samples["sample"]):
        raise ValueError("Cell metadata does not cover exactly the declared sample IDs")
    if records.groupby("sample").size().max() != 1:
        raise ValueError("A sample has conflicting first barcode positions")
    joined = samples.merge(records, on="sample", validate="one_to_one")
    if joined.duplicated(["plate", "barcode_position"]).any() or not joined.barcode_position.between(1, 96).all():
        raise ValueError("Round-1 barcode is not a valid one-to-one well key per plate")
    out.mkdir(parents=True, exist_ok=False)
    joined.sort_values(["plate", "number"]).to_csv(out / "positions.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", dict(
        cells=int(joined.assigned_cells.sum()), wells=len(joined),
        rank_agreements=int(joined.sample_rank.eq(joined.barcode_position).sum()),
        disagreements_by_plate={str(k): int(v) for k, v in joined.loc[
            joined.sample_rank.ne(joined.barcode_position), "plate"].value_counts().items()},
        sample_metadata_sha256=sha256(sample_path), obs_metadata_bytes=obs_path.stat().st_size,
        positions_sha256=sha256(out / "positions.tsv"),
        coordinate_kind="Published first-barcode index; no explicit treatment A1-H12 map supplied",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.out)


if __name__ == "__main__":
    main()
