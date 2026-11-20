"""Inspect additional-dose and cell-transfer support without treated expression."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from src.r11_chemical_probe.common import config, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY, table
from src.r11_chemical_probe.prepare import dose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    r11 = Path(cfg["r11_assets"])
    args.out.mkdir(parents=True, exist_ok=False)
    samples = table(r11 / "metadata/samples.tsv")
    active = samples[samples.plate_id.isin(cfg["plates"]) & samples.split.eq("train")]
    entities = active[["compound_entity_id", GROUP]].drop_duplicates()
    raw = pq.read_table(Path(cfg["tahoe_root"]) / "metadata/sample_metadata.parquet").to_pandas()
    raw = raw.rename(columns={"sample": KEY[0], "plate": "plate_id", "drug": "compound_entity_id"})
    raw["dose_uM"] = raw.drugname_drugconc.map(dose)
    all_extra = raw.loc[raw.dose_uM.isin([0.05, 5.0])].merge(
        entities, on="compound_entity_id", validate="many_to_one"
    )
    all_extra = all_extra.assign(time_h=cfg["time_h"], control_role="treated")
    collapsed = all_extra.groupby(["plate_id", "dose_uM"]).agg(wells=(KEY[0], "nunique"),
                                                               groups=(GROUP, "nunique"))
    collapsed.to_csv(args.out / "dose_support.tsv", sep="\t")
    extra = all_extra[all_extra.plate_id.isin(cfg["plates"])]
    extra[[KEY[0], "plate_id", "compound_entity_id", GROUP, "dose_uM", "time_h", "control_role"]].to_csv(
        args.out / "dose_wells.tsv", sep="\t", index=False
    )
    repeats = active.groupby(GROUP).agg(wells=(KEY[0], "nunique"),
                                        entities=("compound_entity_id", "nunique"),
                                        plates=("plate_id", "nunique")).reset_index()
    repeats.to_csv(args.out / "cell_transfer_support.tsv", sep="\t", index=False)
    write_json(args.out / "receipt.json", dict(
        source_groups=active[GROUP].nunique(), groups_with_multiple_wells=int(repeats.wells.ge(2).sum()),
        extra_dose_groups=extra[GROUP].nunique(), extra_dose_wells=len(extra),
        extra_dose_groups_other_plates=all_extra.loc[~all_extra.plate_id.isin(cfg["plates"]), GROUP].nunique()))
    print(collapsed.to_string(), flush=True)


if __name__ == "__main__":
    main()
