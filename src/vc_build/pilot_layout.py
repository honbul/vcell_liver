"""Generate a blind ready planning layout for the unapproved VCB-06 pilot."""

from __future__ import annotations

import argparse
import csv
import random
from itertools import product
from pathlib import Path

MODEL_SLOTS = ("DEV01", "DEV02", "BLIND01", "BLIND02")
DRUG_SLOTS = tuple(f"P{n:02d}" for n in range(1, 25))
RNA_SLOTS = DRUG_SLOTS[:12]
FUNCTION_DOSES = tuple(f"F{n}" for n in range(1, 6))
RNA_DOSES = ("F2", "F4")
HEADER = ("model_slot", "role", "independent_day", "measurement", "plate_slot", "well",
          "condition", "drug_slot", "dose_slot", "time_h", "sister_condition")


def layout(seed=14822):
    rng = random.Random(seed)
    wells = [f"{row}{col:02d}" for row, col in product("BCDEFGHIJKLMNO", range(2, 24))]
    previous = {}
    records = []
    for model, day in product(MODEL_SLOTS, range(1, 4)):
        role = "development" if model.startswith("DEV") else "blind_confirm"
        key = f"{model}-D{day}"
        treatment = list(product(DRUG_SLOTS, FUNCTION_DOSES))
        rna = list(product(RNA_SLOTS, RNA_DOSES))
        for measurement, items, time, controls in (
            ("function_signal", treatment, 72, [("vehicle", 12), ("positive_control", 8), ("blank", 8)]),
            ("rna", rna, 24, [("vehicle_rna", 2)]),
            ("predose_rna", [], 0, [("predose_rna", 1)]),
            ("predose_count", [], 0, [("predose_count", 4)]),
        ):
            plate = f"{key}-{measurement}"
            conditions = [("treated", drug, dose) for drug, dose in items]
            conditions += [(kind, "", str(i + 1)) for kind, count in controls for i in range(count)]
            available = wells.copy()
            rng.shuffle(available)
            for kind, drug, dose in conditions:
                identifier = (model, measurement, kind, drug, dose)
                seen = previous.setdefault(identifier, set())
                well = next(slot for slot in available if slot not in seen)
                available.remove(well)
                seen.add(well)
                pairing = f"{key}-{drug}-{dose}" if kind == "treated" else ""
                records.append((model, role, day, measurement, plate, well, kind, drug, dose, time, pairing))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = layout()
    with args.out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
