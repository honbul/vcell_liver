"""Build a metadata-only HepaRG parent target list for the LINCS profile bank."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def build(src, dst):
    observed = defaultdict(list)
    with src.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["parent_key"] and row["smiles"]:
                observed[row["parent_key"]].append((row["smiles"], row["chemical"]))
    with dst.open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["parent_key", "smiles", "names"])
        for key, members in sorted(observed.items()):
            smiles = min(Counter(s for s, _ in members).items(), key=lambda x: (-x[1], x[0]))[0]
            writer.writerow([key, smiles, "|".join(sorted(set(name for _, name in members)))])
    print(f"{len(observed)} verified parent metadata records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.source, args.out)


if __name__ == "__main__":
    main()
