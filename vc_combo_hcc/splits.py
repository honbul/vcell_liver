"""Register independent lineage-cold, unordered-pair-cold and scaffold-cold assays."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


KEYS = ["id", "source", "cell_id", "cell_group", "liver", "adult_hcc",
        "compound_a", "compound_b", "connectivity_a", "connectivity_b", "scaffold_a", "scaffold_b"]
OUTCOMES = ["zip", "bliss", "css"]


def record(path):
    h = sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def group_hash(value, n_folds, salt):
    return int.from_bytes(sha256(f"track8:{salt}:{value}".encode()).digest()[:8], "big") % n_folds


def split_groups(rows, fields):
    return set().union(*(set(rows[name]) for name in fields))


def pair_key(left, right):
    return np.where(left < right, left + "|" + right, right + "|" + left)


def validate_assays(frame):
    if set(frame.columns) != set(KEYS + OUTCOMES):
        raise ValueError("Expected registered combination identities plus ZIP, Bliss and CSS")
    if frame[KEYS].isna().any().any() or not np.isfinite(frame[OUTCOMES].to_numpy()).all():
        raise ValueError("Combination eligibility requires exact assay identities and complete endpoints")
    if not frame.id.is_unique or (frame.connectivity_a == frame.connectivity_b).any():
        raise ValueError("Repeated assay IDs or same-parent drug combinations")
    if not frame.loc[frame.adult_hcc, "liver"].all():
        raise ValueError("HCC status disagrees with the registered tissue")
    identities = frame[["compound_a", "connectivity_a", "scaffold_a"]].rename(
        columns={"compound_a": "compound", "connectivity_a": "parent", "scaffold_a": "scaffold"},
    )
    others = frame[["compound_b", "connectivity_b", "scaffold_b"]].rename(
        columns={"compound_b": "compound", "connectivity_b": "parent", "scaffold_b": "scaffold"},
    )
    families = pd.concat((identities, others)).drop_duplicates()
    if families.compound.duplicated().any() or families[["parent", "scaffold"]].drop_duplicates().parent.duplicated().any():
        raise ValueError("An exact compound identity or standardized parent splits across scaffold groups")
    return frame.assign(pair=pair_key(frame.connectivity_a, frame.connectivity_b))


def branch_sets(frame):
    return split_groups(frame, ("scaffold_a", "scaffold_b"))


def validate_split(fit, valid, query, axis, reserved=None):
    for left, right in ((fit, valid), (fit, query), (valid, query)):
        if set(left.id) & set(right.id):
            raise ValueError("Repeated measurement across combination assay roles")
    if not query.adult_hcc.all():
        raise ValueError("External assay contains a non-HCC label")
    if axis == "lco_hcc":
        if ((set(fit.cell_group) & set(valid.cell_group))
                or (set(query.cell_group) & (set(fit.cell_group) | set(valid.cell_group)))
                or fit.liver.any() or valid.liver.any()):
            raise ValueError("Liver group or validation lineage leaked into LCO")
    if axis == "lpo":
        if set(fit.pair) & set(valid.pair) or set(query.pair) & (set(fit.pair) | set(valid.pair)):
            raise ValueError("Unordered compound pair crossed an LPO boundary")
    if axis == "ldo":
        seen = branch_sets(pd.concat([fit, valid]))
        if seen & reserved or not (query.scaffold_a.isin(reserved) | query.scaffold_b.isin(reserved)).all():
            raise ValueError("LDO external assay has no unseen scaffold or a held scaffold crossed into fitting")
        valid_unseen = branch_sets(valid) - branch_sets(fit)
        if not (valid.scaffold_a.isin(valid_unseen) | valid.scaffold_b.isin(valid_unseen)).all():
            raise ValueError("LDO validation lacks unseen scaffolds")


def lco(frame, seed, n):
    liver_groups = set(frame.loc[frame.liver, "cell_group"])
    source = frame[~frame.liver & ~frame.cell_group.isin(liver_groups)]
    groups = pd.Series({x: group_hash(x, n, seed + 31) for x in source.cell_group.unique()})
    role = source.cell_group.map(groups)
    return [(source[role.ne(0)], source[role.eq(0)], frame[frame.adult_hcc], None)]


def lpo(frame, seed, n):
    assign = pd.Series({x: group_hash(x, n, seed + 71) for x in frame.pair.unique()})
    roles = frame.pair.map(assign)
    return [
        (frame[~roles.isin((i, (i + 1) % n))], frame[roles.eq((i + 1) % n)],
         frame[roles.eq(i) & frame.adult_hcc], None)
        for i in range(n)
    ]


def ldo(frame, seed, n):
    ids = branch_sets(frame)
    assign = {x: group_hash(x, n, seed + 97) for x in ids}
    a, b = frame.scaffold_a.map(assign), frame.scaffold_b.map(assign)
    # If two cold scaffolds belong to different folds, choose the anchor
    # using only canonical pair identity. Every query is scored once.
    choose_first = pd.Series({x: group_hash(x, 2, seed + 123) == 0 for x in frame.pair.unique()})
    first_is_a = frame.connectivity_a.lt(frame.connectivity_b)
    anchor = np.where(frame.pair.map(choose_first).eq(first_is_a), a, b)
    result = []
    for i in range(n):
        hold = (a == i) | (b == i)
        next_fold = (i + 1) % n
        inner = (a == next_fold) | (b == next_fold)
        result.append((frame[~hold & ~inner], frame[~hold & inner],
                       frame[(anchor == i) & frame.adult_hcc],
                       {scaffold for scaffold, fold in assign.items() if fold == i}))
    return result


def register(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame = validate_assays(pd.read_parquet(a.assays))
    result = {}
    with tempfile.TemporaryDirectory(prefix=f".{a.output.name}.", dir=a.output.parent) as d:
        staged = Path(d)
        for axis, build in (("lco_hcc", lco), ("lpo", lpo), ("ldo", ldo)):
            assessed = set()
            for i, (fit, valid, query, held) in enumerate(build(frame, a.seed, a.folds)):
                name = f"{axis}-{i}"
                if min(len(fit), len(valid)) < 100 or len(query) < 10:
                    raise ValueError(f"Insufficient independent registered assays for {name}")
                validate_split(fit, valid, query, axis, held)
                if assessed & set(query.id):
                    raise ValueError(f"Repeated across outer folds for {axis}")
                assessed.update(query.id)
                scope = staged / name
                scope.mkdir()
                paths = {}
                for role, table in (("fit", fit), ("validation", valid),
                                    ("query", query[KEYS + ["pair"]]), ("labels", query[["id", *OUTCOMES]])):
                    # Forecast input files carry no synergy, combination efficacy,
                    # or observed single-agent measurements from held-out wells.
                    file = scope / f"{role}.parquet"
                    table.to_parquet(file, index=False)
                    paths[file.name] = record(file)
                result[name] = {
                    "axis": axis, "train_rows": len(fit), "validation_rows": len(valid), "test_rows": len(query),
                    "query_lineages": query.cell_group.nunique(), "query_pairs": query.pair.nunique(),
                    "query_studies": query.source.value_counts().to_dict(),
                    "reserved_scaffolds": sorted(held) if held is not None else [],
                    "files": paths,
                }
                (scope / "receipt.json").write_text(json.dumps(result[name], indent=2, sort_keys=True) + "\n")
            if assessed != set(frame.loc[frame.adult_hcc, "id"]):
                raise ValueError(f"{axis} misses registered adult HCC assay rows")
        (staged / "receipt.json").write_text(json.dumps({
            "assays": record(a.assays), "seed": a.seed, "panels": result,
            "panel_receipts": {p.name: record(p / "receipt.json") for p in staged.iterdir() if p.is_dir()},
        }, indent=2, sort_keys=True) + "\n")
        staged.rename(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assays", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=81122)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    if a.folds < 3:
        p.error("At least three folds are required for independent scaffold and pair validation")
    register(a)


if __name__ == "__main__":
    main()
