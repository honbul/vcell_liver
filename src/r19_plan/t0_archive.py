"""Reconcile the frozen R15 transfer and its archived, score-only scale diagnostic."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json

FREEZE = "32c46e8d7a9c90eb29f6ee62cde650ae2765300920ddb0266ffe0db40a21aac0"
SCORE = "44be0e43e85dba15b4695d9a2c4c27096829e4abbb07cb96a8083affe0be9622"
KEY = ["minimum_cells", "cohort", "stratum", "arm"]
BASES = {"B0", "B_pos-all", "B_cell", "R15-A", "R15-B"}


def verified_tables(folder, freeze_sha256):
    stamp = read_json(folder / "receipt.json")
    if stamp["freeze_sha256"] != freeze_sha256:
        raise ValueError(f"The archived table has different forecast lineage: {folder}")
    for name, digest in stamp["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"The archived table changed: {folder / name}")
    return stamp


def decompose(points, direction):
    rows = points.merge(direction, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not len(rows) or not rows._merge.eq("both").all():
        raise ValueError("The archived error and direction tables use different support")
    rows = rows.drop(columns="_merge")
    rho = rows.pooled_cosine.to_numpy()
    c = rows.prediction_to_observed_rms.to_numpy()
    sst = rows.sst.to_numpy()
    sse = rows.sse.to_numpy()
    if (not np.isfinite(np.stack((rho, c, sst, sse))).all() or (np.abs(rho) > 1.).any()
            or (c < 0.).any() or (sst <= 0.).any() or (sse < 0.).any()
            or ((c == 0.) & (rho != 0.)).any()):
        raise ValueError("The archived contrast aggregates are invalid")
    reconstructed = 2 * rho * c - c ** 2
    if (not np.allclose(reconstructed, rows.r2_pooled, rtol=1e-10, atol=1e-11)
            or not np.allclose(1. - sse / sst, rows.r2_pooled, rtol=1e-10, atol=1e-11)):
        raise ValueError("Archived direction and norms disagree with the pooled R²")
    optimum = np.divide(np.maximum(rho, 0.), c, out=np.zeros_like(c), where=c > 0.)
    fixed_direction_ceiling = np.square(np.maximum(rho, 0.))
    return rows.assign(
        reconstructed_r2=reconstructed,
        absolute_reconstruction_error=np.abs(reconstructed - rows.r2_pooled),
        r2_at_equal_norm=np.where(c > 0., 2 * rho - 1., np.nan),
        diagnostic_nonnegative_optimum=optimum,
        diagnostic_fixed_direction_r2_ceiling=fixed_direction_ceiling,
        r2_gain_from_scale_upper_bound=fixed_direction_ceiling - rows.r2_pooled,
    )


def benchmark_gaps(rows):
    comparisons = []
    for key, block in rows.groupby(KEY[:-1]):
        alternatives = block.loc[~block.arm.isin(BASES)].sort_values(["r2_pooled", "arm"], ascending=[False, True])
        if len(alternatives) != 10 or not {"B0", "B_pos-all"} <= set(block.arm):
            raise ValueError(f"The original Tahoe comparator registry changed: {key}")
        best = alternatives.iloc[0]
        scores = block.set_index("arm")
        if (block.parent_groups.nunique() != 1 or block.rows_roles.nunique() != 1
                or block.sst.nunique() != 1):
            raise ValueError(f"External comparators do not share archived support: {key}")
        for arm in ("R15-A", "R15-B", "B_cell"):
            if arm not in scores.index:
                continue
            row = scores.loc[arm]
            references = [(name, scores.loc[name]) for name in ("B0", "B_pos-all")]
            for label, reference in [*references, ("best-T2-by-raw-R2", best)]:
                comparisons.append(dict(
                    zip(KEY[:-1], key, strict=True), arm=arm, reference=label,
                    reference_arm=best.arm if label == "best-T2-by-raw-R2" else label,
                    parent_groups=row.parent_groups, rows_roles=row.rows_roles,
                    arm_raw_r2=row.r2_pooled, reference_raw_r2=reference.r2_pooled,
                    arm_scale_r2_ceiling=row.diagnostic_fixed_direction_r2_ceiling,
                    scale_ceiling_minus_reference_raw_r2=row.diagnostic_fixed_direction_r2_ceiling - reference.r2_pooled,
                ))
    return pd.DataFrame(comparisons)


def run(archive, output):
    if sha256(archive / "freeze.json") != FREEZE or sha256(archive / "scored/receipt.json") != SCORE:
        raise ValueError("The original R15 external freeze or primary score receipt changed")
    verified_tables(archive / "scored", FREEZE)
    descriptive = verified_tables(archive / "descriptive", FREEZE)
    if descriptive.get("descriptive_only") is not True:
        raise ValueError("The archived direction and norm record must be score-only")
    points = pd.read_csv(archive / "scored/points.tsv", sep="\t")
    direction = pd.read_csv(archive / "descriptive/direction_scale.tsv", sep="\t")
    rows = decompose(points, direction)
    comparisons = benchmark_gaps(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".t0-diagnostic-") as temp:
        stage = Path(temp)
        rows.to_csv(stage / "direction_decomposition.tsv", index=False, sep="\t", na_rep="NA")
        comparisons.to_csv(stage / "baseline_gaps.tsv", index=False, sep="\t")
        write_json(stage / "receipt.json", dict(
            source_freeze_sha256=FREEZE, source_score_receipt_sha256=SCORE,
            source_direction_receipt_sha256=sha256(archive / "descriptive/receipt.json"),
            code_sha256=sha256(__file__), archived_support_rows=len(rows),
            maximum_absolute_reconstruction_error=float(rows.absolute_reconstruction_error.max()),
            score_only=True, training=False, treatment_arrays_read=False,
            files={file.name: sha256(file) for file in sorted(stage.iterdir())},
        ))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.archive, args.out)
