"""Project the prospective 10 µM benchmark precision from verified development scores."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan.hcc_confirm_design import MARGINAL_POWER, PRACTICAL, pass_probability, required_parents


METRICS = ("relative_rmse_gain", "delta_r2")
BLOCKS = (8, 16, 24)
BLOCK_SDS = (0., .25, .5, 1.)
SOURCES = {"JHH5": (305, 276, 6), "JHH7": (88, 88, 1)}


def verified_table(folder, filename):
    path = folder / filename
    if sha256(path) != read_json(folder / "receipt.json")["files"][filename]:
        raise ValueError(f"Summary table differs from its pinned score receipt: {path}")
    return pd.read_csv(path, sep="\t")


def calibration(data):
    records, digests = [], {}
    for cell, counts in SOURCES.items():
        folder = data / "jhh_off_position_robustness" / cell
        digest = sha256(folder / "receipt.json")
        receipt = read_json(folder / "receipt.json")
        old = data / (cell.lower()+"_off_position_native")
        if (not receipt["original_score_replayed"]
                or receipt["original_freeze_sha256"] != sha256(old/"freeze.json")
                or receipt["original_score_sha256"] != sha256(old/"score/receipt.json")):
            raise ValueError("Calibration must use the frozen matched public comparator score")
        table = verified_table(folder, "clustered.tsv")
        table = table.loc[table.reference.eq("best_VC")]
        blocks = {"scaffold", "scaffold_x_plate"} if counts[2] > 1 else {"scaffold"}
        if (len(table) != 2*len(blocks) or set(table.design) != blocks or set(table.metric) != set(METRICS)
                or set(map(tuple, table[["parents", "scaffolds", "plates"]].to_numpy())) != {counts}):
            raise ValueError("Unexpected score or clustering support")
        for row in table.itertuples():
            point, lo, hi = 100*row.point, 100*row.low99, 100*row.high99
            width = max(point-lo, hi-point)
            records.append(dict(development_cell=cell, design=row.design, metric=row.metric,
                 parents=row.parents, scaffolds=row.scaffolds, rna_plates=row.plates, observed_gain=point,
                 low99=lo, high99=hi, halfwidth=width,
                 equivalent_scaffold_sd=width*np.sqrt(row.scaffolds)/norm.ppf(.995) if row.design=="scaffold" else np.nan))
        digests[cell] = digest
    return pd.DataFrame(records), digests


def support(data):
    folder = data / "hcc_off_position_pilot" / "audit"
    receipt = read_json(folder/"receipt.json")
    names = verified_table(folder, "slots.tsv")
    plate = verified_table(folder, "plate_checks.tsv")
    raw = verified_table(folder, "support.tsv")
    if (receipt["measured"] or not receipt["metadata_only"] or len(names) != receipt["selected"]
            or not names.scaffold_unit.is_unique or not plate.plate_slot.is_unique or not plate.treatments.eq(36).all()
            or not plate.vehicles.eq(12).all() or receipt["actual_treatment_slots"] != plate.treatments.sum()
            or receipt["actual_vehicle_slots"] != plate.vehicles.sum() or len(plate) != 16):
        raise ValueError("Planning assumptions disagree with the existing pilot roster or layout")
    upper = raw.loc[raw.minimum_cells.eq(8)].set_index("phase_scope", verify_integrity=True).distinct_position_eligible_scaffolds
    confirmed_counts = dict(pilot=int(receipt["selected"]), registered_pool=int(upper["registered_phases"]),
                            all_phases_pool=int(upper["all_phases"]))
    confirmed_counts.update(separate_registered=confirmed_counts["registered_pool"]-confirmed_counts["pilot"],
                            separate_all_phases=confirmed_counts["all_phases_pool"]-confirmed_counts["pilot"])
    return confirmed_counts, sha256(folder/"receipt.json")


def workload(parents, blocks):
    if not math.isfinite(parents):
        return dict(treatment_wells_two_lines=math.inf, minimum_plates_two_lines=math.inf, vehicle_wells_two_lines=math.inf)
    plates = 2*max(blocks, math.ceil(3*parents/36))
    return dict(treatment_wells_two_lines=6*parents, minimum_plates_two_lines=plates, vehicle_wells_two_lines=12*plates)


def maximum_se(effect, blocks):
    if effect <= PRACTICAL:
        return 0.
    z = norm.ppf(MARGINAL_POWER)
    return min((effect-PRACTICAL)/z, effect/(t.ppf(.995, blocks-1)+z))


def designs(calibrated, supports):
    limits, grid = [], []
    for cell, chunk in calibrated.loc[calibrated.design.eq("scaffold")].groupby("development_cell"):
        source = chunk.set_index("metric", verify_integrity=True).loc[list(METRICS)]
        sd = source.equivalent_scaffold_sd.to_numpy()
        observed = source.observed_gain.to_numpy()
        effects = {"development_point": observed, "half_development_point": .5*observed,
                   "one_one": (1., 1.), "one_two": (1., 2.), "two_two": (2., 2.)}
        for effect_case, (rmse, r2) in effects.items():
            true_effect = [rmse, r2]
            for blocks in BLOCKS:
                for block in BLOCK_SDS:
                    common = dict(development_cell=cell, effect_case=effect_case, true_rmse_gain=rmse,
                                  true_r2_gain=r2, independent_blocks_per_line=blocks, assumed_block_sd=block)
                    n_rmse, n_r2 = [required_parents(s, block, blocks, e) for s,e in zip(sd, true_effect, strict=True)]
                    n = max(n_rmse,n_r2,3)
                    limits.append(dict(**common, minimum_parents_rmse=n_rmse, minimum_parents_r2=n_r2,
                                        minimum_parents_both=n, within_registered_holdout_cap=n<=supports["separate_registered"],
                                        within_all_phases_holdout_cap=n<=supports["separate_all_phases"], **workload(n,blocks)))
                    for support_name, size in supports.items():
                        p1,p2 = [pass_probability(s, block, blocks, size, e) for s,e in zip(sd,true_effect,strict=True)]
                        ready = [blocks*(maximum_se(e,blocks)**2-s**2/size) for s,e in zip(sd,true_effect,strict=True)]
                        grid.append(dict(**common, support_case=support_name, parents=size, rmse_pass_probability=p1,
                            r2_pass_probability=p2, joint_lower_bound_one_line=max(0.,p1+p2-1),
                            max_block_sd_at_goal_rmse=np.sqrt(ready[0]) if ready[0]>=0 else np.nan,
                            max_block_sd_at_goal_r2=np.sqrt(ready[1]) if ready[1]>=0 else np.nan,
                            **workload(size,blocks)))
    return pd.DataFrame(limits), pd.DataFrame(grid)


def run(args):
    calibrated, score_receipts = calibration(args.data)
    supports, pilot_receipt = support(args.data)
    minima, grid = designs(calibrated,supports)
    pilot = grid.loc[grid.support_case.eq("pilot") & grid.independent_blocks_per_line.eq(8)]
    if not (pilot.minimum_plates_two_lines.eq(16).all() and pilot.treatment_wells_two_lines.eq(576).all()
            and pilot.vehicle_wells_two_lines.eq(192).all()):
        raise ValueError("Pilot projection workload differs from the frozen layout")
    args.out.mkdir(exist_ok=False)
    for name, df in (("calibration", calibrated), ("pilot", pilot), ("minima", minima), ("power_grid", grid)):
        df.to_csv(args.out/(name+".tsv"), sep="\t", index=False, na_rep="NA", float_format="%.10g")
    helper = Path(__file__).with_name("hcc_confirm_design.py")
    write_json(args.out/"receipt.json", dict(protocol_sha256=sha256(args.protocol), code_sha256=sha256(__file__),
        helper_sha256=sha256(helper), development_receipts=score_receipts, pilot_audit_receipt_sha256=pilot_receipt,
        chemical_support=supports, measured_expression_opened=False, files={f.name:sha256(f) for f in args.out.iterdir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data","protocol","out"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
