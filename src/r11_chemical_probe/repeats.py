"""Describe source-only measurement repeats without selecting models or holdout rows."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .common import arguments, config, sha256, write_json
from .dataset import ENTITY, GROUP, KEY, LINE, POOL, checked_table, load_data, table, weights_for
from .metrics import correlations


def center(vec, table_rows):
    out = vec.astype(np.float64).copy()
    for idx in table_rows.groupby(POOL, sort=False).indices.values():
        weights = weights_for(table_rows.iloc[idx], (GROUP, ENTITY, KEY[0]))
        out[idx] -= np.average(out[idx], axis=0, weights=weights)
    return out


def technical(source, legacy):
    wells = table(legacy / "wells.tsv")
    rhs = wells.rename(columns={"sample": KEY[0]})[[*KEY, "wid", "n_A", "n_B"]]
    rows = source.merge(rhs, on=KEY, validate="one_to_one", how="left")
    rows = rows[rows.n_A.ge(10) & rows.n_B.ge(10)].reset_index(drop=True)
    values = np.load(legacy / "treated_halves.npy", mmap_mode="r")
    a, b = (center(values[i, rows.wid.astype(int)], rows) for i in range(2))
    cosine, reason = correlations(a, b)
    rows = rows[[*KEY, "plate_id", GROUP, ENTITY, "n_A", "n_B"]].copy()
    rows["cosine"] = cosine
    rows["reason"] = reason
    return rows


def physical_rows(roots, source, treatment, panel, role):
    control_rows, controls = load_data(roots, role, panel)
    refs = control_rows[[LINE, "plate_id", KEY[0]]].copy()
    refs["control_index"] = np.arange(len(refs))
    refs = refs.rename(columns={KEY[0]: "control_sample"})
    rows = source.copy()
    rows["treated_index"] = np.arange(len(rows))
    rows = rows.merge(refs, on=[LINE, "plate_id"], how="inner", validate="many_to_one")
    deltas = treatment[rows.treated_index].toarray() - controls[rows.control_index].toarray()
    centered = center(deltas, rows)
    pairs = []
    for _, repeated in rows.groupby([ENTITY, LINE], sort=False):
        for a, b in itertools.combinations(repeated.index, 2):
            if rows.at[a, KEY[0]] == rows.at[b, KEY[0]]:
                continue
            cosine, reason = correlations(centered[[a]], centered[[b]])
            pairs.append(dict(role=role, cell_line_id=rows.at[a, LINE], compound_group_id=rows.at[a, GROUP],
                              entity=rows.at[a, ENTITY], well_a=rows.at[a, KEY[0]], well_b=rows.at[b, KEY[0]],
                              plate_a=rows.at[a, "plate_id"], plate_b=rows.at[b, "plate_id"],
                              same_plate=rows.at[a, "plate_id"] == rows.at[b, "plate_id"],
                              centered_cosine=cosine[0], cosine_reason=reason[0],
                              delta_disagreement_rmse=np.sqrt(np.mean((deltas[a] - deltas[b]) ** 2))))
    return pd.DataFrame(pairs)


def main():
    parser = arguments(__doc__)
    parser.add_argument("--legacy-halves", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    source = args.assets / "source"
    panel = table(args.assets / "development/final_panel.tsv").token_id.to_numpy()
    rows, values = load_data(source, "treated", panel)
    out = args.assets / "repeat_diagnostics"
    out.mkdir(exist_ok=False)
    pairs = pd.concat([physical_rows(source, rows, values, panel, role) for role in ("A", "B")], ignore_index=True)
    tech = technical(rows, args.legacy_halves)
    pairs.to_csv(out / "physical.tsv", sep="\t", index=False, na_rep="NA")
    tech.to_csv(out / "technical.tsv", sep="\t", index=False, na_rep="NA")
    R = [dict(scope="all", role=role, same_plate=bool(same), pairs=len(part),
              groups=part[GROUP].nunique(), cosine_median=part.centered_cosine.median(),
              delta_rmse_median=part.delta_disagreement_rmse.median())
         for (role, same), part in pairs.groupby(["role", "same_plate"])]
    for line, name in cfg["data"]["liver"].items():
        for (role, same), part in pairs[pairs[LINE].eq(line)].groupby(["role", "same_plate"]):
            R.append(dict(scope=name, role=role, same_plate=bool(same), pairs=len(part),
                          groups=part[GROUP].nunique(), cosine_median=part.centered_cosine.median(),
                          delta_rmse_median=part.delta_disagreement_rmse.median()))
    pd.DataFrame(R).to_csv(out / "summary.tsv", sep="\t", index=False, na_rep="NA")
    treated = checked_table(source, "treated_rows.tsv")
    write_json(out / "receipt.json", dict(
        metadata_eligible_source_rows=int(treated.eligible_cell_count.sum()),
        technical_rows=len(tech), technical_groups=tech[GROUP].nunique(),
        technical_cosine_median=float(tech.cosine.median()),
        technical_normalization="legacy R6 mean of per-cell log1p CP10K on legacy panel",
        physical_normalization="R11 log1p CP10K of per-well counts on final source-chosen panel",
        physical_scope="all metadata/technical-eligible source pairs of the same entity and line, contrasted within their own plate pools",
        independent_culture_confirmed=False, legacy_hash=sha256(args.legacy_halves / "treated_halves.npy"),
        files={f.name: sha256(f) for f in sorted(out.iterdir())}
    ))


if __name__ == "__main__":
    main()
