"""Verify actual E4 rosters, frozen H2 replays, PSD Grams and drug-constant offsets."""
import argparse
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from followup_h2.common import panel_hash, parquet_lock, sha256
from nextgen_h2.assets import KEY
from nextgen_h2.run import META, read_json
from nextgen_h2_v2.freeze import gram_receipt
from vc_build.io import publish_json

from .data import COLD, cpu_threads, lincs_profiles, observed, registry, source_metadata, utc_now
from .forecast import (
    E1_COMPONENTS, adult_predictions, frozen_components, licob18, outcomes_locked, read_forecast, selection_lock,
)
from .model import FULL, KERNEL_ONLY
from .split import construct, overlaps


def rank_invariance(panel):
    panel = panel.reset_index(drop=True)
    left, right = panel[KERNEL_ONLY].to_numpy(), panel[FULL].to_numpy()
    count = {"all": 0, "lincs_observed": 0, "lincs_unobserved": 0}
    for rows in panel.groupby(["source_id", "endpoint_id", KEY], sort=False).indices.values():
        if not np.array_equal(rankdata(left[rows]), rankdata(right[rows])):
            raise AssertionError(f"Drug-constant intercept altered ranks for {panel.iloc[rows[0]][KEY]}")
        present = panel.lincs_observed.iloc[rows]
        assert present.nunique() == 1
        count["all"] += 1
        count["lincs_observed" if present.iloc[0] else "lincs_unobserved"] += 1
    return count


def licence():
    note = read_json(COLD / "receipt.json")
    return parquet_lock(COLD / "query.parquet", note["files"]["query.parquet"])


def verify(preflight, selected, forecasts, out):
    cpu_threads()
    split, selection, selection_hash = selection_lock(preflight, selected)
    outcomes_locked()
    reg, native = registry(), source_metadata()

    class Contracts(unittest.TestCase):
        def test_all_nested_screens_and_scaffolds_are_separated(self):
            profiles = lincs_profiles(reg.index)
            rebuilt, _ = construct(reg, native, observed(profiles), observed(profiles, "modz24"))
            self.assertEqual(rebuilt["roles"], split["roles"])
            self.assertEqual(rebuilt["selections"], split["selections"])
            self.assertEqual(rebuilt["evaluations"], split["evaluations"])
            for name, role in split["roles"].items():
                if role["kind"] == "loso":
                    self.assertTrue(set(role["fit_screens"]).isdisjoint(role["query_screens"]))
                self.assertEqual(set(overlaps(reg, role["fit_parents"], role["query_parents"]).values()), {0})
                if name.startswith("T2/") and role["kind"] == "loso":
                    fold = name.split("/")[1]
                    outer = split["roles"][f"T2/{fold}/outer"]["query_parents"]
                    self.assertEqual(set(overlaps(reg, role["fit_parents"] + role["query_parents"], outer).values()), {0})

        def test_frozen_baselines_and_offset_invariance_on_all_supports(self):
            for index, (panel, note) in enumerate(panels):
                track = note["track"]
                if track == "T1":
                    evidence["rank_groups"][f"T1/{note['fold']}"] = rank_invariance(panel)
                    continue
                parts, replay = frozen_components(note["fold"])
                for key in ("F00_max_abs_diff", "F11_max_abs_diff"):
                    self.assertLess(replay[key], 1e-6)
                self.assertEqual(replay, note["frozen_replay"])
                self.assertTrue(panel[META].equals(parts[META]))
                for arm in ("C1", "H2-v1", "H2-v2"):
                    np.testing.assert_array_equal(panel[arm], parts[arm])
                evidence["replays"][str(index)] = replay
                if track == "T2":
                    for screen, rows in panel.groupby("source_id"):
                        evidence["rank_groups"][f"T2/{note['fold']}/{screen}"] = rank_invariance(rows)
                else:
                    for cohort, rows in panel.groupby("source_id"):
                        if cohort == "LICOB":
                            rows = adult_predictions(rows, licence())
                            novel = licob18(rows)
                            self.assertEqual((len(novel), novel[KEY].nunique()), (684, 18))
                            self.assertEqual((len(rows), rows[KEY].nunique()), (2888, 76))
                            evidence["rank_groups"]["T3/LICOB18"] = rank_invariance(novel)
                        evidence["rank_groups"][f"T3/{cohort}"] = rank_invariance(rows)

        def test_selected_grams_and_roster_denominators(self):
            for index, (panel, note) in enumerate(panels):
                if index < len(split["evaluations"]):
                    role = split["roles"][split["evaluations"][index]]
                    self.assertEqual(note["roster_sha256"], role["query_roster_sha256"])
                    self.assertEqual(len(panel), role["query"]["rows"])
                    self.assertEqual(panel.loc[panel.lincs_observed, KEY].nunique(), role["query"]["lincs_parents"])
                path = forecasts / str(index) / "selected_drug_gram.npy"
                pinned = note["selected_drug_gram"]
                self.assertEqual(sha256(path), pinned["sha256"])
                gram = np.load(path, allow_pickle=False)
                check = gram_receipt(torch.from_numpy(gram))
                self.assertGreater(check["minimum_eigenvalue"], 0.)
                self.assertAlmostEqual(check["minimum_eigenvalue"], pinned["minimum_eigenvalue"], places=9)
                self.assertEqual(check["size"], pinned["size"])
                evidence["gram_min_eigenvalues"][str(index)] = pinned
            for track in ("T1", "T2"):
                rows = pd.concat([p for p, n in panels if n["track"] == track]).sort_values("row_id", ignore_index=True)
                self.assertEqual(len(rows), 670830)
                self.assertEqual(panel_hash(rows), split["source_roster_sha256"])
                self.assertEqual(rows[KEY].nunique(), 1186)
                self.assertEqual(rows.loc[rows.lincs_observed, KEY].nunique(), 724)
            ext, note = panels[-1]
            evidence["external_holdout"] = note["external_holdout"]
            self.assertEqual(note["external_holdout"]["GDSC1_transitive_scaffolds"], 66)
            self.assertEqual(set(note["external_holdout"]["source_and_GDSC1_overlap"].values()), {0})
            gdsc = ext.loc[ext.source_id.eq("GDSC1")]
            self.assertEqual((len(gdsc), gdsc[KEY].nunique(), gdsc.cell_id.nunique()), (39545, 66, 713))
            self.assertEqual(gdsc.loc[gdsc.lincs_observed, KEY].nunique(), 48)

    panels = [read_forecast(forecasts / str(i), selection_hash) for i in range(len(split["evaluations"]) + 1)]
    evidence = {"replays": {}, "rank_groups": {}, "gram_min_eigenvalues": {}}
    results = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not results.wasSuccessful():
        raise SystemExit(1)
    out.mkdir(exist_ok=False, parents=True)
    publish_json(out / "test_receipt.json", evidence | {
        "tests": results.testsRun, "status": "PASS", "passed_at_utc": utc_now(), "query_outcomes_opened": 0,
        "frozen_components_directory": str(E1_COMPONENTS), "replay_abs_error_limit": 1e-6,
        "LOSO_Full_minus_Kernel_plus_V2_Int_within_drug_delta": 0.,
        "loso_roles": sum(x["kind"] == "loso" for x in split["roles"].values()),
        "track2_outer_fit_roles_audited": sum(n.startswith("T2/") and r["kind"] == "loso"
                                              for n, r in split["roles"].items()),
        "all_role_overlaps": {n: v["overlap"] for n, v in split["roles"].items()},
        "forecast_receipt_sha256": {str(i): sha256(forecasts / str(i) / "receipt.json") for i in range(len(panels))},
        "split_sha256": selection["split_sha256"], "selection_sha256": selection_hash,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("preflight", "selection", "forecasts", "out"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    verify(args.preflight, args.selection, args.forecasts, args.out)
