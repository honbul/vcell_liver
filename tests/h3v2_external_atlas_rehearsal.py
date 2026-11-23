"""Build the external feature bank and verify OOF offsets without response queries."""

import numpy as np

from h3v2.atlas import read, save
from h3v2.data import ROOT
from h3v2.features import SourceFeatures
from h3v2.final_fit import external_data, fixed_kernel
from vc_build.io import publish_json, sha256


def main():
    fit, query = external_data()
    if len(fit) != 670830 or fit.compound_id.nunique() != 1186:
        raise ValueError("External fit must use all registered source observations")
    if set(("endpoint_value", "truth", "observed")) & set(query):
        raise ValueError("External feature construction requires an outcome-free query")
    target = ROOT / "features/external-prepared-v4"
    source = SourceFeatures.read(extra=query)
    save(fit, query, source, target)
    atlas, meta = read(target, fit, query)
    base_fit, base_query, kernel = fixed_kernel("external", fit, query)
    if (len(base_fit), len(base_query)) != (len(fit), len(query)) or not all(
        np.isfinite(values).all() for values in (base_fit, base_query)
    ):
        raise ValueError("External H2 offsets must have one finite score for each exact row")
    if sha256(target / "schema.pkl") != meta["schema_sha256"] or not meta["lincs_full_unclipped"]:
        raise ValueError("External feature schema or full LINCS views could not be reproduced")
    views = ("rna", "activity", "proxy", "paths", "bits", "props", "targets", "target_match")
    if not all(np.isfinite(atlas[key]).all() for key in views):
        raise ValueError("A required common feature has an unregistered missing value")
    measured = atlas["lincs_mask"]
    if any(not np.array_equal(np.isfinite(atlas[key]).all(axis=1), measured)
           for key in ("lincs", "perturb_paths")):
        raise ValueError("Both LINCS views must use the same natural support")
    evidence = {"fit_rows":len(base_fit), "query_rows":len(base_query),
                "cohorts":query.source_id.value_counts().to_dict(), "external_outcomes_opened":False,
                "kernel_outcome_free_sha256":kernel, "atlas":meta}
    publish_json(ROOT / "external_atlas_preflight.json", evidence)
    print("Verified full-source external feature bank and exact finite OOF/query offsets", evidence["cohorts"], flush=True)


if __name__ == "__main__":
    main()
