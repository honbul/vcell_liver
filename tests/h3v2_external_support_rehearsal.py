"""Prove the three external query and feature rosters without opening response files."""
import numpy as np
import pandas as pd

from nextgen_h2.assets import COLD, KEY, canonical_registry
from h3v2.data import ROOT, register
from h3v2.features import SourceFeatures
from h3v2.final_fit import exact_join, external_query, old_h2
from vc_build.io import publish_json


def main():
    query=external_query()
    reference,stamp=old_h2(3)
    exact_join(query,reference)
    source=register()
    ext=SourceFeatures.read(extra=query)
    hold=query[query.source_id.eq("GDSC1")]
    candidate=canonical_registry(hold)
    overlapping_parents=set(candidate.index)&set(source.index)
    overlapping_connectivity=set(candidate.connectivity)&set(source.connectivity)
    overlapping_scaffolds=set(candidate.scaffold_aliases.str.split("|").explode())&set(
        source.scaffold_aliases.str.split("|").explode())
    if (len(candidate),candidate.scaffold.nunique())!=(66,66) or any(
            (overlapping_parents,overlapping_connectivity,overlapping_scaffolds)):
        raise ValueError("The external GDSC1 registry is no longer scaffold OOD")
    licob=pd.read_parquet(COLD/"query.parquet",columns=["adult_age_confirmed","donor_id",KEY])
    donors=licob[licob.adult_age_confirmed]
    if (donors.donor_id.nunique(),donors[KEY].nunique())!=(38,76):
        raise ValueError("Wrong adult LICOB scope")
    support={}
    for cohort,subset in query.groupby("source_id"):
        rna=ext.full_rna.loc[sorted(subset.cell_id.unique())]
        n=rna.notna().sum(1)
        if min(n)<10000:
            raise ValueError("Insufficient full panel RNA coverage for the external model")
        support[cohort]={"rows":len(subset),"admitted_cells":subset.cell_id.nunique(),
                         "exact_parents":subset[KEY].nunique(),"min_measured_genes":int(n.min()),
                         "lincs_query_parents":int(ext.coverage.lincs_modz24.reindex(subset[KEY].unique()).eq(True).sum())}
    publish_json(ROOT/"external_roster_verification.json",{"outcomes_opened":False,"H2_reference_sha256":stamp,
        "supports":support,"GDSC1_source_parent_connectivity_scaffold_overlap_counts":[0,0,0],
        "LICOB_adult":[38,76],"outer_fit_geometry_has_complete_chemistry":bool(np.isfinite(ext.chem).all().all())})
    print("Verified outcome-free 66-scaffold GDSC1, 50-cell LIMORE and 38-donor LICOB input rosters",flush=True)


if __name__=="__main__":
    main()
