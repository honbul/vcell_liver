"""Metric integration fixture only: never read external cells or write scientific run artifacts."""
import contextlib,io,tempfile
import anndata as ad
from research_core import *
import research_core as core

with tempfile.TemporaryDirectory(prefix='vcell_metric_fixture_') as folder:
    core.REPORT=Path(folder)/'reports';core.ASSETS=Path(folder)/'assets'
    obs=pd.DataFrame(dict(cell_id=['fixture1','fixture2','fixture3','fixture4'],patient_id=['known','known','unlabeled','unlabeled'],cell_type_coarse=['B','T/NK','Unknown','Unknown']),index=['fixture1','fixture2','fixture3','fixture4'])
    a=ad.AnnData(obs=obs);labels=['B','T/NK'];pred=np.array(['B','T/NK','B','Unknown'])
    c=dict(labels=labels,evaluation_regime='metric_fixture',dataset_id='synthetic_fixture_not_an_experiment',split_id='fixture',split_manifest='manifests/research_phase2/development_donor4_v2.tsv',seed=42,data_variant='fixture',model_variant='fixture')
    with contextlib.redirect_stdout(io.StringIO()):
        result=core.save_evaluation(a,np.arange(4),pred,None,c,'fixture',dict(model_id='metric_fixture'))
    assert result['metrics']['macro_f1']==1. and result['metrics']['n_known']==2 and result['metrics']['evaluation_coverage']==.5
    report=pd.read_csv(core.REPORT/'runs/fixture/per_class.tsv',sep='\t',index_col=0);assert report.loc['B','precision']==1.,'Unknown truth must not be counted as a B false positive'
    matrix=pd.read_csv(core.REPORT/'runs/fixture/confusion_matrix.tsv',sep='\t',index_col=0);assert matrix.to_numpy().sum()==4 and matrix.loc['Unknown','B']==1
    donor=pd.read_csv(core.REPORT/'runs/fixture/by_donor.tsv',sep='\t').set_index('patient_id');assert pd.isna(donor.loc['unlabeled','macro_f1_common_ontology']) and donor.loc['known','macro_f1_common_ontology']==1.
# Only the check status is persisted, with no invented model scores or external observations.
write_json(REPORT/'metric_contract_check.json',dict(status='verified',project_commit=git_commit(),source_sha256=sha(Path(__file__)),fixture_only=True,external_data_read=False,scientific_runs_written=False,
    checks=['Primary and per-class metrics use only known truth','Full confusion preserves unknown truth and all rows','Unknown-only donor has no invented F1','Coverage counts unknown truth explicitly']))
print('UNKNOWN_METRIC_CONTRACT_VERIFIED',flush=True)
