"""Document every required HepG2 cell-contract field, including explicit missing sample IDs."""
import anndata as ad
from research_core import *

records=[]
for dataset in ['hepg2_v1','hepg2_foundation_common_v1']:
    root=ASSETS/'response'/dataset;contract=json.loads((root/'contract.json').read_text());a=ad.read_h5ad(root/'trainval_controls.h5ad',backed='r');columns=set(a.obs.columns)
    fields={
        'cell_id':dict(source='obs_names and original cell_id',status='observed'),
        'dataset_id':dict(source='immutable prepared contract dataset_id',value=contract['dataset_id'],status='known_dataset_constant'),
        'cell_line':dict(source='Author HepG2-specific source file, GEO and study Methods; obs.cell_line',value='HepG2',status='known_dataset_constant'),
        'sample_id':dict(source='No explicit biological sample identity in supplied metadata; gem_group is not substituted',value='unknown',status='unknown'),
        'replicate_id':dict(source='obs.replicate_id; original per-biological-replicate assignment unavailable',value='unknown',status='unknown'),
        'intervention_type':dict(source='Study Methods and obs.intervention_type',value='CRISPRi',status='known_dataset_constant'),
        'target_gene_ids':dict(source='obs.gene_id validated Ensembl for treated targets; obs.target supplies author symbol',status='observed',control='not_applicable_non_targeting'),
        'guide_ids':dict(source='obs.sgID_AB, author-filtered assigned guide string/pair; transcript variants grouped by target',status='observed'),
        'control_type':dict(source='obs.role==control and obs.control_type',value='non-targeting within same gem_group',status='observed'),
        'time':dict(source='Study day7 readout and obs.time_days',value=7,units='days',status='known_dataset_constant'),
        'batch':dict(source='obs.gem_group and obs.batch; technical library batch',status='observed'),
        'library_size':dict(source='obs.library_size_panel9624, row sum before output-panel selection; author UMI_count retained separately',status='observed')}
    for name in ['gene_id','sgID_AB','role','target','batch','library_size_panel9624','replicate_id','time_days','intervention_type','control_type']:assert name in columns,(dataset,name)
    assert a.obs_names.is_unique and set(a.obs.role.astype(str))=={'train','val','control'}
    records.append(dict(dataset_id=contract['dataset_id'],physical_trainval_path=str(root/'trainval_controls.h5ad'),prepared_contract_sha256=sha(root/'contract.json'),n_cells_in_physical_trainval_file=a.n_obs,n_genes=a.n_vars,fields=fields,raw_counts='layers[counts] of selected observed output genes; denominator from original9624-gene panel; source complete matrix retained separately',expression='X is prepared log1p10k expression; not mislabeled raw counts',guide_efficiency='unknown; no invented efficiency threshold',multiple_infection='Use author-filtered assignment; do not infer a second gene from same-gene dual sgRNA design',heldout_policy='Test-treated cells stored physically separately; not present in this inspected training/validation file'))
    a.file.close()
write_json(REPORT/'response_cell_schema.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),records=records));print('RESPONSE_CELL_SCHEMA_AUDITED',flush=True)
