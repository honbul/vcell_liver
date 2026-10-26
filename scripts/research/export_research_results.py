"""Small task-separated tables from actual immutable runs; never infer missing results."""
from research_core import *

audits={}
path=REPORT/'model_parameter_audit/scvi.json'
if path.exists():audits.update({r['run_id']:r for r in json.loads(path.read_text())['records']})
records=[]
for path in sorted((REPORT/'runs').glob('*/run.json')):
    run=json.loads(path.read_text());assert run['status']=='executed';audit=audits.get(run['run_id'],{});metrics=run.get('metrics',{});params=run.get('trainable_parameters');numeric=params if isinstance(params,(int,float)) else audit.get('trainable_parameters')
    records.append(dict(run_id=run['run_id'],task=run.get('task','unknown_in_original_record'),model_id=run.get('model_id'),regime=run.get('regime'),dataset=run.get('dataset_version'),split_id=run.get('split_id'),seed=run.get('seed'),fold=run.get('fold'),data_variant=run.get('data_variant'),model_variant=run.get('model_variant'),feature_dimension=run.get('feature_dimension'),trainable_parameters=numeric,parameter_count_supplement='model_parameter_audit/scvi.json' if audit else None,n_training=run.get('n_outer_training',run.get('n_training_cells')),n_evaluation=run.get('n_evaluation',run.get('n_test_cells',run.get('n_evaluation_cells'))),macro_f1=metrics.get('macro_f1'),balanced_accuracy=metrics.get('balanced_accuracy'),evaluation_coverage=metrics.get('evaluation_coverage'),prediction_coverage=metrics.get('prediction_coverage'),wall_training_seconds=run.get('wall_training_seconds'),wall_inference_seconds=run.get('wall_inference_seconds'),peak_process_gpu_gib=run.get('peak_process_gpu_gib'),gpu_name=run.get('gpu_name'),run_metadata_path=str(path.relative_to(ROOT)),run_metadata_sha256=sha(path),prediction_sha256=run.get('prediction_sha256'),project_commit=run.get('project_commit'),pretraining_overlap=run.get('pretraining_overlap','unknown_or_not_applicable_see_original_run')))
pd.DataFrame(records).to_csv(REPORT/'run_registry.tsv',sep='\t',index=False)
datasets=[
dict(dataset_id='GSE149614_development_full',task='annotation',species='human',n_cells=63036,n_genes=25712,n_donors=8,role='development_grouped4fold',status='executed',contract='configs/research/protocol_annotation_full.yaml',limitation='HCC02/HCC09 excluded; repeated development inspection is not external final'),
dict(dataset_id='official_scGPT_MS',task='official_annotation_reproduction',species='human',n_cells=21312,n_genes=3000,n_donors=None,role='official_reference7844_query13468',status='executed_modified_reproduction',contract='reports/research_phase2/official_reproduction.md',limitation='Official stored numerical score not matched; compatibility changes recorded'),
dict(dataset_id='official_GEARS_Norman',task='genetic_response',species='human_K562',n_cells=89357,n_genes=5045,n_donors=None,role='official_simulation_seed1',status='executed',contract='reports/research_phase2/runs/R01_norman_simulation_seed1_v2/run.json',limitation='Source91205cells; native GO filtering retains89357; CRISPRa; original DE dictionary retained; not liver or strict HepG2'),
dict(dataset_id='GSE264667_HepG2_strict',task='genetic_response',species='human_HepG2',n_cells=140772,n_genes=2000,n_donors=None,role='unseen_target1714_train214_val214_test',status='executed',contract='reports/research_phase2/hepg2_data_contract/contract.json',limitation='CRISPRi day7; same-gene dual sgRNA; hepatoblastoma-derived, not adult patient HCC; biological replicate unknown'),
dict(dataset_id='HepG2_native_common',task='genetic_response',species='human_HepG2',n_cells=35695,n_genes=3303,n_donors=None,role='matched_native_input191_test_targets_primary1977_genes',status='executed' if (REPORT/'response_evaluation_native_common/provenance.json').exists() else 'matched_models_running',contract='reports/research_phase2/foundation_response_contract/contract.json',limitation='Different coverage/population from full HepG2; four treated training cells per eligible target; parent-training gene panel retained'),
dict(dataset_id='official_nault2021_singleDose',task='chemical_response',species='mouse_liver',n_cells=18865,n_genes=5000,n_donors=None,role='portal_treated1074_holdout_controls_supplied',status='executed_modified_native_flow',contract='reports/research_phase2/runs/C00_scvidr_mouse_TCDD_portal_seed42_v3/data_contract.json',limitation='Source30927cells22213genes; six author nonimmune types retained; TCDD30ug/kg every4days28days; per-mouse assignment unavailable'),
dict(dataset_id='GSE151530_HCC_external_final',task='annotation',species='human',n_cells=50023,n_genes=18667,n_donors=25,role='reserved_external_final',status='executed_external_final' if (REPORT/'external_final/evaluation.json').exists() else 'prepared_unscored',contract='reports/research_phase2/external_metadata/contract.json',limitation='32biopsies grouped25donors; no gold Hepatocyte lineage; malignant/unclassified Unknown, known27212cells; fixed-six maximum5/6')]
pd.DataFrame(datasets).to_csv(REPORT/'dataset_registry.tsv',sep='\t',index=False)
for regime,directory in [('inductive','annotation_full_summary'),('query_adaptive','annotation_full_query_adaptive_summary')]:
    source=REPORT/directory/'pooled_metrics.tsv';frame=pd.read_csv(source,sep='\t');frame.insert(0,'task','annotation');frame.insert(1,'dataset','GSE149614_development_full');frame.insert(2,'regime',regime)
    for i,row in frame.iterrows():
        code={'pca':'A00','scgpt':'A01','geneformer':'A02','geneformer_clcancer':'A05','scvi':'A04'}.get(row.model)
        ids=[f'{code}_full_fold{fold}_seed42_v2' if code else f'S00_liverct_full_fold{fold}_seed42_v4'+('_scanvi_control' if row.model=='LiverCT_SCANVI_control' else '') for fold in range(4)];runs=[json.loads((REPORT/'runs'/rid/'run.json').read_text()) for rid in ids];first=runs[0];feature=first.get('feature_extraction_metadata') or {}
        dimension=first.get('feature_dimension')
        if not code:
            path=REPORT/'model_parameter_audit/liverct.json'
            if path.exists():
                audit=next(r for r in json.loads(path.read_text())['records'] if r['run_id']==ids[0].replace('_scanvi_control',''));dimension=audit.get('feature_dimension');frame.loc[i,'external_reference_registry_cells']=audit.get('reference_registry_n_cells');frame.loc[i,'external_reference_label_categories']=len(audit.get('reference_label_categories',[]));frame.loc[i,'exact_native_head_training_label_count']='unknown; VAE reference registry size is not ensemble fitting sample count';frame.loc[i,'native_head_type']=audit.get('native_broad_classifier_type')
            frame.loc[i,'label_information']='External labeled liver reference; no HCC query truth used. Reference label/cell amount not matched to inductive development training.'
            frame.loc[i,'unlabeled_query_fit']=True;frame.loc[i,'query_adaptation_epochs']=20;frame.loc[i,'training_cell_budget']='External reference checkpoint plus all unlabeled heldout query cells'
        else:
            frame.loc[i,'label_information']='Same six labeled development training donors per fold; pretrained corpus overlap unknown where applicable';frame.loc[i,'unlabeled_query_fit']=False;frame.loc[i,'training_cell_budget']='12000 sampled development training cells per fold'
        frame.loc[i,'feature_dimension']=dimension;frame.loc[i,'n_evaluation_rows_excluded']=0;frame.loc[i,'fold_macro_f1_std']=float(np.std([r['metrics']['macro_f1'] for r in runs],ddof=1));frame.loc[i,'mean_current_process_wall_seconds']=float(np.mean([r['wall_total_seconds'] for r in runs]));frame.loc[i,'resource_scope']='Head fitting only plus separately linked frozen extraction' if feature else 'Fold-specific full pipeline; recovered LiverCT fold0 excludes original adaptation, see resource_scope.md'
        if feature:
            frame.loc[i,'one_shared_frozen_extraction_inference_seconds']=feature.get('wall_inference_seconds');frame.loc[i,'frozen_extraction_gpu']=feature.get('gpu_name');frame.loc[i,'frozen_extraction_precision']=feature.get('precision')
    frame.to_csv(REPORT/f'leaderboard_annotation_{regime}.tsv',sep='\t',index=False)
response=[]
for task in ['norman','hepg2','native_common']:
    path=REPORT/f'response_evaluation_{task}/condition_macro_metrics.tsv'
    if path.exists():
        frame=pd.read_csv(path,sep='\t');frame.insert(0,'dataset',task);frame.insert(0,'task','measured_genetic_response');response.append(frame)
pd.concat(response,ignore_index=True).to_csv(REPORT/'leaderboard_genetic_response.tsv',sep='\t',index=False)
chemical=REPORT/'runs/C00_scvidr_mouse_TCDD_portal_seed42_v3/metrics.tsv'
if chemical.exists():
    frame=pd.read_csv(chemical,sep='\t');frame.insert(0,'dataset','mouse_liver_TCDD_portal_holdout');frame.insert(0,'task','chemical_response');frame.to_csv(REPORT/'leaderboard_chemical_response.tsv',sep='\t',index=False)
hypotheses=json.loads((ROOT/'configs/research/hypotheses_v1.json').read_text())['hypotheses']
h3_path=ROOT/'configs/research/hypothesis_DH3_kg_compression_v1.json'
if h3_path.exists():
    h3=json.loads(h3_path.read_text());result=REPORT/'kg_compression_diagnostic/provenance.json';h3['current_status']='executed_exploratory' if result.exists() else 'preregistered_running';h3['conclusion']='No support for fixed32 compression; see paired effects. Excluded from current external selection.' if result.exists() else 'Await fixed4fold execution';hypotheses.append(h3)
for h in hypotheses:
    if h['hypothesis_id']=='D-H1':h.update(current_status='frozen_control_confirmation_executed',conclusion='Mean gain−.003921945; only2/8 donors nonnegative; preregistered success rule not met. Adapted-model factors tracked separately.')
    elif h['hypothesis_id']=='D-H2':h.update(current_status='screened_not_selected',conclusion='Donor×class balancing lowered heldout inner validation relative to D0 and D1; not promoted or declared improved.')
    elif h['hypothesis_id']=='M-H1':
        complete=(REPORT/'factorial_confirmation/provenance.json').exists();h.update(current_status='confirmation_complete' if complete else 'selected_partial_confirmation_running')
        if complete:
            effects=pd.read_csv(REPORT/'factorial_confirmation/paired_effects.tsv',sep='\t').set_index('effect');m=effects.loc['Model gain'];h['conclusion']=f"Partial chosen only by inner validation.3seed4fold model-only gain {m.pooled_mean_difference:.6f}, paired donor95CI[{m.donor_bootstrap_lower95:.6f},{m.donor_bootstrap_upper95:.6f}],{int(m.n_nonnegative_donors)}/8 donors nonnegative. Combined gain and interaction remain separate; no clear confirmed adapter benefit. Corrected LoRA v2 retained; v1 excluded."
        else:h['conclusion']='Partial chosen by inner validation; await complete confirmation; corrected-seed LoRA v2 included, v1 excluded.'
pd.DataFrame(hypotheses).to_csv(REPORT/'hypothesis_registry.tsv',sep='\t',index=False)
write_json(REPORT/'registry_provenance.json',dict(status='exported_actual_available_results',project_commit=git_commit(),n_immutable_executed_runs=len(records),source_sha256=sha(Path(__file__)),missing_values='Blank means unavailable/not applicable in original record, never an invented measurement',separation='Different tasks, information regimes, training budgets and response gene/target panels remain explicitly separated'))
print('RESEARCH_TABLES_EXPORTED',len(records),flush=True)
