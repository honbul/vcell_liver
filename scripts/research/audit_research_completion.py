"""Requirement-level closing evidence from verified executions and final reports; no model or data mutation."""
import csv,hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/research_phase2';requirements=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((REPORT/p).read_text())
def run(name):
    value=read('runs/'+name+'/run.json');assert value['status']=='executed';return value
def record(name,paths,conclusion):
    assert all((REPORT/p).exists() for p in paths),name
    requirements.append(dict(requirement=name,status='verified',evidence=[dict(path=p,sha256=sha(REPORT/p)) for p in paths],conclusion=conclusion))

base='243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79';legacy=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',base,'envs'],text=True).splitlines()+[str(p.relative_to(ROOT)) for p in (ROOT/'notebooks').glob('0[0-5]*.ipynb')]+['scripts/common.py','scripts/geneformer_utils.py']
assert not subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',base,'--',*legacy],text=True)
assert run('R00_ms_full_seed0_v2')['reproduction_status']=='modified_reproduction';run('R01_norman_simulation_seed1_v2');assert read('official_norman_population/audit.json')['n_official_GO_eligible_cells']==89357
assert all(read('official_scgpt_gap_audit/source_comparison.json')['checks'].values())
record('V2_official_execution_and_V3_7_bounded_gap_check',['official_reproduction.md','runs/R00_ms_full_seed0_v2/run.json','runs/R01_norman_simulation_seed1_v2/run.json','official_norman_population/audit.json','official_scgpt_gap_audit/audit.json','official_scgpt_gap_audit/source_comparison.json'],'Official flows executed. MS numerical mismatch and unseparated cause retained; no quantitative-reproduction success claim.')
for code in ['A00','A01','A02','A04','A05']:
    records=[run(f'{code}_full_fold{fold}_seed42_v2') for fold in range(4)];assert sum(v['n_evaluation'] for v in records)==63036 and all(v['regime']=='inductive' and v['n_outer_training']==12000 for v in records)
assert read('scfoundation_access.json')['status']=='blocked_official_weight_download'
record('V2_basic_annotation_or_concrete_scFoundation_blocker',['annotation_full_summary/provenance.json','annotation_uncertainty_inductive/provenance.json','leaderboard_annotation_inductive.tsv','scfoundation_access.json'],'Five real fourfold pipelines; scFoundation is a documented official-weight-access blocker, not an executed score.')
for suffix in ['', '_scanvi_control']:
    records=[run(f'S00_liverct_full_fold{fold}_seed42_v4{suffix}') for fold in range(4)];assert sum(v['n_evaluation'] for v in records)==63036 and all(v['regime']=='query_adaptive_reference_mapping' for v in records)
record('V2_specialized_models_and_information_regimes',['model_parameter_audit/liverct.json','liverct_output_audit/provenance.json','liverct_native_head.md','leaderboard_annotation_query_adaptive.tsv'],'CLcancer executed in the inductive comparison; LiverCT reference/query adaptation and ontology coverage remain separate.')
site=read('tissue_error_analysis/provenance.json');context=read('tissue_error_analysis/context_provenance.json');assert site['metadata_mapping']['n_exact_value_matches']==63036 and context['within_context_comparison']['n_supported_contexts']==59 and not read('tissue_error_analysis/metadata_schema_resolution.json')['conflicts']
record('V3_2_site_mapping_and_existing_OOF_errors',['tissue_error_analysis/provenance.json','tissue_error_analysis/metadata_schema_resolution.json','tissue_error_analysis/canonical_schema_mapping_observation.json','tissue_error_analysis/support_by_tissue_site.tsv','tissue_error_analysis/errors_by_donor_tissue_lineage.tsv','tissue_error_analysis/within_donor_tissue_comparisons.tsv','tissue_error_analysis/context_provenance.json','development_insights.md'],'Original site values retained across all63036cells; donor/site/lineage and UMI/B error strata use existing predictions, without causal claims.')
confirmation=read('factorial_confirmation/provenance.json');assert confirmation['n_executed_runs']==48
for d in confirmation['sources']:assert sha(REPORT/'runs'/d['run_id']/'run.json')==d['run_sha256']
selection=json.loads((ROOT/'configs/research/confirmation_selection_v2.json').read_text());assert selection['selected_model']=='partial' and selection['confirmation_seeds']==[42,43,44] and not selection['is_confirmed_improvement']
probe=run('M03_lora_module_gradient_reload_probe_v1');assert len(probe['target_modules'])==22 and probe['save_reload_logits_max_abs_difference']==0 and any(s['nonzero_gradient_tensors']>0 for s in probe['gradient_steps'])
for name in ['M01_full4096_fold0_seed42_v1','M02_full4096_fold0_seed42_v1','M03_full4096_fold0_seed42_v2']:run(name)
record('V2_data_model_controls_LoRA_and_V3_4_all48_factors',['sampling_screen/audit.json','runs/M01_full4096_fold0_seed42_v1/run.json','runs/M02_full4096_fold0_seed42_v1/run.json','runs/M03_full4096_fold0_seed42_v2/run.json','runs/M03_lora_module_gradient_reload_probe_v1/run.json','factorial_confirmation/provenance.json','factorial_confirmation/pooled_by_seed.tsv','factorial_confirmation/paired_effects.tsv','factorial_confirmation/per_class_mean_std.tsv','factorial_confirmation/resources_by_factor.tsv'],'48 original executions complete; candidate selection is unchanged. LoRA integration and model efficacy, development and external effects are distinguished.')
assert run('B03_hepg2_strict_unseen_target_seed42_v1')['n_test_interventions']==214
for name in ['B04_pretrained_native_common_seed42_v1','B04_random_native_common_seed42_v1','B03_GEARS_native_common_seed42_v1']:
    d=run(name);assert d['n_test_interventions']==191 and d['n_primary_genes']==1977
native=read('response_evaluation_native_common/provenance.json');assert len(native['model_ids'])==6 and native['n_cells']==12839
record('V2_measured_response_and_V3_4_native_comparison',['hepg2_data_contract/contract.json','response_evaluation_hepg2/provenance.json','response_evaluation_native_common/provenance.json','response_evaluation_native_common/condition_macro_metrics.tsv','response_evaluation_native_common/paired_model_comparisons.tsv'],'Full214-target and matched191-target comparisons completed. Norman K562, HepG2 parent and common panels remain separate.')
assert run('C00_scvidr_mouse_TCDD_portal_seed42_v3')['n_evaluation_cells']==1074
record('V2_conditional_chemical_route',['runs/C00_scvidr_mouse_TCDD_portal_seed42_v3/run.json','leaderboard_chemical_response.tsv','data_card.md'],'Executed same-TCDD mouse context transfer only; no human efficacy or animal-replicate confidence interval claim.')
sens=read('control_sensitivity/aggregation_provenance.json');p=read('control_sensitivity/native_first_target.json');assert sens['n_test_targets']==191 and p['n_models']==2 and p['n_treated_cells']==38 and not p['new_training']
record('V3_6_full_mean_and_sampled_control_sensitivity',['control_sensitivity.md','control_sensitivity/aggregation_provenance.json','control_sensitivity/aggregation_metrics.tsv','control_sensitivity/simple_predictor_resampling.tsv','control_sensitivity/native_first_target.json'],'Original predictions preserved;191-target additive baseline sensitivity and bounded saved-model probe completed. Mean-MSE selection, mean-RMSE reporting and top20 versus FDR DEG distinguished.')
verification=read('external_final/completion_verification.json');evaluation=read('external_final/evaluation.json');closed=read('external_final/attempts/external_categorical_recovery_v1/completed.json');assert verification['status']=='verified_original_completed_batch' and verification['source_sha256']==sha(ROOT/'scripts/research/audit_external_completion.py')
assert verification['evaluation_sha256']==closed['evaluation_sha256']==sha(REPORT/'external_final/evaluation.json') and verification['freeze_sha256']==closed['freeze_sha256']==sha(REPORT/'external_final/freeze.json') and not closed['model_or_data_selection_changed']
assert verification['n_candidates']==8 and evaluation['n_known']==27212 and evaluation['n_unknown_truth']==22811 and evaluation['primary_pipeline']=='geneformer_frozen_linear_12000'
for d in verification['records']:assert sha(REPORT/'runs'/f"E00_{d['candidate']}_GSE151530_seed42_v1"/'run.json')==d['external_run_metadata_sha256']
assert read('external_final/current_status.json')['status']=='evaluated'
record('V3_1_completed_jobs_and_V3_3_immutable_release',['external_final/current_status.json','external_final/attempts/external_categorical_recovery_v1/completed.json','external_final/completion_verification.json','external_release_dependencies.json','slurm_jobs.tsv'],'Original8-model batch completed without duplicate inference or release regeneration. Optional KG gate removed; original frozen inputs/fits/predictions retained. Old scheduler handle expiry is explicit.')
record('V2_external_HCC_and_V3_5_five_nonmalignant_lineages',['external_final/evaluation.json','external_final/metrics.tsv','external_final/paired_donor_bootstrap.tsv','external_final/resource_summary.tsv','external_final.md'],'Eight fixed candidates verified; original fixed-six/five,54.40%known truth coverage and22811Unknown retained. No gold hepatocyte/malignant-lineage validation.')
notebooks=read('tutorial_execution.json');assert len(notebooks)==5
for d in notebooks:
    p=ROOT/d['notebook'];assert sha(p)==d['sha256'] and d['status']=='executed';code=[c for c in json.loads(p.read_text())['cells'] if c['cell_type']=='code'];assert all(c['execution_count'] is not None for c in code) and not any(o['output_type']=='error' for c in code for o in c['outputs'])
record('V3_8_final_reports_single_index_and_existing_tutorials',['RESULTS_INDEX.md','final_summary.md','progress_review_resolution.md','tutorial_execution.json'],'Final reports include actual external results, unchanged primary, separated budgets/tasks, uncertainty/cost and one next experiment. Existing06–10 only; changed06/08/09 ran in fresh kernels,07/10 retain verified outputs.')
for p in [ROOT/'README.md',REPORT/'RESULTS_INDEX.md',REPORT/'final_summary.md',REPORT/'external_final.md',REPORT/'progress_review_resolution.md']:
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if not target.startswith(('http:','https:','#')):assert (p.parent/target).exists(),(p.name,target)
result=dict(status='verified_requested_execution_and_reporting_scope',verified_at_utc=datetime.now(timezone.utc).isoformat(),project_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),branch='research/phase2-v2',objective='V2 research plan with all8 V3 corrections and completed original external final; not narrowed to reading the attachment',original_V1_tracked_notebooks_environment_definitions_and_core_helpers_unchanged=True,requirements=requirements,unexecuted_or_unresolved_limits=[dict(item='scFoundation official checkpoint',status='blocked_documented',reason='Both official weight links403; example archive lacks weights'),dict(item='Official scGPT MS numerical agreement and cause attribution',status='not_claimed',reason='Modified reproduction .67558 vs stored notebook .732; bounded checks did not identify a cause'),dict(item='Human HCC efficacy, gold hepatocyte/malignant external truth and biological replicate response validation',status='outside_measured_evidence',reason='Explicit data/endpoint limitations')],publication='Final commit/push and exact remote SHA equality checked after committing this artifact; see repository history and final completion response')
(REPORT/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('REQUIREMENT_AUDIT_VERIFIED',len(requirements),'groups; five research notebook hashes/outputs valid; V1 preserved; external8 original batch verified')
