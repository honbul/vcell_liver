"""Read-only metadata supplements; keep original execution records immutable and sources explicit."""
import joblib
from research_core import *

audits={};liver={}
for name,destination in [('scvi',audits),('liverct',liver)]:
    path=REPORT/'model_parameter_audit'/f'{name}.json'
    if path.exists():
        for record in json.loads(path.read_text())['records']:
            destination[record['run_id']]=record
            if 'paired_control_run_id' in record:destination[record['paired_control_run_id']]=record
records=[];parameter_rows=[]
for path in sorted((REPORT/'runs').glob('*/run.json')):
    run=json.loads(path.read_text());assert run['status']=='executed';rid=run['run_id'];resolved=dict(run);evidence=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path))];source_contract=run.get('source_contract',{});feature=run.get('feature_extraction_metadata') or run.get('feature_metadata') or {}
    task=run.get('task','unknown');annotation=task=='annotation';probe=any(word in (task+' '+rid).lower() for word in ['probe','smoke','replay'])
    resolved['original_run_metadata_path']=str(path.relative_to(ROOT));resolved['original_run_metadata_sha256']=sha(path);resolved['scientific_performance_run']=not probe and task not in ['annotation_final_fit']
    resolved.setdefault('dataset_version',run.get('dataset_id',source_contract.get('dataset_id')))
    resolved.setdefault('data_variant','Original prepared dataset; see linked source contract')
    resolved.setdefault('model_variant',run.get('model_id','See original execution source'))
    resolved.setdefault('label_mapping','Not applicable: continuous gene-expression response' if 'response' in task else 'See original task labels; no inferred replacement labels')
    resolved.setdefault('pretraining_overlap','unknown');resolved.setdefault('base_checkpoint_revision',feature.get('checkpoint_revision',run.get('upstream_commit')))
    resolved.setdefault('input_contract',feature.get('input_contract'))
    if rid.startswith('R00_ms_'):
        resolved.setdefault('model_id','scGPT_whole_human_official_MS_annotation');resolved.setdefault('split_id','official_MS_reference_query_with_original_reference_random_validation_seed0')
        resolved['input_contract']='Official provided3000-gene MS matrices; original tutorial preprocessing, sequence length3001, label ontology and split; compatibility changes in official_reproduction.md'
        resolved['label_mapping']='Original official MS celltype categories, not HCC six-lineage mapping'
        resolved.setdefault('allowed_target_information','Original official annotation tutorial information contract, own query counts and labels for evaluation; not HCC grouped-donor validation')
        evidence.append(dict(path='manifests/research_phase2/official_scgpt_ms.json',sha256=sha(ROOT/'manifests/research_phase2/official_scgpt_ms.json')))
    elif rid.startswith('R01_norman_'):
        resolved.setdefault('allowed_target_information','Official processed release including original DE dictionaries; native train/validation/test protocol, not strict HepG2 policy');resolved['input_contract']='Official K562 CRISPRa normalized5045-gene release; native GEARS GO filtering, simulation seed1, hidden64; original DE dictionary retained'
    elif rid.startswith('B00_B02_'):
        native='native_common' in rid;norman='norman' in rid;root=ASSETS/'response'/('hepg2_foundation_common_v1' if native else 'hepg2_v1')
        resolved.setdefault('regime',source_contract.get('regime') or ('strict_unseen_target_matched_native_panel' if native else 'official_reproduction_matched_baselines' if norman else 'strict_unseen_target_with_matched_controls'));resolved.setdefault('split_id',source_contract.get('split_sha256',source_contract.get('target_split_sha256','target_roles_in_linked_contract')))
        resolved['input_contract']='Actual intervention-level mean expression; authentic external GO multi-hot features, train-only response PCA32 and validation-selected ridge; no test response used in features'
        resolved.setdefault('allowed_target_information','Declared target identity/external GO prior and provided matched controls; original Norman DE release retained only in official reproduction' if norman else 'Declared target identity/external GO prior and provided same-gem_group controls; heldout treated means evaluator-only')
        model_path=ASSETS/'runs'/rid/'go_ridge.joblib';model=joblib.load(model_path);ridge=model['ridge'];pca=model['pca'];ngenes=int(pca.components_.shape[1]);counts={'B00_no_change':0,'B01_training_mean_delta':ngenes,'B02_ridge_coefficients_and_intercept':int(ridge.coef_.size+ridge.intercept_.size),'B02_PCA_components_and_mean':int(pca.components_.size+pca.mean_.size)}
        resolved['trainable_parameters']=counts;resolved['trainable_parameter_scope']='Statistical fitted component sizes, not a neural network count; B01 mean vector, B02 ridge and PCA separately; combined source run shares fitting cost'
        resolved['prediction_path']={model_id:str(ASSETS/'runs'/rid/f'{model_id}_predictions.npz') for model_id in ['B00_no_change','B01_train_mean','B02_GO_PCA_ridge']};resolved['saved_baseline_model_sha256']=sha(model_path)
        parameter_rows.extend(dict(run_id=rid,component=component,n_parameters=value,model_sha256=resolved['saved_baseline_model_sha256']) for component,value in counts.items())
    elif rid.startswith('B03_') or rid.startswith('B04_'):
        native='native_common' in rid or 'native_response_smoke' in rid;contract_path=ASSETS/'response'/('hepg2_foundation_common_v1' if native else 'hepg2_v1')/'contract.json';contract=json.loads(contract_path.read_text())
        if not resolved.get('dataset_version'):resolved['dataset_version']=contract['dataset_id']
        resolved.setdefault('split_id','target_split_sha256:'+sha(contract_path.parent/'target_split.tsv'));resolved.setdefault('allowed_target_information','Declared heldout target identity/external GO or pretrained vocabulary and supplied matched controls; no heldout treated-response fitting')
        resolved['input_contract']='Strict HepG2 observed count-derived log1p10k expression and saved gene order; target-group split. Native common uses3303 input/output genes and1977 primary evaluation genes; full main uses2000 genes' if native else 'Strict HepG2 train-derived2000-gene log1p10k panel; authentic GO prior; same-batch supplied controls; no heldout DE loss or graph fitting'
        evidence.append(dict(path=str(contract_path),sha256=sha(contract_path)))
    elif rid.startswith('C00_scvidr_'):
        resolved['input_contract']='Official mouse raw counts normalized10k/log1p pointwise; six author nonimmune types; train-only5000HVGs; all portal-treated cells excluded before HVG/VAE fit'
        resolved.setdefault('allowed_target_information','Known target celltype and its vehicle controls; same TCDD30ug/kg28day response in other celltypes; no target treated cells during fit')
        evidence.append(dict(path=str(path.parent/'data_contract.json'),sha256=sha(path.parent/'data_contract.json')))
    elif annotation:
        if run.get('model_id')=='pca':resolved['input_contract']='Raw counts normalized10k/log1p; separate inner/outer train-only variance2000HVGs/scaler/PCA50; fixed head family'
        if 'LiverCT' in str(run.get('model_id')):resolved['input_contract']='Original external reference feature alignment; duplicate symbols summed, missing genes raw0; unlabeled query20epoch adaptation then native SVM or same-adapted SCANVI head'
        if not resolved.get('input_contract') and 'geneformer' in str(run.get('model_id')).lower():resolved['input_contract']='Official V2 raw-count/Ensembl rank tokenizer, CLS/EOS4096 and hidden_states[11] CLS; exact frozen/partial/LoRA scope, precision and source feature path in original record'
        if not resolved.get('input_contract') and run.get('model_id')=='scgpt':resolved['input_contract']='Official whole-human frozen CLS/L2 representation, DataCollator gene sampling length1200; raw-count/vocabulary alignment from linked source feature run'
    if task=='annotation_final_fit':
        resolved['dataset_version']='GSE149614_development_full';resolved.setdefault('split_id','all8_development_donors_final_fit_no_external');resolved.setdefault('allowed_target_information','Development counts/labels only; external_data_accessed=False in original fit record')
        if not resolved.get('input_contract'):resolved['input_contract']=feature.get('input_contract','Original fixed development preprocessing/tokenization; source final-fit script and serialized bundle define exact pipeline')
    if rid in audits:
        audit=audits[rid];resolved['trainable_parameters']=audit['trainable_parameters'];resolved['trainable_parameter_scope']='Actual saved scVI module plus fitted logistic head';evidence.append(dict(path='reports/research_phase2/model_parameter_audit/scvi.json',sha256=sha(REPORT/'model_parameter_audit/scvi.json')))
    if rid in liver:
        audit=liver[rid];resolved['trainable_parameters']=audit['requires_grad_parameters_under_reconstructed_query_contract'];resolved['trainable_parameter_scope']='Requires_grad tensor scalars under reconstructed query contract; gradient hooks mask columns, so upper bound on updated entries. Native reference SVM fixed. Adaptation shared with paired control.';evidence.append(dict(path='reports/research_phase2/model_parameter_audit/liverct.json',sha256=sha(REPORT/'model_parameter_audit/liverct.json')))
    if 'gpu_name' not in resolved:
        resolved['gpu_name']=None;resolved['gpu_measurement_scope']='No GPU measurement in this original process record; frozen feature-extraction GPU, when used, is separately linked'
        if feature:resolved['feature_extraction_gpu_name']=feature.get('gpu_name')
    for key in ['peak_cuda_allocated_gib','peak_cuda_reserved_gib','peak_process_gpu_gib']:resolved.setdefault(key,None)
    resolved.setdefault('metric_protocol','See original run metrics plus fixed-label annotation or response_evaluation_* evaluator provenance; different gene/target universes are separate tasks')
    resolved.setdefault('training_donors','Not applicable or not retained in original task; see source split/role manifests');resolved.setdefault('validation_donors','See source split/role manifests');resolved.setdefault('evaluation_donors_or_perturbations','See source split/role manifests and recorded prediction target IDs')
    resolved.setdefault('failure_or_limitations',['Integration/replay only, no full performance claim'] if probe else ['See linked task/data/evaluation contracts; unavailable fields remain explicit rather than invented'])
    resolved['metadata_resolution_evidence']=evidence;resolved['original_execution_record_changed']=False;records.append(resolved)
write_json(REPORT/'resolved_run_contracts.json',dict(status='derived_metadata_supplement',project_commit=git_commit(),source_sha256=sha(Path(__file__)),n_records=len(records),policy='Original immutable executions plus explicit source-backed additions. Null resource fields mean unavailable/not applicable, never fabricated zero. Probe records are separate from scientific performance runs.',records=records))
pd.DataFrame(parameter_rows).to_csv(REPORT/'model_parameter_audit/statistical_baselines.tsv',sep='\t',index=False)
print('RUN_METADATA_RESOLVED',len(records),flush=True)
