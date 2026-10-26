"""Verify matched-size, fixed-head, fixed-validation data-only comparisons from executed runs."""
from research_core import *
ids=['D00_full4096_fold0_seed42_v1','D1_donor_balanced_full4096_fold0_seed42_v1','D2_donor_class_balanced_full4096_fold0_seed42_v1']
runs=[json.loads((REPORT/'runs'/name/'run.json').read_text()) for name in ids];base=runs[0];rows=[]
for run in runs:
    assert run['status']=='executed' and run['seed']==42 and run['fold']==0
    for key in ['model_id','checkpoint_sha256','n_inner_training','n_inner_validation','n_outer_training','n_evaluation','validation_manifest_sha256']:
        assert run[key]==base[key],key
    for key in ['selected_C','selected_class_weight']:assert run['head_selection'][key]==base['head_selection'][key],key
    pred=pd.read_csv(run['prediction_path'],sep='\t');reference=pd.read_csv(base['prediction_path'],sep='\t');assert pred.cell_id.tolist()==reference.cell_id.tolist()
    # Verify the recorded manifests against actual files, and training/evaluation donor separation.
    raw=Path(run['prediction_path']).parent;training=pd.read_csv(raw/'outer_training_cell_ids.tsv',sep='\t');val=pd.read_csv(raw/'inner_validation_cell_ids.tsv',sep='\t')
    assert not set(training.cell_id)&set(pred.cell_id)
    assert sha(raw/'outer_training_cell_ids.tsv')==run['training_manifest_sha256'] and sha(raw/'inner_validation_cell_ids.tsv')==run['validation_manifest_sha256']
    rows.append(dict(run_id=run['run_id'],data_variant=run['data_variant'],n_outer_training=run['n_outer_training'],n_evaluation=run['n_evaluation'],inner_validation_macro_f1=run['head_selection']['validation_macro_f1'],development_outer_macro_f1=run['metrics']['macro_f1'],selected_C=run['head_selection']['selected_C'],selected_class_weight=run['head_selection']['selected_class_weight']))
out=REPORT/'sampling_screen';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'metrics.tsv',sep='\t',index=False)
# Select data candidates using the registered inner-validation criterion; no outer score tie breaking.
choice=max(rows[1:],key=lambda row:row['inner_validation_macro_f1'])
write_json(out/'audit.json',dict(status='executed',project_commit=git_commit(),checks=['same seed/fold/checkpoint/head hyperparameters','identical training cell count','identical validation manifest hash','identical ordered evaluation population','no outer-training/evaluation overlap','actual manifest hashes verified'],selected_data_candidate=choice['data_variant'],selection_metric='inner-validation fixed-six macroF1 only',is_confirmed_improvement=False,confirmation='all donor folds and seeds42,43,44 still required',external_final_scores_observed=False))
print(pd.DataFrame(rows).to_string(index=False));print('DATA_CANDIDATE',choice['data_variant'])
