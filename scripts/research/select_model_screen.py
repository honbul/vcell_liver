"""Freeze the adapted-model candidate using preregistered inner-validation scores only."""
from research_core import *
import yaml
paths={kind:REPORT/'runs'/f'M{code}_full4096_fold0_seed42_{"v2" if kind=="lora" else "v1"}/selection.json' for kind,code in [('frozen_nn','01'),('partial','02'),('lora','03')]}
selections={kind:json.loads(path.read_text()) for kind,path in paths.items()}
assert all(j['status']=='frozen_before_outer_evaluation' for j in selections.values())
candidates=['partial','lora'];chosen=max(candidates,key=lambda kind:selections[kind]['inner_validation_macro_f1'])
# Outer scores are deliberately never opened by this selector.
result=dict(status='frozen_confirmation_design',selection_rule='largest fold0 seed42 inner-validation fixed-six macroF1 among partial/LoRA; fixed-head frozen NN is the matched control',selected_model=chosen,
    selected_data='D1_donor_balanced',data_selection_source='reports/research_phase2/sampling_screen/audit.json',scores={kind:s['inner_validation_macro_f1'] for kind,s in selections.items()},
    selection_source_hashes={kind:sha(path) for kind,path in paths.items()},confirmation_seeds=[42,43,44],confirmation_folds=[0,1,2,3],
    train_cell_cap=4096,backbone='Geneformer-V2-104M',model_epochs_policy='max3, inner-validation patience2; final refit uses selected epoch count per condition',
    factorial_control='M0=frozen same NN head (M01), M1=selected adapted encoder with identical NN head; M00 linear screen is reported separately',
    is_confirmed_improvement=False,external_final_score_access=False,project_commit=git_commit())
p=ROOT/'configs/research/confirmation_selection_v2.json';assert not p.exists(),'Selection is immutable; no reselection from subsequent outcomes';write_json(p,result)
manifest=ROOT/'manifests/research_phase2/adapted_confirmation_runs_v2.tsv';rows=[]
for seed in [42,43,44]:
 for fold in range(4):
  for factor,variant in [('DM01','D0_natural'),('DM11','D1_donor_balanced')]:
   run=f'{factor}_full4096_fold{fold}_seed{seed}_v1'
   if seed==42 and fold==0 and factor=='DM01':run='M02_full4096_fold0_seed42_v1' if chosen=='partial' else 'M03_full4096_fold0_seed42_v2'
   rows.append(dict(seed=seed,fold=fold,factor=factor,data_variant=variant,model=chosen,run_id=run,shard=len(rows)%2))
pd.DataFrame(rows).to_csv(manifest,sep='\t',index=False)
print('CONFIRMATION_FROZEN',chosen,result['scores'],flush=True)
