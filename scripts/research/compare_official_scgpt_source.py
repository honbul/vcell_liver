"""Bounded AST supplement: verify unchanged optimization/checkpoint-selection logic without rerunning it."""
import ast
from research_core import *

out=REPORT/'official_scgpt_gap_audit/source_comparison.json';assert not out.exists();nb=ROOT/'third_party/scGPT/tutorials/Tutorial_Annotation.ipynb';original='\n\n'.join(''.join(c['source']) for c in json.loads(nb.read_text())['cells'] if c['cell_type']=='code');executed=ASSETS/'official/scGPT_runs/R00_ms_full_seed0_v2/official_annotation_patched.py';patched=executed.read_text();before=ast.parse(original);after=ast.parse(patched)
def same(nodes1,nodes2):return bool(nodes1) and bool(nodes2) and [ast.dump(n,include_attributes=False) for n in nodes1]==[ast.dump(n,include_attributes=False) for n in nodes2]
checks={}
for name in ['train','evaluate']:
    checks[name+'_function_AST_equal']=same([n for n in ast.walk(before) if isinstance(n,ast.FunctionDef) and n.name==name],[n for n in ast.walk(after) if isinstance(n,ast.FunctionDef) and n.name==name])
checks['epoch_loop_and_checkpoint_selection_AST_equal']=same([n for n in before.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch'],[n for n in after.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch'])
for name in ['accuracy_score','precision_score','recall_score','f1_score']:
    checks[name+'_call_AST_equal']=same([n for n in ast.walk(before) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name],[n for n in ast.walk(after) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name])
assert all(checks.values());write_json(out,dict(status='executed_read_only_source_comparison',project_commit=git_commit(),source_sha256=sha(Path(__file__)),original_notebook_sha256=sha(nb),executed_script_sha256=sha(executed),checks=checks,checkpoint_selection='Unchanged original best_model selected by lower validation val_loss; not reselected by test macroF1',new_training=False,conclusion='Optimization, checkpoint-selection loop and metric calls unchanged in this limited source comparison; backend/loader/runtime contribution to numerical gap remains unidentified'))
print('OFFICIAL_SOURCE_COMPARISON_COMPLETE',checks,flush=True)
