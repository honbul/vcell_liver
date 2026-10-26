"""Actual official GO-filtered Norman population; reconcile source and used cell counts."""
import pickle
import anndata as ad
from gears.utils import filter_pert_in_go
from research_core import *

root=ASSETS/'official/GEARS';source=root/'norman/perturb_processed.h5ad';a=ad.read_h5ad(source,backed='r');conditions=a.obs.condition.astype(str)
with open(root/'essential_all_data_pert_genes.pkl','rb') as f:essential=pickle.load(f)
with open(root/'gene2go_all.pkl','rb') as f:gene2go=pickle.load(f)
pert_names=np.array(sorted({gene for gene in essential if gene in gene2go}));keep=conditions.map(lambda value:filter_pert_in_go(value,pert_names)).to_numpy();run=json.loads((REPORT/'runs/R01_norman_simulation_seed1_v2/run.json').read_text());assert int(keep.sum())==run['n_cells']==89357 and a.n_obs==91205
split=json.loads((ASSETS/'runs/R01_norman_simulation_seed1_v2/split.json').read_text());test=keep&conditions.isin(set(split['test'])-{'ctrl'}).to_numpy();assert int(test.sum())==run['n_test_cells']==28754
counts=conditions[~keep].value_counts();out=REPORT/'official_norman_population';out.mkdir(exist_ok=True);counts.rename_axis('excluded_condition').rename('n_cells').to_csv(out/'excluded_conditions.tsv',sep='\t')
assert keep[conditions.to_numpy()=='ctrl'].all()
write_json(out/'audit.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),input_data_sha256=run['data_sha256'],upstream_filter_source_sha256=sha(ROOT/'third_party/GEARS/gears/utils.py'),upstream_loading_source_sha256=sha(ROOT/'third_party/GEARS/gears/pertdata.py'),n_source_cells=a.n_obs,n_official_GO_eligible_cells=int(keep.sum()),n_excluded_cells=int((~keep).sum()),n_genes=a.n_vars,n_test_interventions=run['n_test_interventions'],n_test_cells=int(test.sum()),n_perturbation_graph_nodes=len(pert_names),control_cells_retained=True,
    reason='Exact pinned GEARS filter_pert_in_go with original essential gene list intersected with authentic GO prior; source91205 and analysis89357 are distinct counts',baseline_comparability='Matched baseline and evaluator use the exact saved official condition split; control population unchanged. Evaluator separately verifies actual GEARS test truth means against baseline observations.'))
a.file.close();print('NORMAN_POPULATION_AUDIT',int(keep.sum()),'test',int(test.sum()),flush=True)
