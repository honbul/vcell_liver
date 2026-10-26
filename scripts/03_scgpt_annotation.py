"""Official whole-human frozen CLS embeddings; raw counts binned once by collator."""
import json, subprocess
import anndata as ad
import numpy as np
import pandas as pd
import torch
from common import DATA, ROOT, Resources, config, evaluate, provenance, save_features, sha, write_json

c=config();np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(8)
r=Resources()
from scgpt.tasks import embed_data
a=ad.read_h5ad(DATA/'subsets'/f"{c['subset']}.h5ad")
checkpoint=DATA/'checkpoints/scgpt-whole-human'
vocab=json.loads((checkpoint/'vocab.json').read_text())
match=np.array([g in vocab for g in a.var.gene_symbol])
valid=np.asarray((a.layers['counts'][:,match]>0).sum(1)).ravel()
assert (valid>0).all()
# Upstream embed_data uses only X/var for tokens. Metadata labels never enter model.
a.X=a.layers['counts'].copy()
b=embed_data(a.copy(),checkpoint,gene_col='gene_symbol',max_length=c['max_length'],batch_size=c['batch_size'],device='cuda' if torch.cuda.is_available() else 'cpu',use_fast_transformer=False)
assert np.array_equal(a.obs_names,b.obs_names)
features=b.obsm['X_scGPT']
assert features.shape[0]==a.n_obs and np.isfinite(features).all()
save_features(a,features,c['run_id'])
resource_results=r.finish()
metrics=evaluate(features,a,c['run_id'])
out=ROOT/'results'/c['run_id']
pd.DataFrame(dict(cell_id=a.obs_names,n_vocab_nonzero=valid,n_input_tokens=np.minimum(valid+1,c['max_length']))).to_csv(DATA/'outputs'/c['run_id']/'token_counts.tsv',sep='\t',index=False)
run=provenance(c,a,'scGPT-whole-human frozen CLS + supervised logistic head',resource_results,metrics=metrics,
       upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/scGPT'),'rev-parse','HEAD'],text=True).strip(),
       checkpoint_revision='official whole-human Google Drive folder 1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y; hash pinned',
       checkpoint_sha256=sha(checkpoint/'best_model.pt'),batch_size=c['batch_size'],max_sequence_length=c['max_length'],dtype='autocast float16 on CUDA; features float32',
       gene_mapping_coverage=float(match.mean()),n_vocab_genes=int(match.sum()),embedding_shape=features.shape,
       token_count_quantiles=np.quantile(valid,[0,.25,.5,.75,1]).tolist(),preprocessing='raw nonzero count rank-quantile bins in official DataCollator, once; official random sequence sampling, seed 42',
       limitations=['HCC application, not reproduction of original paper benchmark','pretraining overlap unknown','zero-shot embedding + supervised head'])
run.update(resource_results)
write_json(out/'run.json',run)
print(json.dumps(run),flush=True)
