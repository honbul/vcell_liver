import anndata as ad
import numpy as np
import torch
from common import DATA,ROOT,Resources,config,evaluate,provenance,save_features,sha,write_json
from geneformer_utils import CHECKPOINT,REV,embeddings,load_model,tokenizer_and_data

c=config();r=Resources();np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(8)
a=ad.read_h5ad(DATA/'subsets'/f"{c['subset']}.h5ad")
ds,tokens,audit=tokenizer_and_data(a,c['run_id'])
model,parameter_count=load_model()
x=embeddings(model,ds,tokens,c['batch_size'])
assert x.shape==(a.n_obs,768) and np.isfinite(x).all()
save_features(a,x,c['run_id'])
resource_results=r.finish()
metrics=evaluate(x,a,c['run_id'])
run=provenance(c,a,'Geneformer-V2-104M frozen CLS + supervised logistic head',resource_results,
    metrics=metrics,upstream_commit=REV,checkpoint_revision=REV,checkpoint_sha256=sha(CHECKPOINT/'model.safetensors'),
    actual_parameter_count=parameter_count,checkpoint_config=vars(model.config),embedding_shape=x.shape,
    gene_mapping_coverage=audit['n_original_vocab_genes']/a.n_vars,tokenization=audit,
    batch_size=c['batch_size'],max_sequence_length=4096,dtype='float32',hidden_states_index=11,resource_measurement_scope='official tokenization, checkpoint loading and frozen embedding; classifier CPU time excluded; full CLI in reports/stages',
    limitations=['official default emb_layer=-1 means second-to-last transformer layer','Different official input pipelines; not architecture-only comparison','pretraining overlap unknown'])
write_json(ROOT/'results'/c['run_id']/'run.json',run)
print({k:v for k,v in run.items() if k not in ('checkpoint_config','metrics')},flush=True)
