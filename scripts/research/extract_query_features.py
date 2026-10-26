"""Frozen expression-only query features for the deployable annotation interface."""
import argparse,pickle,time
import anndata as ad
from scipy import sparse
import torch
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',choices=['scgpt','geneformer','geneformer_clcancer'],required=True);p.add_argument('--cache-name',required=True);args=p.parse_args()
assert Path(args.cache_name).name==args.cache_name
out=ASSETS/'query_features'/f'{args.model}_{args.cache_name}.npz';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
r=Resources();start=time.monotonic();torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);original=ad.read_h5ad(args.input)
counts=sparse.csr_matrix(original.layers['counts'] if 'counts' in original.layers else original.X,dtype=np.float32);assert original.obs_names.is_unique and np.isfinite(counts.data).all() and (counts.data>=0).all() and np.all(counts.data==np.round(counts.data))
# Restrict metadata at the boundary: model/tokenizer never receives query truth or diagnoses.
obs=pd.DataFrame(index=original.obs_names.copy());obs['cell_id']=obs.index.astype(str);obs['n_counts']=np.asarray(counts.sum(1)).ravel();assert (obs.n_counts>0).all()
a=ad.AnnData(counts.copy(),obs=obs,var=original.var.copy());a.layers['counts']=counts;del original
assert {'ensembl_id','gene_symbol'}<=set(a.var.columns),'Supply both mapped Ensembl IDs and original gene symbols'
# H5AD may store duplicate symbol columns as categorical even without missing values.
# Convert storage dtype only; identifiers, counts and frozen preprocessing remain unchanged.
for column in ['ensembl_id','gene_symbol']:a.var[column]=a.var[column].astype(object)
if args.model=='scgpt':
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained
    from scgpt.tasks import get_batch_cell_embeddings
    checkpoint=DATA/'checkpoints/scgpt-whole-human';weights=checkpoint/'best_model.pt';config=json.loads((checkpoint/'args.json').read_text());vocab=GeneVocab.from_file(checkpoint/'vocab.json')
    for token in ['<pad>','<cls>','<eoc>']:
        if token not in vocab:vocab.append_token(token)
    vocab.set_default_index(vocab['<pad>']);match=np.array([str(gene) in vocab for gene in a.var.gene_symbol]);b=a[:,match].copy();b.X=b.layers['counts'].copy();gene_ids=np.array(vocab(b.var.gene_symbol.astype(str).tolist()),dtype=int)
    model=TransformerModel(ntoken=len(vocab),d_model=config['embsize'],nhead=config['nheads'],d_hid=config['d_hid'],nlayers=config['nlayers'],nlayers_cls=config['n_layers_cls'],n_cls=1,vocab=vocab,dropout=config['dropout'],pad_token=config['pad_token'],pad_value=config['pad_value'],do_mvc=True,use_fast_transformer=False)
    load_pretrained(model,torch.load(weights,map_location='cpu',weights_only=True),verbose=False);model=model.eval().requires_grad_(False).cuda();preprocess=time.monotonic()-start;start=time.monotonic()
    features=get_batch_cell_embeddings(b,model=model,vocab=vocab,max_length=1200,batch_size=32,model_configs=config,gene_ids=gene_ids)
    coverage=dict(n_original_genes=a.n_vars,n_vocabulary_matched_genes=int(match.sum()),count_fraction=float(b.layers['counts'].sum()/counts.sum()));contract='Same official length1200 random-sampling CLS/L2 representation as development; FP16 autocast, frozen FP32 weights'
else:
    from datasets import load_from_disk
    from transformers import BertForMaskedLM
    from geneformer import TOKEN_DICTIONARY_FILE
    from geneformer.emb_extractor import get_embs
    from geneformer_utils import tokenizer_and_data
    token_run='query_'+args.cache_name;path=DATA/'tokenized'/token_run/'cells.dataset'
    if path.exists():
        assert json.loads((path.parent/'input_contract.json').read_text())['input_sha256']==sha(args.input),'Token cache name already belongs to different input counts'
        ds=load_from_disk(str(path));assert ds['cell_id']==a.obs_names.tolist()
        audit=json.loads((DATA/'outputs'/token_run/'tokenization_audit.json').read_text())
    else:
        ds,_,audit=tokenizer_and_data(a,token_run)
        write_json(path.parent/'input_contract.json',dict(input_sha256=sha(args.input),tokenizer='official_V2_4096_CLS_EOS'))
    with open(TOKEN_DICTIONARY_FILE,'rb') as f:tokens=pickle.load(f)
    checkpoint=DATA/'checkpoints'/('Geneformer-V2-104M_CLcancer' if args.model=='geneformer_clcancer' else 'Geneformer-V2-104M');weights=checkpoint/'model.safetensors'
    full=BertForMaskedLM.from_pretrained(checkpoint,output_hidden_states=True,local_files_only=True);model=full.bert.eval().requires_grad_(False).cuda();del full
    preprocess=time.monotonic()-start;start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):features=get_embs(model,ds,emb_mode='cls',layer_to_quant=11,pad_token_id=tokens['<pad>'],forward_batch_size=8,token_gene_dict={v:k for k,v in tokens.items()},special_token=True).float().detach().cpu().numpy()
    coverage=audit;contract='Same official V2 rank tokenizer, CLS/EOS4096, hidden_states[11] CLS as development; BF16 autocast; provided all-gene query library sum before Ensembl filtering'
inference=time.monotonic()-start;assert len(features)==a.n_obs and np.isfinite(features).all();np.savez_compressed(out,X=features,cell_ids=a.obs_names.to_numpy(dtype=str))
res=r.finish();write_json(out.with_suffix('.json'),dict(status='executed',task='frozen_query_representation',model_id=args.model,project_commit=git_commit(),source_sha256=sha(Path(__file__)),input_data_sha256=sha(args.input),checkpoint_sha256=sha(weights),feature_sha256=sha(out),shape=features.shape,input_contract=contract,coverage=coverage,query_labels_supplied=False,query_fitting_or_adaptation=False,pretraining_overlap='unknown',wall_preprocessing_seconds=preprocess,wall_inference_seconds=inference,wall_training_seconds=0,wall_total_seconds=res.pop('wall_seconds'),**res))
print('FROZEN_QUERY_FEATURES_COMPLETE',args.model,features.shape,flush=True)
