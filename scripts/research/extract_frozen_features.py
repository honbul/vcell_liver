"""Full development-only frozen embeddings, explicit compute precision and shared V2 tokens."""
import argparse,json,pickle,sys,time
from pathlib import Path
from research_core import *
import anndata as ad
import torch
from datasets import load_from_disk

REV='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5'
p=argparse.ArgumentParser();p.add_argument('--model',choices=['scgpt','geneformer','geneformer_clcancer'],required=True);p.add_argument('--dataset',choices=['pilot','full'],default='full');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--precision',choices=['float32','bf16','fp16'],default='bf16');p.add_argument('--tokenize-only',action='store_true');p.add_argument('--precision-check',action='store_true');args=p.parse_args()
torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42)
dataset_id=f'GSE149614_development_{args.dataset}'
out=ASSETS/'features'/f'{args.model}_{dataset_id}.npz';out.parent.mkdir(parents=True,exist_ok=True)
assert args.tokenize_only or not out.exists(),'Immutable feature file already exists'
r=Resources();started=time.monotonic();a=ad.read_h5ad(ASSETS/'data'/f'development_{args.dataset}.h5ad')
assert not a.obs.patient_id.isin(['HCC02','HCC09']).any()
token_run=f'research_V2_tokens_{args.dataset}_v1'
ds_path=DATA/'tokenized'/token_run/'cells.dataset'
if args.model!='scgpt':
    from geneformer_utils import tokenizer_and_data
    from geneformer import TOKEN_DICTIONARY_FILE
    if args.tokenize_only:
        ds,tokens,audit=tokenizer_and_data(a,token_run)
        write_json(REPORT/f'tokenization_{args.dataset}.json',dict(status='executed',dataset_id=dataset_id,path=str(ds_path),n_cells=len(ds),fingerprint=ds._fingerprint,tokenization_audit=audit,**r.finish()))
        print('TOKENIZATION_COMPLETE',len(ds),flush=True);sys.exit(0)
    ds=load_from_disk(str(ds_path));assert ds['cell_id']==a.obs_names.tolist()
    with open(TOKEN_DICTIONARY_FILE,'rb') as f:tokens=pickle.load(f)
preprocessing=time.monotonic()-started
start=time.monotonic()
if args.model=='scgpt':
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained
    from scgpt.tasks import get_batch_cell_embeddings
    checkpoint=DATA/'checkpoints/scgpt-whole-human';weights=checkpoint/'best_model.pt'
    configs=json.loads((checkpoint/'args.json').read_text());vocab=GeneVocab.from_file(checkpoint/'vocab.json')
    for t in ['<pad>','<cls>','<eoc>']:
        if t not in vocab:vocab.append_token(t)
    vocab.set_default_index(vocab['<pad>'])
    match=np.array([g in vocab for g in a.var.gene_symbol]);b=a[:,match].copy();b.X=b.layers['counts'].copy()
    gene_ids=np.array(vocab(b.var.gene_symbol.tolist()),dtype=int)
    model=TransformerModel(ntoken=len(vocab),d_model=configs['embsize'],nhead=configs['nheads'],d_hid=configs['d_hid'],nlayers=configs['nlayers'],nlayers_cls=configs['n_layers_cls'],n_cls=1,vocab=vocab,dropout=configs['dropout'],pad_token=configs['pad_token'],pad_value=configs['pad_value'],do_mvc=True,use_fast_transformer=False)
    load_pretrained(model,torch.load(weights,map_location='cpu',weights_only=True),verbose=False)
    model=model.eval().requires_grad_(False).cuda();parameter_count=sum(p.numel() for p in model.parameters())
    loading=time.monotonic()-start;start=time.monotonic()
    x=get_batch_cell_embeddings(b,model=model,vocab=vocab,max_length=1200,batch_size=args.batch_size,model_configs=configs,gene_ids=gene_ids)
    inference=time.monotonic()-start
    method='official DataCollator, random sampling length1200, final-layer CLS, L2 normalization'
    actual_precision='official CUDA autocast float16; weights/feature storage float32'
    upstream='cebd6fae655b9c585a4807daa3ac31bb764f06b4';check=None
else:
    from transformers import BertForMaskedLM
    from geneformer.emb_extractor import get_embs
    checkpoint=DATA/'checkpoints'/('Geneformer-V2-104M_CLcancer' if args.model=='geneformer_clcancer' else 'Geneformer-V2-104M')
    weights=checkpoint/'model.safetensors'
    full=BertForMaskedLM.from_pretrained(checkpoint,output_hidden_states=True,local_files_only=True)
    parameter_count=sum(p.numel() for p in full.parameters());model=full.bert.eval().requires_grad_(False).cuda()
    assert parameter_count==104386867 and model.config.max_position_embeddings==4096
    loading=time.monotonic()-start
    def extract(data,precision):
        dtype=torch.bfloat16 if precision=='bf16' else torch.float16
        with torch.autocast('cuda',enabled=precision!='float32',dtype=dtype):
            t=get_embs(model,data,emb_mode='cls',layer_to_quant=11,pad_token_id=tokens['<pad>'],forward_batch_size=args.batch_size,token_gene_dict={v:k for k,v in tokens.items()},special_token=True)
        return t.float().detach().cpu().numpy()
    check=None
    if args.precision_check:
        small=ds.select(list(range(min(64,len(ds)))))
        fp=extract(small,'float32');mixed=extract(small,args.precision)
        cosine=np.sum(fp*mixed,axis=1)/(np.linalg.norm(fp,axis=1)*np.linalg.norm(mixed,axis=1))
        check=dict(n_cells=len(small),comparison=args.precision+'_vs_float32',cosine_min=float(cosine.min()),cosine_mean=float(cosine.mean()),maximum_absolute_difference=float(np.abs(fp-mixed).max()),note='Numerical characterization without labels; no score-based precision selection')
        assert np.isfinite(mixed).all();print('PRECISION_CHECK',check,flush=True)
    start=time.monotonic();x=extract(ds,args.precision);inference=time.monotonic()-start
    method='official V2 rank tokenizer; unchanged special tokens/4096 limit; hidden_states[11] CLS; MLM logits omitted'
    actual_precision=f'weights float32; CUDA matmul autocast {args.precision}; stored features float32'
    upstream=REV
assert len(x)==a.n_obs and np.isfinite(x).all()
np.savez_compressed(out,X=x,cell_ids=a.obs_names.to_numpy(dtype=str))
resources=r.finish()
write_json(out.with_suffix('.json'),dict(status='executed',model_id=args.model,dataset_id=dataset_id,shape=x.shape,input_data_sha256=sha(ASSETS/'data'/f'development_{args.dataset}.h5ad'),
    project_commit=git_commit(),upstream_commit=upstream,checkpoint_revision=REV if args.model!='scgpt' else 'official whole-human snapshot',checkpoint_sha256=sha(weights),
    actual_parameter_count=parameter_count,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),input_contract=method,precision=actual_precision,precision_check=check,
    batch_size=args.batch_size,wall_preprocessing_seconds=preprocessing,wall_model_loading_seconds=loading,wall_inference_seconds=inference,wall_training_seconds=0,
    wall_total_seconds=resources.pop('wall_seconds'),resource_scope='Includes optional 64-cell float32 precision characterization; no clinical/efficacy prediction',feature_sha256=sha(out),pretraining_overlap='unknown',**resources))
print('FEATURES_COMPLETE',args.model,x.shape,out,flush=True)
