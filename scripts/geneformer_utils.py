"""Use official V2 tokenizer and official CLS extraction without MLM logits."""
import pickle, shutil
import anndata as ad
import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertForMaskedLM
from common import DATA,ROOT,write_json

REV='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5'
SOURCE=ROOT/'third_party/Geneformer'
CHECKPOINT=DATA/'checkpoints/Geneformer-V2-104M'

def tokenizer_and_data(a,run_id):
    from geneformer import TranscriptomeTokenizer
    from geneformer import TOKEN_DICTIONARY_FILE
    with open(TOKEN_DICTIONARY_FILE,'rb') as f:tokens=pickle.load(f)
    d=DATA/'geneformer_inputs'/run_id;d.mkdir(parents=True,exist_ok=True)
    # No HVG selection or binning. Unmapped symbols cannot be invented as Ensembl IDs.
    mapped=a.var.ensembl_id.fillna('').astype(str).to_numpy()!=''
    b=a[:,mapped].copy();b.X=b.layers['counts'].copy()
    b.layers.clear()
    b.write_h5ad(d/'counts.h5ad')
    tk=TranscriptomeTokenizer({'cell_id':'cell_id'},nproc=1,chunk_size=128,model_version='V2',model_input_size=4096,special_token=True,collapse_gene_ids=True)
    output=DATA/'tokenized'/run_id;output.mkdir(parents=True,exist_ok=True)
    path=output/'cells.dataset'
    # Recreate local derived token data for a reproducible script rerun.
    if path.exists():shutil.rmtree(path)
    tk.tokenize_data(d,output,'cells',file_format='h5ad')
    ds=load_from_disk(str(path))
    assert ds['cell_id']==a.obs_names.tolist()
    assert all(x[0]==tokens['<cls>'] and x[-1]==tokens['<eos>'] and len(x)<=4096 and len(x)>2 for x in ds['input_ids'])
    full_counts=np.asarray(a.layers['counts'].sum(1)).ravel()
    mapped_counts=np.asarray(b.X.sum(1)).ravel()
    audit=dict(n_original_genes=a.n_vars,n_mapped_genes=b.n_vars,n_original_vocab_genes=int(np.isin(a.var.ensembl_id,list(tokens)).sum()),
       n_counts_basis='original all-gene raw library sum, before mapping/filtering',
       total_original_counts=int(full_counts.sum()),total_mapped_counts=int(mapped_counts.sum()),
       mapped_count_fraction=float(mapped_counts.sum()/full_counts.sum()),
       token_length_quantiles=np.quantile(ds['length'],[0,.25,.5,.75,1]).tolist(),
       model_input_size=4096,special_token=True,collapse_gene_ids=True,
       duplicate_mapped_ids=int(b.var.ensembl_id.duplicated().sum()))
    write_json(DATA/'outputs'/run_id/'tokenization_audit.json',audit)
    return ds,tokens,audit

def load_model():
    full=BertForMaskedLM.from_pretrained(CHECKPOINT,output_hidden_states=True,local_files_only=True)
    count=sum(p.numel() for p in full.parameters())
    assert full.config.num_hidden_layers==12 and full.config.hidden_size==768 and full.config.max_position_embeddings==4096
    # Same encoder weights and hidden_states as MLM; avoid materializing unused vocabulary logits.
    model=full.bert.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    return model,count

def embeddings(model,ds,tokens,batch_size):
    from geneformer.emb_extractor import get_embs
    if torch.cuda.is_available():
        result=get_embs(model,ds,emb_mode='cls',layer_to_quant=11,pad_token_id=tokens['<pad>'],forward_batch_size=batch_size,token_gene_dict={v:k for k,v in tokens.items()},special_token=True)
        return result.detach().cpu().numpy()
    # Official helper hardcodes CUDA. CPU fallback preserves second-to-last-layer CLS semantics.
    result=[]
    for seq in ds['input_ids']:
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([seq]),attention_mask=torch.ones(1,len(seq),dtype=torch.long))
        result.append(out.hidden_states[11][0,0].numpy())
    return np.stack(result)
