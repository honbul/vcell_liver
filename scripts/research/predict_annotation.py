"""Label-free inference with a fitted annotation bundle; never fits a query statistic."""
import argparse,time
import anndata as ad
import joblib
from scipy import sparse
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--bundle',required=True);p.add_argument('--features');p.add_argument('--token-cache');p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);assert not out.exists(),'Prediction destinations are immutable';out.mkdir(parents=True);r=Resources();start=time.monotonic()
a=ad.read_h5ad(args.input);bundle=joblib.load(args.bundle);counts=a.layers['counts'] if 'counts' in a.layers else a.X
counts=sparse.csr_matrix(counts,dtype=np.float32);assert a.obs_names.is_unique and a.var_names.is_unique and np.isfinite(counts.data).all() and (counts.data>=0).all() and np.all(counts.data==np.round(counts.data))
coverage=None;representation_seconds=0.;neural=bundle.get('kind')=='geneformer_nn'
if neural and bundle['adaptation']=='partial':
    import pickle,torch
    from datasets import load_from_disk
    from transformers import BertForMaskedLM
    from geneformer import TOKEN_DICTIONARY_FILE
    cache=Path(args.token_cache);assert json.loads((cache.parent/'input_contract.json').read_text())['input_sha256']==sha(args.input)
    ds=load_from_disk(str(cache));assert ds['cell_id']==a.obs_names.tolist();base=DATA/'checkpoints/Geneformer-V2-104M';assert sha(base/'model.safetensors')==bundle['feature_metadata']['checkpoint_sha256']
    full=BertForMaskedLM.from_pretrained(base,local_files_only=True,output_hidden_states=True);encoder=full.bert.eval().requires_grad_(False).cuda();del full
    named=dict(encoder.named_parameters())
    with torch.no_grad():
        for name,value in bundle['encoder_state'].items():named[name].copy_(value.to('cuda'))
    with open(TOKEN_DICTIONARY_FILE,'rb') as f:pad=pickle.load(f)['<pad>']
    results=[];rep_start=time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for offset in range(0,len(ds),8):
            sequences=ds[offset:offset+8]['input_ids'];tokens=torch.full((len(sequences),max(map(len,sequences))),pad,dtype=torch.long,device='cuda')
            for i,seq in enumerate(sequences):tokens[i,:len(seq)]=torch.tensor(seq,device='cuda')
            results.append(encoder(input_ids=tokens,attention_mask=(tokens!=pad).long(),output_hidden_states=True).hidden_states[11][:,0,:].float().cpu().numpy())
    features=np.concatenate(results);representation_seconds=time.monotonic()-rep_start;coverage=dict(token_cache=str(cache),n_query_genes=a.n_vars,representation='Base Geneformer with fitted blocks9/10 overlay; own query rank tokens only')
    del encoder;torch.cuda.empty_cache()
elif bundle['model_id']=='pca':
    prep=bundle['preprocessor'];gene_ids=a.var.ensembl_id.astype(object).fillna('').astype(str).to_numpy() if 'ensembl_id' in a.var else a.var_names.astype(str).to_numpy();assert not pd.Series(gene_ids[gene_ids!='']).duplicated().any()
    lookup={gene:i for i,gene in enumerate(gene_ids) if gene};hvg=prep['hvg_ensembl'];assert 'hvg_gene_symbols' in prep,'Use the corrected PCA v2 bundle with explicit Ensembl/symbol alignment'
    symbols=a.var.gene_symbol.astype(object).fillna('').astype(str).to_numpy() if 'gene_symbol' in a.var else np.array(['']*a.n_vars);symbol_counts=pd.Series(symbols).value_counts();symbol_lookup={gene:i for i,gene in enumerate(symbols) if gene and symbol_counts[gene]==1};present=[];n_id=0;n_symbol=0
    for j,(gene,symbol) in enumerate(zip(hvg,prep['hvg_gene_symbols'])):
        if gene and gene in lookup:present.append((j,lookup[gene]));n_id+=1
        elif symbol and symbol in symbol_lookup:present.append((j,symbol_lookup[symbol]));n_symbol+=1
    coverage=dict(n_reference_hvg=len(hvg),n_present_hvg=len(present),n_matched_ensembl=n_id,n_matched_unique_symbol_fallback=n_symbol,missing_gene_policy='Prefer nonempty Ensembl ID, then unique original gene symbol; raw zero fill only if neither can match; original provided query-library sum retained')
    assert present,'No reference HVGs overlap the query Ensembl IDs'
    library=np.asarray(counts.sum(1)).ravel();assert (library>0).all();normalized=counts.multiply((prep['normalization_target']/library)[:,None]).tocsr();normalized.data=np.log1p(normalized.data)
    small=np.zeros((a.n_obs,len(hvg)),dtype=np.float32);columns,source=zip(*present);small[:,columns]=normalized[:,source].toarray();features=prep['pca'].transform(prep['scaler'].transform(small))
else:
    assert args.features,'Supply frozen checkpoint features extracted from this unlabeled query'
    d=np.load(args.features);assert np.array_equal(d['cell_ids'],a.obs_names.astype(str));features=d['X'];meta=json.loads(Path(args.features).with_suffix('.json').read_text())
    assert meta['model_id']==bundle.get('feature_model_id',bundle['model_id']) and meta['checkpoint_sha256']==bundle['feature_metadata']['checkpoint_sha256'] and meta['input_data_sha256']==sha(args.input) and meta['feature_sha256']==sha(args.features)
    coverage=meta['coverage']
preprocess=time.monotonic()-start-representation_seconds;start=time.monotonic();x=bundle['scaler'].transform(features);labels=bundle['labels']
if neural:
    import torch
    head=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(128,6)).cuda().eval();head.load_state_dict(bundle['head_state']);parts=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for offset in range(0,len(x),256):parts.append(torch.softmax(head(torch.tensor(x[offset:offset+256],dtype=torch.float32,device='cuda')).float(),1).cpu().numpy())
    prob=np.concatenate(parts);prediction=np.array(labels)[prob.argmax(1)]
else:
    head=bundle['classifier'];prediction=head.predict(x);native=head.predict_proba(x);prob=np.zeros((a.n_obs,len(labels)))
    for j,label in enumerate(head.classes_):prob[:,labels.index(label)]=native[:,j]
frame=pd.DataFrame(dict(cell_id=a.obs_names.astype(str),prediction=prediction,confidence=prob.max(1)))
if 'patient_id' in a.obs:frame['patient_id']=a.obs.patient_id.to_numpy()
frame.to_csv(out/'predictions.tsv',sep='\t',index=False);np.savez_compressed(out/'probabilities.npz',cell_ids=a.obs_names.to_numpy(dtype=str),probabilities=prob,labels=np.array(labels))
res=r.finish();write_json(out/'inference.json',dict(status='executed',task='annotation_inference',regime='inductive',model_id=bundle['model_id'],project_commit=git_commit(),source_sha256=sha(Path(__file__)),input_data_sha256=sha(args.input),bundle_path=args.bundle,bundle_sha256=sha(args.bundle),feature_path=args.features,n_cells=a.n_obs,feature_dimension=features.shape[1],prediction_sha256=sha(out/'predictions.tsv'),probability_sha256=sha(out/'probabilities.npz'),coverage=coverage,
    allowed_query_information='Own raw counts and gene identifiers; cell ID/patient ID copied for output only. No query labels read, no query fit or adaptation',wall_preprocessing_seconds=preprocess,wall_representation_seconds=representation_seconds,wall_inference_seconds=time.monotonic()-start+representation_seconds,wall_training_seconds=0,wall_total_seconds=res.pop('wall_seconds'),**res))
print('ANNOTATION_INFERENCE_COMPLETE',bundle['model_id'],a.n_obs,flush=True)
