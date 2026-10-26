"""Refit the four preregistered matched4096 factors on all eight development donors."""
import argparse,math,pickle,time
import anndata as ad
import joblib,torch
from datasets import load_from_disk
from transformers import BertForMaskedLM
from geneformer import TOKEN_DICTIONARY_FILE
from sampling import balanced_without_replacement
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--factor',choices=['DM00','DM10','DM01','DM11'],required=True);p.add_argument('--run-id',required=True);args=p.parse_args()
confirmation=json.loads((REPORT/'factorial_confirmation/provenance.json').read_text());assert confirmation['status']=='executed' and confirmation['n_executed_runs']==48
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True);raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True);r=Resources();started=time.monotonic();torch.set_num_threads(8)
a=ad.read_h5ad(ASSETS/'data/development_full.h5ad');assert a.n_obs==63036 and not a.obs.patient_id.isin(['HCC02','HCC09']).any();all_indices=np.arange(a.n_obs)
balanced=args.factor in ['DM10','DM11'];adapted=args.factor in ['DM01','DM11']
training=balanced_without_replacement(a.obs,all_indices,4096,['patient_id'],42+20000) if balanced else np.sort(np.random.default_rng(42).choice(all_indices,4096,replace=False))
pd.DataFrame({'cell_id':a.obs_names[training]}).to_csv(raw/'training_cell_ids.tsv',sep='\t',index=False)
manifest=pd.concat([pd.read_csv(ROOT/'manifests/research_phase2'/name,sep='\t') for name in ['frozen_confirmation_runs.tsv','adapted_confirmation_runs_v2.tsv']]);selected=manifest[manifest.factor==args.factor];assert len(selected)==12
epochs=[];sources=[]
for row in selected.itertuples():
    path=REPORT/'runs'/row.run_id/'run.json';run=json.loads(path.read_text());assert run['status']=='executed';epochs.append(int(run['chosen_epochs']));sources.append(dict(run_id=row.run_id,run_sha256=sha(path),chosen_epochs=run['chosen_epochs']))
epochs=int(math.ceil(np.median(epochs)));assert 1<=epochs<=3
labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];y=np.array([labels.index(str(t)) for t in a.obs.cell_type_coarse]);feature_path=ASSETS/'features/geneformer_GSE149614_development_full.npz';d=np.load(feature_path);assert np.array_equal(d['cell_ids'],a.obs_names.astype(str));frozen=d['X'];feature_metadata=json.loads(feature_path.with_suffix('.json').read_text())
scaler=StandardScaler().fit(frozen[training]);mu=torch.tensor(scaler.mean_,dtype=torch.float32,device='cuda');scale=torch.tensor(scaler.scale_,dtype=torch.float32,device='cuda');torch.manual_seed(42);encoder=None
if adapted:
    full=BertForMaskedLM.from_pretrained(DATA/'checkpoints/Geneformer-V2-104M',local_files_only=True,output_hidden_states=True);encoder=full.bert.eval().requires_grad_(False).cuda();del full
    for index in [9,10]:encoder.encoder.layer[index].requires_grad_(True)
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});ds=load_from_disk(str(DATA/'tokenized/research_V2_tokens_full_v1/cells.dataset'));assert ds['cell_id']==a.obs_names.tolist()
    with open(TOKEN_DICTIONARY_FILE,'rb') as f:pad=pickle.load(f)['<pad>']
torch.manual_seed(42);head=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(128,6)).cuda();parameters=[] if encoder is None else [p for p in encoder.parameters() if p.requires_grad]
groups=[dict(params=list(head.parameters()),lr=1e-3)]
if parameters:groups.append(dict(params=parameters,lr=1e-5))
optimizer=torch.optim.AdamW(groups,weight_decay=.01);preprocess=time.monotonic()-started;updates=0;gradient=None;start=time.monotonic()
def logits(indices):
    if encoder is None:features=torch.tensor(frozen[indices],device='cuda')
    else:
        sequences=[ds[int(i)]['input_ids'] for i in indices];tokens=torch.full((len(sequences),max(map(len,sequences))),pad,dtype=torch.long,device='cuda')
        for i,seq in enumerate(sequences):tokens[i,:len(seq)]=torch.tensor(seq,device='cuda')
        encoder.encoder.training=torch.is_grad_enabled();assert all(not m.training for m in encoder.modules() if isinstance(m,torch.nn.Dropout))
        with torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(input_ids=tokens,attention_mask=(tokens!=pad).long(),output_hidden_states=True).hidden_states[11][:,0,:].float()
    with torch.autocast('cuda',dtype=torch.bfloat16):return head((features-mu)/scale).float()
for epoch in range(epochs):
    head.train();order=np.random.default_rng(42+1000+epoch).permutation(training)
    for offset in range(0,len(order),16):
        block=order[offset:offset+16];optimizer.zero_grad()
        for pos in range(0,len(block),2):
            batch=block[pos:pos+2];loss=torch.nn.functional.cross_entropy(logits(batch),torch.tensor(y[batch],device='cuda'))*len(batch)/len(block);assert torch.isfinite(loss);loss.backward()
        if gradient is None:
            gradient=dict(head_gradient_norm=sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None),nonzero_encoder_gradient_tensors=sum(p.grad is not None and bool(torch.any(p.grad!=0)) for p in parameters));assert gradient['head_gradient_norm']>0
            if adapted:assert gradient['nonzero_encoder_gradient_tensors']>0 and all(p.grad is None for p in encoder.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(list(head.parameters())+parameters,1.);optimizer.step();updates+=1
    print('FINAL_FACTORIAL_EPOCH',args.factor,epoch+1,flush=True)
training_time=time.monotonic()-start;head.eval()
with torch.no_grad():expected=torch.softmax(logits(training[:8]),1).cpu().numpy()
state=dict(head={n:p.detach().cpu().clone() for n,p in head.state_dict().items()},encoder={} if encoder is None else {n:p.detach().cpu().clone() for n,p in encoder.named_parameters() if p.requires_grad})
bundle=dict(schema_version=2,kind='geneformer_nn',model_id='Geneformer_'+args.factor,feature_model_id='geneformer',labels=labels,scaler=scaler,head_state=state['head'],encoder_state=state['encoder'],adaptation='partial' if adapted else 'frozen',feature_metadata=feature_metadata,training_manifest_sha256=sha(raw/'training_cell_ids.tsv'),epochs=epochs)
joblib.dump(bundle,raw/'annotation_bundle.joblib');loaded=joblib.load(raw/'annotation_bundle.joblib');head.load_state_dict(loaded['head_state'])
if encoder is not None:
    named=dict(encoder.named_parameters())
    with torch.no_grad():
        for name,value in loaded['encoder_state'].items():named[name].copy_(value.to('cuda'))
with torch.no_grad():actual=torch.softmax(logits(training[:8]),1).cpu().numpy()
difference=float(np.abs(actual-expected).max());assert difference<=1e-5
write_json(out/'selection.json',dict(rule='Ceiling of median chosen inner-validation epoch over all12 development confirmation runs for this factor; no external scores',epochs=epochs,sources=sources))
res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='annotation_final_fit',regime='inductive',factor=args.factor,model_id=bundle['model_id'],project_commit=git_commit(),source_sha256=sha(Path(__file__)),seed=42,n_training_cells=len(training),training_donors=sorted(a.obs.iloc[training].patient_id.unique()),training_manifest_sha256=bundle['training_manifest_sha256'],feature_dimension=768,feature_metadata=feature_metadata,epochs=epochs,optimizer_updates=updates,gradient_check=gradient,trainable_parameters=sum(p.numel() for p in head.parameters())+sum(p.numel() for p in parameters),trainable_head_parameters=sum(p.numel() for p in head.parameters()),trainable_encoder_parameters=sum(p.numel() for p in parameters),save_reload_probability_max_abs_difference=difference,
    bundle_path=str(raw/'annotation_bundle.joblib'),bundle_sha256=sha(raw/'annotation_bundle.joblib'),data_sha256=sha(ASSETS/'data/development_full.h5ad'),wall_preprocessing_seconds=preprocess,wall_training_seconds=training_time,wall_inference_seconds=0,wall_total_seconds=res.pop('wall_seconds'),external_data_accessed=False,precision='FP32 weights/BF16 matmul; same head/optimizer as development confirmation',**res))
print('FINAL_FACTORIAL_FIT_COMPLETE',args.factor,flush=True)
