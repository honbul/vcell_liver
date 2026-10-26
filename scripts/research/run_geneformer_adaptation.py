"""Matched-head frozen, partial and real LoRA supervised lineage adaptation on development donors."""
import argparse,copy,math,time,traceback
import anndata as ad
import torch,yaml
from datasets import load_from_disk
from transformers import BertForMaskedLM
from peft import LoraConfig,get_peft_model
from research_core import *
from sampling import sampling_roles

p=argparse.ArgumentParser();p.add_argument('--model',choices=['frozen_nn','partial','lora'],required=True);p.add_argument('--data-variant',choices=['D0_natural','D1_donor_balanced','D2_donor_class_balanced'],default='D0_natural');p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=3);p.add_argument('--fixed-epochs',type=int);p.add_argument('--microbatch',type=int,default=2);args=p.parse_args()
c=yaml.safe_load((ROOT/'configs/research/protocol_improvement.yaml').read_text());c['seed']=args.seed;c['data_variant']=args.data_variant;c['model_variant']={'frozen_nn':'M01_frozen_same_NN_head','partial':'M02_partial_same_NN_head','lora':'M03_LoRA_same_NN_head'}[args.model]
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True);raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
launch_commit=git_commit();source_hash=sha(Path(__file__))
torch.set_num_threads(8);r=Resources();start=time.monotonic();a=ad.read_h5ad(DATA/c['dataset_path']);roles=sampling_roles(a.obs,c,args.fold,args.data_variant)
for role,indices in roles.items():pd.DataFrame({'cell_id':a.obs_names[indices]}).to_csv(raw/f'{role}_cell_ids.tsv',sep='\t',index=False)
frozen=np.load(ASSETS/'features/geneformer_GSE149614_development_full.npz');assert np.array_equal(frozen['cell_ids'],a.obs_names.to_numpy(dtype=str));frozen=frozen['X']
ds=load_from_disk(str(DATA/'tokenized/research_V2_tokens_full_v1/cells.dataset'));assert ds['cell_id']==a.obs_names.tolist()
from geneformer import TOKEN_DICTIONARY_FILE
import pickle
with open(TOKEN_DICTIONARY_FILE,'rb') as f:pad=pickle.load(f)['<pad>']
y=np.array([c['labels'].index(str(t)) for t in a.obs.cell_type_coarse]);base_path=DATA/'checkpoints/Geneformer-V2-104M';phase={'preprocessing':time.monotonic()-start,'training':0.,'inference':0.};gradient_checks=[];epochs_history=[];update_counts={}

def make_model():
    # Seed adapter initialization as well as the head, independently for both fits.
    torch.manual_seed(args.seed)
    encoder=None;targets=[]
    if args.model!='frozen_nn':
        full=BertForMaskedLM.from_pretrained(base_path,local_files_only=True,output_hidden_states=True);encoder=full.bert.eval().requires_grad_(False).cuda();del full
        if args.model=='partial':
            for index in [9,10]:encoder.encoder.layer[index].requires_grad_(True)
        else:
            targets=[name for name,module in encoder.named_modules() if isinstance(module,torch.nn.Linear) and any(name==f'encoder.layer.{i}.attention.self.{projection}' for i in range(11) for projection in ['query','value'])]
            assert len(targets)==22
            encoder=get_peft_model(encoder,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=targets,bias='none')).eval()
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # All conditions disable encoder dropout, including training; autograd remains enabled.
        encoder.eval()
    torch.manual_seed(args.seed)
    head=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(128,6)).cuda()
    return encoder,head,targets

def tokenize(indices):
    seqs=[ds[int(i)]['input_ids'] for i in indices];width=max(map(len,seqs));batch=torch.full((len(seqs),width),pad,dtype=torch.long,device='cuda')
    for i,seq in enumerate(seqs):batch[i,:len(seq)]=torch.tensor(seq,device='cuda')
    return batch

def logits(encoder,head,indices,mu,scale):
    if encoder is None:features=torch.tensor(frozen[indices],device='cuda')
    else:
        tokens=tokenize(indices)
        core=encoder.base_model.model if args.model=='lora' else encoder
        core.encoder.training=torch.is_grad_enabled() # checkpoint control only; child dropout modules remain in eval mode
        assert all(not m.training for m in encoder.modules() if isinstance(m,torch.nn.Dropout))
        with torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(input_ids=tokens,attention_mask=(tokens!=pad).long(),output_hidden_states=True).hidden_states[11][:,0,:].float()
    features=(features-mu)/scale
    with torch.autocast('cuda',dtype=torch.bfloat16):return head(features).float()

def predict(encoder,head,indices,mu,scale):
    start=time.monotonic();head.eval();probs=[]
    with torch.no_grad():
        for offset in range(0,len(indices),8):probs.append(torch.softmax(logits(encoder,head,indices[offset:offset+8],mu,scale),1).cpu().numpy())
    phase['inference']+=time.monotonic()-start;prob=np.concatenate(probs);assert np.isfinite(prob).all();return prob

def snapshot(encoder,head):
    return dict(head={n:p.detach().cpu().clone() for n,p in head.state_dict().items()},encoder={} if encoder is None else {n:p.detach().cpu().clone() for n,p in encoder.named_parameters() if p.requires_grad})

def restore(encoder,head,state):
    head.load_state_dict(state['head'])
    if encoder is not None:
        named=dict(encoder.named_parameters())
        with torch.no_grad():
            for name,value in state['encoder'].items():named[name].copy_(value.to('cuda'))

def fit(indices,stage,epochs,validation=None):
    start=time.monotonic();encoder,head,targets=make_model();scaler=StandardScaler().fit(frozen[indices]);mu=torch.tensor(scaler.mean_,dtype=torch.float32,device='cuda');scale=torch.tensor(scaler.scale_,dtype=torch.float32,device='cuda')
    params=[] if encoder is None else [p for p in encoder.parameters() if p.requires_grad];groups=[{'params':list(head.parameters()),'lr':1e-3}]
    if params:groups.append({'params':params,'lr':1e-4 if args.model=='lora' else 1e-5})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);phase['preprocessing']+=time.monotonic()-start
    best=None;patience=0;updates=0;seen=0
    for epoch in range(epochs):
        head.train();order=np.random.default_rng(args.seed+1000*(stage=='outer')+epoch).permutation(indices);started=time.monotonic()
        for offset in range(0,len(order),16):
            block=order[offset:offset+16];optimizer.zero_grad()
            for pos in range(0,len(block),args.microbatch):
                batch=block[pos:pos+args.microbatch];output=logits(encoder,head,batch,mu,scale);loss=torch.nn.functional.cross_entropy(output,torch.tensor(y[batch],device='cuda'))*len(batch)/len(block);loss.backward();seen+=len(batch)
            if updates<2:
                check=dict(stage=stage,update=updates,head_gradient_norm=sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None),encoder_nonzero_gradient_tensors=sum(p.grad is not None and bool(torch.any(p.grad!=0)) for p in params))
                assert check['head_gradient_norm']>0
                if params:assert check['encoder_nonzero_gradient_tensors']>0
                if encoder is not None:assert all(p.grad is None for p in encoder.parameters() if not p.requires_grad)
                gradient_checks.append(check)
            torch.nn.utils.clip_grad_norm_(list(head.parameters())+params,1.);optimizer.step();updates+=1
        phase['training']+=time.monotonic()-started
        if validation is not None:
            prob=predict(encoder,head,validation,mu,scale);score=float(f1_score(y[validation],prob.argmax(1),labels=list(range(6)),average='macro',zero_division=0));epochs_history.append(dict(stage=stage,epoch=epoch+1,validation_macro_f1=score,optimizer_updates=updates,seen_examples=seen));print('EPOCH',args.run_id,epoch+1,score,flush=True)
            if best is None or score>best[0]:best=(score,epoch+1,snapshot(encoder,head));patience=0
            else:patience+=1
            if patience>=2:break
    update_counts[stage]=dict(optimizer_updates=updates,seen_examples=seen)
    if best is not None:restore(encoder,head,best[2]);chosen=best[1];score=best[0]
    else:chosen=epochs;score=None
    torch.save(snapshot(encoder,head),raw/f'{stage}_trainable_state.pt')
    import joblib
    joblib.dump(scaler,raw/f'{stage}_scaler.joblib')
    return encoder,head,mu,scale,chosen,score,targets,len(params) and sum(p.numel() for p in params)

try:
    encoder,head,mu,scale,chosen,score,targets,nparams=fit(roles['inner_training'],'inner',args.epochs,roles['inner_validation'])
    if args.fixed_epochs is not None:chosen=args.fixed_epochs
    write_json(out/'selection.json',dict(status='frozen_before_outer_evaluation',chosen_epochs=chosen,inner_validation_macro_f1=score,history=epochs_history,external_final_access=False))
    del encoder,head,mu,scale;torch.cuda.empty_cache()
    encoder,head,mu,scale,_,_,targets,nparams=fit(roles['outer_training'],'outer',chosen)
    prob=predict(encoder,head,roles['evaluation'],mu,scale);pred=np.array(c['labels'])[prob.argmax(1)]
    # Reload the exact trainable overlay plus head and verify real evaluation logits.
    expected=prob[:8];saved=torch.load(raw/'outer_trainable_state.pt',map_location='cpu',weights_only=True);restore(encoder,head,saved)
    restored=predict(encoder,head,roles['evaluation'][:8],mu,scale);reload_difference=float(np.abs(expected-restored).max());assert np.allclose(expected,restored,atol=1e-5,rtol=1e-5)
    if args.model=='lora':encoder.save_pretrained(raw/'outer_adapter',safe_serialization=True)
    res=r.finish();save_evaluation(a,roles['evaluation'],pred,prob,c,args.run_id,dict(model_id='Geneformer_'+args.model,fold=args.fold,project_commit=launch_commit,source_sha256=source_hash,initialization_seed=args.seed,fold_training_seed_policy='Seed before encoder/adapter construction and again before shared head initialization in both inner/outer fits',base_checkpoint_revision='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5',checkpoint_sha256=sha(base_path/'model.safetensors'),
      chosen_epochs=chosen,inner_validation_macro_f1=score,training_donors=sorted(a.obs.iloc[roles['outer_training']].patient_id.unique()),validation_donors=sorted(a.obs.iloc[roles['inner_validation']].patient_id.unique()),evaluation_donors_or_perturbations=sorted(a.obs.iloc[roles['evaluation']].patient_id.unique()),
      n_inner_training=len(roles['inner_training']),n_inner_validation=len(roles['inner_validation']),n_outer_training=len(roles['outer_training']),n_evaluation=len(roles['evaluation']),trainable_encoder_parameters=nparams,trainable_head_parameters=sum(p.numel() for p in head.parameters()),trainable_parameters=nparams+sum(p.numel() for p in head.parameters()),
      target_modules=targets,rank=8 if args.model=='lora' else None,alpha=16 if args.model=='lora' else None,adapter_dropout=0.,partial_blocks=[9,10] if args.model=='partial' else None,
      head_architecture='Linear768→128, GELU, Dropout0.1, Linear128→6; same initialization/optimizer across conditions',head_learning_rate=1e-3,encoder_learning_rate=0 if encoder is None else 1e-4 if args.model=='lora' else 1e-5,
      microbatch=args.microbatch,effective_batch=16,update_counts=update_counts,gradient_checks=gradient_checks,precision='FP32 weights; BF16 matmul, no quantization',encoder_dropout='disabled uniformly; autograd enabled for selected parameters',pooling='hidden_states[11] CLS, full4096 V2 input; unused block11 excluded from adaptation',
      training_manifest_sha256=sha(raw/'outer_training_cell_ids.tsv'),validation_manifest_sha256=sha(raw/'inner_validation_cell_ids.tsv'),save_reload_probability_max_abs_difference=reload_difference,
      wall_preprocessing_seconds=phase['preprocessing'],wall_training_seconds=phase['training'],wall_inference_seconds=phase['inference'],wall_total_seconds=res.pop('wall_seconds'),
      failure_or_limitations=['Separate matched4096 training budget; compare against its own frozen controls, not12000-cell A-series','Frozen NN uses batch8 BF16 cached embeddings; adapted encoders use online microbatch2 BF16 during training','Pretraining overlap unknown; development OOF only'],**res))
except Exception as exc:
    write_json(out/'failure.json',dict(status='failed',run_id=args.run_id,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),**r.finish()));raise
