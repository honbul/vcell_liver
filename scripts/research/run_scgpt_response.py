"""Native scGPT TransformerGenerator measured-response fine-tuning and matched random-init control."""
import argparse,copy,time,traceback
import anndata as ad
import torch
from research_core import *
from response_metrics import score_response
from scgpt.model import TransformerGenerator
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained

p=argparse.ArgumentParser();p.add_argument('--initialization',choices=['pretrained','random'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke',action='store_true');args=p.parse_args()
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True);raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
launch=git_commit();source_hash=sha(Path(__file__));r=Resources();torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed);started=time.monotonic();root=ASSETS/'response/hepg2_foundation_common_v1'
phase={'preprocessing':0.,'training':0.,'inference':0.};history=[];gradient_check=None
try:
    contract=json.loads((root/'contract.json').read_text());a=ad.read_h5ad(root/'trainval_controls.h5ad');assert not a.obs.role.isin(['test']).any()
    genes=a.var_names.to_numpy(dtype=str);primary=a.var.is_primary.to_numpy();lookup={g:i for i,g in enumerate(genes)};controls=np.flatnonzero(a.obs.role.to_numpy()=='control');control_lookup={str(a.obs_names[i]):int(i) for i in controls}
    checkpoint=DATA/'checkpoints/scgpt-whole-human';config=json.loads((checkpoint/'args.json').read_text());vocab=GeneVocab.from_file(checkpoint/'vocab.json')
    for token in ['<pad>','<cls>','<eoc>']:
        if token not in vocab:vocab.append_token(token)
    assert all(g in vocab for g in genes);gene_ids=torch.tensor(vocab(genes.tolist()),dtype=torch.long,device='cuda')
    model=TransformerGenerator(len(vocab),config['embsize'],config['nheads'],config['d_hid'],config['nlayers'],nlayers_cls=config['n_layers_cls'],n_cls=1,vocab=vocab,dropout=0.,pad_token='<pad>',pad_value=0,pert_pad_id=0,use_fast_transformer=False)
    verified=0
    if args.initialization=='pretrained':
        state=torch.load(checkpoint/'best_model.pt',map_location='cpu',weights_only=True);state={k:v for k,v in state.items() if any(k.startswith(prefix) for prefix in ['encoder','value_encoder','transformer_encoder'])};load_pretrained(model,state,verbose=False)
        for name,value in state.items():
            mapped=name.replace('Wqkv.','in_proj_')
            if 'transformer_encoder' in mapped and 'in_proj_' in mapped:
                assert torch.equal(model.state_dict()[mapped],value);verified+=1
        assert verified==24
        del state
    model.cls_decoder.requires_grad_(False);model=model.cuda();optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    training=np.flatnonzero(a.obs.role.isin(['train','control']).to_numpy());validation=np.flatnonzero(a.obs.role.to_numpy()=='val')
    if args.smoke:training=training[:32];validation=validation[:32]
    def forward(adata,indices,selected):
        obs=adata.obs.iloc[indices];matches=np.array([control_lookup[str(cell)] for cell in obs.control_match_id]);values=torch.tensor(a.X[matches][:,selected].toarray(),dtype=torch.float32,device='cuda')
        target_positions=np.array([-1 if target=='non-targeting' else lookup[str(target)] for target in obs.target]);flags=torch.tensor(selected[None,:]==target_positions[:,None],dtype=torch.long,device='cuda')
        assert np.array_equal(flags.sum(1).cpu().numpy(),(target_positions>=0).astype(int)), 'Every treated target must be present; no silent all-zero perturbation flags'
        mapped=gene_ids[torch.tensor(selected,device='cuda')].repeat(len(indices),1)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(mapped,values,flags,src_key_padding_mask=torch.zeros_like(values,dtype=torch.bool),CLS=False,CCE=False,MVC=False,ECS=False)['mlm_output'].float()
        return pred
    def aggregate(adata,indices):
        start=time.monotonic();model.eval();sums={};counts={};selected=np.arange(len(genes))
        with torch.no_grad():
            for offset in range(0,len(indices),8):
                ix=indices[offset:offset+8];values=forward(adata,ix,selected).cpu().numpy()
                for target,row in zip(adata.obs.iloc[ix].target.astype(str),values):sums[target]=sums.get(target,0)+row;counts[target]=counts.get(target,0)+1
        phase['inference']+=time.monotonic()-start
        return {target:value/counts[target] for target,value in sums.items()},counts
    phase['preprocessing']+=time.monotonic()-started;best=None;patience=0;updates=0;seen=0
    for epoch in range(1 if args.smoke else args.epochs):
        model.train();order=np.random.default_rng(args.seed+epoch).permutation(training);gene_rng=np.random.default_rng(args.seed+1000+epoch);start=time.monotonic()
        for offset in range(0,len(order),32):
            block=order[offset:offset+32];required=np.array(sorted({lookup[str(t)] for t in a.obs.iloc[block].target if str(t)!='non-targeting'}),dtype=int);others=np.setdiff1d(np.arange(len(genes)),required)
            selected=np.concatenate([required,gene_rng.choice(others,min(1536-len(required),len(others)),replace=False)]);optimizer.zero_grad()
            for pos in range(0,len(block),4):
                ix=block[pos:pos+4];pred=forward(a,ix,selected);truth=torch.tensor(a.X[ix][:,selected].toarray(),dtype=torch.float32,device='cuda');loss=torch.nn.functional.mse_loss(pred,truth)*len(ix)/len(block);assert torch.isfinite(loss);loss.backward();seen+=len(ix)
            if gradient_check is None:
                gradient_check=dict(encoder_gradient_norm=float(model.encoder.embedding.weight.grad.norm()),perturbation_flag_gradient_norm=float(model.pert_encoder.weight.grad[1].norm()),loss=float(loss));assert gradient_check['encoder_gradient_norm']>0 and gradient_check['perturbation_flag_gradient_norm']>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
        scheduler.step();phase['training']+=time.monotonic()-start
        predicted,ns=aggregate(a,validation);targets=sorted(predicted);truth=np.stack([np.asarray(a.X[validation[a.obs.iloc[validation].target.astype(str).to_numpy()==target]].mean(0)).ravel() for target in targets]);values=np.stack([predicted[target] for target in targets]);error=float(np.mean((values[:,primary]-truth[:,primary])**2));history.append(dict(epoch=epoch+1,validation_primary_gene_intervention_mean_mse=error));print('SCGPT_RESPONSE_EPOCH',args.initialization,history[-1],flush=True)
        if best is None or error<best[0]:best=(error,epoch+1,copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}));patience=0
        else:patience+=1
        if patience>=2:break
    model.load_state_dict(best[2]);torch.save(best[2],raw/'best_model.pt');write_json(out/'selection.json',dict(status='smoke_training_only' if args.smoke else 'frozen_before_heldout_response_open',chosen_epoch=best[1],history=history,test_treated_read_before_selection=False))
    if args.smoke:
        res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='measured_response_smoke',project_commit=launch,source_sha256=source_hash,initialization=args.initialization,gradient_check=gradient_check,pretrained_attention_tensors_verified=verified,heldout_response_opened=False,**res));print('SCGPT_RESPONSE_SMOKE_PASSED',flush=True)
    else:
        test=ad.read_h5ad(root/'heldout_treated.h5ad');assert np.array_equal(test.var_names,a.var_names);predicted,ns=aggregate(test,np.arange(test.n_obs));observed=np.load(root/'test_aggregate.npz');targets=observed['targets'];prediction=np.stack([predicted[str(t)] for t in targets]);assert sum(ns.values())==test.n_obs
        np.savez_compressed(raw/'predictions_mean.npz',targets=targets,prediction=prediction,n_cells=observed['n_cells'],gene_names=genes,primary_mask=primary)
        rows=score_response(targets,observed['truth'][:,primary],observed['control'][:,primary],prediction[:,primary],observed['n_cells'],gene_names=genes[primary]);pd.DataFrame(rows).to_csv(out/'per_intervention_primary.tsv',sep='\t',index=False)
        res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='measured_genetic_response',regime='strict_unseen_target_matched_native_panel',model_id='scGPT_TransformerGenerator_'+args.initialization,project_commit=launch,source_sha256=source_hash,upstream_commit='cebd6fae655b9c585a4807daa3ac31bb764f06b4',initialization=args.initialization,seed=args.seed,
          checkpoint_sha256=sha(checkpoint/'best_model.pt') if args.initialization=='pretrained' else None,pretrained_attention_tensors_verified=verified,data_contract_sha256=sha(root/'contract.json'),n_genes=len(genes),n_primary_genes=int(primary.sum()),n_training_cells=len(training),n_validation_cells=len(validation),n_test_cells=test.n_obs,n_test_interventions=len(targets),
          max_epochs=args.epochs,chosen_epoch=best[1],actual_epochs=len(history),optimizer_updates=updates,seen_examples=seen,gradient_check=gradient_check,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),microbatch=4,effective_batch=32,lr=1e-4,precision='BF16 matmul with FP32 weights; no flash; no quantization',
          sampling_contract='Training genes sampled to1536 while always including every current intervention target; all actual observed genes used at inference; valid target flags asserted in every pass',
          prediction_path=str(raw/'predictions_mean.npz'),prediction_sha256=sha(raw/'predictions_mean.npz'),metrics={k:float(v) for k,v in pd.DataFrame(rows).mean(numeric_only=True).items()},wall_preprocessing_seconds=phase['preprocessing'],wall_training_seconds=phase['training'],wall_inference_seconds=phase['inference'],wall_total_seconds=res.pop('wall_seconds'),
          failure_or_limitations=['New strict HepG2 task fine-tuning using native architecture, not original tutorial numerical reproduction','Matched few-cell training-update budget retains parent training-derived gene panel','Same model/input/optimizer random-init control required to attribute pretrained-weight benefit','Unpaired mean expression prediction; no drug or clinical efficacy claim'],**res));print('SCGPT_RESPONSE_COMPLETE',args.run_id,flush=True)
except Exception as exc:
    write_json(out/'failure.json',dict(status='failed',run_id=args.run_id,source_sha256=source_hash,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),**r.finish()));raise
