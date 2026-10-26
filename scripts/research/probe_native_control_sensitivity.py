"""Saved-model inference only: one fixed real target, sampled versus full same-batch controls."""
import anndata as ad
import torch
from scgpt.model import TransformerGenerator
from scgpt.tokenizer import GeneVocab
from research_core import *
from response_metrics import score_response

out=REPORT/'control_sensitivity';assert not (out/'native_first_target.json').exists();out.mkdir(exist_ok=True);raw=ASSETS/'probes/control_sensitivity_native_v3';assert not raw.exists();raw.mkdir(parents=True);r=Resources();torch.set_num_threads(8)
root=ASSETS/'response/hepg2_foundation_common_v1';a=ad.read_h5ad(root/'trainval_controls.h5ad');test=ad.read_h5ad(root/'heldout_treated.h5ad',backed='r');observed=np.load(root/'test_aggregate.npz');target=str(sorted(observed['targets'].astype(str))[0]);position=list(observed['targets']).index(target);part=test.obs[test.obs.target.astype(str)==target];genes=a.var_names.to_numpy(dtype=str);primary=a.var.is_primary.to_numpy();assert np.array_equal(genes,test.var_names) and primary.sum()==1977
controls=np.flatnonzero(a.obs.role.to_numpy()=='control');lookup={str(a.obs_names[i]):i for i in controls};bybatch={b:controls[a.obs.iloc[controls].batch.astype(str).to_numpy()==b] for b in part.batch.astype(str).unique()};means={b:np.asarray(a.X[ix].mean(0)).ravel() for b,ix in bybatch.items()}
sampled=a.X[[lookup[str(v)] for v in part.control_match_id]].toarray().astype(np.float32);full=np.stack([means[b] for b in part.batch.astype(str)]).astype(np.float32);assert len(part)==int(observed['n_cells'][position]);assert np.allclose(full.mean(0),observed['control'][position],atol=1e-5)
checkpoint=DATA/'checkpoints/scgpt-whole-human';config=json.loads((checkpoint/'args.json').read_text());vocab=GeneVocab.from_file(checkpoint/'vocab.json')
for token in ['<pad>','<cls>','<eoc>']:
    if token not in vocab:vocab.append_token(token)
gene_ids=torch.tensor(vocab(genes.tolist()),dtype=torch.long,device='cuda');target_index=list(genes).index(target);rows=[];evidence=[]
for initialization in ['pretrained','random']:
    rid=f'B04_{initialization}_native_common_seed42_v1';run=json.loads((REPORT/'runs'/rid/'run.json').read_text());path=ASSETS/'runs'/rid/'best_model.pt';weights_hash=sha(path);model=TransformerGenerator(len(vocab),config['embsize'],config['nheads'],config['d_hid'],config['nlayers'],nlayers_cls=config['n_layers_cls'],n_cls=1,vocab=vocab,dropout=0.,pad_token='<pad>',pad_value=0,pert_pad_id=0,use_fast_transformer=False);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval();predictions={}
    for mode,inputs in [('recorded_sampled',sampled),('full_same_batch_mean',full)]:
        outputs=[]
        with torch.no_grad():
            for offset in range(0,len(part),8):
                values=torch.tensor(inputs[offset:offset+8],device='cuda');flags=torch.zeros_like(values,dtype=torch.long);flags[:,target_index]=1
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(gene_ids.repeat(len(values),1),values,flags,src_key_padding_mask=torch.zeros_like(values,dtype=torch.bool),CLS=False,CCE=False,MVC=False,ECS=False)['mlm_output'].float()
                outputs.append(pred.cpu().numpy())
        prediction=np.concatenate(outputs).mean(0);predictions[mode]=prediction
        rows.extend(dict(initialization=initialization,input_control_policy=mode,**row) for row in score_response([target],observed['truth'][position:position+1,primary],observed['control'][position:position+1,primary],prediction[None,primary],[len(part)],gene_names=genes[primary]))
    original=np.load(run['prediction_path']);i=list(original['targets']).index(target);difference=float(np.max(np.abs(predictions['recorded_sampled']-original['prediction'][i])));assert sha(path)==weights_hash
    np.savez_compressed(raw/f'{initialization}.npz',gene_names=genes,target=np.array([target]),**predictions)
    evidence.append(dict(source_run_id=rid,fitted_checkpoint_sha256=weights_hash,original_prediction_sha256=sha(run['prediction_path']),sampled_replay_vs_original_mean_prediction_max_abs_difference=difference,mean_prediction_input_policy_discrepancy_RMSE=float(np.sqrt(np.mean((predictions['recorded_sampled'][primary]-predictions['full_same_batch_mean'][primary])**2)))))
    del model;torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(out/'native_first_target_metrics.tsv',sep='\t',index=False);resources=r.finish()
write_json(out/'native_first_target.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),config_sha256=sha(ROOT/'configs/research/control_sensitivity_v3.json'),target=target,n_treated_cells=len(part),n_primary_genes=int(primary.sum()),n_models=2,n_inference_policies=2,new_training=False,target_selection='Lexicographically first fixed test target, before inspecting probe scores',evidence=evidence,reporting='Same original full matched-control evaluator for both input policies; fixed primary1977 genes',limitations=['One target only; no rank or biological replicate generalization','Both models remain trained on sampled controls; full-mean inference is an input sensitivity check, not a retrained fair-input benchmark','Probe groups the selected target into batches8 and may use a different GPU; measured replay numerical discrepancy is reported, not hidden','Original aggregate predictions and selection records are unchanged'],**resources))
print('NATIVE_CONTROL_SENSITIVITY_COMPLETE',pd.DataFrame(rows)[['initialization','input_control_policy','target','delta_rmse','pearson_delta']].to_string(index=False),flush=True)
