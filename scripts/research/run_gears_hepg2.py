"""Strict unseen-target HepG2 GEARS: matched batch controls, training-only graph/loss, frozen-test gate."""
import argparse,copy,pickle,random,time,traceback
import anndata as ad
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gears import PertData,GEARS
from gears.utils import loss_fct
from research_core import *
from response_metrics import score_response

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True);raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
launch_commit=git_commit();source_hash=sha(Path(__file__));r=Resources();start=time.monotonic();torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
root=ASSETS/'response/hepg2_v1';contract=json.loads((root/'contract.json').read_text());split=pd.read_csv(root/'target_split.tsv',sep='\t');assets=ASSETS/'official/GEARS';cache=ASSETS/'gears_cache/hepg2_strict_v1';(cache/'hepg2').mkdir(parents=True,exist_ok=True)
for name in ['gene2go_all.pkl','essential_all_data_pert_genes.pkl','go_essential_all']:
    link=cache/name
    if not link.exists():link.symlink_to(assets/name,target_is_directory=name=='go_essential_all')
phase={'preprocessing':0.,'training':0.,'inference':0.};history=[]
try:
    # This file physically excludes every held-out treated cell.
    a=ad.read_h5ad(root/'trainval_controls.h5ad');assert set(a.obs.role.astype(str))=={'train','val','control'}
    assert not set(a.obs.target)&set(split.loc[split.role=='test','target'])
    a.obs['condition']=np.where(a.obs.role=='control','ctrl',a.obs.target.astype(str)+'+ctrl');a.obs['condition_name']='HepG2_'+a.obs.condition.astype(str)+'_1'
    # Uniform training loss support is independent of evaluation-condition DE or nonzero genes.
    a.uns['non_zeros_gene_idx']={name:np.arange(a.n_vars) for name in a.obs.condition_name.unique()}
    data=PertData(str(cache));data.adata=a;data.set_pert_genes();data.gene_names=a.var.gene_name;data.node_map={name:i for i,name in enumerate(data.gene_names)}
    data.dataset_name='hepg2';data.dataset_path=str(cache/'hepg2');data.split='strict_unseen_target';data.seed=args.seed;data.train_gene_set_size=.8;data.subgroup=None
    data.set2conditions={role:[target+'+ctrl' for target in split.loc[split.role==role,'target']] for role in ['train','val','test']};data.set2conditions['train'].append('ctrl')
    assert set(split.target).issubset(set(data.pert_names))
    controls=np.flatnonzero(a.obs.role.to_numpy()=='control');control_by_batch={batch:controls[a.obs.iloc[controls].batch.astype(str).to_numpy()==batch] for batch in a.obs.batch.astype(str).unique()};assert all(len(v)>0 for v in control_by_batch.values())
    training_mask=a.obs.role.to_numpy()=='train';training_mean=np.asarray(a.X[training_mask].mean(0)).ravel();control_mean=np.asarray(a.X[controls].mean(0)).ravel();fixed_top20=np.argsort(-np.abs(training_mean-control_mean),kind='stable')[:20]
    def graphs(adata,indices,seed):
        rng=np.random.default_rng(seed);result=[]
        for index in indices:
            obs=adata.obs.iloc[index];target=str(obs.target);batch=str(obs.batch)
            truth=np.asarray(adata.X[index].toarray()).ravel().astype(np.float32)
            if target=='non-targeting':control=truth;pert_idx=[-1];condition='ctrl'
            else:
                matched=int(rng.choice(control_by_batch[batch]));control=np.asarray(a.X[matched].toarray()).ravel().astype(np.float32);pert_idx=[data.node_map_pert[target]];condition=target+'+ctrl'
            result.append(Data(x=torch.from_numpy(control[:,None].copy()),y=torch.from_numpy(truth[None,:].copy()),pert_idx=pert_idx,de_idx=fixed_top20,pert=condition))
        return result
    train_graphs=graphs(a,np.flatnonzero(a.obs.role.isin(['train','control']).to_numpy()),args.seed)
    val_graphs=graphs(a,np.flatnonzero(a.obs.role.to_numpy()=='val'),args.seed+1)
    data.dataloader={'train_loader':DataLoader(train_graphs,batch_size=32,shuffle=True,drop_last=True),'val_loader':DataLoader(val_graphs,batch_size=128,shuffle=False)}
    phase['preprocessing']+=time.monotonic()-start;start=time.monotonic();model=GEARS(data,device='cuda:0',weight_bias_track=False);model.model_initialize(hidden_size=64);phase['preprocessing']+=time.monotonic()-start
    assert model.adata is a and not set(model.adata.obs.target)&set(split.loc[split.role=='test','target'])
    optimizer=torch.optim.Adam(model.model.parameters(),lr=1e-3,weight_decay=5e-4);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.5)
    def aggregate(loader,network):
        started=time.monotonic();network.eval();sums={};ns={}
        with torch.no_grad():
            for batch in loader:
                batch=batch.to('cuda:0');pred=network(batch).detach().cpu().numpy()
                for condition,row in zip(batch.pert,pred):
                    key=condition.replace('+ctrl','');sums[key]=sums.get(key,0)+row;ns[key]=ns.get(key,0)+1
        phase['inference']+=time.monotonic()-started
        return {key:np.asarray(value/ns[key]) for key,value in sums.items()},ns
    val=np.load(root/'trainval_aggregate.npz');is_val=val['split']=='val';val_targets=val['targets'][is_val];val_truth=val['truth'][is_val]
    best=None;patience=0;first_gradient=None
    for epoch in range(args.epochs):
        started=time.monotonic();model.model.train()
        for step,batch in enumerate(data.dataloader['train_loader']):
            batch=batch.to('cuda:0');optimizer.zero_grad();pred=model.model(batch)
            loss=loss_fct(pred,batch.y,batch.pert,ctrl=model.ctrl_expression,dict_filter=model.dict_filter,direction_lambda=.1);loss.backward()
            if first_gradient is None:
                first_gradient=sum(float(p.grad.norm()) for p in model.model.parameters() if p.grad is not None);assert first_gradient>0
            torch.nn.utils.clip_grad_value_(model.model.parameters(),1.);optimizer.step()
            if step%500==0:print('TRAIN',epoch+1,step,float(loss),flush=True)
        scheduler.step();phase['training']+=time.monotonic()-started
        predicted,ns=aggregate(data.dataloader['val_loader'],model.model);values=np.stack([predicted[str(t)] for t in val_targets]);error=float(np.mean((values-val_truth)**2));history.append(dict(epoch=epoch+1,validation_all_gene_intervention_mean_mse=error));print('VALIDATION',history[-1],flush=True)
        if best is None or error<best[0]:best=(error,epoch+1,copy.deepcopy(model.model.state_dict()));patience=0
        else:patience+=1
        torch.save(dict(current=model.model.state_dict(),best=best[2],optimizer=optimizer.state_dict(),epoch=epoch+1,history=history),raw/'last_training_state.pt')
        if patience>=5:break
    model.model.load_state_dict(best[2]);model.best_model=copy.deepcopy(model.model);model.save_model(str(raw/'model'))
    write_json(out/'selection.json',dict(status='frozen_before_heldout_response_open',chosen_epoch=best[1],validation_mse=best[0],history=history,test_treated_read_before_selection=False,
        graph_scope='Training-target cells and supplied control cells only; validation excluded from coexpression; no test treated file opened',loss_support='all2000 output genes for training targets, not heldout DE ranks'))
    # The held-out response files are opened only after model selection and checkpoint save.
    test=ad.read_h5ad(root/'heldout_treated.h5ad');assert not set(test.obs.target)&set(a.obs.target)
    assert np.array_equal(test.var_names,a.var_names)
    test_graphs=graphs(test,np.arange(test.n_obs),args.seed+2);test_loader=DataLoader(test_graphs,batch_size=128,shuffle=False);predicted,ns=aggregate(test_loader,model.best_model)
    observed=np.load(root/'test_aggregate.npz');targets=observed['targets'];pred=np.stack([predicted[str(t)] for t in targets]);assert sum(ns.values())==test.n_obs
    np.savez_compressed(raw/'predictions_mean.npz',targets=targets,prediction=pred,n_cells=np.array([ns[str(t)] for t in targets]),gene_names=a.var_names.to_numpy(dtype=str))
    rows=score_response(targets,observed['truth'],observed['control'],pred,observed['n_cells'],gene_names=a.var_names);pd.DataFrame(rows).to_csv(out/'per_intervention.tsv',sep='\t',index=False)
    metrics={key:float(value) for key,value in pd.DataFrame(rows).mean(numeric_only=True).items()};res=r.finish()
    write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='measured_genetic_response',regime='strict_unseen_target_with_matched_controls',model_id='GEARS_HepG2',project_commit=launch_commit,source_sha256=source_hash,upstream_commit='f374e43e197b295016d80395d7a54ddb81cc6769',seed=args.seed,
        dataset_id=contract['dataset_id'],data_contract_sha256=sha(root/'contract.json'),target_split_sha256=sha(root/'target_split.tsv'),n_train_interventions=int((split.role=='train').sum()),n_validation_interventions=int((split.role=='val').sum()),n_test_interventions=len(targets),n_test_cells=test.n_obs,n_genes=a.n_vars,
        trainable_parameters=sum(p.numel() for p in model.model.parameters() if p.requires_grad),hidden_size=64,batch_size=32,max_epochs=args.epochs,chosen_epoch=best[1],executed_epochs=len(history),first_gradient_norm=first_gradient,
        validation_selection='all-gene equal-intervention mean MSE; patience5; no test DE ranks',metrics=metrics,prediction_path=str(raw/'predictions_mean.npz'),prediction_sha256=sha(raw/'predictions_mean.npz'),
        wall_preprocessing_seconds=phase['preprocessing'],wall_training_seconds=phase['training'],wall_inference_seconds=phase['inference'],wall_total_seconds=res.pop('wall_seconds'),
        failure_or_limitations=['Native GEARS architecture, modified strict data/training/evaluation policy; not original-paper numerical reproduction','One random same-batch control per treated cell; simple B00 uses the exact full matched control mean','HepG2 single-target CRISPRi only; biological replicate and guide efficiency unknown','Unpaired intervention-mean prediction, not prospective single-cell trajectory or adult HCC treatment response'],**res))
    print('GEARS_HEPG2_COMPLETE',metrics,flush=True)
except Exception as exc:
    write_json(out/'failure.json',dict(status='failed',run_id=args.run_id,project_commit=launch_commit,source_sha256=source_hash,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),**r.finish()));raise
