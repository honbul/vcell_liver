"""R2: execute the Fig. 3a author script with recorded compatibility edits."""
import argparse, logging, math, os, time, types
from run_support import *
from scgpt_author import author_sources, GeneVocab, ZENODO, GEARS

ap=argparse.ArgumentParser()
ap.add_argument('--run-id',required=True)
ap.add_argument('--resume',action='store_true')
a=ap.parse_args()
assert torch.cuda.is_available()
out,run_lock=claim_run('R2',a.run_id,ROOT/'protocols/reproduction/scgpt_norman.yaml',dict(split_seed=1,training_seed=0,source=ZENODO,loader=GEARS,adapter_sha256=digest(__file__),support_sha256=digest(Path(__file__).with_name('run_support.py')),author_compat_sha256=digest(Path(__file__).with_name('scgpt_author.py'))),a.resume)
code=author_sources(out)
logging.basicConfig(level=logging.INFO)
data_root=ROOT/'assets/cache/R2'
data_lock=link_dataset(data_root)
start=time.monotonic()
def replace(old,new):
    global code
    assert code.count(old)==1, old
    code=code.replace(old,new,1)
for line in ['import wandb','from torchtext._torchtext import Vocab as VocabPybind','from torchtext.vocab import Vocab','sys.path.insert(0, "../")','from scgpt.tokenizer.gene_tokenizer import GeneVocab']:
    replace(line+'\n','')
replace('    load_model="/scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-May23-08-36-2023",','    load_model=str(PUBLIC / "scgpt"),')
replace('    use_fast_transformer=True,','    use_fast_transformer=False,')
replace('pert_data = PertData("/scratch/ssd004/scratch/chloexq/perturb_analysis")','pert_data = PertData(str(data_root))')
replace('pert_data.prepare_split(split=split, seed=1)','pert_data.prepare_split(split=split, seed=1)\ndata_lock.close()')
replace('pert_data.get_dataloader(batch_size=config.batch_size, test_batch_size=eval_batch_size)','write(out / "split.json", split_record(pert_data))\npert_data.get_dataloader(batch_size=config.batch_size, test_batch_size=eval_batch_size)')
replace('save_dir = Path(f"./save/dev_perturb_{data_name}-{time.strftime(\'%b%d-%H-%M\')}/")','save_dir = out')
replace('with open(save_dir / "git_diff.diff", "w") as f:\n    f.write(scg.utils.get_git_diff())','')
replace('\n    pretrained_dict = torch.load(model_file)\n','\n    pretrained_dict = torch.load(model_file, map_location="cpu", weights_only=True)\n')
replace('    for k, v in pretrained_dict.items():\n        logger.info(f"Loading params {k} with shape {v.shape}")','    pretrained_dict = {k.replace("Wqkv.", "in_proj_"):v for k,v in pretrained_dict.items()}\n    assert all(k in model_dict and model_dict[k].shape==v.shape for k,v in pretrained_dict.items())\n    for k, v in pretrained_dict.items():\n        logger.info(f"Loading params {k} with shape {v.shape}")')
replace('    model.load_state_dict(model_dict)\nelif','    model.load_state_dict(model_dict)\n    assert all(torch.equal(model.state_dict()[k].cpu(), v) for k,v in pretrained_dict.items())\n    write(out / "checkpoint_load.json", dict(loaded_keys=sorted(pretrained_dict), tensor_values_verified=True, vocab_matched=int((gene_ids_in_vocab>=0).sum()), total=n_genes))\nelif')
replace('            input_pert_flags = pert_flags[:, input_gene_ids]','            input_pert_flags = pert_flags[:, input_gene_ids]\n            sampling_counts[0] += (pert_flags==1).sum()\n            sampling_counts[1] += (input_pert_flags==1).sum()\n            sampling_counts[2] += len(pert_flags)')
code=code.replace('.astype(np.float)', '.astype(np.float64)')
replace('for epoch in range(1, config.epochs + 1):','start_epoch, best_val_corr, best_model, patience = rf_restore(model,optimizer,scheduler,scaler)\nfor epoch in range(start_epoch, config.epochs + 1):')
replace('            break\n\n    # torch.save(', '            rf_save(epoch,model,best_model,optimizer,scheduler,scaler,best_val_corr,patience,True)\n            break\n\n    # torch.save(')
replace('    scheduler.step()  # TODO: have this back','    scheduler.step()  # TODO: have this back\n    rf_save(epoch,model,best_model,optimizer,scheduler,scaler,best_val_corr,patience,False)')
# Visualization between training and final evaluation is not needed to score panel a.
i=code.index('for p in perts_to_plot:\n    fig = plot_perturbation(')
j=code.index('test_loader = pert_data.dataloader["test_loader"]',i)
code=code[:i]+code[j:]
replace('test_res = eval_perturb(test_loader, best_model, device)\n# test_metrics','test_res = eval_perturb(test_loader, best_model, device)\nnp.savez(out / "prediction.npz", **test_res, gene_names=np.asarray(genes,dtype=str))\n# test_metrics')
assert code.count('print(test_metrics)')==2 or code.count('print(test_metrics)')==1
(out/'manuscript_runtime.py').write_text(code)
sampling_counts=torch.zeros(3,dtype=torch.int64,device='cuda')
def rf_restore(model,opt,scheduler,scaler):
    path=out/'resume.pt'
    if not path.exists():return 1,0,None,0
    assert a.resume
    s=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(s['current']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);scaler.load_state_dict(s['scaler'])
    import copy
    best=copy.deepcopy(model);best.load_state_dict(s['best'])
    restore_rng(s['rng'])
    sampling_counts.copy_(s['sampling_counts'])
    return (16 if s['stopped'] else s['epoch']+1),s['score'],best,s['patience']
def rf_save(epoch,model,best,opt,scheduler,scaler,score,patience,stopped):
    checkpoint(out/'resume.pt',dict(epoch=epoch,current=model.state_dict(),best=best.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),score=score,patience=patience,stopped=stopped,sampling_counts=sampling_counts.cpu()))
    print('EPOCH_CHECKPOINT',epoch,score,flush=True)
class LocalLogging:
    def init(self,**kw):self.config=types.SimpleNamespace(**kw['config']);return self
    def log(self,*args,**kw):pass
    def watch(self,*args,**kw):pass
    def finish(self):pass
    Settings=lambda self,**kw: None

env=dict(globals(),wandb=LocalLogging())
exec(compile(code,str(out/'manuscript_runtime.py'),'exec'),env)
state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
write(out/'result.json',dict(status='modified_reproduction_executed',epochs=state['epoch'],split_sha256=digest(out/'split.json'),prediction_sha256=digest(out/'prediction.npz'),model_sha256=digest(out/'best_model.pt'),runtime_sha256=digest(out/'manuscript_runtime.py'),gpu=torch.cuda.get_device_name(),peak_gib=torch.cuda.max_memory_allocated()/2**30,total_seconds=time.monotonic()-start,metrics={k:float(v) for k,v in env['test_metrics'].items() if np.isscalar(v) and np.isfinite(v)},target_flags_before_sampling=int(sampling_counts[0]),target_flags_after_sampling=int(sampling_counts[1]),training_examples=int(sampling_counts[2])))
print('R2_FINISHED',out,flush=True)
