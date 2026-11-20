"""R1: original logged GEARS source and an explicitly modified public-data replay."""
import argparse, csv, inspect, json, os, sys, textwrap, time
from pathlib import Path
from run_support import *

p=argparse.ArgumentParser()
p.add_argument('--split-seed',type=int,choices=range(1,6),required=True)
p.add_argument('--run-id',required=True)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
assert torch.cuda.is_available()
upstream=ROOT/'.upstream/gears-paper'
revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
assert revision=='9d27bd619935cebe6f1519c3b53f61d2777dee59'
assert not subprocess.check_output(['git','-C',str(upstream),'diff','HEAD','--','gears'])
sys.path.insert(0,str(upstream))
from gears import GEARS,PertData
from gears.inference import non_dropout_analysis,compute_metrics,evaluate
import torch
protocol=ROOT/'protocols/reproduction/gears_norman.yaml'
out,run_lock=claim_run('R1',a.run_id,protocol,dict(seed=a.split_seed,source=revision,adapter_sha256=digest(__file__),support_sha256=digest(Path(__file__).with_name('run_support.py')),epochs=20,batch=32),a.resume)
t0=time.monotonic()
data_root=ROOT/'assets/cache/R1'
lock=link_dataset(data_root)
seed_all(0) # deterministic new graph/control-pair cache; private original cache unavailable
data=PertData(str(data_root))
data.load(data_name='norman')
data.gene_names=data.adata.var.gene_name # original loader only sets this on a cache miss
data.prepare_split(split='simulation',seed=a.split_seed)
lock.close()
write(out/'split.json',split_record(data))
data.get_dataloader(batch_size=32,test_batch_size=32)
seed_all(a.split_seed)
g=GEARS(data,device='cuda:0',weight_bias_track=False)
g.model_initialize(hidden_size=64)

def before(model, optimizer, scheduler, path):
    best = model
    if path.exists():
        if not a.resume:
            raise RuntimeError('Existing epoch checkpoint requires --resume')
        state = torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['current'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best = __import__('copy').deepcopy(model)
        best.load_state_dict(state['best'])
        restore_rng(state['rng'])
        return state['epoch'],state['min_val'],best
    return 0,np.inf,best

def after(model, best, optimizer, scheduler, epoch, minimum, path):
    checkpoint(path, dict(current=model.state_dict(),best=best.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,min_val=minimum,rng=rng_state()))
    print('EPOCH_CHECKPOINT',epoch,minimum,flush=True)

# Inject only epoch-boundary persistence into the author train/selection loop.
source=textwrap.dedent(inspect.getsource(GEARS.train))
old='    for epoch in range(epochs):'
assert source.count(old)==1
source=source.replace(old,'    start_epoch, min_val, best_model = before(self.model, optimizer, scheduler, resume_path)\n    for epoch in range(start_epoch, epochs):')
old='    print_sys("Done!")\n    self.best_model = best_model'
assert source.count(old)==1
source=source.replace(old,'        after(self.model, best_model, optimizer, scheduler, epoch + 1, min_val, resume_path)\n'+old)
# Route native final evaluation to the explicit export below, avoiding duplicate inference.
assert '    if \'test_loader\' not in self.dataloader:' in source
source=source[:source.index("    if 'test_loader' not in self.dataloader:")]
ns=dict(sys.modules[GEARS.__module__].__dict__,before=before,after=after,resume_path=out/'resume.pt')
(out/'training_runtime.py').write_text(source)
exec(compile(source,str(out/'training_runtime.py'),'exec'),ns)
train_start=time.monotonic()
ns['train'](g,epochs=20)
train_time=time.monotonic()-train_start
g.save_model(str(out/'model'))
test=g.dataloader['test_loader']
infer_start=time.monotonic()
pred=evaluate(test,g.best_model,g.config['uncertainty'],'cuda:0')
pred['pert_cat']=np.asarray(pred['pert_cat'],dtype=str)
np.savez(out/'prediction.npz',**pred,gene_names=np.array(g.gene_list,dtype=str))
# Same supplied test control IDs/pairs as the trained model, identity output.
ctrl=np.concatenate([batch.x[:,0].reshape(-1,len(g.gene_list)).numpy() for batch in test])
none={**pred,'pred':ctrl}
obs=non_dropout_analysis(g.adata,pred)
baseline=non_dropout_analysis(g.adata,none)
metric='mse_top20_de_non_dropout'
singles={next(v for v in p.split('+') if v!='ctrl') for p in g.set2conditions['train'] if p!='ctrl' and 'ctrl' in p.split('+')}
rows=[]
for category,perts in g.subgroup['test_subgroup'].items():
    for pert in perts:
        if pert not in obs or pert not in baseline:
            raise RuntimeError(f'Author metric missing for {pert}')
        rows.append(dict(group=category,condition=pert,n_train_singles=sum(gene in singles for gene in pert.split('+') if gene!='ctrl'),gears=float(obs[pert][metric]),no_perturb=float(baseline[pert][metric])))
with (out/'fig2f_conditions.tsv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
metrics,_=compute_metrics(pred)
write(out/'result.json',dict(status='modified_reproduction_executed',split_seed=a.split_seed,n_train_batches=len(g.dataloader['train_loader']),n_test_cells=len(pred['pert_cat']),gpu=torch.cuda.get_device_name(),peak_gib=torch.cuda.max_memory_allocated()/2**30,train_seconds=train_time,inference_seconds=time.monotonic()-infer_start,total_seconds=time.monotonic()-t0,metrics={k:float(v) for k,v in metrics.items() if np.isfinite(v)},prediction_sha256=digest(out/'prediction.npz'),model_sha256=digest(out/'model/model.pt'),split_sha256=digest(out/'split.json')))
print('R1_FINISHED',out,flush=True)
