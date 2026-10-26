"""Execute the pinned official scGPT MS notebook code with recorded compatibility patches."""
import argparse,difflib,json,os,re,sys,time
from pathlib import Path
from research_core import *
import torch

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--smoke',action='store_true');args=p.parse_args()
out=REPORT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=True)
assert not (out/'run.json').exists(),'Immutable run ID exists'
runroot=ASSETS/'official/scGPT_runs'/args.run_id
(runroot/'tutorials').mkdir(parents=True,exist_ok=True);(runroot/'data').mkdir(exist_ok=True);(runroot/'save').mkdir(exist_ok=True)
for link,target in [(runroot/'data/ms',ASSETS/'official/scGPT_MS'),(runroot/'save/scGPT_human',DATA/'checkpoints/scgpt-whole-human')]:
    if not link.exists():link.symlink_to(target,target_is_directory=True)
path=ROOT/'third_party/scGPT/tutorials/Tutorial_Annotation.ipynb';nb=json.loads(path.read_text())
original='\n\n'.join(''.join(cell['source']) for cell in nb['cells'] if cell['cell_type']=='code')
source=original[:original.rindex('\n# plot\n')]
source=source.replace('    import scvi\n','').replace('import scvi\n','')
source=source.replace('from torchtext.vocab import Vocab','from scgpt.tokenizer.vocab_compat import BuiltinVocab as Vocab')
source=re.sub(r'from torchtext\._torchtext import \(\s*Vocab as VocabPybind,?\s*\)', 'VocabPybind = None # unused when loading official pretrained vocabulary', source)
source=source.replace('epochs=10,',f'epochs={args.epochs},').replace('batch_size=32,',f'batch_size={args.batch_size},').replace('fast_transformer=True,','fast_transformer=False,')
source=source.replace('run = wandb.init(', 'run = wandb.init(mode="disabled",')
source=source.replace('config = wandb.config','config = run.config')
source=source.replace('num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),','num_workers=min(4, eval_batch_size // 2),')
start=source.index('if config.load_model is not None:\n    try:\n        model.load_state_dict')
end=source.index('pre_freeze_param_count',start)
source=source[:start]+'''if config.load_model is not None:
    from scgpt.utils import load_pretrained
    official_state = torch.load(model_file, map_location="cpu", weights_only=True)
    load_pretrained(model, official_state, verbose=False)
    # The official helper maps flash Wqkv keys to PyTorch in_proj keys.
    checked_attention = 0
    for original_key, value in official_state.items():
        key = original_key.replace("Wqkv.", "in_proj_")
        if "transformer_encoder" in key and "in_proj_" in key and key in model.state_dict():
            assert torch.equal(model.state_dict()[key].cpu(), value.cpu()), key
            checked_attention += 1
    assert checked_attention == 24, checked_attention
    print("PRETRAINED_ATTENTION_VERIFIED", checked_attention, flush=True)

'''+source[end:]
if args.smoke:
    source=source.replace('adata = sc.read(data_dir / "c_data.h5ad")','adata = sc.read(data_dir / "c_data.h5ad")[:64].copy()')
    source=source.replace('adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")','adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")[:64].copy()')
source=source.replace('for epoch in range(1, epochs + 1):','_research_train_start = time.monotonic()\n_research_preprocessing_seconds = _research_train_start - _research_start\nfor epoch in range(1, epochs + 1):')
source=source.replace('# %% inference\ndef test','_research_training_seconds = time.monotonic() - _research_train_start\n\n# %% inference\ndef test')
source=source.replace('predictions, labels, results = test(best_model, adata_test)','_research_inference_start = time.monotonic()\npredictions, labels, results = test(best_model, adata_test)\n_research_inference_seconds = time.monotonic() - _research_inference_start')
dest=runroot/'official_annotation_patched.py';dest.write_text(source)
patch=''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='official_Tutorial_Annotation.py',tofile='executed_compatibility_patch.py'))
(out/'compatibility.patch').write_text(patch)
write_json(out/'execution_contract.json',dict(status='running',source_notebook_sha256=sha(path),executed_script_sha256=sha(dest),
    upstream_commit='cebd6fae655b9c585a4807daa3ac31bb764f06b4',seed=0,epochs=args.epochs,batch_size=args.batch_size,max_sequence_length=3001,
    modifications=['non-flash backend','official weight-key compatibility loader; all24 QKV tensors checked','remove unused torchtext/scvi imports','wandb disabled; no network logging','omit post-metric visualizations; save model/predictions separately']+(['64reference/64query smoke subset'] if args.smoke else [])))
torch.set_num_threads(8);os.environ['WANDB_MODE']='disabled';os.environ['MPLBACKEND']='Agg'
resources=Resources();started=time.monotonic();os.chdir(runroot/'tutorials')
namespace={'__name__':'official_scgpt_reproduction','_research_start':started}
try:
    exec(compile(source,str(dest),'exec'),namespace)
    final=namespace['results'];pred=namespace['predictions'];labels=namespace['labels'];id2type=namespace['id2type']
    cells=namespace['adata_test'].obs_names
    pd.DataFrame(dict(cell_id=cells,truth=[id2type[int(i)] for i in labels],prediction=[id2type[int(i)] for i in pred])).to_csv(runroot/'predictions.tsv',sep='\t',index=False)
    torch.save(namespace['best_model'].state_dict(),runroot/'best_model.pt')
    res=resources.finish();run=dict(run_id=args.run_id,status='executed',task='annotation',regime='official_reproduction',dataset_version='official_MS_c_data_and_filtered_ms_adata',
        reproduction_status='smoke_passed' if args.smoke else 'modified_reproduction',project_commit=git_commit(),upstream_commit='cebd6fae655b9c585a4807daa3ac31bb764f06b4',
        base_checkpoint_revision='official whole-human snapshot',checkpoint_sha256=sha(DATA/'checkpoints/scgpt-whole-human/best_model.pt'),seed=0,n_reference=namespace['adata'].n_obs,n_evaluation=len(labels),
        actual_epochs=args.epochs,batch_size=args.batch_size,trainable_parameters=sum(p.numel() for p in namespace['model'].parameters() if p.requires_grad),
        pretrained_attention_tensors_verified=namespace['checked_attention'],metrics=final,prediction_path=str(runroot/'predictions.tsv'),prediction_sha256=sha(runroot/'predictions.tsv'),
        wall_preprocessing_seconds=namespace['_research_preprocessing_seconds'],wall_training_seconds=namespace['_research_training_seconds'],wall_inference_seconds=namespace['_research_inference_seconds'],wall_total_seconds=res.pop('wall_seconds'),
        failure_or_limitations=['attention backend and runtime versions differ; not identical numerical reproduction','official random reference validation split retained; this is not HCC donor CV','original tutorial reported values and available published variation will be compared explicitly'],**res)
    write_json(out/'run.json',run);print('OFFICIAL_REPRODUCTION_COMPLETE',json.dumps(run),flush=True)
except Exception as exc:
    write_json(out/'failure.json',dict(run_id=args.run_id,status='failed',error=f'{type(exc).__name__}: {exc}',source_notebook_sha256=sha(path),executed_script=str(dest),**resources.finish()))
    raise
