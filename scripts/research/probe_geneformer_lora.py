"""Real V2/4096 CLS LoRA integration proof on development training cells only."""
import pickle,time
import anndata as ad
import torch,yaml
from datasets import load_from_disk
from transformers import BertForMaskedLM
from peft import LoraConfig,get_peft_model,PeftModel
from research_core import *

run_id='M03_lora_module_gradient_reload_probe_v1';out=REPORT/'runs'/run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True)
raw=ASSETS/'runs'/run_id;raw.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42);torch.set_num_threads(8);r=Resources()
c=yaml.safe_load((ROOT/'configs/research/protocol_annotation_full.yaml').read_text());a=ad.read_h5ad(DATA/c['dataset_path']);ids=donor_roles(a.obs,c,0)['inner_training'][:2]
ds=load_from_disk(str(DATA/'tokenized/research_V2_tokens_full_v1/cells.dataset'));assert ds['cell_id']==a.obs_names.tolist()
ckpt=DATA/'checkpoints/Geneformer-V2-104M'
def load_base():
    full=BertForMaskedLM.from_pretrained(ckpt,local_files_only=True,output_hidden_states=True)
    return full.bert.eval().requires_grad_(False).cuda()
base=load_base();head=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.GELU(),torch.nn.Linear(128,6)).cuda()
from geneformer import TOKEN_DICTIONARY_FILE
with open(TOKEN_DICTIONARY_FILE,'rb') as f:pad=pickle.load(f)['<pad>']
seqs=[ds[int(i)]['input_ids'] for i in ids];size=max(map(len,seqs));batch=torch.full((len(seqs),size),pad,dtype=torch.long,device='cuda')
for i,s in enumerate(seqs):batch[i,:len(s)]=torch.tensor(s,device='cuda')
labels=torch.tensor([c['labels'].index(str(a.obs.iloc[i].cell_type_coarse)) for i in ids],device='cuda')
def output(model):
    with torch.autocast('cuda',dtype=torch.bfloat16):return model(input_ids=batch,attention_mask=(batch!=pad).long(),output_hidden_states=True).hidden_states[11][:,0,:].float()
with torch.no_grad():before=output(base).detach().clone()
# hidden_states[11] uses encoder blocks 0..10. The unused final block11 is excluded.
targets=[name for name,module in base.named_modules() if isinstance(module,torch.nn.Linear) and any(name==f'encoder.layer.{layer}.attention.self.{projection}' for layer in range(11) for projection in ['query','value'])]
assert len(targets)==22
model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=targets,bias='none'));model.eval()
with torch.no_grad():after=output(model)
initial_max=float(torch.max(torch.abs(before-after)));assert initial_max==0.,initial_max
trainable={n:p for n,p in model.named_parameters() if p.requires_grad};assert trainable and all('lora_' in n for n in trainable)
optimizer=torch.optim.AdamW(list(trainable.values())+list(head.parameters()),lr=1e-4)
original={n:p.detach().clone() for n,p in trainable.items()};steps=[]
for step in range(2):
    optimizer.zero_grad();features=output(model);logits=head(features);loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward()
    gradients={n:float(p.grad.norm()) if p.grad is not None else 0. for n,p in trainable.items()}
    assert any(v>0 for n,v in gradients.items() if 'lora_B' in n)
    if step==1:assert any(v>0 for n,v in gradients.items() if 'lora_A' in n)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
    steps.append(dict(step=step,loss=float(loss),nonzero_gradient_tensors=sum(v>0 for v in gradients.values()),gradients=gradients));optimizer.step()
changed={n:float((p-original[n]).abs().max()) for n,p in trainable.items()};assert any(v>0 for v in changed.values())
head.eval()
with torch.no_grad():expected=head(output(model)).detach().cpu()
model.save_pretrained(raw/'adapter',safe_serialization=True);torch.save(head.state_dict(),raw/'head.pt')
del model,base,optimizer;torch.cuda.empty_cache()
restored=PeftModel.from_pretrained(load_base(),raw/'adapter',is_trainable=False).eval();head.load_state_dict(torch.load(raw/'head.pt',weights_only=True))
with torch.no_grad():actual=head(output(restored)).detach().cpu()
reload_max=float((expected-actual).abs().max());assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-6),reload_max
res=r.finish();write_json(out/'run.json',dict(run_id=run_id,status='executed',task='adapter_integration_check',model_id='Geneformer_V2_LoRA',project_commit=git_commit(),base_revision='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5',checkpoint_sha256=sha(ckpt/'model.safetensors'),seed=42,
    selected_training_cell_ids=a.obs_names[ids].tolist(),forbidden_legacy_donors=['HCC02','HCC09'],pooling='hidden_states[11] CLS; final encoder block11 deliberately excluded',input_lengths=list(map(len,seqs)),precision='BF16 matmul; FP32 weights and head',
    rank=8,alpha=16,dropout=0.,target_modules=targets,adapter_parameter_count=sum(p.numel() for p in trainable.values()),head_parameter_count=sum(p.numel() for p in head.parameters()),
    initial_output_max_abs_difference=initial_max,gradient_steps=steps,parameter_max_abs_updates=changed,save_reload_logits_max_abs_difference=reload_max,
    adapter_sha256=sha(raw/'adapter/adapter_model.safetensors'),head_sha256=sha(raw/'head.pt'),failure_or_limitations=['Integration proof only, not model efficacy; actual task fine-tuning and matched controls remain required'],**res))
print('LORA_INTEGRATION_PASSED',initial_max,reload_max,flush=True)
