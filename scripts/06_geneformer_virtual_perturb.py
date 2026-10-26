"""Independent token deletion and matched controls; exploratory embedding shifts only."""
import copy, json
import anndata as ad
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from geneformer.perturber_utils import delete_indices
from common import DATA,ROOT,Resources,config,provenance,sha,write_json
from geneformer_utils import CHECKPOINT,REV,embeddings,load_model,tokenizer_and_data

c=config();r=Resources();np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(8)
a=ad.read_h5ad(DATA/'subsets'/f"{c['subset']}.h5ad")
path=DATA/'tokenized'/c['token_run']/'cells.dataset'
if path.exists():
    import pickle
    from geneformer import TOKEN_DICTIONARY_FILE
    ds=load_from_disk(str(path))
    with open(TOKEN_DICTIONARY_FILE,'rb') as f:tokens=pickle.load(f)
    assert ds['cell_id']==a.obs_names.tolist()
else:ds,tokens,_=tokenizer_and_data(a,c['token_run'])
eligible=np.flatnonzero((a.obs.split.to_numpy()=='train') & (a.obs.cell_type_coarse.to_numpy()==c['lineage']))
assert len(eligible)>0
selected=np.sort(np.random.default_rng(42).choice(eligible,size=min(c['max_cells'],len(eligible)),replace=False))
cell_data=ds.select(selected.tolist())
mean=np.asarray(a.layers['counts'][eligible].mean(0)).ravel()
gene_lookup={g:i for i,g in enumerate(a.var.gene_symbol)}
vocab_ix=[i for i,g in enumerate(a.var.ensembl_id) if g in tokens and mean[i]>0 and a.var.gene_symbol.iloc[i] not in c['targets']]
targets=[];rng=np.random.default_rng(42)
for gene in c['targets']:
    i=gene_lookup[gene];eid=a.var.ensembl_id.iloc[i]
    assert eid in tokens and mean[i]>0,('target not expressed/in vocabulary',gene)
    candidates=sorted(vocab_ix,key=lambda j:(abs(np.log1p(mean[j])-np.log1p(mean[i])),a.var.gene_symbol.iloc[j]))[:10]
    control=int(rng.choice(candidates))
    targets.append(dict(gene=gene,ensembl_id=eid,token=tokens[eid],condition='independent_delete',train_lineage_mean_count=float(mean[i])))
    targets.append(dict(gene=a.var.gene_symbol.iloc[control],ensembl_id=a.var.ensembl_id.iloc[control],token=tokens[a.var.ensembl_id.iloc[control]],condition='expression_matched_random_delete',matched_target=gene,train_lineage_mean_count=float(mean[control]),target_train_mean_count=float(mean[i])))
write_json(ROOT/'results'/c['run_id']/'prespecified_selection.json',dict(lineage=c['lineage'],targets=c['targets'],basis='ALB: hepatocyte identity marker; HNF4A: hepatic transcription factor. Tutorial hypotheses, not validated treatment targets. Selection and expression matching use train only.',controls=targets,n_cells=len(selected),seed=42))
model,parameter_count=load_model()
original=embeddings(model,cell_data,tokens,c['batch_size'])
noop=embeddings(model,cell_data,tokens,c['batch_size'])
def cosine_shift(x,y):return np.maximum(0,1-np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)))
rows=[]
for j,s in enumerate(cosine_shift(original,noop)):
    rows.append(dict(cell_id=cell_data[j]['cell_id'],gene='none',condition='no_op',token_present=True,cosine_shift=float(s),interpretation='실측 섭동 검증 전 탐색'))
for target in targets:
    examples=[];has=[]
    for record in cell_data:
        record=copy.deepcopy(record)
        indices=[i for i,t in enumerate(record['input_ids']) if t==target['token']]
        has.append(bool(indices));record['perturb_index']=indices
        record=delete_indices(record)  # official deletion function, one gene per operation
        assert target['token'] not in record['input_ids']
        examples.append(record)
    perturbed=Dataset.from_list(examples)
    changed=embeddings(model,perturbed,tokens,c['batch_size'])
    shifts=cosine_shift(original,changed)
    for j,s in enumerate(shifts):
        rows.append(dict(cell_id=cell_data[j]['cell_id'],gene=target['gene'],condition=target['condition'],matched_target=target.get('matched_target',target['gene']),token_present=has[j],cosine_shift=float(s),interpretation='실측 섭동 검증 전 탐색'))
table=pd.DataFrame(rows)
table.to_csv(ROOT/'reports'/f"virtual_perturbation_embedding_shift_{c['subset']}.tsv",sep='\t',index=False)
summary=table.groupby(['gene','condition','token_present'],dropna=False).cosine_shift.agg(['count','mean','median','max'])
summary.to_csv(ROOT/'reports'/f"virtual_perturbation_summary_{c['subset']}.tsv",sep='\t')
if c['subset']=='pilot':
    table.to_csv(ROOT/'reports/virtual_perturbation_embedding_shift.tsv',sep='\t',index=False)
    summary.to_csv(ROOT/'reports/virtual_perturbation_summary.tsv',sep='\t')
run=provenance(c,a,'Geneformer-V2-104M independent token-deletion CLS shift',r,
    upstream_commit=REV,checkpoint_revision=REV,checkpoint_sha256=sha(CHECKPOINT/'model.safetensors'),actual_parameter_count=parameter_count,
    batch_size=c['batch_size'],max_sequence_length=4096,dtype='float32',n_perturbed_cells=len(selected),lineage=c['lineage'],task_detail='embedding_shift',
    limitations=['실측 섭동 검증 전 탐색','No expression prediction, logFC, viability or drug efficacy','No reliable target state label; shift magnitude only','No classifier confidence interpreted as normalization'])
run['task']='embedding_shift'
run['n_input_dataset_cells']=a.n_obs
run['n_cells']=len(selected)
write_json(ROOT/'results'/c['run_id']/'run.json',run)
(ROOT/'reports/virtual_perturbation_limitations.md').write_text('''# 가상 삭제 결과의 한계\n\n**실측 섭동 검증 전 탐색.** 출력은 Geneformer V2의 rank token을 하나씩 삭제한 전후 CLS embedding cosine distance(1−cosine)다. raw expression을 0으로 만든 실측 knockout이 아니며 post-perturbation expression, logFC, 생존율, 약효를 예측하지 않는다.\n\nTrain hepatocyte만 최대 100세포 선택, ALB(계열 marker)와 HNF4A(hepatic TF)를 사전 지정. 실제 발현·vocabulary를 확인하고 각 조작에서 token 존재를 기록한다. 미존재 token은 변화가 없으며 실제 삭제된 결과와 구분한다. 동일 세포 no-op과 train expression 수준이 가까운 무작위 gene 대조(seed42)를 포함한다. 다유전자 공동 삭제를 하지 않음. 공식 delete_indices와 공식 CLS extraction을 사용하되 변화 크기만 보고하며 전체 InSilicoPerturber 상태전이 protocol을 재현한 결과는 아니다.\n\n신뢰할 정상/질환 상태 라벨과 목표 reference가 없어 방향성, 정상화, HCC 치료 표적 우열을 주장하지 않는다. cell type head confidence를 정상화로 해석하지 않는다. embedding 변화가 크다고 생물학적 영향이 크다는 뜻이 아니다. 실제 genetic perturbation 데이터와 독립 실험으로 확인하기 전 가설 생성용이다.\n''')
print(table.groupby(['gene','condition','token_present']).cosine_shift.agg(['count','mean','max']).to_string(),flush=True)
from plot_perturbation import plot_shifts
plot_shifts(table,c['subset'])
