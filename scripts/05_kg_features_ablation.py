"""Reactome membership proxies and degree/size-preserving random control."""
import json
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from common import DATA,ROOT,Resources,config,evaluate,load_features,provenance,sha,write_json

def randomized_sets(sets,n_genes):
    rng=np.random.default_rng(42)
    edges=[(g,j) for j,genes in enumerate(sets) for g in genes]
    edge_set=set(edges);accepted=0
    for _ in range(len(edges)*20):
        i,k=rng.integers(0,len(edges),size=2)
        g,a=edges[i];h,b=edges[k]
        if g==h or a==b or (g,b) in edge_set or (h,a) in edge_set:continue
        edge_set.remove((g,a));edge_set.remove((h,b));edge_set.add((g,b));edge_set.add((h,a))
        edges[i]=(g,b);edges[k]=(h,a);accepted+=1
    new=[set() for _ in sets]
    for g,j in edges:new[j].add(g)
    assert [len(s) for s in sets]==[len(s) for s in new]
    before=np.bincount([g for s in sets for g in s],minlength=n_genes)
    after=np.bincount([g for s in new for g in s],minlength=n_genes)
    assert np.array_equal(before,after)
    return [sorted(s) for s in new],accepted

c=config();r=Resources();a=ad.read_h5ad(DATA/'subsets'/f"{c['subset']}.h5ad")
train=a.obs.split.to_numpy()=='train';val=a.obs.split.to_numpy()=='val'
y=a.obs.cell_type_coarse.astype(str).to_numpy();labels=sorted(set(y[train]))
x=a.layers['counts'].astype(np.float32)
x=x.multiply((10000/np.maximum(np.asarray(x.sum(1)).ravel(),1))[:,None]).tocsr();x.data=np.log1p(x.data)
mu=np.asarray(x[train].mean(0)).ravel();sd=np.sqrt(np.maximum(np.asarray(x[train].power(2).mean(0)).ravel()-mu**2,0));sd[sd<1e-6]=1
mapping=DATA/'annotation/Ensembl2Reactome_All_Levels.human.tsv'
df=pd.read_csv(mapping,sep='\t',header=None,names=['ensembl','pathway','url','name','evidence','species'])
gene_index={g:i for i,g in enumerate(a.var.ensembl_id) if g}
df=df[df.ensembl.isin(gene_index)].drop_duplicates(['ensembl','pathway'])
evidence_lookup={(row.ensembl,row.pathway):(row.evidence,row.url) for row in df.itertuples()}
groups=[]
for pid,g in df.groupby('pathway',sort=True):
    genes=sorted({gene_index[x] for x in g.ensembl})
    if 10<=len(genes)<=200:groups.append((pid,g.iloc[0]['name'],genes))
assert len(groups)>=32
def proxies(sets):
    rows=[];cols=[];weights=[]
    for j,genes in enumerate(sets):
        rows.extend(genes);cols.extend([j]*len(genes));weights.extend([1/len(genes)]*len(genes))
    W=sparse.csr_matrix((weights,(rows,cols)),shape=(a.n_vars,len(sets)),dtype=np.float32)
    return np.asarray((x.multiply(1/sd)@W).toarray()-((mu/sd)@W))
features=proxies([g[2] for g in groups])
order=np.argsort(-features[train].var(0),kind='stable')
selection=[];best=None
for n in (32,64,128):
    n=min(n,len(groups));F=features[:,order[:n]]
    scaler=StandardScaler().fit(F[train]);F=scaler.transform(F)
    for C in (.01,.1,1.):
        for weight in (None,'balanced'):
            head=LogisticRegression(C=C,class_weight=weight,max_iter=2000,random_state=42).fit(F[train],y[train])
            score=f1_score(y[val],head.predict(F[val]),labels=labels,average='macro',zero_division=0)
            selection.append(dict(n_pathways=n,C=C,class_weight=weight,validation_macro_f1=score))
            if best is None or score>best[0]:best=(score,n)
n=best[1];selected=[groups[i] for i in order[:n]];p=features[:,order[:n]]
random_sets,swaps=randomized_sets([g[2] for g in selected],a.n_vars)
random_features=proxies(random_sets)
foundation=load_features(a,c['foundation_run'])
inputs={'pathway':p,'foundation_pathway':np.concatenate([foundation,p],axis=1),'foundation_random':np.concatenate([foundation,random_features],axis=1)}
results=[];runs=[]
base_metrics=json.loads((ROOT/'results'/c['foundation_run']/'metrics.json').read_text())
base_pred=pd.read_csv(DATA/'outputs'/c['foundation_run']/'predictions.tsv',sep='\t')
for name,F in inputs.items():
    rid=c['run_id']+'_'+name
    metrics=evaluate(F,a,rid)
    pred=pd.read_csv(DATA/'outputs'/rid/'predictions.tsv',sep='\t')
    assert np.array_equal(pred.cell_id,base_pred.cell_id)
    test=pred.split=='test'
    metrics['n_changed_test_predictions_vs_foundation']=int((pred.loc[test,'prediction'].to_numpy()!=base_pred.loc[test,'prediction'].to_numpy()).sum())
    metrics['delta_macro_f1_vs_foundation']=metrics['macro_f1']-base_metrics['macro_f1']
    write_json(ROOT/'results'/rid/'metrics.json',metrics)
    run=provenance(dict(c,run_id=rid),a,name,{},metrics=metrics,
       foundation_run=c['foundation_run'],n_pathways=n,source_version='Reactome 92',seed_random_sets=42,
       pathway_source_sha256=sha(mapping),data_preprocessing='log1p(10000 * counts/original library sum); train-only gene zscore; mean over members')
    runs.append((ROOT/'results'/rid/'run.json',run))
    results.append(dict(run_id=rid,**{k:metrics[k] for k in ['macro_f1','balanced_accuracy','selected_C','selected_class_weight','n_changed_test_predictions_vs_foundation','delta_macro_f1_vs_foundation']}))
for rid in (c['baseline_run'],c['foundation_run']):
    metrics=json.loads((ROOT/'results'/rid/'metrics.json').read_text())
    results.append(dict(run_id=rid,**{k:metrics[k] for k in ['macro_f1','balanced_accuracy','selected_C','selected_class_weight']}))
pd.DataFrame(results).to_csv(ROOT/'reports'/f"kg_ablation_{c['subset']}.tsv",sep='\t',index=False)
if c['subset']=='pilot':pd.DataFrame(results).to_csv(ROOT/'reports/kg_ablation.tsv',sep='\t',index=False)
resources=r.finish()
for path,run in runs:
    run.update(resources);run['resource_measurement_scope']='combined KG feature preparation and all three ablation heads'
    write_json(path,run)
write_json(ROOT/'results'/c['run_id']/'selection.json',dict(selected_n=n,validation_grid=selection,random_seed=42,successful_degree_preserving_swaps=swaps,
    size_preserved=True,gene_membership_degree_preserved=True,**resources))
rows=[];random_rows=[]
for j,(pid,name,genes) in enumerate(selected):
    for i in genes:
        evidence,url=evidence_lookup[(a.var.ensembl_id.iloc[i],pid)]
        rows.append(dict(head_id=a.var.ensembl_id.iloc[i],head_type='human_gene',relation='member_of',tail_id=pid,tail_type='pathway',source_name='Reactome',source_version='92',evidence_id=evidence,species='Homo sapiens',pathway_name=name,source_url=url))
    for i in random_sets[j]:random_rows.append(dict(random_set=j,gene_id=a.var.ensembl_id.iloc[i],matched_pathway=pid))
pd.DataFrame(rows).to_csv(ROOT/'manifests'/f"kg_edges_{c['subset']}.tsv",sep='\t',index=False)
pd.DataFrame(random_rows).to_csv(ROOT/'manifests'/f"kg_random_membership_{c['subset']}.tsv",sep='\t',index=False)
pd.DataFrame([dict(source_name='Reactome',source_version='92',path=str(mapping),sha256=sha(mapping),license='CC0',url='https://zenodo.org/records/15126939')]).to_csv(ROOT/'manifests/kg_sources.tsv',sep='\t',index=False)
(ROOT/'reports/kg_integration.md').write_text('''# KG 특징 실험\n\n기존 VCWorld export는 provenance가 포함된 표준 gene-pathway membership 테이블이 아니므로 수정하지 않고 보존했다. 공식 Reactome v92 human Ensembl→pathway를 사용한다. gene expression은 환자 관측, membership은 외부 prior다. 경로별 평균 train-zscore를 재현 가능한 proxy로 사용하며 활성화/인과관계로 해석하지 않는다.\n\n10–200개의 관측 gene을 포함하는 경로를 train feature variance로 정렬하고, 32/64/128 경로 수 및 logistic C/weight를 validation에서만 선택한다. test는 선택에 사용하지 않는다. foundation은 scGPT frozen embedding. 크기/유전자 membership degree를 모두 보존한 edge-swap random control(seed42)을 사용한다. pathway membership을 예측 입력에 concatenate하는 실험이며 LLM/RAG나 인과 모형이 아니다.\n\n실제 성능은 kg_ablation_*.tsv. 이득이 없거나 random보다 약해도 그대로 보고한다. 모든 head는 train-only scaling, 동일 환자 split·평가 cell IDs·class 정의를 사용한다. test 환자 2명은 외적 타당성 판단에 제한적이다.\n''')
print(pd.DataFrame(results).to_string(index=False),flush=True)
