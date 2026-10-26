"""Stream gene rows to sparse counts; fixed donor splits and nested subsets."""
import datetime, gzip, json, re, time
from collections import defaultdict
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from common import DATA, RAW, ROOT, config, sha, write_json

def subset_indices(obs,n,seed=42):
    if len(obs)<=n:return np.arange(len(obs))
    groups=list(obs.groupby(['patient_id','cell_type_coarse'],observed=True).indices.values())
    assert len(groups)<=n
    weights=np.array([len(g) for g in groups],float)
    target=weights/weights.sum()*(n-len(groups))
    alloc=np.floor(target).astype(int)+1
    for i in np.argsort(-(target-np.floor(target)))[:n-alloc.sum()]:alloc[i]+=1
    # Cap rare groups and redistribute deterministically.
    alloc=np.minimum(alloc,weights.astype(int))
    while alloc.sum()<n:
        for i in np.argsort(-weights):
            if alloc[i]<weights[i]:alloc[i]+=1
            if alloc.sum()==n:break
    rng=np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(g,size=k,replace=False) for g,k in zip(groups,alloc)]))

def main():
    c=config();start=time.monotonic()
    for p in ('processed','subsets','annotation'): (DATA/p).mkdir(parents=True,exist_ok=True)
    (ROOT/'manifests/splits').mkdir(parents=True,exist_ok=True)
    meta_file=RAW/'GSE149614_HCC.metadata.updated.txt.gz'
    count_file=RAW/'GSE149614_HCC.scRNAseq.S71915.count.txt.gz'
    meta=pd.read_csv(meta_file,sep='\t',index_col='Cell')
    assert meta.index.is_unique
    rows=[];cols=[];values=[];genes=[];indptr=[0]
    with gzip.open(count_file,'rt') as f:
        cells=next(f).rstrip('\n').split('\t')
        assert len(cells)==len(set(cells))
        for i,line in enumerate(f):
            gene,counts=line.rstrip('\n').split('\t',1)
            x=np.fromstring(counts,sep='\t',dtype=np.float64)
            assert len(x)==len(cells) and np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.floor(x)).all(),(i,gene)
            ix=np.flatnonzero(x)
            cols.append(ix.astype(np.int32));values.append(x[ix].astype(np.int32));indptr.append(indptr[-1]+len(ix));genes.append(gene)
            if i%3000==0: print('parsed genes',i,'nonzero counts',indptr[-1],flush=True)
    X=sparse.csr_matrix((np.concatenate(values),np.concatenate(cols),np.asarray(indptr)),shape=(len(genes),len(cells))).T.tocsr()
    del values,cols
    missing_matrix=meta.index.difference(cells);missing_metadata=pd.Index(cells).difference(meta.index)
    assert not len(missing_metadata),'Cannot assign donor/label for unmatched cells'
    obs=meta.loc[cells].rename(columns={'patient':'patient_id','sample':'sample_id','site':'tissue_site','celltype':'cell_type_original'}).copy()
    obs['cell_id']=obs.index.astype(str);obs['cell_type_coarse']=obs.cell_type_original.astype(str)
    pats=np.random.default_rng(42).permutation(sorted(obs.patient_id.unique()))
    splitmap={p:s for s,ps in [('train',pats[:6]),('val',pats[6:8]),('test',pats[8:])] for p in ps}
    obs['split']=obs.patient_id.map(splitmap)
    obs['n_counts']=np.asarray(X.sum(1)).ravel().astype(np.int64)
    obs['n_genes']=np.diff(X.indptr)
    mt=np.array([g.startswith('MT-') for g in genes])
    obs['pct_counts_mt']=np.asarray(X[:,mt].sum(1)).ravel()/np.maximum(obs.n_counts,1)*100
    # Processed cohort: no new upper/lower QC thresholds; exclude empty cells only.
    keep=(obs.n_counts>0).to_numpy();before=len(obs);X=X[keep];obs=obs.loc[keep].copy()
    mapping=defaultdict(set)
    gtf=DATA/'annotation/gencode.v47.basic.annotation.gtf.gz'
    with gzip.open(gtf,'rt') as f:
        for line in f:
            if line.startswith('#'):continue
            parts=line.rstrip().split('\t')
            if parts[2]!='gene':continue
            attrs=dict(re.findall(r'(\w+) "([^"]+)"',parts[8]))
            mapping[attrs['gene_name']].add(attrs['gene_id'].split('.')[0])
    var=pd.DataFrame(index=pd.Index(genes,name='gene'))
    var['gene_id_original']=genes;var['gene_symbol']=genes
    var['ensembl_id']=[next(iter(mapping[g])) if len(mapping[g])==1 else '' for g in genes]
    var['id_mapping_status']=['unique' if len(mapping[g])==1 else 'ambiguous' if len(mapping[g])>1 else 'unmapped' for g in genes]
    a=ad.AnnData(X=X,obs=obs,var=var);a.var_names_make_unique()
    a.layers['counts']=a.X.copy()
    manifest=[]
    for p in (meta_file,count_file,gtf):
        manifest.append(dict(filename=p.name,path=str(p),sha256=sha(p),size_bytes=p.stat().st_size,download_verified=True,
             url=('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149614/suppl/'+p.name) if p.suffix=='.gz' and p.name.startswith('GSE') else 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/'+p.name,
             checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),usage_note='Public research data; redistribute acquisition manifest, not counts'))
    schema={'Cell':'cell_id','patient':'patient_id','sample':'sample_id','site':'tissue_site','celltype':'cell_type_original/coarse'}
    a.uns.update(dataset_accession='GSE149614',source_manifest=json.dumps(manifest),schema_mapping=json.dumps(schema),qc_config=json.dumps(dict(rule='n_counts > 0 only; original processed cohort',n_removed=before-len(obs))))
    a.write_h5ad(DATA/'processed/GSE149614.h5ad',compression='gzip')
    subsets={};b=a[subset_indices(a.obs,10000)].copy();subsets['comparison']=b
    b=b[subset_indices(b.obs,2000)].copy();subsets['pilot']=b
    subsets['smoke']=b[subset_indices(b.obs,64)].copy()
    for name,b in subsets.items():
        b.write_h5ad(DATA/'subsets'/f'{name}.h5ad',compression='gzip')
        b.obs[['cell_id','patient_id','cell_type_coarse','split']].to_csv(ROOT/'manifests/splits'/f'{name}.tsv',sep='\t',index=False)
    pd.DataFrame({'original':sorted(obs.cell_type_original.unique()),'coarse':sorted(obs.cell_type_original.unique())}).to_csv(ROOT/'manifests/label_mapping.tsv',sep='\t',index=False)
    var.id_mapping_status.value_counts().rename_axis('status').to_csv(ROOT/'manifests/gene_mapping_summary.tsv',sep='\t')
    pd.crosstab(obs.patient_id,obs.cell_type_coarse).to_csv(ROOT/'reports/patient_label_counts.tsv',sep='\t')
    obs.groupby(['patient_id','tissue_site'],observed=True)[['n_counts','n_genes','pct_counts_mt']].agg(['count','min','median','max']).to_csv(ROOT/'reports/qc_summary.tsv',sep='\t')
    write_json(ROOT/'manifests/data_sources.json',manifest)
    write_json(ROOT/'manifests/donor_split.json',dict(seed=42,patients=splitmap,locked_before_model_evaluation=True))
    write_json(ROOT/'reports/data_audit.json',dict(matrix_shape=X.shape,orientation='gene x cell in source; cell x gene in AnnData',raw_nonnegative_integers=True,
       metadata_missing_from_matrix=len(missing_matrix),matrix_missing_from_metadata=len(missing_metadata),duplicate_cell_ids=0,duplicate_gene_symbols=int(pd.Index(genes).duplicated().sum()),
       ensembl_versions_removed='GENCODE gene_id suffix stripped; source IDs are symbols',gene_mapping=var.id_mapping_status.value_counts().to_dict(),
       count_sum_before_gene_filter=int(X.sum()),count_sum_after_gene_filter=int(X.sum()),filter='no gene filtering in canonical counts',
       wall_seconds=time.monotonic()-start,subsets={k:dict(shape=v.shape,patients=v.obs.patient_id.nunique(),split_counts=v.obs.split.value_counts().to_dict()) for k,v in subsets.items()}))
    text=f'''# GSE149614 데이터 카드\n\n공개 processed raw count. {a.n_obs:,} cells × {a.n_vars:,} genes.\n\n원본 헤더: Cell, sample, res.3, site, patient, stage, virus, celltype. 전문가 원 annotation의 6개 broad lineage를 그대로 사용한다. Tumor 출처를 malignant 정답으로 바꾸지 않는다. Normal은 인접 비종양 조직이며 건강인 cohort가 아니다.\n\n세포 ID를 원본 그대로 유지. metadata↔matrix 누락 {len(missing_matrix)}/{len(missing_metadata)}. 원 count는 sparse X와 layers[counts]에 보존. 전체 dense 변환 없이 유전자별 스트리밍. 비어 있는 세포만 제외({before-len(obs)}); 원 processed 데이터에 추가 QC cutoff를 적용하지 않음. 전체 QC 분포는 qc_summary.tsv 참조.\n\nGENCODE v47 basic annotation(Ensembl 113)의 정확한 symbol 매칭. ambiguous/unmapped는 임의 매핑하지 않음. Ensembl suffix를 annotation에서 제거하고 유일한 ID만 매핑. Geneformer는 매핑 가능한 유전자만 별도 입력하며 n_counts는 원 전체 library sum을 보존.\n\n고정 환자 split: {json.dumps(splitmap)}. seed 42, 모델 결과 확인 전에 고정. 모든 조직은 같은 환자 split. 환자×라벨 층화 nested 10000/2000/64 subset, 원본 ID/분할은 manifests/splits. train에 없는 라벨은 미평가로 별도 집계.\n\n전처리 HVG/scaler/PCA와 분류기 학습은 train만 사용. validation은 고정 grid 선택에 사용. test는 최종 평가에만 사용. 사전학습 cohort와 중복 여부는 unknown; donor holdout을 pretraining-unseen으로 해석하지 않는다.\n\n실제 파일: {DATA}/processed/GSE149614.h5ad, {DATA}/subsets/{{smoke,pilot,comparison}}.h5ad. 전체 raw와 count는 Git에 포함하지 않음.\n'''
    (ROOT/'reports/data_card.md').write_text(text)
    print((ROOT/'reports/data_audit.json').read_text(),flush=True)

if __name__=='__main__':main()
