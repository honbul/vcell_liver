"""Freeze an unseen-target HepG2 CRISPRi task; train-only output genes and matched technical controls."""
import pickle,time
import anndata as ad
import scipy.sparse as sp
from research_core import *

root=ASSETS/'response/hepg2_v1';assert not (root/'contract.json').exists(),'Immutable prepared task exists';root.mkdir(parents=True,exist_ok=True)
r=Resources();start=time.monotonic();source=Path('/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE264667/GSE264667_hepg2_raw_singlecell_01.h5ad')
a=ad.read_h5ad(source);assert a.shape==(145473,9624)
a.X=sp.csr_matrix(a.X,dtype=np.float32);assert np.isfinite(a.X.data).all() and np.all(a.X.data>=0) and np.all(a.X.data==np.round(a.X.data))
o=a.obs.copy();o['target']=o.gene.astype(str);o['batch']=o.gem_group.astype(str);o['cell_id']=o.index.astype(str)
is_control=o.target=='non-targeting';assert is_control.sum()==4976
prior=pickle.load(open(ASSETS/'official/GEARS/gene2go_all.pkl','rb'))
valid_id=o.gene_id.astype(str).str.match(r'^ENSG\d+(?:\.\d+)?$')
counts=o.loc[~is_control&valid_id].target.value_counts();eligible=sorted(t for t,n in counts.items() if n>=20 and t in prior)
assert len(eligible)>1500
shuffled=np.array(eligible);np.random.default_rng(42).shuffle(shuffled);n=len(shuffled);nval=int(.1*n);ntest=int(.1*n)
role={t:'test' if i<ntest else 'val' if i<ntest+nval else 'train' for i,t in enumerate(shuffled)}
o['role']=o.target.map(role);o.loc[is_control,'role']='control';keep=o.role.notna()&(is_control|valid_id)
a=a[keep].copy();o=o.loc[keep].copy();a.obs=o
assert not set(o[o.role=='test'].target)&set(o[o.role=='train'].target)
# Duplicate gene symbols represent counts of one output gene, so sum them explicitly.
symbols=a.var.gene_name.astype(str).to_numpy();unique=pd.Index(pd.unique(symbols));idx=unique.get_indexer(symbols)
mapping=sp.csr_matrix((np.ones(len(idx)),(np.arange(len(idx)),idx)),shape=(len(idx),len(unique)))
counts=sp.csr_matrix(a.X@mapping,dtype=np.float32)
libraries=np.asarray(counts.sum(1)).ravel();assert np.all(libraries>0)
log=counts.multiply((1e4/libraries)[:,None]).tocsr();log.data=np.log1p(log.data)
train=(o.role.isin(['train','control'])).to_numpy();mu=np.asarray(log[train].mean(0)).ravel();var=np.asarray(log[train].power(2).mean(0)).ravel()-mu**2
selected=np.argsort(-var,kind='stable')[:2000];genes=unique[selected].to_numpy(dtype=str)
b=ad.AnnData(log[:,selected].copy(),obs=o.copy(),var=pd.DataFrame({'gene_name':genes},index=genes));b.layers['counts']=counts[:,selected].copy()
b.obs['library_size_panel9624']=libraries;b.obs['cell_line']='HepG2';b.obs['replicate_id']='unknown';b.obs['time_days']=7;b.obs['intervention_type']='CRISPRi';b.obs['control_type']=np.where(b.obs.role=='control','non-targeting','not_control')
batches=o.batch.unique();batch_control={};support=[]
for batch in batches:
 ix=(o.batch==batch)&(o.role=='control');assert ix.sum()>0,f'No matched controls for batch {batch}'
 batch_control[batch]=np.asarray(b.X[ix].mean(0)).ravel();support.append(dict(batch=batch,n_non_targeting_controls=int(ix.sum()),biological_replicate='unknown'))
pd.DataFrame(support).to_csv(root/'batch_control_support.tsv',sep='\t',index=False)
records=[];aggregates={}
for target in sorted(role):
 ix=o.target==target;weights=o.loc[ix].batch.value_counts();control=sum(batch_control[batch]*n for batch,n in weights.items())/ix.sum()
 truth=np.asarray(b.X[ix].mean(0)).ravel();aggregates[target]=dict(truth=truth,control=control,n_cells=int(ix.sum()))
 ids=sorted(set(o.loc[ix,'gene_id'].astype(str)));assert len(ids)==1,f'Multiple Ensembl IDs for target {target}'
 records.append(dict(target=target,ensembl_id=ids[0],role=role[target],n_cells=int(ix.sum()),n_guide_assignments=o.loc[ix,'sgID_AB'].nunique(),n_batches=len(weights),all_guides_and_transcripts_grouped=True))
manifest=pd.DataFrame(records);manifest.to_csv(root/'target_split.tsv',sep='\t',index=False)
for name,roles in [('trainval',{'train','val'}),('test',{'test'})]:
 targets=manifest[manifest.role.isin(roles)].target.to_numpy(dtype=str)
 np.savez_compressed(root/f'{name}_aggregate.npz',targets=targets,split=np.array([role[t] for t in targets]),gene_names=genes,
   truth=np.stack([aggregates[t]['truth'] for t in targets]),control=np.stack([aggregates[t]['control'] for t in targets]),n_cells=np.array([aggregates[t]['n_cells'] for t in targets]))
b[b.obs.role!='test'].write_h5ad(root/'trainval_controls.h5ad',compression='gzip')
b[b.obs.role=='test'].write_h5ad(root/'heldout_treated.h5ad',compression='gzip')
np.savez_compressed(root/'batch_control_means.npz',batches=np.array(list(batch_control)),mean=np.stack(list(batch_control.values())),gene_names=genes)
res=r.finish();write_json(root/'contract.json',dict(status='prepared',dataset_id='GSE264667_HepG2_unseen_target_v1',task='measured_genetic_response',regime='strict_unseen_target_with_matched_controls',project_commit=git_commit(),seed=42,
  source_path=str(source),source_sha256=sha(source),source_shape=[145473,9624],n_cells=b.n_obs,n_genes=2000,n_eligible_targets=len(eligible),target_split_counts=manifest.role.value_counts().to_dict(),cell_line='HepG2',cell_origin='hepatoblastoma-derived cell line; separate from adult HCC patient cohort',
  intervention_type='CRISPRi dCas9-KRAB; dual sgRNA design targeting the same gene or single assigned guide, not dual-gene knockout',time_days=7,biological_replicate='unknown; gem_group is technical batch',control_policy='Non-targeting controls from the same gem_group are available for training and evaluation; held-out treated responses never enter training',
  assignment_policy='Use author-filtered gene/sgID_AB assignments; group all guide and transcript variants of each target gene; require valid Ensembl ID, authentic GO prior, >=20 cells; no efficiency threshold invented',
  author_QC='Original author-filtered cells retained; Methods HepG2 UMI>3000 and mitochondrial fraction<0.18; no additional response-based QC',
  output_gene_selection='2000 largest log-expression variances on training-target cells and supplied controls only, after collapsing duplicate symbols; upstream author 9624-gene panel remains a limitation',
  normalization='log1p(counts*10000/row sum in original fixed 9624-gene panel), before output-gene selection; original author UMI_count retained separately',
  target_split_sha256=sha(root/'target_split.tsv'),training_data_sha256=sha(root/'trainval_controls.h5ad'),heldout_data_sha256=sha(root/'heldout_treated.h5ad'),
  literature_sources=['https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667','https://pmc.ncbi.nlm.nih.gov/articles/PMC13063516/','https://www.cellosaurus.org/CVCL_0027'],
  failure_or_limitations=['Single-target prediction only; no combination, drug, dose or cross-cell-line generalization','Guide efficacy and biological replicate remain unknown','Fixed upstream gene universe was selected by original authors'],**res))
for name in ['contract.json','target_split.tsv','batch_control_support.tsv']:
 dest=REPORT/'hepg2_data_contract'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
print('HEPG2_PREPARED',manifest.groupby('role').n_cells.agg(['count','sum']).to_string(),flush=True)
