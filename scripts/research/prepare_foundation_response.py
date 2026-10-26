"""Matched native-response panel: actual measured/vocabulary-matched targets, shared control pairs and cell budget."""
import pickle,time
import anndata as ad
import scipy.sparse as sp
from research_core import *

root=ASSETS/'response/hepg2_foundation_common_v1';assert not (root/'contract.json').exists();root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
parent=ASSETS/'response/hepg2_v1';original=json.loads((parent/'contract.json').read_text());split=pd.read_csv(parent/'target_split.tsv',sep='\t')
tv=ad.read_h5ad(parent/'trainval_controls.h5ad',backed='r');test=ad.read_h5ad(parent/'heldout_treated.h5ad',backed='r');obs=pd.concat([tv.obs,test.obs]);assert obs.index.is_unique
vocab=json.loads((DATA/'checkpoints/scgpt-whole-human/vocab.json').read_text());raw_path=Path(original['source_path']);raw=ad.read_h5ad(raw_path)
symbols=raw.var.gene_name.astype(str).to_numpy();raw_symbols=set(symbols);eligible=set(split.target)&raw_symbols&set(vocab);primary=set(tv.var.gene_name.astype(str))&set(vocab)
assert len(primary)==1977 and len(eligible)>1500
# All models receive the same actual gene universe. Target identities are known; their held-out responses are not used for gene selection.
genes=np.array(sorted(primary|eligible),dtype=str);gene_lookup={g:i for i,g in enumerate(genes)};rng=np.random.default_rng(42);selected=[]
for target,part in obs[(obs.role=='train')&obs.target.isin(eligible)].groupby('target',observed=True):
    cells=np.array(sorted(part.index),dtype=str);assert len(cells)>=4;selected.extend(rng.choice(cells,4,replace=False))
keep=obs.target.isin(eligible)|(obs.role=='control');keep&=(obs.role!='train')|obs.index.isin(selected);obs=obs.loc[keep].copy()
ids=raw.obs_names[raw.obs_names.isin(obs.index)];assert len(ids)==len(obs);obs=obs.loc[ids].copy()
positions=raw.obs_names.get_indexer(ids);all_counts=sp.csr_matrix(raw.X[positions],dtype=np.float32);libraries=np.asarray(all_counts.sum(1)).ravel();assert np.all(libraries>0)
valid=np.array([s in gene_lookup for s in symbols]);rows=np.flatnonzero(valid);cols=np.array([gene_lookup[s] for s in symbols[valid]])
mapping=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(symbols),len(genes)));counts=(all_counts@mapping).tocsr();log=counts.multiply((1e4/libraries)[:,None]).tocsr();log.data=np.log1p(log.data)
var=pd.DataFrame(dict(gene_name=genes,is_primary=np.isin(genes,list(primary))),index=genes);b=ad.AnnData(log,obs=obs,var=var);b.layers['counts']=counts;b.obs['library_size_panel9624']=libraries
control_ix=np.flatnonzero(b.obs.role.to_numpy()=='control');batch_controls={batch:control_ix[b.obs.iloc[control_ix].batch.astype(str).to_numpy()==batch] for batch in b.obs.batch.astype(str).unique()};assert all(len(v)>0 for v in batch_controls.values())
match_ids=pd.Series(index=b.obs_names,dtype=object)
for role,seed in [('train',42),('val',43),('test',44),('control',45)]:
    generator=np.random.default_rng(seed)
    for i in np.flatnonzero(b.obs.role.to_numpy()==role):
        match=i if role=='control' else int(generator.choice(batch_controls[str(b.obs.iloc[i].batch)]));match_ids.iloc[i]=str(b.obs_names[match])
b.obs['control_match_id']=match_ids.astype(str);assert set(b.obs.control_match_id).issubset(set(b.obs_names[control_ix]))
records=[];aggregates={}
for target,part in b.obs[b.obs.role!='control'].groupby('target',observed=True):
    ix=b.obs_names.isin(part.index);weights=part.batch.astype(str).value_counts();control=sum(np.asarray(b.X[batch_controls[batch]].mean(0)).ravel()*n for batch,n in weights.items())/len(part)
    aggregates[str(target)]=dict(truth=np.asarray(b.X[ix].mean(0)).ravel(),control=control,n_cells=len(part));row=split.set_index('target').loc[str(target)].to_dict();row.update(target=str(target),n_cells=len(part));records.append(row)
manifest=pd.DataFrame(records).sort_values('target');manifest.to_csv(root/'target_split.tsv',sep='\t',index=False);var.to_csv(root/'gene_panel.tsv',sep='\t')
for name,roles in [('trainval',{'train','val'}),('test',{'test'})]:
    targets=manifest[manifest.role.isin(roles)].target.to_numpy(dtype=str)
    np.savez_compressed(root/f'{name}_aggregate.npz',targets=targets,split=np.array([str(manifest.set_index('target').loc[t,'role']) for t in targets]),gene_names=genes,primary_mask=var.is_primary.to_numpy(),truth=np.stack([aggregates[t]['truth'] for t in targets]),control=np.stack([aggregates[t]['control'] for t in targets]),n_cells=np.array([aggregates[t]['n_cells'] for t in targets]))
b[b.obs.role!='test'].write_h5ad(root/'trainval_controls.h5ad',compression='gzip');b[b.obs.role=='test'].write_h5ad(root/'heldout_treated.h5ad',compression='gzip')
write_json(root/'contract.json',dict(status='prepared',dataset_id='GSE264667_HepG2_native_foundation_common_v1',parent_contract_sha256=sha(parent/'contract.json'),project_commit=git_commit(),seed=42,source_path=str(raw_path),n_cells=b.n_obs,n_genes=b.n_vars,n_primary_genes=int(var.is_primary.sum()),n_controls=len(control_ix),n_training_treated=int((b.obs.role=='train').sum()),n_eligible_targets=len(eligible),target_split_counts=manifest.role.value_counts().to_dict(),
    panel_rule='Parent training-derived output genes in official vocabulary, union every declared target gene measured in raw9624 and in vocabulary. All counts are real; no missing target positions imputed',
    train_cell_budget='Exactly4 treated cells per eligible training target, plus all available non-targeting controls; same selected IDs and fixed matched control IDs for every model',
    limitation='Parent output panel was derived from all parent training-target cells; this matches supervised update examples, not a four-cell-only acquisition experiment. Test treated responses excluded from all selection/fitting.',
    primary_evaluation='1977 vocabulary-matched parent output genes; eligible target coverage reported separately',secondary_evaluation='All shared genes; same output supervision for GEARS, pretrained scGPT and random-init scGPT',
    source_split='Same target roles as parent; no score-based target inclusion; physical test treated file opened only after model selection by runners',
    control_policy=original['control_policy'],cell_line='HepG2',intervention_type='single-target CRISPRi',time_days=7,biological_replicate='unknown',target_split_sha256=sha(root/'target_split.tsv'),gene_panel_sha256=sha(root/'gene_panel.tsv'),wall_seconds=time.monotonic()-started))
for name in ['contract.json','target_split.tsv','gene_panel.tsv']:
    dest=REPORT/'foundation_response_contract'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
print('FOUNDATION_RESPONSE_PREPARED',b.shape,manifest.groupby('role').n_cells.agg(['count','sum']).to_string(),flush=True)
