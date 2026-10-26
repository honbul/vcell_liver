"""Fixed third data hypothesis: train-only pathway PCA32 with matched random prior and frozen GF."""
import argparse,time
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
import yaml
from sampling import sampling_roles
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--baseline-run',required=True);p.add_argument('--run-tag',default='v1');args=p.parse_args();assert Path(args.run_tag).name==args.run_tag
def report_failure(exc_type,exc,tb):
    import traceback
    write_json(REPORT/'kg_compression_audit'/f'fold{args.fold}_{args.run_tag}_failed_call_{os.getpid()}.json',dict(status='failed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),error=f'{exc_type.__name__}: {exc}',traceback=''.join(traceback.format_exception(exc_type,exc,tb))))
    sys.__excepthook__(exc_type,exc,tb)
sys.excepthook=report_failure
c=yaml.safe_load((ROOT/'configs/research/protocol_improvement.yaml').read_text());base=json.loads((REPORT/'runs'/args.baseline_run/'run.json').read_text());assert base['status']=='executed' and base['fold']==args.fold and base['seed']==42 and base['n_outer_training']==4096
c['head_C']=[base['head_selection']['selected_C']];c['head_class_weight']=[base['head_selection']['selected_class_weight']];c['data_variant']='D04_KG_compression_diagnostic';c['model_variant']='frozen_Geneformer_fixed_logistic_head'
hypothesis=ROOT/'configs/research/hypothesis_DH3_kg_compression_v1.json';assert json.loads(hypothesis.read_text())['excluded_from_current_confirmation_and_external_model_selection']
r=Resources();started=time.monotonic();a=ad.read_h5ad(DATA/c['dataset_path']);roles=sampling_roles(a.obs,c,args.fold,'D0_natural');assert len(roles['outer_training'])==4096
feature_path=ASSETS/'features/geneformer_GSE149614_development_full.npz';feature_metadata=json.loads(feature_path.with_suffix('.json').read_text());f=np.load(feature_path);assert np.array_equal(f['cell_ids'],a.obs_names.astype(str));foundation=f['X'];y=a.obs.cell_type_coarse.astype(str).to_numpy();source=DATA/'annotation/Ensembl2Reactome_All_Levels.human.tsv'
df=pd.read_csv(source,sep='\t',header=None,names=['ensembl','pathway','url','name','evidence','species']);index={gene:i for i,gene in enumerate(a.var.ensembl_id.fillna('').astype(str)) if gene};df=df[df.ensembl.isin(index)].drop_duplicates(['ensembl','pathway']);groups=[]
for pid,part in df.groupby('pathway',sort=True):
    genes=sorted({index[gene] for gene in part.ensembl})
    if 10<=len(genes)<=200:groups.append((pid,genes))
assert len(groups)>=128
x=a.layers['counts'].astype(np.float32);x=x.multiply((10000/np.maximum(np.asarray(x.sum(1)).ravel(),1))[:,None]).tocsr();x.data=np.log1p(x.data)
def membership(sets):
    row=[];col=[];weights=[]
    for j,genes in enumerate(sets):row.extend(genes);col.extend([j]*len(genes));weights.extend([1/len(genes)]*len(genes))
    return sparse.csr_matrix((weights,(row,col)),shape=(a.n_vars,len(sets)),dtype=np.float32)
all_weights=membership([genes for _,genes in groups])
def randomize(sets):
    rng=np.random.default_rng(42);edges=[(g,j) for j,genes in enumerate(sets) for g in genes];edge_set=set(edges);accepted=0
    for _ in range(20*len(edges)):
        i,k=rng.integers(0,len(edges),2);g,u=edges[i];h,v=edges[k]
        if g==h or u==v or (g,v) in edge_set or (h,u) in edge_set:continue
        edge_set.remove((g,u));edge_set.remove((h,v));edge_set.add((g,v));edge_set.add((h,u));edges[i]=(g,v);edges[k]=(h,u);accepted+=1
    new=[set() for _ in sets]
    for gene,j in edges:new[j].add(gene)
    assert [len(v) for v in new]==[len(v) for v in sets]
    assert np.array_equal(np.bincount([g for group in sets for g in group],minlength=a.n_vars),np.bincount([g for group in new for g in group],minlength=a.n_vars))
    return [sorted(v) for v in new],accepted
def fitted_features(train):
    mu=np.asarray(x[train].mean(0)).ravel();sd=np.sqrt(np.maximum(np.asarray(x[train].power(2).mean(0)).ravel()-mu**2,0));sd[sd<1e-6]=1
    train_proxies=np.asarray((x[train].multiply(1/sd)@all_weights).toarray()-((mu/sd)@all_weights));order=np.argsort(-train_proxies.var(0),kind='stable')[:128];selected=[groups[i] for i in order];sets=[genes for _,genes in selected];random,accepted=randomize(sets)
    def proxies(genes):
        W=membership(genes);return np.asarray((x.multiply(1/sd)@W).toarray()-((mu/sd)@W))
    real=proxies(sets);random_values=proxies(random)
    def compress(values):
        scale=StandardScaler().fit(values[train]);scaled=scale.transform(values);pca=PCA(n_components=32,random_state=42,whiten=False).fit(scaled[train]);return pca.transform(scaled),dict(scaler=scale,pca=pca)
    real32,real_pca=compress(real);random32,random_pca=compress(random_values)
    preprocessor=dict(gene_mu=mu,gene_sd=sd,pathway_ids=[pid for pid,_ in selected],real_sets=sets,random_sets=random,real_compression=real_pca,random_compression=random_pca)
    audit=dict(n_training=len(train),pathway_ids=preprocessor['pathway_ids'],n_pathways=128,n_unique_member_sets=len({tuple(v) for v in sets}),n_edges=sum(map(len,sets)),real_PCA32_explained_variance=float(real_pca['pca'].explained_variance_ratio_.sum()),random_PCA32_explained_variance=float(random_pca['pca'].explained_variance_ratio_.sum()),successful_edge_swaps=accepted,pathway_sizes_preserved=True,gene_membership_degrees_preserved=True)
    return {'real128':np.concatenate([foundation,real],axis=1),'real32':np.concatenate([foundation,real32],axis=1),'random32':np.concatenate([foundation,random32],axis=1),'pathway32':real32},preprocessor,audit
inner,_,inner_audit=fitted_features(roles['inner_training']);final,preprocessor,outer_audit=fitted_features(roles['outer_training']);preprocessing=time.monotonic()-started
outputs=[]
for condition in ['real128','real32','random32','pathway32']:
    rid=f'D04_{condition}_full4096_fold{args.fold}_seed42_{args.run_tag}';assert not (REPORT/'runs'/rid/'run.json').exists()
    pred,prob,head,selection,training,inference=fit_head(inner[condition],final[condition],y,roles,c,42);head['preprocessor']=preprocessor;raw=ASSETS/'runs'/rid;raw.mkdir(parents=True,exist_ok=True)
    for role,indices in roles.items():pd.DataFrame({'cell_id':a.obs_names[indices]}).to_csv(raw/f'{role}_cell_ids.tsv',sep='\t',index=False)
    assert sha(raw/'outer_training_cell_ids.tsv')==base['training_manifest_sha256'] and sha(raw/'inner_validation_cell_ids.tsv')==base['validation_manifest_sha256']
    outputs.append((rid,condition,pred,prob,head,selection,training,inference))
res=r.finish();audit_name=f'fold{args.fold}_seed42'+('' if args.run_tag=='v1' else '_'+args.run_tag)+'.json';audit_path=REPORT/'kg_compression_audit'/audit_name;assert not audit_path.exists();write_json(audit_path,dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),hypothesis_sha256=sha(hypothesis),Reactome_source_sha256=sha(source),inner=inner_audit,outer=outer_audit,baseline_run=args.baseline_run,external_data_accessed=False,selection_eligible_for_current_external_batch=False))
for rid,condition,pred,prob,head,selection,training,inference in outputs:
    save_evaluation(a,roles['evaluation'],pred,prob,c,rid,dict(model_id='Geneformer_KG_'+condition if condition!='pathway32' else 'Reactome_only_PCA32',fold=args.fold,feature_dimension=final[condition].shape[1],project_commit=git_commit(),source_sha256=sha(Path(__file__)),hypothesis_id='D-H3',baseline_run=args.baseline_run,feature_source=str(feature_path) if condition!='pathway32' else 'Reactome92 membership proxies only',feature_extraction_metadata=feature_metadata if condition!='pathway32' else None,checkpoint_sha256=feature_metadata['checkpoint_sha256'] if condition!='pathway32' else None,upstream_commit=feature_metadata['upstream_commit'] if condition!='pathway32' else 'Reactome92',dataset_sha256=sha(DATA/c['dataset_path']),trainable_backbone_parameters=0,n_inner_training=len(roles['inner_training']),n_inner_validation=len(roles['inner_validation']),n_outer_training=len(roles['outer_training']),n_evaluation=len(roles['evaluation']),training_donors=sorted(a.obs.iloc[roles['outer_training']].patient_id.unique()),validation_donors=sorted(a.obs.iloc[roles['inner_validation']].patient_id.unique()),evaluation_donors_or_perturbations=sorted(a.obs.iloc[roles['evaluation']].patient_id.unique()),trainable_parameters=int(head['classifier'].coef_.size+head['classifier'].intercept_.size),head_selection=selection,training_manifest_sha256=sha(ASSETS/'runs'/rid/'outer_training_cell_ids.tsv'),validation_manifest_sha256=sha(ASSETS/'runs'/rid/'inner_validation_cell_ids.tsv'),input_contract='Frozen GF768 plus Reactome92 mean train-gene-zscores;128 pathways selected by train variance; compressed32 via train scaler/PCA without whitening; same source genes and training IDs as baseline',feature_audit_sha256=sha(audit_path),wall_preprocessing_seconds=preprocessing,wall_training_seconds=training,wall_inference_seconds=inference,wall_total_seconds=res['wall_seconds'],resource_measurement_scope='Combined feature preparation and all4 heads in this fold process; per-condition head train/inference times separate; do not sum shared preparation four times',failure_or_limitations=['Exploratory fixed-seed4fold diagnostic, not3seed confirmation','Same-head family/C/class_weight; feature dimensions and linear parameter counts differ. Random32 is dimension-matched to real32','Not eligible to change current D1/partial confirmation or external candidate batch','Pathway scores are observed-expression proxies, not causal activity measurements'],**{k:v for k,v in res.items() if k!='wall_seconds'}),head=head)
print('KG_COMPRESSION_FOLD_COMPLETE',args.fold,flush=True)
