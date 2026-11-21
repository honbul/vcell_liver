"""Reviewer post-hoc diagnostic 5 (read-only; project-seen Tahoe rows).
(A) why Codex's frozen B_pos gives -2.35% while diag3 gave -5.9%: step-by-step protocol changes.
(B) structure of the position-index effect: 8x12 layout, adjacency, edge, cell count, depth, genes."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, os
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
def load(d, name):
    n=np.load(d+name+'_counts.npz'); m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t'); assert m.shape[0]==len(r); return m,r
d05=R+'vcell_liver_r11_chemical_probe/assets/source/'; dxx=R+'vcell_liver_r12_multiaxis/assets/doses/source/'
m05,r05=load(d05,'treated'); mxx,rxx=load(dxx,'treated')
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64); lib[lib==0]=1
    if genes is not None: m=m[:,genes]
    return np.log1p(m.toarray().astype(np.float64)/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
xa=lognorm(m05[ok05]); var=xa.var(0); genes=np.sort(np.argsort(-var)[:2000]); del xa
lib_all=np.concatenate([np.asarray(m05.sum(1)).ravel(),np.asarray(mxx.sum(1)).ravel()])
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]; lib_all=lib_all[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]; lib_all=lib_all[keep]
G=X.shape[1]
C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items(): C[idx]=X[idx]-X[idx].mean(0)
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
rows['pos']=rows.physical_well_id.map(dict(zip(sm['sample'],sm.pos))).values
# Codex's frozen outer / inner folds
man=pd.read_csv(R+'review_claude_20261121/split_manifest.tsv',sep='\t')
of=man[man.usage=='outer_query'].drop_duplicates('compound_group_id').set_index('compound_group_id').outer_fold
rows['fold']=rows.compound_group_id.map(of).values
assert rows.fold.notna().all(), rows[rows.fold.isna()].compound_group_id.unique()[:5]
inner={}
for f in range(3):
    s=man[(man.evaluation_outer==f)&(man.inner_fold>=0)].drop_duplicates('compound_group_id').set_index('compound_group_id').inner_fold
    inner[f]=rows.compound_group_id.map(s).fillna(-1).astype(int).values
fold=rows.fold.astype(int).values; gi=rows.compound_group_id.values; pl=rows.plate_id.values; ds=rows.dose_uM.values
ln=rows.cell_line_id.values; pos=rows.pos.values
out={'usage_values':man.usage.unique().tolist(),'rows':int(len(rows)),'fold_group_counts':pd.Series(fold).groupby(gi).first().value_counts().to_dict()}
def rm(a): return np.sqrt((a**2).mean(1))
def agg(v,g): return float(pd.Series(np.asarray(v)).groupby(np.asarray(g)).mean().mean())
def center(idx):
    """contrast of rows idx centred within plate|line|dose among idx only; valid if pool has >=3 groups."""
    k=pd.Series(rows.pool.values[idx]); o=np.zeros((len(idx),G),np.float32); v=np.zeros(len(idx),bool)
    for p,ii in k.groupby(k).indices.items():
        if len(set(gi[idx[ii]]))>=3: o[ii]=X[idx[ii]]-X[idx[ii]].mean(0); v[ii]=True
    return o,v
def recentre(P,idx):
    k=pd.Series(rows.pool.values[idx]); o=P.copy()
    for p,ii in k.groupby(k).indices.items(): o[ii]=P[ii]-P[ii].mean(0)
    return o
def predict(te,dn,D,per_row_excl):
    key=pd.DataFrame({'p':pos[dn],'l':ln[dn]}).groupby(['p','l']).indices
    P=np.zeros((len(te),G),np.float32); has=np.zeros(len(te),bool)
    for n,i in enumerate(te):
        kk=key.get((pos[i],ln[i]))
        if kk is None: continue
        j=dn[kk]; m=pl[j]!=pl[i]
        if per_row_excl: m&=gi[j]!=gi[i]
        kk=kk[m]
        if len(kk)==0: continue
        # plate-equal mean
        pj=pl[dn[kk]]; up=np.unique(pj); P[n]=np.mean([D[kk[pj==u]].mean(0) for u in up],0); has[n]=True
    return P,has
FINE=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0]); COARSE=np.array([0,0.25,0.5,0.75,1.0])
tg=np.where(ds==0.5)[0]
def one(te,dn,fold_excl,foldwise):
    if foldwise:
        T,v=center(te); te=te[v]; T=T[v]; D,vd=center(dn); dn=dn[vd]; D=D[vd]
    else: T=C[te]; D=C[dn]
    P,has=predict(te,dn,D,per_row_excl=not fold_excl)
    if foldwise: P=recentre(P,te)
    return te,T,P,has
def variant(doses,zero_fill,grid,fold_excl,inner_sel,foldwise):
    dmask=np.isin(ds,doses); res=[]; chosen=[]
    for f in range(3):
        te=tg[fold[tg]==f]; dn=np.where(dmask&(fold!=f))[0] if fold_excl else np.where(dmask)[0]
        res.append(one(te,dn,fold_excl,foldwise))
    def score(parts,s):
        r=[];g=[]
        for te,T,P,has in parts:
            k=np.ones(len(te),bool) if zero_fill else has
            r.append(rm(T[k]-s*P[k])); g.append(gi[te][k])
        return agg(np.concatenate(r),np.concatenate(g))
    for f in range(3):
        if inner_sel:
            parts=[]
            for k in (0,1):
                iv=tg[(fold[tg]!=f)&(inner[f][tg]==k)]; idn=np.where(dmask&(fold!=f)&(inner[f]!=k)&(inner[f]>=0))[0]
                parts.append(one(iv,idn,True,foldwise))
            sc=[score(parts,s) for s in grid]
        else:
            sc=[score([res[j] for j in range(3) if j!=f],s) for s in grid]
        chosen.append(float(grid[int(np.argmin(sc))]))
    r=[];b=[];g=[];sup=0;tot=0;cs=[]
    for f,(te,T,P,has) in enumerate(res):
        k=np.ones(len(te),bool) if zero_fill else has
        r.append(rm(T[k]-chosen[f]*P[k])); b.append(rm(T[k])); g.append(gi[te][k]); sup+=int(has.sum()); tot+=len(te)
        cs.append(((T[has]*P[has]).sum(1)/np.sqrt((T[has]**2).sum(1)*(P[has]**2).sum(1)+1e-30)))
    r=np.concatenate(r);b=np.concatenate(b);g=np.concatenate(g)
    oracle={float(s):100*(score(res,s)/agg(b,g)-1) for s in grid}
    return {'rows_scored':int(len(r)),'groups':int(len(set(g))),'supported_rows':sup,'query_rows':tot,'scales':chosen,'B0':agg(b,g),'rmse':agg(r,g),'rel_%':100*(agg(r,g)/agg(b,g)-1),'cos_median_supported':float(np.median(np.concatenate(cs))),'rel_%_by_fixed_scale':oracle}
ALL=[0.05,0.5,5.0]
steps=[('1_diag3_like: any-dose donors, own-group excl, supported rows, fine grid',dict(doses=ALL,zero_fill=False,grid=FINE,fold_excl=False,inner_sel=False,foldwise=False)),
       ('2_+0.5uM donors only',dict(doses=[0.5],zero_fill=False,grid=FINE,fold_excl=False,inner_sel=False,foldwise=False)),
       ('3_+exclude whole test fold from donors',dict(doses=[0.5],zero_fill=False,grid=FINE,fold_excl=True,inner_sel=False,foldwise=False)),
       ('4_+fold-wise centring (query pool / train pool)',dict(doses=[0.5],zero_fill=False,grid=FINE,fold_excl=True,inner_sel=False,foldwise=True)),
       ('5_+zero-fill unsupported rows in denominator',dict(doses=[0.5],zero_fill=True,grid=FINE,fold_excl=True,inner_sel=False,foldwise=True)),
       ('6_+inner-fold scale selection, fine grid',dict(doses=[0.5],zero_fill=True,grid=FINE,fold_excl=True,inner_sel=True,foldwise=True)),
       ('7_+coarse grid 0.25 steps (Codex-like)',dict(doses=[0.5],zero_fill=True,grid=COARSE,fold_excl=True,inner_sel=True,foldwise=True)),
       ('8_Codex-like but any-dose donors (fold excluded at all doses)',dict(doses=ALL,zero_fill=True,grid=FINE,fold_excl=True,inner_sel=True,foldwise=True))]
out['decomposition']={}
for name,kw in ([] if os.environ.get('SKIP') else steps):
    out['decomposition'][name]=variant(**kw); print(name,json.dumps(out['decomposition'][name]),flush=True)

# ---------- (B) structure of the position effect ----------
B={}
plates=sorted(set(pl)); B['plates']=plates
# line-averaged well contrast per (plate,pos): shared-across-lines component
wk=pd.DataFrame({'plate':pl,'pos':pos}); W={}
for (p,q),ii in wk.groupby(['plate','pos']).indices.items(): W[(p,int(q))]=C[ii].mean(0)
def posvec(pset):
    E=np.full((97,G),np.nan,np.float32)
    for q in range(1,97):
        v=[W[(p,q)] for p in pset if (p,q) in W]
        if v: E[q]=np.mean(v,0)
    return E
rng=np.random.default_rng(1); sims=[]
def cosm(A,Bm):
    a=A/np.sqrt((A**2).sum(1,keepdims=True)+1e-30); b=Bm/np.sqrt((Bm**2).sum(1,keepdims=True)+1e-30); return a@b.T
S=np.zeros((96,96)); nS=0
# drug-set-disjoint halves: plates 1-3 / 4-6(+14) / 7-9 / 10-12 carry the same drug set at three doses
dset=lambda p: 1 if p=='plate14' else (int(p.replace('plate',''))-1)//3
B['drug_set_of_plate']={p:dset(p) for p in plates}
for sa in [(0,1),(0,2),(0,3)]:
    h1=[p for p in plates if dset(p) in sa]; h2=[p for p in plates if dset(p) not in sa]
    E1,E2=posvec(h1)[1:],posvec(h2)[1:]; okp=~(np.isnan(E1).any(1)|np.isnan(E2).any(1))
    M=cosm(np.nan_to_num(E1),np.nan_to_num(E2)); M[~okp]=np.nan; M[:,~okp]=np.nan; S+=np.nan_to_num((M+M.T)/2); nS+=1
S/=nS
idx=np.arange(96); d=np.abs(idx[:,None]-idx[None,:]); rr_=idx//12; cc_=idx%12
def m(mask): return float(np.nanmean(S[mask]))
B['split_half_cosine_same_position_mean']=float(np.mean(np.diag(S)))
B['split_half_cosine_same_position_by_row(row-major 12/row)']=[float(np.mean(np.diag(S)[rr_==r])) for r in range(8)]
B['split_half_cosine_same_position_by_col(row-major)']=[float(np.mean(np.diag(S)[cc_==c])) for c in range(12)]
off=~np.eye(96,dtype=bool)
B['cross_position_cosine']={'all_other_pairs':m(off),'index_diff_1':m(d==1),'index_diff_2':m(d==2),'index_diff_8':m(d==8),'index_diff_12':m(d==12),'index_diff_24':m(d==24),
  'rowmajor_same_row':m(off&(rr_[:,None]==rr_[None,:])),'rowmajor_same_col':m(off&(cc_[:,None]==cc_[None,:])),
  'colmajor_same_col(8/col)':m(off&((idx//8)[:,None]==(idx//8)[None,:])),'colmajor_same_row':m(off&((idx%8)[:,None]==(idx%8)[None,:])),
  'rowmajor_2D_neighbours':m(off&(np.abs(rr_[:,None]-rr_[None,:])<=1)&(np.abs(cc_[:,None]-cc_[None,:])<=1))}
B['cross_position_cosine_by_index_distance']={int(k):m(d==k) for k in [1,2,3,4,6,8,11,12,13,16,24,36,48,60,84]}
E=posvec(plates)[1:]; okp=~np.isnan(E).any(1); En=np.nan_to_num(E)
amp=np.sqrt((En**2).mean(1)); B['position_effect_rms_grid_rowmajor_x1000']=[[int(round(1000*amp[r*12+c])) for c in range(12)] for r in range(8)]
edge=(rr_==0)|(rr_==7)|(cc_==0)|(cc_==11); B['rms_edge_vs_inner_rowmajor']=[float(amp[edge].mean()),float(amp[~edge].mean())]
# PCA of position vectors
En[~okp]=En[okp].mean(0); U,sv,Vt=np.linalg.svd(En-En[okp].mean(0),full_matrices=False); ev=sv**2/np.sum(sv**2)
B['pca_var_share_top6']=[float(x) for x in ev[:6]]
for k in range(3):
    sc=U[:,k]*sv[k]; B[f'PC{k+1}_grid_rowmajor_x100']=[[int(round(100*sc[r*12+c])) for c in range(12)] for r in range(8)]
gn=pd.read_csv(R+'vcell_liver_r12_multiaxis/assets/doses/metadata/genes.tsv',sep='\t').gene_symbol.values[genes]
for k in range(2):
    o=np.argsort(Vt[k]); B[f'PC{k+1}_genes_neg']=gn[o[:20]].tolist(); B[f'PC{k+1}_genes_pos']=gn[o[-20:]].tolist()
# covariates per position: cells, depth
cov=pd.DataFrame({'plate':pl,'pos':pos,'n':rows.n_cells.values,'lib':lib_all}).groupby(['plate','pos']).agg(cells=('n','sum'),lib=('lib','sum')).reset_index()
cov['umi_per_cell']=cov.lib/cov.cells
cov=cov.merge(sm[['plate','pos','mean_gene_count','mean_tscp_count']],on=['plate','pos'],how='left')
for c in ['cells','umi_per_cell','mean_gene_count','mean_tscp_count']:
    cov[c+'_rel']=np.log(cov[c])-np.log(cov[c]).groupby(cov.plate).transform('mean')
pm=cov.groupby('pos')[[c for c in cov.columns if c.endswith('_rel')]].mean()
B['covariate_position_reproducibility(plate-pair corr of rel log value)']={}
for c in pm.columns:
    pv=cov.pivot(index='pos',columns='plate',values=c); cm=pv.corr().values; B['covariate_position_reproducibility(plate-pair corr of rel log value)'][c]=float(np.median(cm[np.triu_indices_from(cm,1)]))
sc1=pd.Series(U[:,0]*sv[0],index=np.arange(1,97)); sc2=pd.Series(U[:,1]*sv[1],index=np.arange(1,97)); a=pd.Series(amp,index=np.arange(1,97))
B['positions_without_treated_rows']=[int(i+1) for i in np.where(~okp)[0]]
B['corr_with_position_mean_covariates']={c:{'PC1':float(sc1.corr(pm[c])),'PC2':float(sc2.corr(pm[c])),'rms':float(a.corr(pm[c]))} for c in pm.columns}
B['corr_PC_with_index_row_col']={'PC1':[float(np.corrcoef(U[:,0],v)[0,1]) for v in (idx,rr_,cc_)],'PC2':[float(np.corrcoef(U[:,1],v)[0,1]) for v in (idx,rr_,cc_)]}
for c in ['cells_rel','umi_per_cell_rel']:
    B[c+'_grid_rowmajor_x100']=[[(None if not np.isfinite(pm[c].get(r*12+c2+1,np.nan)) else int(round(100*pm[c].get(r*12+c2+1)))) for c2 in range(12)] for r in range(8)]
# does regressing out per-well covariates remove the position effect?  per (plate,pos) well-mean contrast W; regress gene-wise on cells_rel & umi_rel
keys=[(p,int(q)) for p,q in zip(cov.plate,cov.pos)]; Wm=np.array([W[k] for k in keys]); Z=np.column_stack([np.ones(len(cov)),cov.cells_rel.values,cov.umi_per_cell_rel.values,cov.mean_gene_count_rel.fillna(0).values])
beta,_,_,_=np.linalg.lstsq(Z,Wm,rcond=None); Wr=Wm-Z@beta
B['well_mean_contrast_MS_explained_by_cells+depth_%']=float(100*(1-(Wr**2).mean()/(Wm**2).mean()))
def lopo(Wx):
    cs=[];pp_=cov.plate.values;qq=cov.pos.values;dsv=np.array([dset(p) for p in pp_])
    for i in range(len(cov)):
        mk=(qq==qq[i])&(dsv!=dsv[i])
        if mk.sum()==0: continue
        v=Wx[mk].mean(0); cs.append(float((v*Wx[i]).sum()/np.sqrt((v**2).sum()*(Wx[i]**2).sum()+1e-30)))
    return float(np.median(cs))
B['well_mean: cosine with same-position other-plate mean']={'raw':lopo(Wm),'after_regressing_cells+depth':lopo(Wr)}
out['layout']=B
print(json.dumps(out,indent=1,default=str))
