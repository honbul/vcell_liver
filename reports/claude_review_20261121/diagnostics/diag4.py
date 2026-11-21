"""Reviewer post-hoc diagnostic 3: well-position effect (read-only; project-seen Tahoe source rows)."""
import numpy as np, pandas as pd, scipy.sparse as sp, json
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
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]
C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items(): C[idx]=X[idx]-X[idx].mean(0)
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
rows['pos']=rows.physical_well_id.map(dict(zip(sm['sample'],sm.pos))).values
def agg(v,r): return float(pd.Series(np.asarray(v),index=r.index).groupby(r.compound_group_id).mean().mean())
def rm(a): return np.sqrt((a**2).mean(1))
def cosr(a,b): return (a*b).sum(1)/np.sqrt((a**2).sum(1)*(b**2).sum(1)+1e-30)
SC=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0)
def evalp(Tm,Pm,tt):
    b=agg(rm(Tm),tt); cs={s:agg(rm(Tm-s*Pm),tt) for s in SC}; s=min(cs,key=cs.get)
    gid=pd.factorize(tt.compound_group_id.values)[0]; rng=np.random.default_rng(0); gf=rng.integers(0,3,gid.max()+1)[gid]; r=np.zeros(len(tt)); chosen=[]
    for f in range(3):
        tr=gf!=f; c2={s2:agg(rm(Tm[tr]-s2*Pm[tr]),tt[tr]) for s2 in SC}; s2=min(c2,key=c2.get); chosen.append(s2); r[~tr]=rm(Tm[~tr]-s2*Pm[~tr])
    cv=agg(r,tt)
    # group bootstrap CI of (cv - b0)
    per=pd.DataFrame({'g':tt.compound_group_id.values,'d':r-rm(Tm)}).groupby('g').d.mean().values
    bs=[rng.choice(per,len(per)).mean() for _ in range(2000)]
    return {'rows':int(len(tt)),'groups':int(tt.compound_group_id.nunique()),'cos_median':float(np.median(cosr(Tm,Pm))),'B0':b,'insample_best_scale':s,'insample_rel_%':100*(cs[s]/b-1),'cv_scales':chosen,'cv_rmse':cv,'cv_rel_%':100*(cv/b-1),'cv_diff':float(per.mean()),'cv_diff_95CI':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]}
gi=rows.compound_group_id.values; pl=rows.plate_id.values; ds=rows.dose_uM.values
# plate -> set of groups
pg=rows.groupby('plate_id').compound_group_id.apply(set).to_dict()
def position_pred(tt, ref_mask, exclude='same_group', mincells=20):
    rr=rows[ref_mask&(rows.n_cells>=mincells)]
    key=rr.groupby(['pos','cell_line_id']).indices; ri=rr.index.values
    P=np.zeros((len(tt),C.shape[1]),np.float32); has=np.zeros(len(tt),bool); nref=np.zeros(len(tt))
    for n,(i,po,ln) in enumerate(zip(tt.index.values,tt.pos.values,tt.cell_line_id.values)):
        kk=key.get((po,ln))
        if kk is None: continue
        ii=ri[kk]
        if exclude=='same_group': ii=ii[(gi[ii]!=gi[i])&(pl[ii]!=pl[i])]
        elif exclude=='disjoint_plate': # plates sharing no compound group with target plate
            ii=np.array([j for j in ii if pl[j]!=pl[i] and gi[i] not in pg[pl[j]] and gi[j]!=gi[i]],dtype=int)
        elif exclude=='only_same_group': ii=ii[(gi[ii]==gi[i])&(pl[ii]!=pl[i])]
        if len(ii)==0: continue
        # plate-balanced mean
        P[n]=C[ii].mean(0); has[n]=True; nref[n]=len(set(pl[ii]))
    return has,P,nref

out={}
pos=rows.pos.values
def sister(same_pos, doses, target_dose=0.5):
    tt=rows[rows.dose_uM==target_dose]; rr=rows[rows.dose_uM.isin(doses)]
    key=rr.groupby(['compound_group_id','cell_line_id']).indices; ri=rr.index.values
    P=np.zeros((len(tt),C.shape[1]),np.float32); has=np.zeros(len(tt),bool)
    for n,(i,g,ln) in enumerate(zip(tt.index.values,tt.compound_group_id.values,tt.cell_line_id.values)):
        kk=key.get((g,ln))
        if kk is None: continue
        ii=ri[kk]; ii=ii[pl[ii]!=pl[i]]
        ii=ii[(pos[ii]==pos[i])] if same_pos else ii[(pos[ii]!=pos[i])]
        if len(ii): P[n]=C[ii].mean(0); has[n]=True
    return tt[has],C[tt.index.values][has],P[has]
for td,doses in [(0.5,[0.05,5.0]),(5.0,[0.05,0.5]),(0.05,[0.5,5.0])]:
    for sp in [True,False]:
        t2,T,P=sister(sp,doses,td)
        if len(t2)<50: out[f'target{td}|same_compound|same_pos={sp}']={'rows':int(len(t2))}; continue
        r=evalp(T,P,t2); r['wells']=int(t2.physical_well_id.nunique()); r['compounds']=sorted(set(t2.compound_entity_id))[:40]
        out[f'target{td}|same_compound|same_pos={sp}']=r
# any target dose pooled: same compound different position
print(json.dumps(out,indent=1,default=str))
