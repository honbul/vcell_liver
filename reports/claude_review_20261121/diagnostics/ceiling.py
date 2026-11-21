"""Post-hoc read-only ceiling diagnostic (reviewer). Uses project-seen Tahoe source rows only."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, sys
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
def load(d, name):
    n=np.load(d+name+'_counts.npz')
    m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t')
    assert m.shape[0]==len(r),(m.shape,len(r))
    return m,r
d05=R+'vcell_liver_r11_chemical_probe/assets/source/'
dxx=R+'vcell_liver_r12_multiaxis/assets/doses/source/'
m05,r05=load(d05,'treated'); print('0.5 rows',m05.shape, r05.columns.tolist()); print(r05.groupby('dose_uM').physical_well_id.nunique())
mxx,rxx=load(dxx,'treated')
A05,ra05=load(d05,'A'); B05,rb05=load(d05,'B')
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64)
    if genes is not None: m=m[:,genes]
    x=m.toarray().astype(np.float64)
    return np.log1p(x/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
x_all=lognorm(m05[ok05])
var=x_all.var(0); genes=np.sort(np.argsort(-var)[:2000]); del x_all
print('panel var min', var[genes].min())
X05=lognorm(m05,genes); Xxx=lognorm(mxx,genes); XA=lognorm(A05,genes); XB=lognorm(B05,genes)
rows=pd.concat([r05.assign(src='s05'),rxx.assign(src='sxx')],ignore_index=True); X=np.vstack([X05,Xxx])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
# pools need >=3 compound groups
ng=rows.groupby('pool').compound_group_id.transform('nunique'); keep=(ng>=3).values
rows=rows[keep].reset_index(drop=True); X=X[keep]
C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items():
    C[idx]=X[idx]-X[idx].mean(0)
rows['rms']=np.sqrt((C**2).mean(1))
def agg(v, r):  # row -> group mean -> mean over groups
    return pd.Series(v,index=r.index).groupby(r.compound_group_id).mean().mean()
out={}
for dose in [0.05,0.5,5.0]:
    s=rows.dose_uM==dose
    out[f'B0_contrast_rmse_{dose}']=float(agg(rows.rms[s].values, rows[s])); out[f'n_rows_{dose}']=int(s.sum()); out[f'n_groups_{dose}']=int(rows[s].compound_group_id.nunique()); out[f'n_wells_{dose}']=int(rows[s].physical_well_id.nunique())
# DMSO null: A-B within plate x line
ka=ra05.eligible_cell_count.values.astype(bool); kb=rb05.eligible_cell_count.values.astype(bool)
a=ra05.assign(i=np.arange(len(ra05)))[ka].set_index(['plate_id','cell_line_id']); b=rb05.assign(i=np.arange(len(rb05)))[kb].set_index(['plate_id','cell_line_id'])
j=a.join(b,lsuffix='_a',rsuffix='_b',how='inner')
D=XA[j.i_a.values]-XB[j.i_b.values]
j['null_rms']=np.sqrt((D**2).mean(1))/np.sqrt(2)
out['dmso_pairs']=int(len(j)); out['dmso_single_well_null_rms_mean']=float(j.null_rms.mean()); out['dmso_single_well_null_rms_median']=float(j.null_rms.median())
out['dmso_cells_median']=float(np.median(np.r_[j.n_cells_a,j.n_cells_b])); out['treated05_cells_median']=float(rows[rows.dose_uM==0.5].n_cells.median())
main=rows[(rows.dose_uM==0.5)]
pl=main.plate_id.unique(); jj=j.reset_index(); jj=jj[jj.plate_id.isin(pl)]
out['dmso_null_rms_mean_mainplates']=float(jj.null_rms.mean()); out['dmso_null_ms_mean_mainplates']=float((jj.null_rms**2).mean())
out['treated05_contrast_ms_rowmean']=float((main.rms**2).mean()); out['treated05_contrast_rms_rowmean']=float(main.rms.mean())
# cell-count dependence
q=pd.qcut(main.n_cells,5); out['B0_rms_by_cellcount_quintile']={str(k):float(v) for k,v in main.groupby(q,observed=True).rms.mean().items()}
# well-mean (line-shared) component
rows['row']=np.arange(len(rows))
wm={}; share=[]
for (w,dz),idx in rows.groupby(['physical_well_id','dose_uM']).indices.items():
    m=C[idx].mean(0); wm[(w,dz)]=m
well=rows.drop_duplicates(['physical_well_id','dose_uM'])[['physical_well_id','dose_uM','compound_group_id','plate_id']].reset_index(drop=True)
well['wm_rms']=[float(np.sqrt((wm[(w,d)]**2).mean())) for w,d in zip(well.physical_well_id,well.dose_uM)]
Wm=np.stack([wm[(w,d)] for w,d in zip(rows.physical_well_id,rows.dose_uM)])
rows['shared_ms']=(Wm**2).mean(1); rows['resid_rms']=np.sqrt(((C-Wm)**2).mean(1))
m5=rows.dose_uM==0.5
out['share_of_contrast_ms_in_wellmean_0.5']=float(rows.shared_ms[m5].sum()/(rows.rms[m5]**2).sum())
out['line_specific_residual_rmse_0.5(same-well-oracle)']=float(agg(rows.resid_rms[m5].values, rows[m5]))
# cross-dose reference oracle: same group, same line, other dose (other plate)
def ref(target_dose, ref_doses):
    t=rows[rows.dose_uM==target_dose]
    rr=rows[rows.dose_uM.isin(ref_doses)]
    # mean contrast per (group,line) over ref rows on other plates
    key=rr.groupby(['compound_group_id','cell_line_id']).indices
    P=np.zeros((len(t),C.shape[1]),np.float32); has=np.zeros(len(t),bool)
    rr_idx=rr.index.values; rr_pl=rr.plate_id.values
    for n,(g,l,p) in enumerate(zip(t.compound_group_id,t.cell_line_id,t.plate_id)):
        k=key.get((g,l))
        if k is None: continue
        k=[i for i in k if rr_pl[i]!=p]
        if not k: continue
        P[n]=C[rr_idx[k]].mean(0); has[n]=True
    return t,P,has
def report(tag,target_dose,ref_doses):
    t,P,has=ref(target_dose,ref_doses); T=C[t.index.values]
    t=t[has]; T=T[has]; P=P[has]
    res={'rows':int(len(t)),'groups':int(t.compound_group_id.nunique())}
    cos=(T*P).sum(1)/np.sqrt((T**2).sum(1)*(P**2).sum(1)+1e-30)
    res['cos_median']=float(np.median(cos)); res['cos_groupmean']=float(agg(cos,t))
    rng=np.random.default_rng(0); perm=rng.permutation(len(t))
    cosn=(T*P[perm]).sum(1)/np.sqrt((T**2).sum(1)*(P[perm]**2).sum(1)+1e-30); res['cos_null_median(shuffled ref)']=float(np.median(cosn))
    b0=agg(np.sqrt((T**2).mean(1)),t); res['B0']=float(b0)
    curve={}
    for s in [0.1,0.2,0.3,0.4,0.5,0.75,1.0]:
        curve[s]=float(agg(np.sqrt(((T-s*P)**2).mean(1)),t))
    res['oracle_rmse_by_scale']=curve
    sb=min(curve,key=curve.get); res['best_scale']=sb; res['best_rel_change_%']=float(100*(curve[sb]-b0)/b0)
    # leave-group-fold-out scale selection (3 folds by group id)
    gid=t.compound_group_id.values; folds=(pd.factorize(gid)[0]%3); rm=np.zeros(len(t))
    for f in range(3):
        tr=folds!=f; best=None
        for s in [0,0.1,0.2,0.3,0.4,0.5,0.75,1.0]:
            v=agg(np.sqrt(((T[tr]-s*P[tr])**2).mean(1)),t[tr])
            if best is None or v<best[0]: best=(v,s)
        rm[~tr]=np.sqrt(((T[~tr]-best[1]*P[~tr])**2).mean(1))
    res['oracle_cv_scale_rmse']=float(agg(rm,t)); res['oracle_cv_rel_change_%']=float(100*(res['oracle_cv_scale_rmse']-b0)/b0)
    # strata by independent activity proxy: reference contrast norm (well/line), tertiles over groups
    pn=pd.Series(np.sqrt((P**2).mean(1)),index=t.index).groupby(t.compound_group_id).mean()
    ter=pd.qcut(pn,[0,.5,.8,1.0],labels=['low50','mid30','top20'])
    st=t.compound_group_id.map(ter).values
    res['strata']={}
    for lab in ['low50','mid30','top20']:
        k=st==lab
        b=agg(np.sqrt((T[k]**2).mean(1)),t[k]); cs={s:float(agg(np.sqrt(((T[k]-s*P[k])**2).mean(1)),t[k])) for s in [0.25,0.5,0.75,1.0]}
        sb=min(cs,key=cs.get)
        res['strata'][lab]={'groups':int(t[k].compound_group_id.nunique()),'B0':float(b),'best_scale':sb,'oracle':cs[sb],'rel_%':float(100*(cs[sb]-b)/b),'cos_median':float(np.median(cos[k]))}
    for name,cv in [('SNU-423','CVCL_0366'),('HepG2/C3A','CVCL_1098')]:
        k=(t.cell_line_id==cv).values
        if k.sum():
            b=agg(np.sqrt((T[k]**2).mean(1)),t[k]); cs={s:float(agg(np.sqrt(((T[k]-s*P[k])**2).mean(1)),t[k])) for s in [0.25,0.5,0.75,1.0]}; sb=min(cs,key=cs.get)
            res[name]={'rows':int(k.sum()),'B0':float(b),'best_scale':sb,'oracle':cs[sb],'rel_%':float(100*(cs[sb]-b)/b),'cos_median':float(np.median(cos[k]))}
    out[tag]=res
report('target0.5_ref5',0.5,[5.0]); report('target0.5_ref0.05',0.5,[0.05]); report('target0.5_ref_both',0.5,[0.05,5.0]); report('target5_ref0.5',5.0,[0.5]); report('target5_ref0.05',5.0,[0.05])
# concentration of signal: share of total contrast MS by top groups at 0.5
gms=(main.rms**2).groupby(main.compound_group_id).mean().sort_values(ascending=False)
out['share_of_groupMS_top10%_groups_0.5']=float(gms.iloc[:int(.1*len(gms))].sum()/gms.sum()); out['share_top20%']=float(gms.iloc[:int(.2*len(gms))].sum()/gms.sum())
out['group_rms_quantiles_0.5']={str(q):float(np.sqrt(gms).quantile(q)) for q in [0.1,0.25,0.5,0.75,0.9,0.99]}
m5r=rows[rows.dose_uM==5.0]; gms5=(m5r.rms**2).groupby(m5r.compound_group_id).mean()
out['group_rms_quantiles_5']={str(q):float(np.sqrt(gms5).quantile(q)) for q in [0.1,0.25,0.5,0.75,0.9,0.99]}
print(json.dumps(out,indent=1))
