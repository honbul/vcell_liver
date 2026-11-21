"""Reviewer post-hoc diagnostic 2 (read-only; project-seen Tahoe source rows)."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, re
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
def load(d, name):
    n=np.load(d+name+'_counts.npz'); m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t'); assert m.shape[0]==len(r); return m,r
d05=R+'vcell_liver_r11_chemical_probe/assets/source/'; dxx=R+'vcell_liver_r12_multiaxis/assets/doses/source/'
m05,r05=load(d05,'treated'); mxx,rxx=load(dxx,'treated')
ctl=[load(d05,'A'),load(d05,'B'),load(dxx,'A'),load(dxx,'B')]
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64); lib[lib==0]=1
    if genes is not None: m=m[:,genes]
    return np.log1p(m.toarray().astype(np.float64)/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
xa=lognorm(m05[ok05]); var=xa.var(0); genes=np.sort(np.argsort(-var)[:2000]); del xa
gsym=pd.read_csv(R+'vcell_liver_r12_multiaxis/assets/doses/metadata/genes.tsv',sep='\t').gene_symbol.values[genes]
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]
# controls: mean of available DMSO wells per plate x line (dedupe by well)
cr=pd.concat([r for _,r in ctl],ignore_index=True); CX=np.vstack([lognorm(m,genes) for m,_ in ctl])
k=cr.eligible_cell_count.values.astype(bool); cr=cr[k].reset_index(drop=True); CX=CX[k]
cr['key']=cr.physical_well_id+'|'+cr.cell_line_id; first=~cr.key.duplicated(); cr=cr[first.values].reset_index(drop=True); CX=CX[first.values]
ctl_mean={k:CX[i].mean(0) for k,i in cr.groupby(['plate_id','cell_line_id']).indices.items()}
C=np.empty_like(X); M=np.zeros_like(X); hasm=np.zeros(len(rows),bool)
for p,idx in rows.groupby('pool').indices.items():
    mu=X[idx].mean(0); C[idx]=X[idx]-mu
    pl,ln,_=p.split('|'); cm=ctl_mean.get((pl,ln))
    if cm is not None: M[idx]=mu-cm; hasm[idx]=True
rows['rms']=np.sqrt((C**2).mean(1))
def agg(v,r): return float(pd.Series(np.asarray(v),index=r.index).groupby(r.compound_group_id).mean().mean())
def rm(a): return np.sqrt((a**2).mean(1))
def cosr(a,b): return (a*b).sum(1)/np.sqrt((a**2).sum(1)*(b**2).sum(1)+1e-30)
out={}
# ---- 1. layout
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
sm['drug']=sm.drug.str.strip()
pp=sm.groupby('drug').pos.agg(['nunique','count'])
out['layout']={'n_drugs':int(len(pp)),'drugs_same_position_on_all_plates':int((pp['nunique']==1).sum()),'dmso_positions':sm[sm.drug.str.contains('DMSO')].groupby('plate').pos.apply(list).to_dict()}
well_pos=dict(zip(sm['sample'],sm.pos))
# ---- 2. rank-1 common axis (pool mean response) at 0.5
t=rows[(rows.dose_uM==0.5)&hasm]; T=C[t.index.values]; Mt=M[t.index.values]
a=(T*Mt).sum(1)/(Mt**2).sum(1); proj=a[:,None]*Mt
out['rank1']={'rows':int(len(t)),'pool_mean_rms(mean)':float(rm(Mt).mean()),'cos(c,-m)_median':float(np.median(cosr(T,-Mt))),'share_MS_on_axis':float((proj**2).sum()/(T**2).sum()),
 'B0':agg(rm(T),t),'oracle_scalar_rmse':agg(rm(T-proj),t)}
out['rank1']['oracle_scalar_rel_%']=100*(out['rank1']['oracle_scalar_rmse']/out['rank1']['B0']-1)
# ---- 3. cross-dose reference, with/without axis, by min cells
def ref(target_dose, ref_doses, mincells=20, same_plate_ok=False):
    tt=rows[(rows.dose_uM==target_dose)&hasm&(rows.n_cells>=mincells)]
    rr=rows[rows.dose_uM.isin(ref_doses)&hasm&(rows.n_cells>=mincells)]
    key=rr.groupby(['compound_group_id','cell_line_id']).indices; ri=rr.index.values; rp=rr.plate_id.values
    P=np.zeros((len(tt),C.shape[1]),np.float32); PA=np.zeros(len(tt)); has=np.zeros(len(tt),bool)
    for n,(g,l,p) in enumerate(zip(tt.compound_group_id,tt.cell_line_id,tt.plate_id)):
        kk=key.get((g,l))
        if kk is None: continue
        kk=[i for i in kk if rp[i]!=p]
        if not kk: continue
        ii=ri[kk]; P[n]=C[ii].mean(0); PA[n]=np.mean((C[ii]*M[ii]).sum(1)/(M[ii]**2).sum(1)); has[n]=True
    return tt[has],C[tt.index.values][has],P[has],M[tt.index.values][has],PA[has]
def best(Tm,Pm,tt,scales=(0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0)):
    b=agg(rm(Tm),tt); cs={s:agg(rm(Tm-s*Pm),tt) for s in scales}; s=min(cs,key=cs.get); return {'B0':b,'best_scale':s,'oracle':cs[s],'rel_%':100*(cs[s]/b-1)}
cd={}
for mc in [20,200,500]:
    tt,Tm,Pm,Mm,PA=ref(0.5,[0.05,5.0],mc)
    res={'rows':int(len(tt)),'groups':int(tt.compound_group_id.nunique()),'cos_median':float(np.median(cosr(Tm,Pm))),'full':best(Tm,Pm,tt)}
    at=(Tm*Mm).sum(1)/(Mm**2).sum(1)
    res['axis_scalar_spearman(target vs ref)']=float(pd.Series(at).corr(pd.Series(PA),method='spearman'))
    res['axis_only_ref_scalar']=best(Tm,PA[:,None]*Mm,tt)
    Tr=Tm-at[:,None]*Mm; ap=(Pm*Mm).sum(1)/(Mm**2).sum(1); Pr=Pm-ap[:,None]*Mm
    res['offaxis_cos_median']=float(np.median(cosr(Tr,Pr))); res['offaxis']=best(Tr,Pr,tt)
    cd[f'mincells{mc}']=res
out['cross_dose_0.5_from_both']=cd
# ---- 4. MoA / target neighbours (other compounds), LOCO, 0.5 uM
dm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet'); dm['drug']=dm.drug.str.strip()
moa=dict(zip(dm.drug,dm['moa-fine'])); tg=dict(zip(dm.drug,dm.targets))
rows['moa']=rows.compound_entity_id.str.strip().map(moa); rows['tg']=rows.compound_entity_id.str.strip().map(tg)
out['moa_label_coverage_rows']=float(rows.moa.notna().mean())
def neigh(label, ref_doses, mode, shuffle=False, target_dose=0.5):
    tt=rows[(rows.dose_uM==target_dose)]; rr=rows[rows.dose_uM.isin(ref_doses)]
    lab=rows[label].copy()
    if shuffle:
        g=rows.drop_duplicates('compound_group_id').set_index('compound_group_id')[label]; rng=np.random.default_rng(1); g2=pd.Series(rng.permutation(g.values),index=g.index); lab=rows.compound_group_id.map(g2)
    bad={'unclear','None','nan','',None}
    rl=lab[rr.index].values; key={}
    for i,(l,ln) in zip(rr.index.values,zip(rl,rr.cell_line_id.values)):
        if l in bad or (isinstance(l,float)): continue
        key.setdefault((l,ln),[]).append(i)
    gi=rows.compound_group_id.values; pl=rows.plate_id.values
    P=np.zeros((len(tt),C.shape[1]),np.float32); has=np.zeros(len(tt),bool); nn=np.zeros(len(tt))
    for n,(i,l,ln) in enumerate(zip(tt.index.values,lab[tt.index].values,tt.cell_line_id.values)):
        kk=key.get((l,ln))
        if not kk: continue
        if mode=='other_plate': kk=[j for j in kk if gi[j]!=gi[i] and pl[j]!=pl[i]]
        elif mode=='same_plate': kk=[j for j in kk if gi[j]!=gi[i] and pl[j]==pl[i]]
        else: kk=[j for j in kk if gi[j]!=gi[i]]
        if not kk: continue
        # group-balanced mean
        df=pd.DataFrame({'g':gi[kk],'j':kk}); P[n]=np.mean([C[v.values].mean(0) for _,v in df.groupby('g').j],axis=0); has[n]=True; nn[n]=df.g.nunique()
    t2=tt[has]; Tm=C[tt.index.values][has]; Pm=P[has]
    if len(t2)==0: return {'rows':0}
    res={'rows':int(len(t2)),'groups_covered':int(t2.compound_group_id.nunique()),'n_neighbors_median':float(np.median(nn[has])),'cos_median':float(np.median(cosr(Tm,Pm))),'cos_groupmean':agg(cosr(Tm,Pm),t2)}
    res.update(best(Tm,Pm,t2))
    # effect on ALL 0.5 rows if uncovered rows fall back to zero
    allb=agg(rm(C[tt.index.values]),tt); pred=np.zeros((len(tt),C.shape[1]),np.float32); pred[has]=res['best_scale']*Pm
    res['all_rows_B0']=allb; res['all_rows_with_fallback']=agg(rm(C[tt.index.values]-pred),tt); res['all_rows_rel_%']=100*(res['all_rows_with_fallback']/allb-1)
    return res
nb={}
for label in ['moa','tg']:
    for mode in ['other_plate','same_plate','any']:
        nb[f'{label}|0.5only|{mode}']=neigh(label,[0.5],mode)
    nb[f'{label}|alldoses|other_plate']=neigh(label,[0.05,0.5,5.0],'other_plate')
    nb[f'{label}|alldoses|other_plate|SHUFFLED']=neigh(label,[0.05,0.5,5.0],'other_plate',shuffle=True)
    nb[f'{label}|target5|alldoses|other_plate']=neigh(label,[0.05,0.5,5.0],'other_plate',target_dose=5.0)
out['label_neighbors']=nb
# ---- 5. DMSO positional control: (A-B) across plates
cw=cr.copy(); cw['pos']=cw.physical_well_id.map(well_pos); cw['i']=np.arange(len(cw))
dd={}
for (pl,ln),g in cw.groupby(['plate_id','cell_line_id']):
    g=g.sort_values('pos')
    if len(g)==2: dd[(pl,ln)]=CX[g.i.values[0]]-CX[g.i.values[1]]
plates=sorted({k[0] for k in dd}); lines=sorted({k[1] for k in dd}); cs=[]
for i,p in enumerate(plates):
    for q in plates[i+1:]:
        v=[float(cosr(dd[(p,l)][None],dd[(q,l)][None])[0]) for l in lines if (p,l) in dd and (q,l) in dd]
        if v: cs.append((p,q,float(np.median(v))))
out['dmso_AminusB_crossplate_cos']={'pairs':len(cs),'median_of_medians':float(np.median([c[2] for c in cs])),'abs_median':float(np.median([abs(c[2]) for c in cs])),'min':min(c[2] for c in cs),'max':max(c[2] for c in cs)}
# within-well cross-line consistency of DMSO A-B (well technical effect shared by lines)
v=[]
for p in plates:
    L=[dd[(p,l)] for l in lines if (p,l) in dd]
    if len(L)>2:
        L=np.stack(L); mu=L.mean(0); v.append(float((mu**2).sum()*len(L)/(L**2).sum()))
out['dmso_AminusB_share_MS_shared_across_lines']=float(np.mean(v))
# ---- 6. gene concentration
t=rows[rows.dose_uM==0.5]; T=C[t.index.values]; gm=(T**2).mean(0); o=np.argsort(-gm)
out['gene_MS_share']={'top20':float(gm[o[:20]].sum()/gm.sum()),'top100':float(gm[o[:100]].sum()/gm.sum()),'top500':float(gm[o[:500]].sum()/gm.sum()),'top20_genes':[f'{gsym[i]}:{gm[i]/gm.sum():.3f}' for i in o[:20]]}
out['rows_by_mincells_0.5']={mc:{'rows':int((t.n_cells>=mc).sum()),'B0':agg(rm(T[(t.n_cells>=mc).values]),t[t.n_cells>=mc])} for mc in [20,100,200,500,1000]}
print(json.dumps(out,indent=1,default=str))
