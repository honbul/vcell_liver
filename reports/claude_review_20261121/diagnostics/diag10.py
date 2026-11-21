"""Reviewer diagnostic 10 (descriptive): how concentrated is the reproducible drug response across compounds?
Per compound group: cross-dose reproducible energy (0.5 vs 5 uM, same line), with a generic-position cross-term removed
using independent donor halves; listed with MoA. Also liver-line view and response-size predictability by MoA class."""
import numpy as np, pandas as pd, scipy.sparse as sp, json
R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
def load(d, name):
    n=np.load(d+name+'_counts.npz'); m=sp.csr_matrix((n['data'],n['indices'],n['indptr']),shape=tuple(n['shape']))
    r=pd.read_csv(d+name+'_rows.tsv',sep='\t'); return m,r
m05,r05=load(R+'vcell_liver_r11_chemical_probe/assets/source/','treated'); mxx,rxx=load(R+'vcell_liver_r12_multiaxis/assets/doses/source/','treated')
def lognorm(m, genes=None):
    lib=np.asarray(m.sum(1)).ravel().astype(np.float64); lib[lib==0]=1
    if genes is not None: m=m[:,genes]
    return np.log1p(m.toarray().astype(np.float64)/lib[:,None]*1e4).astype(np.float32)
ok05=r05.eligible_cell_count.values.astype(bool)
xa=lognorm(m05[ok05]); genes=np.sort(np.argsort(-xa.var(0))[:2000]); del xa
rows=pd.concat([r05,rxx],ignore_index=True); X=np.vstack([lognorm(m05,genes),lognorm(mxx,genes)])
keep=rows.eligible_cell_count.values.astype(bool); rows=rows[keep].reset_index(drop=True); X=X[keep]
rows['pool']=rows.plate_id+'|'+rows.cell_line_id+'|'+rows.dose_uM.astype(str)
keep=(rows.groupby('pool').compound_group_id.transform('nunique')>=3).values; rows=rows[keep].reset_index(drop=True); X=X[keep]
C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items(): C[idx]=X[idx]-X[idx].mean(0)
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
rows['pos']=rows.physical_well_id.map(dict(zip(sm['sample'],sm.pos))).values
dsetf=lambda p: 1 if p=='plate14' else (int(p.replace('plate',''))-1)//3
gi=rows.compound_group_id.values.astype(int); pl=rows.plate_id.values; ds=rows.dose_uM.values; ln=rows.cell_line_id.values; pos=rows.pos.values; st=rows.plate_id.map(dsetf).values
dm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet').set_index('drug')
gs=rows.drop_duplicates('compound_group_id').set_index('compound_group_id')
# generic position estimate from OTHER drug sets, two independent halves of donor plates (by plate parity within the donor list)
key=pd.DataFrame({'p':pos,'l':ln}).groupby(['p','l']).indices
P1=np.zeros_like(C); P2=np.zeros_like(C); ok=np.zeros(len(rows),bool)
for (q,l),kk in key.items():
    for s in np.unique(st[kk]):
        tgt=kk[st[kk]==s]; dn=kk[st[kk]!=s]
        if len(dn)<2: continue
        up=np.unique(pl[dn]); 
        if len(up)<2: continue
        h1=up[::2]; h2=up[1::2]
        v1=np.mean([C[dn[pl[dn]==u]].mean(0) for u in h1],0); v2=np.mean([C[dn[pl[dn]==u]].mean(0) for u in h2],0)
        P1[tgt]=v1; P2[tgt]=v2; ok[tgt]=True
# pairs: same group, same line, 0.5 vs 5 uM (same set & position)
df=pd.DataFrame({'i':np.arange(len(rows)),'g':gi,'l':ln})
a=df[(ds==5.0)&ok]; b=df[(ds==0.5)&ok]; m=a.merge(b,on=['g','l'],suffixes=('','_b')); ia=m.i.values; ib=m.i_b.values
k=(st[ia]==st[ib])&(pos[ia]==pos[ib]); ia=ia[k]; ib=ib[k]
ip_raw=np.einsum('ij,ij->i',C[ia].astype(np.float64),C[ib].astype(np.float64))
ip_cor=np.einsum('ij,ij->i',(C[ia]-P1[ia]).astype(np.float64),(C[ib]-P2[ib]).astype(np.float64))   # independent position estimates on the two sides
e5=(C[ia].astype(np.float64)**2).sum(1); e05=(C[ib].astype(np.float64)**2).sum(1)
liv=np.isin(ln[ia],['CVCL_0366','CVCL_1098'])
t=pd.DataFrame({'g':gi[ia],'ip_raw':ip_raw,'ip_cor':ip_cor,'e5':e5,'e05':e05,'liv':liv})
G_=t.groupby('g').agg(ip_raw=('ip_raw','mean'),ip_cor=('ip_cor','mean'),e5=('e5','mean'),e05=('e05','mean'),n=('g','size'))
L_=t[t.liv].groupby('g').agg(ip_cor_liver=('ip_cor','mean'),e5_liver=('e5','mean'))
G_=G_.join(L_); G_['name']=[gs.loc[g,'compound_entity_id'] for g in G_.index]; G_['moa']=[str(dm['moa-fine'].get(n,'?')) for n in G_.name]; G_['targets']=[str(dm['targets'].get(n,'?'))[:30] for n in G_.name]
G_['shared_cor']=G_.ip_cor/np.sqrt(G_.e5*G_.e05)
tot=G_.ip_cor.sum(); srt=G_.sort_values('ip_cor',ascending=False)
out={'groups':int(len(G_)),'overall_shared_raw':float(G_.ip_raw.sum()/np.sqrt(G_.e5.sum()*G_.e05.sum())),'overall_shared_position_removed':float(tot/np.sqrt(G_.e5.sum()*G_.e05.sum()))}
cum=srt.ip_cor.cumsum()/tot
out['share_of_reproducible_energy_in_top_k_groups']={int(k_):round(float(cum.iloc[k_-1]),3) for k_ in (1,3,5,10,20,40,80)}
out['groups_with_shared_cor>0.2']=int((G_.shared_cor>0.2).sum()); out['groups_with_shared_cor>0.1']=int((G_.shared_cor>0.1).sum()); out['median_shared_cor']=float(G_.shared_cor.median())
print(json.dumps(out)); pd.set_option('display.width',250)
print(srt.head(40)[['name','moa','targets','ip_cor','shared_cor','e5','ip_cor_liver']].round(4).to_string())
mo=G_.groupby('moa').agg(n=('ip_cor','size'),ip_cor_mean=('ip_cor','mean'),shared_med=('shared_cor','median')).sort_values('ip_cor_mean',ascending=False); print(mo[mo.n>=2].head(25).round(4).to_string())
# predictability of response size by MoA class (leave-one-group-out mean of class) -- spearman
G_['loo_moa']=np.nan
for mname,sub in G_.groupby('moa'):
    if mname in ('unclear','?','None') or len(sub)<2: continue
    G_.loc[sub.index,'loo_moa']=(sub.ip_cor.sum()-sub.ip_cor)/(len(sub)-1)
v=G_.dropna(subset=['loo_moa']); print('MoA-class LOO vs own reproducible energy: n',len(v),'spearman',round(float(v[['loo_moa','ip_cor']].corr(method='spearman').iloc[0,1]),3),'pearson(log1p-clip)',round(float(np.corrcoef(np.log1p(v.loo_moa.clip(0)*1e3),np.log1p(v.ip_cor.clip(0)*1e3))[0,1]),3))
print('liver vs all-lines reproducible energy spearman',round(float(G_[['ip_cor','ip_cor_liver']].corr(method='spearman').iloc[0,1]),3))
G_.to_csv(R+'review_claude_20261121/diag10_groups.tsv',sep='\t'); json.dump(out,open(R+'review_claude_20261121/diag10.json','w'),indent=1)
