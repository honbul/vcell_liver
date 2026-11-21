"""Reviewer exploratory diagnostic 7 (no training; descriptive ceilings only, project-seen rows).
(A) replicate agreement before/after subtracting a full-donor position estimate (other drug sets only):
    same-position 5 uM replicates (plate 6<->14), different-position same-compound pairs, sister-dose pairs.
(B) oracle nearest-neighbour ceiling: does the 203-group library contain a response-similar compound at all?"""
import numpy as np, pandas as pd, scipy.sparse as sp, json
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
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
G=X.shape[1]; C=np.empty_like(X)
for p,idx in rows.groupby('pool').indices.items(): C[idx]=X[idx]-X[idx].mean(0)
sm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/sample_metadata.parquet')
sm['num']=sm['sample'].str.extract(r'(\d+)').astype(int); sm['pos']=sm.groupby('plate').num.rank().astype(int)
rows['pos']=rows.physical_well_id.map(dict(zip(sm['sample'],sm.pos))).values
gi=rows.compound_group_id.values; pl=rows.plate_id.values; ds=rows.dose_uM.values; ln=rows.cell_line_id.values; pos=rows.pos.values
dset=np.array([1 if p=='plate14' else (int(p.replace('plate',''))-1)//3 for p in pl])
LIVER={'CVCL_0366':'SNU-423','CVCL_1098':'HepG2/C3A'}
# full-donor position estimate: same (pos,line), plates of OTHER drug sets, own group excluded, plate-equal mean
key=pd.DataFrame({'p':pos,'l':ln}).groupby(['p','l']).indices
P=np.zeros_like(C); has=np.zeros(len(rows),bool); nd=np.zeros(len(rows),int)
for (q,l),kk in key.items():
    for i in kk:
        j=kk[(dset[kk]!=dset[i])&(gi[kk]!=gi[i])]
        if len(j)==0: continue
        up=np.unique(pl[j]); P[i]=np.mean([C[j[pl[j]==u]].mean(0) for u in up],0); has[i]=True; nd[i]=len(up)
out={'support':float(has.mean()),'donor_plates_median':float(np.median(nd[has]))}
s_ls=float((C[has]*P[has]).sum()/(P[has]**2).sum()); out['LS_scale_of_position_estimate']=s_ls
def rm(a): return np.sqrt((a**2).mean(1))
def gmean(v,g): return float(pd.Series(np.asarray(v)).groupby(np.asarray(g)).mean().mean())
def cosr(U,V): return (U*V).sum(1)/np.sqrt((U**2).sum(1)*(V**2).sum(1)+1e-30)
def pairs(sel_a,sel_b,same_pos):
    """row pairs (a,b): same group, same line, a in sel_a, b in sel_b, different plate, position equal/unequal."""
    df=pd.DataFrame({'i':np.arange(len(rows)),'g':gi,'l':ln}); A=df[sel_a]; Bd=df[sel_b]
    m=A.merge(Bd,on=['g','l'],suffixes=('','_b')); a=m.i.values; b=m.i_b.values
    k=(pl[a]!=pl[b])&((pos[a]==pos[b])==same_pos)&has[a]&has[b]; return a[k],b[k]
def report(a,b,name):
    if len(a)==0: return {'rows':0}
    o={'rows':int(len(a)),'groups':int(len(set(gi[a]))),'wells_target':int(len(set(zip(pl[a],pos[a]))))}
    def block(T,D,tag):
        b0=gmean(rm(T),gi[a]); o[tag+'_B0']=b0; o[tag+'_cos_median']=round(float(np.median(cosr(T,D))),4)
        o[tag+'_rmse_gain_%_at_scale(0.25,0.5,0.75,1)']=[round(100*(gmean(rm(T-c*D),gi[a])/b0-1),3) for c in (0.25,0.5,0.75,1)]
        e=pd.DataFrame({'d':rm(T-0.5*D)-rm(T),'g':gi[a]}).groupby('g').d.mean().values; rng=np.random.default_rng(130113)
        bs=e[rng.integers(0,len(e),(2000,len(e)))].mean(1)/b0*100; o[tag+'_gain_at_0.5_ci95']=[round(float(x),3) for x in np.percentile(bs,[2.5,97.5])]
    block(C[a],C[b],'raw')
    for s in (0.5,1.0): block(C[a]-s*P[a],C[b]-s*P[b],f'poscorr{s}')
    b0=gmean(rm(C[a]),gi[a]); o['position_only_gain_%_at(0.25,0.5,1)']=[round(100*(gmean(rm(C[a]-c*P[a]),gi[a])/b0-1),3) for c in (0.25,0.5,1)]
    o['cos(T,pos)']=round(float(np.median(cosr(C[a],P[a]))),4)
    Z=np.stack([C[b].ravel(),P[a].ravel()],1).astype(np.float64); co,_,_,_=np.linalg.lstsq(Z,C[a].ravel().astype(np.float64),rcond=None)
    o['LS_coef(rep,pos)']=[round(float(x),3) for x in co]; o['LS_combo_gain_%']=round(100*(gmean(rm(C[a]-co[0]*C[b]-co[1]*P[a]),gi[a])/b0-1),3)
    return o
A={}
d5=ds==5.0; d05=ds==0.5; d005=ds==0.05
for lab,lm in [('all_lines',np.ones(len(rows),bool))]+[(n,ln==c) for c,n in LIVER.items()]:
    A[lab]={
     'same_pos_5uM_plate6<->14':report(*pairs(d5&lm&np.isin(pl,['plate6','plate14']),d5&np.isin(pl,['plate6','plate14']),True),'x'),
     'diff_pos_5uM_same_compound':report(*pairs(d5&lm,d5,False),'x'),
     'diff_pos_0.5uM_same_compound':report(*pairs(d05&lm,d05,False),'x'),
     'same_pos_sister_dose_0.5->5':report(*pairs(d5&lm,d05,True),'x'),
     'same_pos_sister_dose_0.05->0.5':report(*pairs(d05&lm,d005,True),'x'),
     'same_pos_sister_dose_5->0.5':report(*pairs(d05&lm,d5,True),'x'),
     'diff_pos_cross_dose_any':report(*pairs(lm,np.ones(len(rows),bool),False),'x')}
    print('A',lab,json.dumps(A[lab]),flush=True)
out['A']=A
# ---------- (B) oracle neighbour ceiling ----------
gs=rows.drop_duplicates('compound_group_id').set_index('compound_group_id'); gids=np.array(sorted(gs.index)); gix={g:i for i,g in enumerate(gids)}
fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(gs.loc[g,'smiles']),2,2048) for g in gids]
TAN=np.array([DataStructs.BulkTanimotoSimilarity(f,fps) for f in fps]); np.fill_diagonal(TAN,-1)
dm=pd.read_parquet('/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet').set_index('drug')
moa=np.array([str(dm['moa-fine'].get(gs.loc[g,'compound_entity_id'],'unclear')) for g in gids])
lines=np.array(sorted(set(ln))); lix={l:i for i,l in enumerate(lines)}
Bres={}
for d in (0.05,0.5,5.0):
    for corr in (0.0,0.5):
        sel=np.where((ds==d)&has)[0]; Y=C[sel]-corr*P[sel]
        M=np.zeros((len(gids),len(lines),G),np.float32); cnt=np.zeros((len(gids),len(lines)),np.float32)
        gg=np.array([gix[g] for g in gi[sel]]); ll=np.array([lix[l] for l in ln[sel]])
        np.add.at(M,(gg,ll),Y); np.add.at(cnt,(gg,ll),1); pres=cnt>0; M/=np.maximum(cnt,1)[:,:,None]
        n2=(M**2).sum(2)                       # g x line
        ip=np.stack([M[:,l,:]@M[:,l,:].T for l in range(len(lines))],2)   # g x h x line
        okg=pres.sum(1)>=10; o={'groups':int(okg.sum())}
        def gain(s,choose):
            """choose: g -> h index (or -1). mean over lines present in both, then groups."""
            num=[];den=[]
            for g in np.where(okg)[0]:
                h=choose[g]; m=pres[g]&(pres[h] if h>=0 else True)
                if m.sum()==0: continue
                b=np.sqrt(n2[g,m]/G)
                e=b if h<0 else np.sqrt(np.maximum(n2[g,m]-2*s*ip[g,h,m]+s*s*n2[h,m],0)/G)
                num.append(e.mean()); den.append(b.mean())
            return round(float(100*(np.mean(num)/np.mean(den)-1)),3)
        for s in (0.25,0.5,1.0):
            # oracle: best single neighbour group per test group (shared across lines)
            best=np.full(len(gids),-1)
            for g in np.where(okg)[0]:
                m=pres[g][None,:]&pres; e=np.sqrt(np.maximum(n2[g][None,:]-2*s*ip[g]+s*s*n2,0)/G); b=np.sqrt(n2[g]/G)[None,:]
                sc=np.where(m,e-b,0).sum(1)/np.maximum(m.sum(1),1); sc[g]=np.inf; sc[m.sum(1)<10]=np.inf; best[g]=int(np.argmin(sc))
            o[f'oracle_nn_gain_%_s{s}']=gain(s,best)
            if s==0.5:
                o['oracle_same_moa_frac']=round(float(np.mean([(moa[g]==moa[best[g]])&(moa[g]!='unclear') for g in np.where(okg)[0]])),3)
                o['oracle_tanimoto_median']=round(float(np.median([TAN[g,best[g]] for g in np.where(okg)[0]])),3)
                o['oracle_tanimoto_rank_median']=float(np.median([(TAN[g]>TAN[g,best[g]]).sum()+1 for g in np.where(okg)[0]]))
                pg=[]
                for g in np.where(okg)[0]:
                    h=best[g]; m=pres[g]&pres[h]; pg.append(float((ip[g,h,m]/np.sqrt(n2[g,m]*n2[h,m]+1e-30)).mean()))
                o['oracle_cos_quantiles(10,50,90)']=[round(float(x),3) for x in np.percentile(pg,[10,50,90])]
            tn=np.argmax(np.where(okg[None,:],TAN,-2),1); o[f'tanimoto_nn_gain_%_s{s}']=gain(s,tn)
            rng=np.random.default_rng(0); rn=np.array([rng.choice(np.setdiff1d(np.where(okg)[0],[g])) for g in range(len(gids))]); o[f'random_nn_gain_%_s{s}']=gain(s,rn)
        Bres[f'dose{d}_poscorr{corr}']=o; print('B',d,corr,json.dumps(o,default=float),flush=True)
out['B']=Bres
json.dump(out,open(R+'review_claude_20261121/diag7.json','w'),indent=1)
