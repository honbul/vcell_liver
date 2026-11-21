"""Extract LINCS L1000 level-5 consensus signatures (978 landmark genes) for compounds in the Tahoe project cohort (name match)."""
import pandas as pd, numpy as np, re, h5py, json
L='/mnt/S1/sdata/pubdata/LINCS/'; R='/mnt/S1/sdata/workbench/Virtual_Cell/gssong/'
s1=pd.read_csv(L+'GSE92742/GSE92742_Broad_LINCS_sig_info.txt',sep='\t',low_memory=False); s2=pd.read_csv(L+'GSE70138/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz',sep='\t',low_memory=False)
s1['dose']=pd.to_numeric(s1.pert_dose,errors='coerce'); s1['time']=pd.to_numeric(s1.pert_time,errors='coerce')
s2['dose']=pd.to_numeric(s2.pert_idose.astype(str).str.extract(r'([\d.]+)')[0],errors='coerce'); s2['time']=pd.to_numeric(s2.pert_itime.astype(str).str.extract(r'([\d.]+)')[0],errors='coerce')
s=pd.concat([s1.assign(src='92742'),s2.assign(src='70138')],ignore_index=True); s=s[s.pert_type=='trt_cp']
norm=lambda x: re.sub(r'[^a-z0-9]','',str(x).lower()); base=lambda x: norm(re.sub(r'\(.*?\)','',str(x)).strip())
s['k']=s.pert_iname.map(norm)
r=pd.concat([pd.read_csv(R+'vcell_liver_r11_chemical_probe/assets/source/treated_rows.tsv',sep='\t'),pd.read_csv(R+'vcell_liver_r12_multiaxis/assets/doses/source/treated_rows.tsv',sep='\t')]).drop_duplicates('compound_entity_id')
r['k']=r.compound_entity_id.map(base); k2g=r.groupby('k').compound_group_id.first().to_dict()
s=s[s.k.isin(k2g)].copy(); s['g']=s.k.map(k2g).astype(int)
gi1=pd.read_csv(L+'GSE92742/GSE92742_Broad_LINCS_gene_info.txt',sep='\t'); lm=gi1[gi1.pr_is_lm==1]; lmid=lm.pr_gene_id.astype(str).values
def read(path,sigs):
    with h5py.File(path,'r') as f:
        cid=np.array([x.decode() for x in f['/0/META/COL/id'][:]]); rid=np.array([x.decode() for x in f['/0/META/ROW/id'][:]])
        ci=pd.Series(np.arange(len(cid)),index=cid); ri=pd.Series(np.arange(len(rid)),index=rid)
        sig=[x for x in sigs if x in ci.index]; cols=np.sort(ci[sig].values); rows=ri[lmid].values; o=np.argsort(rows)
        M=f['/0/DATA/0/matrix']; print(path[-40:],M.shape,len(cols),flush=True)
        out=np.empty((len(cols),len(rows)),np.float32)
        for a in range(0,len(cols),2000):
            blk=M[cols[a:a+2000],:]; out[a:a+2000]=blk[:,rows]
        return cid[cols],out
ids1,M1=read(L+'GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx',s[s.src=='92742'].sig_id.tolist())
ids2,M2=read(L+'GSE70138/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx',s[s.src=='70138'].sig_id.tolist())
ids=np.concatenate([ids1,ids2]); M=np.vstack([M1,M2]); s=s.drop_duplicates('sig_id').set_index('sig_id').loc[ids]
np.savez_compressed(R+'review_claude_20261121/lincs_sigs.npz',M=M,genes=lm.pr_gene_symbol.values.astype(str))
s[['pert_iname','cell_id','dose','time','src','g']].to_csv(R+'review_claude_20261121/lincs_sigs.tsv',sep='\t')
print('sigs',M.shape,'groups',s.g.nunique(),'HEPG2 groups',s[s.cell_id=='HEPG2'].g.nunique(),'24h',int((s.time==24).sum()),'6h',int((s.time==6).sum()))
print(s.groupby('g').size().describe().to_dict())
