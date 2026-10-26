"""Prepare reserved HCC external counts and ontology only; no models or predictions are loaded."""
import csv,gzip,re
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import anndata as ad
import scipy.io as sio
from threadpoolctl import threadpool_limits
from research_core import *

source=Path('/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE151530');out=ASSETS/'external/GSE151530_HCC';assert not (out/'contract.json').exists(),'Immutable unscored external data already prepared';out.mkdir(parents=True,exist_ok=True)
info=pd.read_csv(source/'GSE151530_Info.txt.gz',sep='\t');barcodes=pd.read_csv(source/'GSE151530_barcodes.tsv.gz',header=None)[0].astype(str)
assert len(info)==56721 and np.array_equal(info.Cell.astype(str),barcodes)
# Histology is joined to explicit GEO sample metadata, then cross-checked with original Table S1.
samples={}
for path in source.glob('*series_matrix.txt.gz'):
    with gzip.open(path,'rt') as f:lines=[next(csv.reader([line],delimiter='\t')) for line in f if line.startswith('!Sample_')]
    titles=next(row[1:] for row in lines if row[0]=='!Sample_title');gsm=next(row[1:] for row in lines if row[0]=='!Sample_geo_accession')
    cancer=next(row[1:] for row in lines if row[0]=='!Sample_characteristics_ch1' and all(value.startswith('cancer type:') for value in row[1:]))
    for sid,accession,diagnosis in zip(titles,gsm,cancer):
        assert sid not in samples
        histology='HCC' if diagnosis.split(':',1)[1].strip()=='Hepatocellular carcinoma' else 'iCCA'
        samples[sid]=dict(S_ID=sid,GSM=accession,histology=histology,series_matrix=path.name)
assert set(samples)==set(info.S_ID)
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};xml=ET.fromstring(ZipFile(source/'NIHMS1720233-supplement-1.docx').read('word/document.xml'));clinical={}
for table in xml.findall('.//w:tbl',ns):
    rows=[[''.join(cell.itertext()) for cell in row.findall('w:tc',ns)] for row in table.findall('w:tr',ns)]
    if not rows or 'Table S1.' not in ' '.join(rows[0]):continue
    for row in rows:
        if len(row)>=6 and re.fullmatch(r'[HC]\d+\*?',row[1]):clinical[row[1].rstrip('*')]=dict(histology=row[5],repeated_biopsies=row[1].endswith('*'))
assert len(clinical)==37
sample_rows=[]
for sid,part in info.groupby('S_ID',sort=True):
    assert part.Sample.nunique()==1
    sample=str(part.Sample.iloc[0]);match=re.fullmatch(r'([HC]\d+)([a-z]?)',sample);assert match,sample
    donor,suffix=match.groups();assert donor in clinical and clinical[donor]['histology']==samples[sid]['histology']
    if suffix:assert clinical[donor]['repeated_biopsies'],f'Suffix unverified in TableS1 for {sample}'
    sample_rows.append(dict(**samples[sid],author_sample=sample,author_patient_id=donor,n_cells=len(part),repeated_biopsies=clinical[donor]['repeated_biopsies']))
sample_manifest=pd.DataFrame(sample_rows);sample_manifest.to_csv(out/'sample_manifest.tsv',sep='\t',index=False)
mapping={'T cells':'T/NK','B cells':'B','TAMs':'Myeloid','TECs':'Endothelial','CAFs':'Fibroblast','Malignant cells':'Unknown','unclassified':'Unknown'};assert set(info.Type)==set(mapping)
metadata=info.merge(sample_manifest[['S_ID','author_patient_id','histology']],on='S_ID',how='left',validate='many_to_one',sort=False)
assert np.array_equal(metadata.Cell,barcodes)
metadata['cell_type_coarse']=metadata.Type.map(mapping);metadata['cell_id']='GSE151530|'+metadata.S_ID.astype(str)+'|'+metadata.Cell.astype(str);assert metadata.cell_id.is_unique
metadata['patient_id']='GSE151530_'+metadata.author_patient_id.astype(str);metadata['sample_id']=metadata.S_ID;metadata['dataset_id']='GSE151530_HCC_external_final'
keep=metadata.histology.to_numpy()=='HCC';hcc=metadata.loc[keep].copy();hcc.index=hcc.cell_id.astype(str);hcc.index.name='index'
genes=pd.read_csv(source/'GSE151530_genes.tsv.gz',sep='\t',header=None,names=['ensembl_id','gene_symbol']);assert len(genes)==18667 and genes.ensembl_id.is_unique
with threadpool_limits(limits=8):
    with gzip.open(source/'GSE151530_matrix.mtx.gz','rb') as f:matrix=sio.mmread(f).tocsr().T.tocsr()
assert matrix.shape==(56721,18667) and np.isfinite(matrix.data).all() and np.all(matrix.data>=0)
counts=matrix[keep].astype(np.float32);libraries=np.asarray(counts.sum(1)).ravel();assert np.all(libraries>0)
hcc['n_counts']=libraries;var=genes.set_index('ensembl_id',drop=False);a=ad.AnnData(counts.copy(),obs=hcc,var=var);a.layers['counts']=counts;a.write_h5ad(out/'counts_unscored.h5ad',compression='gzip')
labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];support=a.obs.cell_type_coarse.value_counts().to_dict();sources={p.name:dict(size_bytes=p.stat().st_size,sha256=sha(p)) for p in source.iterdir() if p.name.endswith(('.gz','.docx'))}
contract=dict(status='prepared_unscored',dataset_id='GSE151530_HCC_external_final',project_commit=git_commit(),n_cells=a.n_obs,n_genes=a.n_vars,n_donors=a.obs.patient_id.nunique(),n_samples=a.obs.S_ID.nunique(),histology_policy='GEO cancer-type characteristics joined by S_ID, cross-checked with original clinical TableS1; HCC only',
    donor_policy='Author patient ID from TableS1; sample suffix biopsies grouped using TableS1 repeated-biopsy footnote, never treated as independent donors',label_mapping=mapping,label_support=support,
    primary_metric='Fixed-six macroF1 on cells with known lineage ground truth, absent Hepatocyte class contributes0; report all supports and label coverage',
    secondary_metric='Fixed-five macroF1 for B/Endothelial/Fibroblast/Myeloid/T-NK; specified before any external predictions; does not validate Hepatocyte transfer',
    ontology_limitation='No author hepatocyte lineage label. Malignant and unclassified cells remain Unknown rather than inferred hepatocytes. Fixed-six maximum is5/6 for an otherwise perfect classifier in this cohort',
    fallback_review='GSE189903 Info metadata inspected only; same five named nonmalignant lineages with malignant/unclassified labels, no hepatocyte annotation. Retain originally reserved GSE151530; no performance-based selection',
    permitted_access='metadata, ontology, histology, raw-count schema and deterministic count conversion only; no model inference, training or scoring',model_predictions_created=False,model_scores_observed=False,
    counts_path=str(out/'counts_unscored.h5ad'),counts_sha256=sha(out/'counts_unscored.h5ad'),sample_manifest_sha256=sha(out/'sample_manifest.tsv'),source_files=sources,
    source_links=['https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151530','https://pmc.ncbi.nlm.nih.gov/articles/PMC8604764/','https://scatlaslc.ccr.cancer.gov/'],
    overlap_status='No shared GEO samples with GSE149614; biological identity cannot be proven from public study IDs; foundation pretraining overlap unknown')
write_json(out/'contract.json',contract)
for name in ['contract.json','sample_manifest.tsv']:
    dest=REPORT/'external_metadata'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((out/name).read_bytes())
write_json(REPORT/'external_metadata/lineage_mapping.json',mapping)
print('EXTERNAL_PREPARED_UNSCORED',a.shape,'donors',contract['n_donors'],'samples',contract['n_samples'],support,flush=True)
