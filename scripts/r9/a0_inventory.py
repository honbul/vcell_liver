"""A0: label/donor inventory, cohort overlap and deterministic external-count conversion (no model, no scoring)."""
import csv
import gzip
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio

from r9common import *

t = Timer()
REPORT.mkdir(parents=True, exist_ok=True)
MANIFEST.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# ---------------- GSE149614 ----------------
assert sha(PROCESSED149614) == PROCESSED149614_SHA, 'processed GSE149614 hash mismatch'
a = ad.read_h5ad(PROCESSED149614, backed='r')
obs = a.obs.copy()
raw_meta = pd.read_csv(GEO149614 / 'GSE149614_HCC.metadata.updated.txt.gz', sep='\t')
assert len(raw_meta) == len(obs) == 71915
raw_meta = raw_meta.set_index('Cell').loc[obs.index]
assert (raw_meta.celltype.astype(str).values == obs.cell_type_original.astype(str).values).all()
assert (raw_meta.patient.astype(str).values == obs.patient_id.astype(str).values).all()
assert (raw_meta['res.3'].astype(str).values == obs['res.3'].astype(str).values).all()
meta_cols = list(raw_meta.reset_index().columns)
obs['broad'] = obs.cell_type_original.astype(str)
obs['res3'] = 'c' + obs['res.3'].astype(str).str.zfill(2)
obs['patient'] = obs.patient_id.astype(str)
obs['sample'] = obs.sample_id.astype(str)
obs['site'] = obs.tissue_site.astype(str)
cell_obs = obs[['patient', 'sample', 'site', 'stage', 'virus', 'broad', 'res3']].copy()
cell_obs.index.name = 'cell_id'
cell_obs.to_csv(CACHE / 'gse149614_obs.tsv.gz', sep='\t')
donors = sorted(obs.patient.unique())
N = len(obs)

inv = []
# broad tier
for lab, g in obs.groupby('broad'):
    dc = g.patient.value_counts()
    inv.append(dict(cohort='GSE149614', tier='broad_author', label=lab, parent=PARENT[lab],
                    source='author celltype column (metadata.updated)', n_cells=len(g), frac=len(g) / N,
                    n_donors_ge1=int((dc >= 1).sum()), n_donors_ge10=int((dc >= 10).sum()),
                    n_donors_ge50=int((dc >= 50).sum()), max_donor_share=dc.max() / len(g),
                    top_donor=dc.idxmax(), sites=','.join(sorted(g.site.unique())),
                    dominant_broad=lab, broad_purity=1.0,
                    cross_cohort_map={'B': 'B cells', 'T/NK': 'T cells', 'Myeloid': 'TAMs', 'Endothelial': 'TECs',
                                      'Fibroblast': 'CAFs', 'Hepatocyte': 'none (Malignant cells: unvalidated)'}[lab]))
# res.3 tier (author-released Seurat clusters; computational, computed on all cells jointly)
for lab, g in obs.groupby('res3'):
    dc = g.patient.value_counts()
    bc = g.broad.value_counts()
    inv.append(dict(cohort='GSE149614', tier='res3_cluster', label=lab, parent=bc.idxmax(),
                    source='author res.3 cluster id (unsupervised, all cells jointly; not an expert sub-state name)',
                    n_cells=len(g), frac=len(g) / N, n_donors_ge1=int((dc >= 1).sum()),
                    n_donors_ge10=int((dc >= 10).sum()), n_donors_ge50=int((dc >= 50).sum()),
                    max_donor_share=dc.max() / len(g), top_donor=dc.idxmax(),
                    sites=','.join(sorted(g.site.unique())), dominant_broad=bc.idxmax(),
                    broad_purity=bc.max() / len(g), cross_cohort_map='none (cluster ids are cohort-specific)'))

counts = []
for keys, g in obs.groupby(['patient', 'sample', 'site', 'stage', 'virus']):
    for tier, col in [('broad_author', 'broad'), ('res3_cluster', 'res3')]:
        for lab, n in g[col].value_counts().items():
            counts.append(dict(cohort='GSE149614', patient=keys[0], sample=keys[1], site=keys[2], stage=keys[3],
                               virus=keys[4], treatment='not reported (resection atlas)', histology='HCC',
                               tier=tier, label=lab, n_cells=int(n)))

# Rare definition per outer LOPO fold, from the training donors only (0.5/1/2 %).
rare_rows = []
for test in donors:
    tr = obs[obs.patient != test]
    for tier, col in [('broad_author', 'broad'), ('res3_cluster', 'res3')]:
        vc = tr[col].value_counts()
        for lab, n in vc.items():
            dsub = tr[tr[col] == lab].patient.value_counts()
            te_n = int((obs[(obs.patient == test)][col] == lab).sum())
            rare_rows.append(dict(outer_test_donor=test, tier=tier, label=lab, train_cells=int(n),
                                  train_frac=n / len(tr), rare_0p5=n / len(tr) <= 0.005, rare_1=n / len(tr) <= 0.01,
                                  rare_2=n / len(tr) <= 0.02, train_donors_ge10=int((dsub >= 10).sum()),
                                  train_max_donor_share=dsub.max() / n, test_donor_cells=te_n))
rare = pd.DataFrame(rare_rows)
write_tsv(rare, REPORT / 'rare_definition_by_fold.tsv')

# ---------------- GSE151530 ----------------
info = pd.read_csv(GEO151530 / 'GSE151530_Info.txt.gz', sep='\t')
info.columns = [c.strip() for c in info.columns]
for c in info.columns:
    if info[c].dtype == object:
        info[c] = info[c].astype(str).str.strip()
barcodes = pd.read_csv(GEO151530 / 'GSE151530_barcodes.tsv.gz', header=None)[0].astype(str)
assert len(info) == len(barcodes) and np.array_equal(info.Cell.values, barcodes.values)
samples = {}
for path in sorted(GEO151530.glob('*series_matrix.txt.gz')):
    with gzip.open(path, 'rt') as f:
        lines = [next(csv.reader([line], delimiter='\t')) for line in f if line.startswith('!Sample_')]
    titles = next(r[1:] for r in lines if r[0] == '!Sample_title')
    gsm = next(r[1:] for r in lines if r[0] == '!Sample_geo_accession')
    chars = [r[1:] for r in lines if r[0] == '!Sample_characteristics_ch1']
    for i, sid in enumerate(titles):
        d = dict(S_ID=sid, GSM=gsm[i], series_matrix=path.name)
        for row in chars:
            k, v = row[i].split(':', 1)
            d[k.strip()] = v.strip()
        samples[sid] = d
smeta = pd.DataFrame(samples.values())
assert set(smeta.S_ID) == set(info.S_ID)
smeta['histology'] = smeta['cancer type'].map({'Hepatocellular carcinoma': 'HCC', 'Intrahepatic cholangiocarcinoma': 'iCCA'})
assert smeta.histology.notna().all()
# Table S1 of the public supplement: repeated-biopsy footnote (*) verifies sample-suffix grouping.
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
xml = ET.fromstring(ZipFile(GEO151530 / 'NIHMS1720233-supplement-1.docx').read('word/document.xml'))
clinical = {}
for table in xml.findall('.//w:tbl', ns):
    rows = [[''.join(c.itertext()) for c in r.findall('w:tc', ns)] for r in table.findall('w:tr', ns)]
    if not rows or 'Table S1.' not in ' '.join(rows[0]):
        continue
    for r in rows:
        if len(r) >= 6 and re.fullmatch(r'[HC]\d+\*?', r[1]):
            clinical[r[1].rstrip('*')] = dict(histology_s1=r[5], repeated=r[1].endswith('*'))
srows = []
for sid, part in info.groupby('S_ID', sort=True):
    assert part.Sample.nunique() == 1
    s = part.Sample.iloc[0]
    m = re.fullmatch(r'([HC]\d+)([a-z]?)', s)
    assert m, s
    donor, suf = m.groups()
    hist = smeta.set_index('S_ID').loc[sid, 'histology']
    assert donor in clinical and clinical[donor]['histology_s1'] == hist, (sid, donor)
    if suf:
        assert clinical[donor]['repeated'], s
    srows.append(dict(S_ID=sid, author_sample=s, patient='GSE151530_' + donor, histology=hist,
                      repeated_biopsy=clinical[donor]['repeated'], n_cells=len(part)))
sman = pd.DataFrame(srows).merge(smeta, on=['S_ID', 'histology'], how='left')
write_tsv(sman, MANIFEST / 'gse151530_sample_manifest.tsv')
meta = info.merge(sman[['S_ID', 'patient', 'histology', 'author_sample']], on='S_ID', how='left', validate='many_to_one', sort=False)
assert np.array_equal(meta.Cell.values, barcodes.values)
assert set(meta.Type) <= set(EXT_MAP), set(meta.Type)
meta['mapped'] = meta.Type.map(EXT_MAP)
meta['cell_id'] = 'GSE151530|' + meta.S_ID + '|' + meta.Cell
hcc = meta.histology.values == 'HCC'
ext_N = int(hcc.sum())
for lab, g in meta[hcc].groupby('Type'):
    dc = g.patient.value_counts()
    inv.append(dict(cohort='GSE151530_HCC', tier='broad_author', label=lab, parent=PARENT.get(EXT_MAP[lab], 'unmapped'),
                    source='author Type column (Info.txt)', n_cells=len(g), frac=len(g) / ext_N,
                    n_donors_ge1=int((dc >= 1).sum()), n_donors_ge10=int((dc >= 10).sum()),
                    n_donors_ge50=int((dc >= 50).sum()), max_donor_share=dc.max() / len(g), top_donor=dc.idxmax(),
                    sites='tumor biopsy', dominant_broad=EXT_MAP[lab], broad_purity=np.nan,
                    cross_cohort_map=EXT_MAP[lab]))
for keys, g in meta.groupby(['patient', 'S_ID', 'author_sample', 'histology']):
    sm = smeta.set_index('S_ID').loc[keys[1]]
    for lab, n in g.Type.value_counts().items():
        counts.append(dict(cohort='GSE151530', patient=keys[0], sample=f'{keys[1]}({keys[2]})', site='tumor biopsy',
                           stage='not reported', virus='not reported',
                           treatment='; '.join(f'{k}={sm[k]}' for k in smeta.columns
                                               if k not in ('S_ID', 'GSM', 'series_matrix', 'histology', 'cancer type') and pd.notna(sm[k])) or 'not in series matrix',
                           histology=keys[3], tier='broad_author', label=lab, n_cells=int(n)))
write_tsv(pd.DataFrame(inv), REPORT / 'label_inventory.tsv')
write_tsv(pd.DataFrame(counts), REPORT / 'donor_class_counts.tsv')

# Deterministic HCC-only count conversion (cache, git-ignored). No model touches it in A0.
genes = pd.read_csv(GEO151530 / 'GSE151530_genes.tsv.gz', sep='\t', header=None, names=['ensembl_id', 'gene_symbol'])
ext_path = CACHE / 'gse151530_hcc_counts.h5ad'
if not ext_path.exists():
    with gzip.open(GEO151530 / 'GSE151530_matrix.mtx.gz', 'rb') as f:
        mtx = sio.mmread(f).tocsr().T.tocsr()
    assert mtx.shape == (len(info), len(genes))
    sub = mtx[hcc].astype(np.float32)
    o = meta[hcc].set_index('cell_id')
    o.index.name = None
    e = ad.AnnData(sub, obs=o[['S_ID', 'author_sample', 'patient', 'histology', 'Type', 'mapped']].astype(str),
                   var=genes.set_index('ensembl_id', drop=False))
    e.write_h5ad(ext_path, compression='gzip')
sym149 = set(a.var.gene_symbol.astype(str))
ens149 = set(a.var.ensembl_id.dropna().astype(str)) - {'', 'nan'}
gene_overlap = dict(gse149614_genes=a.n_vars, gse151530_genes=len(genes),
                    symbol_overlap=len(sym149 & set(genes.gene_symbol.astype(str))),
                    ensembl_overlap=len(ens149 & set(genes.ensembl_id.astype(str))))

# ---------------- cohort overlap & exposure ----------------
ov = []
g1_exposure = {d: 'G1 (origin/main 243fa66) train/val donor; phase2-v2 development donor' for d in donors}
for d in G1_TEST_DONORS:
    g1_exposure[d] = ('G1 fixed TEST donor: scored in 64/2000/10000-cell PCA/scGPT/Geneformer/KG comparisons '
                      '(origin/main 243fa66); also held-out development donor in research/phase2-v2 -> NOT sealed')
for d in donors:
    g = obs[obs.patient == d]
    ov.append(dict(accession='GSE149614', patient=d, n_samples=g['sample'].nunique(), samples=','.join(sorted(g['sample'].unique())),
                   n_cells=len(g), histology='HCC', sites=','.join(sorted(g.site.unique())),
                   shared_geo_samples_with_other_cohort='none', r9_role='development (outer LOPO); all-donor training for A4',
                   prior_exposure=g1_exposure[d]))
for p, g in meta.groupby('patient'):
    h = g.histology.iloc[0]
    ov.append(dict(accession='GSE151530', patient=p, n_samples=g.S_ID.nunique(), samples=','.join(sorted(g.author_sample.unique())),
                   n_cells=len(g), histology=h, sites='tumor biopsy', shared_geo_samples_with_other_cohort='none',
                   r9_role='A4 separate-cohort evaluation (HCC only)' if h == 'HCC' else 'excluded (iCCA)',
                   prior_exposure=('research/phase2-v2 E00 external evaluation (2026-10-26): broad-lineage predictions of '
                                   '8 frozen pipelines scored on this HCC subset -> not a blind cohort for broad-lineage '
                                   'classification in this project' if h == 'HCC' else 'not scored (iCCA excluded)')))
write_tsv(pd.DataFrame(ov), REPORT / 'cohort_overlap.tsv')

write_json(MANIFEST / 'a0_data_hashes.json', dict(
    commit=git_commit(), processed_gse149614=dict(path=str(PROCESSED149614), sha256=PROCESSED149614_SHA),
    raw_files={str(p): dict(size=p.stat().st_size, sha256=sha(p)) for p in sorted(list(GEO149614.iterdir()) + list(GEO151530.iterdir())) if p.is_file()},
    gse149614_metadata_columns=meta_cols, gse151530_info_columns=list(info.columns),
    gse151530_series_characteristic_keys=[c for c in smeta.columns if c not in ('S_ID', 'GSM', 'series_matrix')],
    gse151530_type_values=sorted(meta.Type.unique()), ext_label_map=EXT_MAP, parent_map=PARENT,
    gse151530_hcc_cells=ext_N, gse151530_hcc_patients=int(meta[hcc].patient.nunique()),
    gse151530_hcc_samples=int(meta[hcc].S_ID.nunique()),
    ext_cache=dict(path=str(ext_path), sha256=sha(ext_path), size=ext_path.stat().st_size),
    gse149614_obs_cache=dict(path=str(CACHE / 'gse149614_obs.tsv.gz'), sha256=sha(CACHE / 'gse149614_obs.tsv.gz')),
    gene_overlap=gene_overlap, resources=t.finish()))
print('A0 done', gene_overlap, t.finish())
