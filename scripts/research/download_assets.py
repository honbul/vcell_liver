"""Official phase-2 assets, each independently recorded; legacy assets are untouched."""
import concurrent.futures,datetime,hashlib,json,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from common import DATA,sha,write_json
REV='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5'

def fetch(job):
    url,path,revision,note=job;path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    row=dict(url=url,path=str(path),revision=revision,usage_note=note,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print('START',url,flush=True)
    try:
        if not path.exists():
            with urllib.request.urlopen(url,timeout=90) as response,open(str(path)+'.part','wb') as f:
                while b:=response.read(8*1024*1024):f.write(b)
            Path(str(path)+'.part').replace(path)
        row.update(status='downloaded',download_verified=True,size_bytes=path.stat().st_size,sha256=sha(path))
    except Exception as exc:row.update(status='failed',download_verified=False,error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(row),flush=True);return row

jobs=[]
for f in ['config.json','generation_config.json','model.safetensors']:
    jobs.append((f'https://huggingface.co/ctheodoris/Geneformer/resolve/{REV}/Geneformer-V2-104M_CLcancer/{f}',DATA/'checkpoints/Geneformer-V2-104M_CLcancer'/f,REV,'Official Apache-2.0 cancer continual checkpoint; same pinned V2 tokenizer dictionaries'))
jobs += [
 ('https://zenodo.org/records/8239570/files/models.zip?download=1',DATA/'checkpoints/LiverCT/models.zip','Zenodo 8239570','CC-BY-4.0; official LiverCT reference and native classifiers'),
 ('https://zenodo.org/records/13380493/files/Lu2022_subsampled.h5ad?download=1',DATA/'research_phase2/official/LiverCT/Lu2022_subsampled.h5ad','Zenodo 13380493','CC-BY-4.0; official demo overlaps GSE149614, not external final'),
 ('https://dataverse.harvard.edu/api/access/datafile/6154020',DATA/'research_phase2/official/GEARS/norman.zip','Harvard Dataverse file6154020','GEARS official Norman CRISPRa benchmark, not liver/HCC'),
 ('https://dataverse.harvard.edu/api/access/datafile/6153417',DATA/'research_phase2/official/GEARS/gene2go_all.pkl','Harvard Dataverse file6153417','GEARS official gene/GO graph prior')]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,jobs))
write_json(ROOT/'manifests/research_phase2/downloads.json',rows)
for path,dest in [(DATA/'checkpoints/LiverCT/models.zip',DATA/'checkpoints/LiverCT/reference'),(DATA/'research_phase2/official/GEARS/norman.zip',DATA/'research_phase2/official/GEARS')]:
    if path.exists():
        try:
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():assert not Path(n).is_absolute() and '..' not in Path(n).parts
                z.extractall(dest)
            print('EXTRACTED',path,dest,flush=True)
        except Exception as exc:print('EXTRACTION_FAILED',path,str(exc),flush=True)
