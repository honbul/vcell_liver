"""Download only the two official scGPT MS inputs, not unrelated Drive files."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from common import DATA,sha,write_json
import gdown
p=DATA/'research_phase2/official/scGPT_MS';p.mkdir(parents=True,exist_ok=True)
rows=[]
for name,file_id in [('c_data.h5ad','1bV1SHKVZgkcL-RmmuN51_IIUJTSJbXOi'),('filtered_ms_adata.h5ad','1casFhq4InuBNhJLMnGebzkRXM2UTTeQG')]:
    if not (p/name).exists():
        result=gdown.download(id=file_id,output=str(p/name)+'.part')
        if result is None:raise RuntimeError('Official MS download failed: '+name)
        (p/(name+'.part')).replace(p/name)
    rows.append(dict(filename=name,url='https://drive.google.com/uc?id='+file_id,source='scGPT official annotation tutorial',path=str(p/name),size_bytes=(p/name).stat().st_size,sha256=sha(p/name),download_verified=True))
    print(rows[-1],flush=True)
write_json(ROOT/'manifests/research_phase2/official_scgpt_ms.json',rows)
