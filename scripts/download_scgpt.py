"""Official folder IDs resolved from upstream README; save hashes for all three files."""
import datetime
import gdown
from common import DATA,ROOT,sha,write_json

folder=DATA/'checkpoints/scgpt-whole-human';folder.mkdir(parents=True,exist_ok=True)
files={'args.json':'1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1','best_model.pt':'14AebJfGOUF047Eg40hk57HCtrb0fyDTm','vocab.json':'1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC'}
rows=[]
for name,file_id in files.items():
    p=folder/name
    if not p.exists():
        result=gdown.download(id=file_id,output=str(p)+'.part',quiet=False)
        if result is None:raise RuntimeError('Official checkpoint download failed: '+name)
        (folder/(name+'.part')).replace(p)
    rows.append(dict(filename=name,url='https://drive.google.com/uc?id='+file_id,path=str(p),sha256=sha(p),size_bytes=p.stat().st_size,download_verified=True,
          checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),upstream='https://github.com/bowang-lab/scGPT',license_note='scGPT repository MIT; official whole-human linked asset; no separate weight license file in download folder'))
write_json(ROOT/'manifests/scgpt_checkpoint.json',rows)
print(rows,flush=True)
