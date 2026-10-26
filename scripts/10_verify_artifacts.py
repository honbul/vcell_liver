"""Verify acquisition hashes and inventory regenerable large local outputs."""
import hashlib, json, urllib.request
from pathlib import Path
from common import ROOT,DATA,RAW,sha,write_json
REV='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5'

manifest=json.loads((ROOT/'manifests/assets.json').read_text())
info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/ctheodoris/Geneformer/revision/'+REV+'?blobs=true'))
expected={r['rfilename']:r.get('lfs',{}).get('sha256') for r in info['siblings']}
checks=[]
for row in manifest+json.loads((ROOT/'manifests/scgpt_checkpoint.json').read_text())+json.loads((ROOT/'manifests/data_sources.json').read_text()):
    path=Path(row['path']);actual=sha(path);assert actual==row['sha256'],str(path)
    check=dict(path=str(path.resolve()),sha256=actual,size_bytes=path.stat().st_size,manifest_match=True)
    if '/Geneformer/resolve/' in row['url']:
        name=row['url'].split('/'+REV+'/')[1]
        if expected.get(name):
            assert expected[name]==actual,(name,expected[name],actual)
            check['official_hf_lfs_sha256_match']=True
    checks.append(check)
archive=Path('/mnt/S1/sdata/pubdata/Virtual_Cell/reactome/zenodo_15126939/reactome-download-directory.zip')
h=hashlib.md5()
with archive.open('rb') as f:
    while b:=f.read(8*1024*1024):h.update(b)
assert h.hexdigest()=='965f8f7ceda222e68e917c230469bcad'
checks.append(dict(path=str(archive),official_zenodo_md5_match=True,md5=h.hexdigest()))
write_json(ROOT/'reports/asset_verification.json',dict(status='passed',checks=checks))
rows=[]
for section,command in [('processed','python scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml'),('subsets','python scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml'),('checkpoints','python scripts/download_assets.py; python scripts/download_scgpt.py'),('features','python scripts/02_baseline_annotation.py / 03_scgpt_annotation.py / 04_geneformer_annotation.py with matching configs'),('outputs','matching model/head script and run config'),('tokenized','python scripts/04_geneformer_annotation.py with matching config')]:
    for path in sorted((DATA/section).rglob('*')):
        if path.is_file():
            parts=path.relative_to(DATA/section).parts
            run_id=path.stem if section=='features' else parts[0]
            config_id=run_id
            if '_foundation_' in config_id:config_id=config_id.split('_foundation_')[0]
            elif config_id.endswith('_pathway'):config_id=config_id[:-len('_pathway')]
            if section in ('features','outputs','tokenized'):
                script=next((script for prefix,script in [('tutorial_pca','02_baseline_annotation.py'),('pca','02_baseline_annotation.py'),('tutorial_scgpt','03_scgpt_annotation.py'),('scgpt','03_scgpt_annotation.py'),('tutorial_geneformer','04_geneformer_annotation.py'),('geneformer','04_geneformer_annotation.py'),('tutorial_kg','05_kg_features_ablation.py'),('kg','05_kg_features_ablation.py')] if config_id.startswith(prefix)),None)
                if script:command=f'python scripts/{script} --config configs/{config_id}.yaml'
            if section=='checkpoints':command='python scripts/download_scgpt.py' if parts[0].startswith('scgpt') else 'python scripts/download_assets.py'
            rows.append(dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path),recreation_command=command))
write_json(ROOT/'manifests/local_artifacts.json',rows)
print('Verified',len(checks),'source assets;',len(rows),'derived local artifacts inventoried',flush=True)
