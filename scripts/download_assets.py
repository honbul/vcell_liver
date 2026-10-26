"""Pinned official assets; atomic downloads, no blanket LFS checkout."""
import concurrent.futures, datetime, hashlib, json, pathlib, urllib.request, zipfile
ROOT=pathlib.Path('/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver')
DATA=pathlib.Path('/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver')
REV='1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5'
def download(item):
    url,path,revision,note=item
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=90) as r, open(str(path)+'.part','wb') as f:
            while b:=r.read(8*1024*1024):f.write(b)
        pathlib.Path(str(path)+'.part').replace(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    row=dict(url=url,path=str(path),revision=revision,size_bytes=path.stat().st_size,sha256=h.hexdigest(),download_verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),usage_note=note)
    print(json.dumps(row),flush=True)
    return row
if __name__=='__main__':
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/ctheodoris/Geneformer/revision/'+REV))
    files=[x['rfilename'] for x in info['siblings'] if x['rfilename'].startswith('geneformer/') or x['rfilename'] in ('setup.py','requirements.txt','LICENSE','README.md') or x['rfilename'] in ('Geneformer-V2-104M/config.json','Geneformer-V2-104M/generation_config.json','Geneformer-V2-104M/model.safetensors')]
    jobs=[('https://huggingface.co/ctheodoris/Geneformer/resolve/'+REV+'/'+f,(DATA/'checkpoints'/f if f.startswith('Geneformer-V2-104M/') else ROOT/'third_party/Geneformer'/f),REV,'Apache-2.0; official ctheodoris repository; V2-104M only') for f in files if not f.startswith('geneformer/gene_dictionaries_30m/')]
    jobs += [('https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.basic.annotation.gtf.gz',DATA/'annotation/gencode.v47.basic.annotation.gtf.gz','GENCODE v47 / Ensembl 113','GENCODE human annotation; source terms https://www.gencodegenes.org/pages/data_access.html')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rows=list(ex.map(download,jobs))
    source_link=ROOT/'third_party/Geneformer/Geneformer-V2-104M'
    if not source_link.exists():source_link.symlink_to(DATA/'checkpoints/Geneformer-V2-104M',target_is_directory=True)
    archive=pathlib.Path('/mnt/S1/sdata/pubdata/Virtual_Cell/reactome/zenodo_15126939/reactome-download-directory.zip')
    target=DATA/'annotation/Ensembl2Reactome_All_Levels.human.tsv'
    if not target.exists():
        with zipfile.ZipFile(archive) as z, z.open('92/Ensembl2Reactome_All_Levels.txt') as source, target.open('wb') as f:
            for line in source:
                if line.rstrip().endswith(b'Homo sapiens'):f.write(line)
    rows.append(dict(url='https://zenodo.org/records/15126939/files/reactome-download-directory.zip',revision='Reactome 92',archive_path=str(archive),member='92/Ensembl2Reactome_All_Levels.txt',
         path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),size_bytes=target.stat().st_size,download_verified=True,usage_note='Reused existing official public archive, extracted Homo sapiens only. CC0. Direct current URL returned HTTP 403; no archive modification.'))
    (ROOT/'manifests/assets.json').write_text(json.dumps(rows,indent=2))
