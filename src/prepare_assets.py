"""Stage verified public author bytes and re-extract the official Norman archive."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import zipfile

from artifact_io import digest, write

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path('/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver')
OUT = ROOT/'assets/public'


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')


def setup(out, source_dir, public_cache):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.stage_public.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt = []
        expected = {r['file']:r for r in json.loads((ROOT/'manifests/public_receipt.json').read_text())}
        with (ROOT/'manifests/public_assets.tsv').open() as f:
            for row in csv.DictReader(f, delimiter='\t'):
                name = row['file']
                dest = out/name
                if not dest.exists():
                    source = source_dir/name if source_dir is not None else public_cache/row['cache_relative']
                    if not source.is_file():
                        raise FileNotFoundError(f'{source}; obtain author bytes from {row["source"]}')
                    verify(source, row['sha256'])
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with TemporaryDirectory(prefix='.author-download-',dir=dest.parent) as d:
                        tmp = Path(d)/dest.name
                        shutil.copyfile(source,tmp)
                        verify(tmp,row['sha256'])
                        tmp.chmod(0o444)
                        tmp.rename(dest)
                verify(dest,row['sha256'])
                receipt.append(dict(row,bytes=dest.stat().st_size))
        with zipfile.ZipFile(out/'norman.zip') as z:
            for name in ('norman/perturb_processed.h5ad','norman/go.csv'):
                dest = out/name
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with TemporaryDirectory(prefix='.author-extract-',dir=dest.parent) as d:
                        tmp = Path(d)/dest.name
                        with z.open(name) as inp,tmp.open('wb') as fout:
                            shutil.copyfileobj(inp,fout,8 << 20)
                        verify(tmp,expected[name]['sha256'])
                        tmp.chmod(0o444)
                        tmp.rename(dest)
                verify(dest,expected[name]['sha256'])
                if dest.stat().st_size != z.getinfo(name).file_size:
                    raise ValueError(f'Unexpected archive member size: {dest}')
                receipt.append(dict(file=name, source='Norman public archive member',
                                    sha256=expected[name]['sha256'],bytes=dest.stat().st_size))
        if receipt != list(expected.values()):
            raise ValueError('Verified files disagree with the committed public receipt')
        path = out/'receipt.json'
        if path.exists():
            if json.loads(path.read_text()) != receipt:
                raise ValueError(f'Existing receipt disagrees with verified files: {path}')
        else:
            write(path,receipt)
        print('VERIFIED_PUBLIC_ASSETS_READY',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--source-dir',type=Path,help='Directory of author downloads, named as in manifests/public_assets.tsv')
    p.add_argument('--public-cache',type=Path,default=LEGACY,help='Optional cache containing only verified author files at manifest paths')
    a = p.parse_args()
    setup(a.output,a.source_dir,a.public_cache)
