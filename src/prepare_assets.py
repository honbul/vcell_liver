"""Stage only verified author bytes; re-extract Norman from the public archive."""
import csv, hashlib, json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path('/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver')
OUT = ROOT / 'assets/public'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    with (ROOT / 'manifests/public_assets.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            source = LEGACY / row['cache_relative']
            assert sha(source) == row['sha256'], source
            dest = OUT / row['file']
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                # Copy the verified bytes and set read-only permissions in the new namespace.
                tmp = dest.with_suffix(dest.suffix + '.partial')
                shutil.copyfile(source, tmp)
                assert sha(tmp) == row['sha256']
                tmp.chmod(0o444)
                tmp.replace(dest)
            assert sha(dest) == row['sha256'], dest
            receipts.append(dict(row, bytes=dest.stat().st_size))
    with zipfile.ZipFile(OUT / 'norman.zip') as z:
        for name in ['norman/perturb_processed.h5ad', 'norman/go.csv']:
            dest = OUT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                tmp = dest.with_suffix(dest.suffix + '.partial')
                with z.open(name) as src, tmp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)
                tmp.chmod(0o444)
                tmp.replace(dest)
            assert dest.stat().st_size == z.getinfo(name).file_size
            receipts.append(dict(file=name, source='Norman public archive member', sha256=sha(dest), bytes=dest.stat().st_size))
    (OUT / 'receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print('VERIFIED_PUBLIC_ASSETS_READY', flush=True)
