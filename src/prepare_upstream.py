"""Verify the pinned paper sources, or fetch only missing verified artifacts."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from artifact_io import digest

ROOT = Path(__file__).resolve().parents[1]


def verify_tree(path, revision):
    base = ['git','-C',str(path)]
    actual = subprocess.check_output(base+['rev-parse','HEAD'],text=True).strip()
    if actual != revision or subprocess.check_output(base+['status','--porcelain','--','gears']):
        raise ValueError(f'GEARS checkout not pristine at pinned revision: {path}')


def setup(upstream, fetch):
    sources = json.loads((ROOT/'manifests/upstream_sources.json').read_text())
    upstream.mkdir(parents=True, exist_ok=True)
    with (upstream/'.prepare_sources.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gears = sources['gears']
        checkout = upstream/gears['checkout']
        if not checkout.exists():
            if not fetch:
                raise FileNotFoundError(checkout)
            with TemporaryDirectory(prefix='gears-source-',dir=upstream) as d:
                tmp = Path(d)/'source'
                subprocess.run(['git','init','--quiet',str(tmp)],check=True)
                cmd = ['git','-C',str(tmp)]
                subprocess.run(cmd+['fetch','--quiet','--depth=1',gears['url'],gears['revision']],check=True)
                subprocess.run(cmd+['checkout','--quiet','--detach','FETCH_HEAD'],check=True)
                verify_tree(tmp, gears['revision'])
                tmp.rename(checkout)
        verify_tree(checkout, gears['revision'])
        print(checkout.name, gears['revision'])
        for key in ('scgpt_manuscript','cell_gears_loader'):
            src = sources[key]
            path = upstream/src['file']
            if not path.exists():
                if not fetch:
                    raise FileNotFoundError(path)
                with TemporaryDirectory(prefix='source-download-',dir=upstream) as d:
                    tmp = Path(d)/path.name
                    with urlopen(src['url'], timeout=40) as inp, tmp.open('wb') as out:
                        shutil.copyfileobj(inp, out, 8 << 20)
                    if digest(tmp) != src['sha256']:
                        raise ValueError(f'Download hash mismatch: {src["url"]}')
                    tmp.chmod(0o444)
                    tmp.rename(path)
            if digest(path) != src['sha256']:
                raise ValueError(f'Source hash mismatch: {path}')
            print(path.name, src['sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=ROOT/'.upstream')
    p.add_argument('--fetch', action='store_true')
    a = p.parse_args()
    setup(a.upstream,a.fetch)
