"""Load the manuscript release, with only explicit import/backend compatibility."""
import importlib, json, logging, sys, types, zipfile, tarfile
from pathlib import Path
from run_support import ROOT,digest

ZENODO = 'a308a53967a0d279bc9db42533451b03413357e422c855773701028136312bac'
GEARS = 'c9eb8aa1176589717cfa34ee9998eb72c6d380f6957bb538c54257811ba4c45a'
class GeneVocab:
    """Read the author JSON's token indices without rebuilding or reordering them."""
    def __init__(self, indices):
        assert sorted(indices.values())==list(range(len(indices)))
        self.indices=indices
        self.default=None
    @classmethod
    def from_file(cls,path):return cls(json.loads(Path(path).read_text()))
    def __contains__(self,key):return key in self.indices
    def __getitem__(self,key):return self.indices.get(key,self.default)
    def __len__(self):return len(self.indices)
    def append_token(self,key):
        if key in self.indices:raise ValueError(key)
        self.indices[key]=len(self.indices)
    def set_default_index(self,value):self.default=value

def author_sources(out):
    zpath=ROOT/'.upstream/scgpt-paper.zip'
    tar=ROOT/'.upstream/gears002.tar.gz'
    assert digest(zpath)==ZENODO and digest(tar)==GEARS
    runtime=out/'upstream_runtime'
    runtime.mkdir(exist_ok=True)
    z=zipfile.ZipFile(zpath)
    parent='scGPT-release-branch_reproducibility/'
    for name in z.namelist():
        if name.startswith(parent+'scgpt/') and name.endswith('.py'):
            p=runtime/name.removeprefix(parent)
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(z.read(name))
    with tarfile.open(tar) as t:
        for m in t:
            if '/gears/' in m.name and m.isfile() and m.name.endswith('.py'):
                p=runtime/'gears'/Path(m.name).name
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
    # Avoid optional tasks unrelated to this expression-prediction path.
    (runtime/'scgpt/__init__.py').write_text('import logging\nlogger = logging.getLogger("scGPT-paper")\n')
    util=runtime/'scgpt/utils/util.py';s=util.read_text();assert s.count('import scib\n')==1;util.write_text(s.replace('import scib\n',''))
    sys.path.insert(0,str(runtime))
    return z.read(parent+'tutorials/Fig3_AB_PerturbPred.py').decode()
