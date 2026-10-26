"""Use the portable Python3.10 interpreter with one untouched, pinned scientific package environment."""
import argparse,runpy,site,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--env',choices=['scgpt','geneformer','research'],required=True);p.add_argument('script');p.add_argument('arguments',nargs=argparse.REMAINDER);args=p.parse_args()
assert sys.version_info[:2]==(3,10),'Invoke .venvs/vcell-runtime310/bin/python explicitly'
package_dir=ROOT/f'.venvs/vcell-{args.env}/lib/python3.10/site-packages';assert package_dir.is_dir();site.addsitedir(str(package_dir))
script=Path(args.script).resolve();assert script.is_file();sys.path.insert(0,str(script.parent));sys.argv=[str(script)]+args.arguments
runpy.run_path(str(script),run_name='__main__')
