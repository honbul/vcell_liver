"""Execute specified phase2 tutorials in fresh kernels and retain exact notebook hashes."""
import argparse,time
import nbformat
from nbclient import NotebookClient
from research_core import *

p=argparse.ArgumentParser();p.add_argument('notebooks',nargs='+');args=p.parse_args();os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter/share/jupyter')+(':'+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
path=REPORT/'tutorial_execution.json';records=json.loads(path.read_text()) if path.exists() else []
for name in args.notebooks:
    notebook=ROOT/name;assert notebook.parent==ROOT/'notebooks' and int(notebook.name[:2])>=6
    start=time.monotonic();nb=nbformat.read(notebook,as_version=4);NotebookClient(nb,timeout=300,kernel_name='vcell-scgpt',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute();nbformat.write(nb,notebook)
    cells=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in cells) and not any(o.output_type=='error' for c in cells for o in c.outputs)
    records=[row for row in records if row['notebook']!=name];records.append(dict(notebook=name,status='executed',code_cells=len(cells),wall_seconds=time.monotonic()-start,sha256=sha(notebook),default_mode='Read actual executed artifacts; training and query submission disabled'))
    write_json(path,sorted(records,key=lambda row:row['notebook']));print('TUTORIAL_EXECUTED',name,flush=True)
