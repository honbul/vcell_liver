"""Execute each tutorial in a fresh kernel; save real outputs and audit counts."""
import datetime, json, os, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from common import ROOT,write_json

os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter/share/jupyter')+(':'+os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
status=[]
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    start=time.monotonic();nb=nbformat.read(path,as_version=4)
    client=NotebookClient(nb,timeout=1800,kernel_name='vcell-scgpt',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
    try:
        client.execute()
        nbformat.write(nb,path)
        codes=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=='error' for c in codes for o in c.outputs)
        status.append(dict(notebook=str(path.relative_to(ROOT)),status='executed',code_cells=len(codes),execution_counts=[c.execution_count for c in codes],wall_seconds=time.monotonic()-start,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(status[-1]),flush=True)
    except Exception as exc:
        nbformat.write(nb,path)
        status.append(dict(notebook=str(path.relative_to(ROOT)),status='failed',error=str(exc),wall_seconds=time.monotonic()-start))
        write_json(ROOT/'reports/notebook_execution.json',status)
        raise
    write_json(ROOT/'reports/notebook_execution.json',status)
