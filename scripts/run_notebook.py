"""Execute a cleared notebook in a new kernel and retain its actual outputs."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

p=argparse.ArgumentParser();p.add_argument('notebook',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
nb=nbformat.read(a.notebook,as_version=4)
for c in nb.cells:
    if c.cell_type=='code':c.outputs=[];c.execution_count=None
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(a.notebook.resolve().parent)}}).execute()
a.output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,a.output)
print(a.output)
