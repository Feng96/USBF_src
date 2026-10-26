"""Small explicit-path I/O. Importing this module performs no work."""
from pathlib import Path
import hashlib,json,os

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def native(x):
    import numpy as np
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    if isinstance(x,bytes):return x.hex()
    raise TypeError(type(x).__name__)

def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(value,indent=2,allow_nan=False,default=native)+'\n');q.replace(p)

def safe_path(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if root not in p.parents:raise ValueError('Asset path escapes the data directory')
    return p

def verify_files(root,index_name='FILES.json'):
    root=Path(root);index=read(root/index_name);errors=[]
    for relative,spec in index.items():
        p=safe_path(root,relative)
        if not p.is_file():errors.append(relative+': missing')
        elif p.stat().st_size!=spec['bytes'] or sha(p)!=spec['sha256']:errors.append(relative+': checksum mismatch')
    if errors:raise ValueError('Asset verification failed:\n'+'\n'.join(errors))
    return dict(files=len(index),status='passed')
