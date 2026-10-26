"""Versioned, checksum-verified release downloads with an offline check mode."""
from pathlib import Path
import importlib.resources,json,shutil,tarfile,urllib.request
from usbf.io.common import sha,save,safe_path,verify_files

def fetch(data_dir,manifest=None,offline=False,full_inputs=False):
    root=Path(data_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if offline:
        result=verify_files(root)
        if full_inputs:
            result=dict(minimal=result,full=verify_files(root/'full','FULL_FILES.json'),status='passed')
        return result
    if manifest:
        if str(manifest).startswith(('https://','http://')):
            with urllib.request.urlopen(str(manifest),timeout=60) as f:registry=json.load(f)
        else:registry=json.loads(Path(manifest).read_text())
    else:registry=json.loads(importlib.resources.files('usbf').joinpath('assets.json').read_text())
    records=[]
    for asset in registry['assets']:
        if asset.get('optional') and not full_inputs:continue
        if not asset.get('url'):raise RuntimeError('This release asset has not been publicly published: '+asset['name'])
        dest=safe_path(root,asset['destination']);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha(dest)!=asset['sha256']:
            temp=dest.with_name(dest.name+'.partial')
            with urllib.request.urlopen(asset['url'],timeout=120) as source,temp.open('wb') as target:
                shutil.copyfileobj(source,target,4*1024*1024)
            if temp.stat().st_size!=asset['bytes'] or sha(temp)!=asset['sha256']:
                temp.unlink();raise ValueError('Downloaded checksum mismatch: '+asset['name'])
            temp.replace(dest)
        if asset.get('archive')=='tar.gz':
            with tarfile.open(dest,'r:gz') as archive:
                for member in archive.getmembers():
                    if not (member.isfile() or member.isdir()):raise ValueError('Archive links/special files refused')
                    target=safe_path(root,member.name)
                    if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
        records.append(dict(name=asset['name'],sha256=sha(dest),bytes=dest.stat().st_size,url=asset['url']))
    result=dict(status='passed',downloaded_assets=records,verification=verify_files(root))
    if full_inputs:result['full_input_verification']=verify_files(root/'full','FULL_FILES.json')
    save(root/'DOWNLOAD.json',result);return result
