"""Actual ground-image registration; never receives scored footprint shapes."""
from pathlib import Path
import numpy as np
import rasterio
from .registration_sift import grey,estimate as sift
from .registration_ncc import estimate as ncc
from usbf.io.common import read,save,sha

def run(data_dir,output):
    data_dir=Path(data_dir);meta=read(data_dir/'registration/context.json');path=data_dir/'registration/context.npz'
    if sha(path)!=meta['sha256']:raise ValueError('Registration context checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        a=grey(z['spacenet'],z['spacenet_valid']);b=grey(z['naip'],z['naip_valid']);ground=z['ground']
    tr=rasterio.Affine(*meta['transform'][:6]);first,points=sift(a,b,ground,tr)
    supplemental=None
    if first['status']!='accepted':supplemental=ncc(a,b,ground,tr)
    selected=first if first['status']=='accepted' else supplemental
    result=dict(name=meta['name'],sift=first,sift_points=points,ncc=supplemental,selected=selected,
        movement_applies_only_to='SpaceNet reference polygons',scored_footprint_shapes_loaded=False,
        context_sha256=meta['sha256'])
    save(output,result);return result
