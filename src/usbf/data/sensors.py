"""Explicit RGB/signed-nDSM/validity input contract for prepared snapshots."""
from pathlib import Path
import numpy as np
import rasterio
from usbf.io.common import sha

def load_sensor(row,verify=False):
    p=Path(row['path'])
    if verify and sha(p)!=row['sha256']:raise ValueError('Sensor checksum mismatch: '+p.name)
    with np.load(p,allow_pickle=False) as z:
        missing=set(('rgb','signed','valid','rgb_valid'))-set(z.files)
        if missing:raise ValueError('Missing sensor arrays: '+str(sorted(missing)))
        rgb=z['rgb'].copy();signed=z['signed'].copy()
        hv=z['valid'].squeeze()>.5;rv=z['rgb_valid'].squeeze().astype(bool)
    signed=np.asarray(signed).squeeze()
    if signed.ndim!=2 or rgb.shape!=(3,*signed.shape) or hv.shape!=signed.shape or rv.shape!=signed.shape:
        raise ValueError('RGB, height and validity shapes disagree')
    if not np.issubdtype(rgb.dtype,np.floating) or not np.isfinite(rgb[:,rv]).all():
        raise ValueError('RGB must be finite normalized floating-point values in the valid domain')
    if rv.any() and (rgb[:,rv].min()<0 or rgb[:,rv].max()>1):raise ValueError('RGB must be in 0..1')
    hv &= np.isfinite(signed);rv &= np.isfinite(rgb).all(0)
    transform=rasterio.Affine(*row['transform'][:6]);crs=rasterio.crs.CRS.from_user_input(row['crs'])
    if transform.a<=0 or transform.e>=0 or transform.b or transform.d:raise ValueError('North-up raster required')
    return dict(rgb=rgb,signed=np.where(hv,signed,np.nan).astype('float32'),height_valid=hv,
                rgb_valid=rv,crs=crs,transform=transform,hashes={str(p):sha(p)} if verify else {})

def prepare(rgb_path,output,dsm_path=None,dtm_path=None):
    """Align physical DSM/DTM to an existing north-up RGB grid, in metres.

    Prepared paper snapshots are the exact historical input. This entry is for
    new user data; it does not claim to rebuild a historical download bytewise.
    """
    from rasterio.warp import reproject,Resampling
    with rasterio.open(rgb_path) as src:
        if src.count<3:raise ValueError('At least three RGB bands required')
        rgb=src.read([1,2,3]).astype('float32')/255
        rv=(src.read_masks([1,2,3])>0).all(0)
        crs=src.crs;t=src.transform;shape=src.shape
    height=np.full(shape,np.nan,'float32');valid=np.zeros(shape,bool)
    if bool(dsm_path)!=bool(dtm_path):raise ValueError('Supply both DSM and DTM, or neither')
    if dsm_path:
        arrays=[]
        for path in (dsm_path,dtm_path):
            with rasterio.open(path) as src:
                if src.units[0] not in (None,'metre','meter','m'):raise ValueError('Elevation must be metres')
                raw=src.read(1).astype('float32');raw[(src.read_masks(1)==0)|~np.isfinite(raw)]=np.nan
                dst=np.full(shape,np.nan,'float32')
                reproject(raw,dst,src_transform=src.transform,src_crs=src.crs,src_nodata=np.nan,
                          dst_transform=t,dst_crs=crs,dst_nodata=np.nan,resampling=Resampling.bilinear)
                arrays.append(dst)
        valid=np.isfinite(arrays[0])&np.isfinite(arrays[1]);height[valid]=arrays[0][valid]-arrays[1][valid]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,rgb=rgb,signed=height,valid=valid,rgb_valid=rv)
    return dict(path=str(output),sha256=sha(output),crs=str(crs),transform=list(t),kind='ma_npz',
                height_units='metres',valid_height_fraction=float(valid.mean()),missing_height=not bool(valid.any()))
