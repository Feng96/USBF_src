"""Actual frozen B0/B1 inference; CPU and CUDA paths are explicit."""
from pathlib import Path
import time
import numpy as np
import rasterio
import torch
from .rgb_height_r1 import RGBHeightR1,R1Config
from .rgb_height_shape_v1 import RGBHeightShape
from usbf.io.common import sha,save
from usbf.data.sensors import load_sensor

WEIGHTS={
 'B0_control':('B0_control_best_shape_naip_val.pth','1912c45a663ab95962449960791396eacfc2fa9f5236b0abcce6bb5716c71238'),
 'B1_shape':('B1_shape_best_shape_naip_val.pth','65b86df2d146f4074eea700cd6cd8ff2d3839e9a837c3dc0d947211b6e87bcc9')}

def load_models(data_dir,device='cpu'):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    models=[];records={}
    for arm,(name,expected) in WEIGHTS.items():
        p=Path(data_dir)/'weights'/name
        if not p.is_file():raise FileNotFoundError('Missing frozen '+arm+' weight; run usbf fetch first')
        if sha(p)!=expected:raise ValueError('Frozen '+arm+' weight checksum mismatch; refusing load')
        # Deserialization occurs only after the complete trusted release hash check.
        state=torch.load(p,map_location='cpu',weights_only=False)
        model=RGBHeightShape() if arm=='B1_shape' else RGBHeightR1(R1Config('gated'))
        model.load_state_dict(state['model_state_dict'],strict=True);model.to(device).eval()
        models.append(model);records[arm]=dict(file=name,sha256=expected,parameters=sum(p.numel() for p in model.parameters()))
    return models,records

def run(records,data_dir,output,device='cpu',threads=2,missing_height=False):
    start=time.time();torch.set_num_threads(threads);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models,weights=load_models(data_dir,device);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[];missing_checks=[]
    for row in records:
        p=load_sensor(row,verify=True);x=torch.from_numpy(np.nan_to_num(p['rgb']).clip(0,1))[None].to(device)
        valid=np.zeros_like(p['height_valid']) if missing_height else p['height_valid']
        h=torch.from_numpy(np.where(valid,np.clip(p['signed'],0,30)/30,0).astype('float32'))[None,None].to(device)
        v=torch.from_numpy(valid.astype('float32'))[None,None].to(device)
        with torch.inference_mode():
            predictions=[m(x,h,v) for m in models]
            prob=np.stack([a['logits'].sigmoid()[0,0].cpu().numpy() for a in predictions])
            if not results:
                for arm,m in zip(WEIGHTS,models):
                    z=m(x,torch.zeros_like(h),torch.zeros_like(v))
                    error=float((z['logits']-z['rgb_logits']).abs().max())
                    if error!=0:raise AssertionError('All-missing elevation must reproduce the model RGB path')
                    missing_checks.append(dict(model=arm,max_absolute_logit_difference=error))
        if not np.isfinite(prob).all():raise ValueError('Non-finite model probabilities')
        prob[:,~p['rgb_valid']]=-9999
        path=output/(row['name']+'.tif')
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(path,'w',driver='GTiff',height=prob.shape[1],width=prob.shape[2],count=2,
                    dtype='float32',crs=p['crs'],transform=p['transform'],nodata=-9999,compress='deflate') as dst:
                dst.write(prob);dst.write_mask(p['rgb_valid'].astype('uint8')*255)
                dst.set_band_description(1,'frozen_B0_control');dst.set_band_description(2,'frozen_B1_shape')
        with rasterio.open(path) as src:
            if not np.array_equal(src.read(),prob):raise AssertionError('Probability TIFF readback differs')
        results.append(dict(name=row['name'],probability=str(path),sha256=sha(path)))
        print('actual_model_forward',row['name'],device,flush=True)
    report=dict(status='complete',device=device,threads=threads,frames=len(records),weights=weights,
                all_missing_height_checks=missing_checks,elapsed_seconds=time.time()-start,records=results)
    save(output/'INFERENCE.json',report);return report
