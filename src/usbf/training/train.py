"""Explicit local training entry; retraining and fixed-weight replay are separate."""
from pathlib import Path
import csv,json,math,time
import numpy as np
import torch
from .optimizer import optimizer
from .objectives import shape_loss
from .validity import masked_r1_loss
from usbf.io.common import read,save,sha,safe_path
from usbf.models.rgb_height_r1 import RGBHeightR1,R1Config
from usbf.models.rgb_height_shape_v1 import RGBHeightShape

PARENT_SHA='cf96686c920a4bbe27b3ec00161f6d855f73a5a94f1da107fa2353040a0f48f8'

def loss(arm,output,batch):
    return shape_loss(output,batch['target'],batch['supervision_valid']) if arm=='B1_shape' else masked_r1_loss(output,batch['target'],batch['supervision_valid'],'gated')

def initialize(arm,parent,device):
    if sha(parent)!=PARENT_SHA:raise ValueError('The manuscript paired training requires its exact epoch65 parent; selected inference weights are not that parent')
    state=torch.load(parent,map_location='cpu',weights_only=False)['model_state_dict'];torch.manual_seed(20260909)
    model=RGBHeightShape() if arm=='B1_shape' else RGBHeightR1(R1Config('gated'))
    if arm=='B1_shape':model.load_b0(state)
    else:model.load_state_dict(state,strict=True)
    return model.to(device)

def load_record(row,root):
    path=safe_path(root,row['path'])
    if row.get('sha256') and sha(path)!=row['sha256']:raise ValueError('Training patch checksum mismatch: '+row['name'])
    with np.load(path,allow_pickle=False) as z:
        required=('rgb','height','valid','target','supervision_valid')
        if not all(k in z for k in required):raise ValueError('Training NPZ requires '+str(required))
        values={k:torch.from_numpy(np.asarray(z[k]).copy()) for k in required}
    for k in required:
        if k!='rgb' and values[k].ndim==2:values[k]=values[k][None]
        values[k]=values[k].bool() if k=='supervision_valid' else values[k].float()
    if not torch.isfinite(values['rgb']).all() or not torch.isfinite(values['height']).all():raise ValueError('Training tensors must be finite with separate validity')
    return values

def train(manifest,data_dir,parent,output,arm='B1_shape',device='cpu',epochs=60,max_updates=100,batch_size=8):
    """Bounded optimizer/data-flow entry with the published losses and LR groups.

    The complete archived paired run also used domain-specific model selection.
    This configurable loop does not claim to recreate checkpoint selection or
    the full paired H100 run merely by completing a finite update budget.
    """
    output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh training output directory')
    output.mkdir(parents=True)
    with Path(manifest).open() as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty training manifest')
    domains={d:sorted([r for r in rows if r['split']=='train' and r['domain']==d],key=lambda r:r['name']) for d in ('naip','ma')}
    if not sum(map(len,domains.values())):raise ValueError('No training records')
    model=initialize(arm,parent,device);opt=optimizer(model);updates=0;history=[];start=time.time()
    for epoch in range(1,epochs+1):
        schedule=[]
        for domain,rr in domains.items():
            order=np.random.default_rng(20260909+epoch+(10000 if domain=='ma' else 0)).permutation(len(rr))
            schedule.extend((domain,order[i:i+batch_size]) for i in range(0,len(order),batch_size))
        np.random.default_rng(20260909+epoch).shuffle(schedule)
        factor=.1+.9*(1+math.cos(math.pi*(epoch-1)/59))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
        model.train()
        for domain,indices in schedule:
            samples=[]
            for index in indices:
                b=load_record(domains[domain][int(index)],data_dir)
                rng=np.random.default_rng(np.random.SeedSequence([20260909,epoch,int(index),1 if domain=='ma' else 0]))
                turns=int(rng.integers(4));b={k:torch.rot90(v,turns,(-2,-1)) for k,v in b.items()}
                for axis in (-2,-1):
                    if rng.random()<.5:b={k:v.flip(axis) for k,v in b.items()}
                if rng.random()<.4:b['rgb']=(b['rgb']*float(rng.uniform(.9,1.1))+float(rng.uniform(-.04,.04))).clamp(0,1)
                samples.append(b)
            batch={k:torch.stack([b[k] for b in samples]).to(device) for k in samples[0]}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                value,parts=loss(arm,model(batch['rgb'],batch['height'],batch['valid']),batch)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();updates+=1
            history.append(dict(epoch=epoch,updates=updates,loss=float(value.detach())))
            if updates>=max_updates:break
        if updates>=max_updates:break
    checkpoint=output/'last.pth';torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),updates=updates,arm=arm),checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) for k,v in model.state_dict().items()):raise AssertionError('Checkpoint readback mismatch')
    result=dict(status='complete',updates=updates,seconds=time.time()-start,history=history,checkpoint_sha256=sha(checkpoint),
        full_published_training_reproduced=False,checkpoint_selection_replayed=False,original_parent_sha256=PARENT_SHA)
    save(output/'TRAINING.json',result);return result

def smoke(data_dir,output,arm='B1_shape'):
    """Real image + source GBA raster used only as an explicit software control."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from usbf.data.sensors import load_sensor
    from usbf.models.inference import load_models
    data_dir=Path(data_dir);meta=read(data_dir/'example/sensors.json')['records'][0]
    meta=dict(meta,path=str(data_dir/'example'/meta['path']));s=load_sensor(meta,True)
    buildings=gpd.read_parquet(data_dir/'example/neighbors.parquet').to_crs(s['crs'])
    target=rasterize([(g,1) for g in buildings.geometry if g.is_valid],out_shape=s['signed'].shape,transform=s['transform'],dtype='uint8')
    # The crop is fixed, rather than selected from a loss or reference score.
    sl=np.s_[192:320,192:320]
    batch=dict(rgb=torch.from_numpy(s['rgb'][:,192:320,192:320])[None],
        height=torch.from_numpy(np.where(s['height_valid'],np.clip(s['signed'],0,30)/30,0).astype('float32')[sl])[None,None],
        valid=torch.from_numpy(s['height_valid'][sl].astype('float32'))[None,None],
        target=torch.from_numpy(target[sl].astype('float32'))[None,None],supervision_valid=torch.from_numpy(s['rgb_valid'][sl])[None,None])
    torch.set_num_threads(2);models,_=load_models(data_dir,'cpu');model=models[1 if arm=='B1_shape' else 0]
    opt=optimizer(model);model.train();before=model.mask_head.weight.detach().clone()
    value,parts=loss(arm,model(batch['rgb'],batch['height'],batch['valid']),batch)
    if not torch.isfinite(value):raise AssertionError('Nonfinite software-control loss')
    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    delta=float((model.mask_head.weight-before).abs().max())
    if delta<=0:raise AssertionError('No real parameter update')
    result=dict(status='passed',arm=arm,updates=1,loss=float(value.detach()),maximum_mask_head_weight_change=delta,
        labels='Original GBA rasterized on a real image solely for a software-control step',
        part_of_published_training_corpus=False,new_accuracy_evidence=False,frozen_weight_files_modified=False)
    save(output,result);return result
