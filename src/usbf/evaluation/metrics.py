import re
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from shapely.ops import unary_union

def year(value):
    if value is None:return None
    s=str(value).strip()
    if s.lower() in ('','none','nan','null'):return None
    try:n=int(float(s))
    except (ValueError,OverflowError):return None
    if 1900<=n<=2100:return n
    if 19000000<=n<=21009999:return n//10000
    return None

def is_archived(value):return str(value).strip().upper() in ('Y','YES','TRUE','1')

def union(geoms):return unary_union(list(geoms)) if len(geoms) else Polygon()

def iou(a,b):
    u=a.union(b).area
    return a.intersection(b).area/u if u else 1.

def matrix(pred,gt):
    out=np.zeros((len(pred),len(gt)))
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):
            if p.intersects(g):out[i,j]=iou(p,g)
    return out

def match(a,threshold):
    if not a.size:return []
    score=(a>=threshold)*(max(a.shape)+1.)+a
    ix,jx=linear_sum_assignment(score,maximize=True)
    return [(int(i),int(j),float(a[i,j])) for i,j in zip(ix,jx) if a[i,j]>=threshold]

def boundary(pred,gt,core,pixel=.5):
    if core.is_empty:return {k:0 for k in ['pred_n','gt_n','pred_hit_1m','gt_hit_1m','pred_hit_2m','gt_hit_2m']}
    xmin,ymin,xmax,ymax=core.buffer(3).bounds
    w=int(np.ceil((xmax-xmin)/pixel));h=int(np.ceil((ymax-ymin)/pixel));t=from_origin(xmin,ymax,pixel,pixel)
    def burn(gs):
        shapes=[(g,1) for g in gs if not g.is_empty]
        return rasterize(shapes,out_shape=(h,w),transform=t,dtype='uint8').astype(bool) if shapes else np.zeros((h,w),bool)
    allowed=burn([core]);pm=burn(pred);gm=burn(gt)
    # Compute on full geometries before restricting boundary samples to core.
    pe=(pm ^ ndimage.binary_erosion(pm))&allowed;ge=(gm ^ ndimage.binary_erosion(gm))&allowed
    result=dict(pred_n=int(pe.sum()),gt_n=int(ge.sum()))
    dg=ndimage.distance_transform_edt(~ge,sampling=pixel) if ge.any() else None
    dp=ndimage.distance_transform_edt(~pe,sampling=pixel) if pe.any() else None
    for tol in [1,2]:
        result['pred_hit_%dm'%tol]=int((dg[pe]<=tol).sum()) if dg is not None else 0
        result['gt_hit_%dm'%tol]=int((dp[ge]<=tol).sum()) if dp is not None else 0
    return result

def f1(tp_pred,tp_gt,npred,ngt):
    if npred==0 and ngt==0:return 1.
    p=tp_pred/npred if npred else 0.;r=tp_gt/ngt if ngt else 0.
    return 2*p*r/(p+r) if p+r else 0.

def evaluate(pred_all,gt_all,pred_objects,gt_objects,core):
    a=union(pred_all).intersection(core);b=union(gt_all).intersection(core)
    inter=a.intersection(b).area;un=a.union(b).area
    row=dict(area_intersection_m2=inter,area_union_m2=un,area_iou=inter/un if un else 1.,
             prediction_area_m2=a.area,reference_area_m2=b.area,core_area_m2=core.area,
             pred_objects=len(pred_objects),gt_objects=len(gt_objects))
    bc=boundary(pred_all,gt_all,core);row.update(bc)
    for tol in [1,2]:row['boundary_f1_%dm'%tol]=f1(bc['pred_hit_%dm'%tol],bc['gt_hit_%dm'%tol],bc['pred_n'],bc['gt_n'])
    a=matrix(pred_objects,gt_objects)
    for threshold in [.25,.5,.75]:
        n=len(match(a,threshold));suffix=str(int(threshold*100));row['tp_'+suffix]=n
        row['object_f1_'+suffix]=f1(n,n,len(pred_objects),len(gt_objects))
    return row

def aggregate(df):
    den=float(df.area_union_m2.sum());out=dict(patches=len(df),pooled_area_iou=float(df.area_intersection_m2.sum()/den) if den else 1.,
        macro_area_iou=float(df.area_iou.mean()),nonempty_patches=int((df.area_union_m2>0).sum()),
        macro_nonempty_area_iou=float(df.loc[df.area_union_m2>0,'area_iou'].mean()) if (df.area_union_m2>0).any() else None,
        gt_objects=int(df.gt_objects.sum()),pred_objects=int(df.pred_objects.sum()))
    for tol in [1,2]:
        out['micro_boundary_f1_%dm'%tol]=f1(df['pred_hit_%dm'%tol].sum(),df['gt_hit_%dm'%tol].sum(),df.pred_n.sum(),df.gt_n.sum())
        out['macro_boundary_f1_%dm'%tol]=float(df['boundary_f1_%dm'%tol].mean())
    for t in [25,50,75]:out['pooled_object_f1_'+str(t)]=f1(df['tp_'+str(t)].sum(),df['tp_'+str(t)].sum(),df.pred_objects.sum(),df.gt_objects.sum())
    return out
