import numpy as np
from scipy.signal import correlate
from scipy.ndimage import binary_erosion

def corrgrid(a,b,mask,radius=10):
    """Masked NCC of destination b against translated source a."""
    m=mask.astype('float64');s=a.astype('float64')/255-.5;d=b.astype('float64')/255-.5
    def cc(x,y):
        z=correlate(x,y,mode='full',method='fft');cy,cx=np.array(a.shape)-1
        return z[cy-radius:cy+radius+1,cx-radius:cx+radius+1]
    n=cc(m,m);sa=cc(m,m*s);sb=cc(m*d,m)
    va=cc(m,m*s*s)-sa*sa/np.maximum(n,1)
    vb=cc(m*d*d,m)-sb*sb/np.maximum(n,1)
    cov=cc(m*d,m*s)-sa*sb/np.maximum(n,1)
    ncc=np.divide(cov,np.sqrt(np.maximum(va*vb,0)),out=np.full_like(cov,-2.),where=(n>=2000)&(va>1e-8)&(vb>1e-8))
    iy,ix=np.unravel_index(np.argmax(ncc),ncc.shape);peak=float(ncc[iy,ix])
    yy,xx=np.indices(ncc.shape);away=np.hypot(xx-ix,yy-iy)>=3
    other=float(ncc[away].max());dx=float(ix-radius);dy=float(iy-radius)
    def sub(v0,v1,v2):
        den=v0-2*v1+v2
        return float(np.clip(.5*(v0-v2)/den,-.75,.75)) if den<-.000001 else 0.
    edge=ix in [0,2*radius] or iy in [0,2*radius]
    if not edge:
        dx+=sub(ncc[iy,ix-1],ncc[iy,ix],ncc[iy,ix+1]);dy+=sub(ncc[iy-1,ix],ncc[iy,ix],ncc[iy+1,ix])
    return dict(dx=dx,dy=dy,peak=peak,zero=float(ncc[radius,radius]),gain=peak-float(ncc[radius,radius]),
                uniqueness=peak-other,paired_pixels=int(round(n[iy,ix])),edge_peak=edge)

def estimate(a,b,ground,tr):
    ground=binary_erosion(ground,iterations=3);yy,xx=np.indices(ground.shape)
    east=tr.c+xx+.5;north=tr.f-yy-.5
    even=((np.floor(east/100)+np.floor(north/100)).astype('int64')%2)==0
    fit=ground&even;hold=ground&~even
    result=dict(fit_pixels=int(fit.sum()),holdout_pixels=int(hold.sum()))
    if min(fit.sum(),hold.sum())<5000:return dict(result,status='dense_insufficient_ground')
    f=corrgrid(a,b,fit);h=corrgrid(a,b,hold);delta=np.array([f['dx'],f['dy']]);agreement=float(np.hypot(f['dx']-h['dx'],f['dy']-h['dy']))
    result.update({('fit_'+k):v for k,v in f.items()});result.update({('holdout_'+k):v for k,v in h.items()})
    result.update(shift_east_m=f['dx'],shift_north_m=-f['dy'],split_agreement_m=agreement)
    nearzero=np.linalg.norm(delta)<=.75
    if (f['edge_peak'] or h['edge_peak'] or np.linalg.norm(delta)>10 or agreement>1 or
        min(f['peak'],h['peak'])<.65 or min(f['uniqueness'],h['uniqueness'])<.01 or
        min(f['paired_pixels'],h['paired_pixels'])<5000 or (not nearzero and min(f['gain'],h['gain'])<.015)):
        return dict(result,status='dense_rejected_split_or_peak')
    quads=[]
    for iy in range(2):
        for ix in range(2):
            mask=ground&(yy>=iy*a.shape[0]//2)&(yy<(iy+1)*a.shape[0]//2)&(xx>=ix*a.shape[1]//2)&(xx<(ix+1)*a.shape[1]//2)
            if mask.sum()<4000:continue
            q=corrgrid(a,b,mask);q['agreement_m']=float(np.hypot(q['dx']-f['dx'],q['dy']-f['dy']))
            q.update(quadrant=f'{iy}_{ix}',passed=not q['edge_peak'] and q['peak']>=.55 and q['agreement_m']<=1.5)
            quads.append(q)
    result['quadrants']=quads;result['quadrants_passed']=sum(q['passed'] for q in quads)
    return dict(result,status='accepted' if result['quadrants_passed']>=3 else 'dense_rejected_subregions',near_zero_verified=bool(nearzero))
