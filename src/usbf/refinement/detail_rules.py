import numpy as np
from scipy.ndimage import map_coordinates
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity, set_precision
from . import guided as G, raster_rules as R

RULES=dict(version='V6circle_shared_with_V7',primitive_min_area_m2=12,primitive_max_area_m2=6000,
    model_fit_iou=.80,model_vs_rectangle_gain=.025,minimum_equivalent_radius_m=2.2,
    same_fit_thresholds_required=2,image_arc_sector_fraction=.70,image_vs_rectangle_gain=.04,
    V7_base='exact_original_GBA',V7_small_change_min_m2=6,V7_min_pixels=4,
    V7_addition_B0=.65,V7_addition_B1=.70,V7_removal_max_dual=.25,
    V7_RGB_gain=.035,V7_probability_gain=.04,missing_height_is_not_absence=True,
    original_holes_protected=True,small_FP_flag_only=True,GT_access=False)

def sample(array,xy,w):
    xy=np.asarray(xy);return map_coordinates(np.nan_to_num(array,nan=0.),[xy[:,1]/w['gy']-.5,xy[:,0]/w['gx']-.5],order=1,mode='constant',cval=0.)

def parameters(raw,ellipse=False):
    if raw.geom_type!='Polygon' or raw.is_empty:return None
    pts=G.sampled_ring(Polygon(raw.exterior),128)
    if ellipse:
        from skimage.measure import EllipseModel
        e=EllipseModel()
        if not e.estimate(pts) or not np.isfinite(e.params).all():return None
        cx,cy,a,b,theta=e.params
        if a<b:a,b,theta=b,a,theta+np.pi/2
    else:
        keep=np.ones(len(pts),bool)
        for _ in range(3):
            v=pts[keep];cx,cy,k=np.linalg.lstsq(np.column_stack([2*v,np.ones(len(v))]),np.sum(v*v,axis=1),rcond=None)[0]
            a=b=np.sqrt(max(k+cx*cx+cy*cy,0));theta=0.
            residual=np.abs(np.linalg.norm(pts-[cx,cy],axis=1)-a);keep=residual<=np.quantile(residual,.85)
    if b<2.2 or a/b>2.5:return None
    return np.array([cx,cy,a,b,theta],float)

def primitive(p):
    cx,cy,a,b,t=p
    return affinity.translate(affinity.rotate(affinity.scale(Point(0,0).buffer(1,quad_segs=48),a,b),t,origin=(0,0),use_radians=True),cx,cy)

def arc_evidence(p,w):
    """72 angular rays; whole-circle support, not four bright rectangle corners."""
    cx,cy,a,b,theta=p;ang=np.arange(72)*2*np.pi/72
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    offsets=np.column_stack([a*np.cos(ang),b*np.sin(ang)])@rot.T
    normal=np.column_stack([np.cos(ang)/a,np.sin(ang)/b])@rot.T
    normal/=np.linalg.norm(normal,axis=1)[:,None];xy=offsets+[cx,cy]
    gs=max(w['gx'],w['gy']);delta=max(gs,.65)
    rgb=w['rgb'].mean(0);inn=sample(rgb,xy-delta*normal,w);out=sample(rgb,xy+delta*normal,w)
    contrast=np.abs(inn-out)
    valid=(sample(w['rgb_valid'].astype(float),xy-2*delta*normal,w)>.99)&(sample(w['rgb_valid'].astype(float),xy+2*delta*normal,w)>.99)
    # Local normal search diagnoses whether supposed arcs actually lie on
    # another boundary; it is not used to translate exported geometries.
    shifts=np.arange(-2,2.01,.5)*gs
    response=np.stack([np.abs(sample(rgb,xy+(s-delta)*normal,w)-sample(rgb,xy+(s+delta)*normal,w)) for s in shifts])
    best=np.argmax(response,axis=0);peak=response[best,np.arange(72)]
    supported=valid&(contrast>=.025)&(contrast>=.65*peak)&(np.abs(shifts[best])<=1.5*gs)
    sector=supported.reshape(12,6).mean(1)>=.5
    # Radial probability contrast is separate from mere roof texture.
    pd=sample(w['prob'][1],xy-delta*normal,w)-sample(w['prob'][1],xy+delta*normal,w)
    return dict(sector_fraction=float(sector.mean()),valid_fraction=float(valid.mean()),
        rgb_score=float(np.minimum(contrast,.20)[valid].mean()) if valid.any() else 0.,
        probability_edge=float(pd[valid].mean()) if valid.any() else 0.)

def boundary(q,w,roi=None):
    pts=[];normals=[]
    for part in G.parts(q):
        for ring in [part.exterior,*part.interiors]:
            xy=np.asarray(ring.coords)
            for a,b in zip(xy[:-1],xy[1:]):
                d=b-a;l=np.linalg.norm(d)
                if l<.05:continue
                count=max(1,min(256,int(np.ceil(l/max(w['gx'],w['gy'])))))
                v=a+(np.arange(count)+.5)[:,None]/count*d;n=np.array([-d[1],d[0]])/l
                if not part.contains(Point(v[0]+.2*n)):n=-n
                if roi is not None:v=np.array([p for p in v if roi.covers(Point(p))]).reshape(-1,2)
                pts.extend(v);normals.extend([n]*len(v))
    if not pts:return dict(rgb=0.,prob=0.,samples=0)
    pts=np.asarray(pts);normal=np.asarray(normals);d=max(w['gx'],w['gy'])
    inside=pts+d*normal;outside=pts-d*normal
    valid=(sample(w['rgb_valid'].astype(float),inside,w)>.99)&(sample(w['rgb_valid'].astype(float),outside,w)>.99)
    if not valid.any():return dict(rgb=0.,prob=0.,samples=0)
    rg=abs(sample(w['rgb'].mean(0),inside,w)-sample(w['rgb'].mean(0),outside,w))
    pd=sample(w['prob'][1],inside,w)-sample(w['prob'][1],outside,w)
    return dict(rgb=float(np.minimum(rg,.20)[valid].mean()),prob=float(pd[valid].mean()),samples=int(valid.sum()))

def circles(g,raw,alts,neighbors,w,t):
    audit=dict(candidate=False,accepted=False,attempts=[],reason='outside_primitive_scale')
    if raw.is_empty or raw.geom_type!='Polygon' or not 12<=raw.area<=6000:return None,audit
    rect=raw.minimum_rotated_rectangle;short,long,_=R.size(rect)
    if short<4.4 or long/short>2.5:return None,audit
    audit['candidate']=True;proposals=[]
    for ellipse in [False,True]:
        p=parameters(raw,ellipse)
        if p is None:continue
        # Ellipse and circle compete, never classify every rounded rectangle as ellipse.
        if ellipse and p[2]/p[3]<1.2:continue
        proposals.append((p,'raw_fit','ellipse' if ellipse else 'circle'))
    # RGB rescue also proposes from a square model/GBA envelope. It does not
    # require the model mask to be circular before inspecting actual arcs.
    for base in [g,raw]:
        rr=base.minimum_rotated_rectangle;xy=np.asarray(rr.exterior.coords);vec=np.diff(xy,axis=0);ll=np.linalg.norm(vec,axis=1)
        j=int(np.argmax(ll));a=ll[j]/2;b=ll[(j+1)%4]/2;center=np.array(rr.centroid.coords[0]);theta=np.arctan2(vec[j,1],vec[j,0])
        if min(a,b)<2.2 or max(a,b)/min(a,b)>2.5:continue
        for scale in [.85,1.,1.15]:
            for offset in [(0,0),(w['gx'],0),(-w['gx'],0),(0,w['gy']),(0,-w['gy'])]:
                kind='circle' if a/b<1.2 else 'ellipse';aa=bb=(a+b)/2 if kind=='circle' else 0
                if kind=='ellipse':aa,bb=a,b
                proposals.append((np.array([*(center+offset),aa*scale,bb*scale,theta]),'RGB_rescue',kind))
    best=None
    for p,origin,kind in proposals:
        q=primitive(p)
        if q.intersection(g).area/max(g.area,1)<.50 or q.area/g.area>1.8 or q.area/g.area<.5:continue
        if q.intersection(neighbors).area>g.intersection(neighbors).area+.001:continue
        mi=G.poly_iou(raw,q);rect_iou=G.poly_iou(raw,rect)
        stable=0
        for other in [raw,*alts]:
            pp=parameters(other,kind=='ellipse')
            if pp is not None and np.linalg.norm(pp[:2]-p[:2])<=max(1.5*max(w['gx'],w['gy']),.12*p[3]) and max(abs(pp[2:4]/p[2:4]-1))<=.18:stable+=1
        arc=arc_evidence(p,w)
        bm=R.burn(q.buffer(-max(w['gx'],w['gy'])),w['signed'].shape,t)
        if bm.sum()<4:continue
        available=bm&w['rgb_valid']&np.isfinite(w['prob']).all(0)
        if available.sum()<.95*bm.sum():continue
        bm=available
        b0=float(np.nanmean(w['prob'][0][bm]));b1=float(np.nanmean(w['prob'][1][bm]))
        competitor=boundary(q.minimum_rotated_rectangle,w)
        shape_path=(origin=='raw_fit' and mi>=.80 and mi-rect_iou>=.025 and stable>=2 and
            (arc['sector_fraction']>=.4 or (b0>=.8 and mi>=.90)) and b1>=.65 and b0>=.5)
        rgb_path=(arc['sector_fraction']>=.70 and arc['valid_fraction']>=.9 and
            arc['rgb_score']>=competitor['rgb']+.04 and mi>=.60 and b1>=.60 and b0>=.5)
        accepted=bool(shape_path or rgb_path)
        rec=dict(kind=kind,origin=origin,parameters=p.tolist(),model_iou=mi,rectangle_model_iou=rect_iou,
            stable_fits=stable,arc=arc,rectangle_RGB=competitor['rgb'],B0_core=b0,B1_core=b1,accepted=accepted,
            shape_path=bool(shape_path),RGB_rescue=bool(rgb_path))
        audit['attempts'].append(rec)
        score=arc['rgb_score']+max(0,mi-rect_iou)+.03*(stable>=2)
        if accepted and (best is None or score>best[0]):best=(score,q,rec)
    if best:
        q=best[1].difference(G.holes_of(g))
        if q.is_valid and q.geom_type=='Polygon':
            audit.update(accepted=True,reason='accepted_'+best[2]['kind'],selected=best[2]);return q,audit
    audit['reason']='no_joint_arc_shape_instance_support';return None,audit

def local_change_allowed(g,trial,region,kind,raw,alts,w,t):
    roi=region.buffer(2*max(w['gx'],w['gy']));before=boundary(g,w,roi);after=boundary(trial,w,roi)
    mask=R.burn(region,w['signed'].shape,t);core=R.burn(region.buffer(-max(w['gx'],w['gy'])),w['signed'].shape,t)
    if core.sum()<4:core=mask
    available=core&w['rgb_valid']&np.isfinite(w['prob']).all(0)
    if not core.any() or available.sum()<.95*core.sum():
        return False,dict(kind=kind,area_m2=region.area,reasons=['incomplete_local_sensor_context'])
    core=available
    m0=float(np.nanmean(w['prob'][0][core])) if core.any() else 0.;m1=float(np.nanmean(w['prob'][1][core])) if core.any() else 0.
    hv=w['height_valid']&core;fraction=float(hv.sum()/max(core.sum(),1));height=float(np.nanpercentile(w['signed'][hv],25 if kind=='add' else 75)) if hv.any() else None
    color=w['rgb'];green=(color[1]>1.10*color[0])&(color[1]>1.05*color[2]);green_fraction=float(green[core].mean()) if core.any() else 1.
    stable=min((region.intersection(a).area/region.area if kind=='add' else 1-region.intersection(a).area/region.area for a in [raw,*alts]),default=0.)
    reasons=[]
    if region.area<max(6,4*w['gx']*w['gy']):reasons.append('subresolution_change_keep_original_detail')
    if min(before['samples'],after['samples'])<4:reasons.append('insufficient_boundary_observations')
    if after['rgb']<before['rgb']+.035 or after['prob']<before['prob']+.04:reasons.append('no_joint_local_boundary_gain')
    if stable<.75:reasons.append('threshold_unstable_local_change')
    if kind=='add':
        if m0<.65 or m1<.70:reasons.append('weak_dual_model_addition')
        if green_fraction>.35:reasons.append('vegetation_ambiguous')
        if fraction>=.8 and height is not None and height<1.5:reasons.append('observed_low_height')
        elif fraction<.8 and (m0<.85 or m1<.85 or after['rgb']<before['rgb']+.06):reasons.append('missing_height_requires_stronger_image_evidence')
    else:
        # A missed mask is not proof a GBA projection should be deleted.
        if max(m0,m1)>.25:reasons.append('possible_real_GBA_detail')
        if fraction<.9 or height is None or height>1.5:reasons.append('no_reliable_absence_height')
    return not reasons,dict(kind=kind,area_m2=region.area,before=before,after=after,B0=m0,B1=m1,height_valid=fraction,height_quantile=height,
        green_fraction=green_fraction,stability=stable,reasons=reasons)

def retain_details(g,v6,raw,alts,neighbors,w,t,protected_shape=False):
    if protected_shape:return v6,dict(decision='shared_qualified_primitive',edits=[])
    final=g;edits=[]
    # Only accepted V6 courtyard holes are carried over; never introduce holes
    # merely because raster prediction has a void.
    holes=G.holes_of(v6).difference(G.holes_of(g))
    for hole in G.parts(holes):
        trial=final.difference(hole)
        if trial.is_valid and trial.geom_type=='Polygon':final=trial;edits.append(dict(kind='preserve_qualified_hollow',area_m2=hole.area))
    for kind,regions in [('add',G.parts(v6.difference(g))),('remove',G.parts(g.difference(v6).difference(holes)))]:
        for region in sorted(regions,key=lambda p:-p.area):
            trial=final.union(region) if kind=='add' else final.difference(region)
            if not trial.is_valid or trial.geom_type!='Polygon':edits.append(dict(kind=kind,accepted=False,reasons=['topology']));continue
            ok,rec=local_change_allowed(final,trial,region,kind,raw,alts,w,t)
            if trial.intersection(neighbors).area>g.intersection(neighbors).area+.001:ok=False;rec['reasons'].append('neighbor_instance_overlap')
            if G.holes_of(g).difference(G.holes_of(trial)).area>.001:ok=False;rec['reasons'].append('original_hole_protection')
            rec['accepted']=ok;edits.append(rec)
            if ok:final=trial
    # No simplify/buffer-based whole-GBA cleaning. Collapse only numerical slits.
    if not final.equals_exact(g,0):
        snapped=set_precision(final,1e-6)
        if snapped.geom_type=='Polygon' and snapped.is_valid and snapped.symmetric_difference(final).area<.001:final=snapped
    return final,dict(decision='local_edits' if final.symmetric_difference(g).area>.001 else 'keep_original_details',edits=edits)
