import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from . import decisions as B, local as N, raster_rules as R, guided as G, adaptive as V
W = B.W

def largest(g):
    ps=G.parts(g);return max(ps,key=lambda p:p.area) if ps else Polygon()

def baseline(g,raw,alts,neighbors,w,t,measured):
    circle,kind,_=G.primitive_branch(raw,alts)
    if circle is not None:
        inside=R.burn(g,w['signed'].shape,t)
        unsplit=largest(R.geom(R.component((w['prob'][1]>=.5)&w['rgb_valid'],inside),t))
        if G.fit_primitive(unsplit)[1]!=kind:circle=None
    holes=[m['region'] for m in measured if m['family']=='hollow' and 'S' in m['levels']]
    if circle is not None or holes:
        q,flags=W.special_shapes(g,circle,holes)
        if q.is_valid and q.geom_type=='Polygon' and q.intersection(neighbors).area<=g.intersection(neighbors).area+.01:
            return q,dict(decision='qualified_'+('circle' if circle is not None else 'hollow'),shape=bool(circle is not None),hollow=bool(holes),fit_failed=False)
        return g,dict(decision='KEEP_special_geometry_conflict',shape=False,hollow=False,fit_failed=True)
    ga=G.directions(g)[0]%(np.pi/2);ma=V.robust_axis(raw)[0];preliminary=[]
    for name,axis in [('GBA',ga),('model',ma)]:
        trial,_=V.fit_axis(raw,axis)
        if trial.is_valid and trial.geom_type=='Polygon':preliminary.append((V.contour_score(trial,w,t)['score'],name,axis,trial))
    if not preliminary:return g,dict(decision='KEEP_no_seed',shape=False,hollow=False,fit_failed=True)
    _,name,axis,seed=max(preliminary,key=lambda x:x[0]);options=[];seen=[]
    for source,a in [(name,axis),('fallback_GBA',ga),('fallback_model',ma)]:
        if any(V.angle_distance(a,b)<np.deg2rad(2) for b in seen):continue
        seen.append(a)
        q,fit=W.regularize(seed,Polygon(raw.exterior),alts,a,max(w['gx'],w['gy']),w,t)
        q=q.difference(G.holes_of(g));overlap=q.intersection(neighbors).area
        if overlap>.001:
            clipped=q.difference(neighbors.buffer(.005))
            if clipped.geom_type=='Polygon' and clipped.is_valid and overlap/q.area<.03:q=clipped
            else:continue
        if not q.is_valid or q.geom_type!='Polygon':continue
        if fit['fallback']:continue
        ok,why=B.decide(g,q,raw,alts,neighbors,w,t,False)
        return (q if ok else g),dict(decision='ADOPT_contour' if ok else 'KEEP:'+','.join(why['veto']),
                 shape=False,hollow=False,fit_failed=False,axis_deg=float(np.rad2deg(a)),cautions=why['cautions'])
    return g,dict(decision='KEEP_no_safe_wall_fit',shape=False,hollow=False,fit_failed=True)
