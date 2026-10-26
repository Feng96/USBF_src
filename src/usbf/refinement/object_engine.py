import numpy as np
import rasterio
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from . import context as S, baseline as E, curvature as Q
G, R, N = E.G, E.R, E.N
CTX = COHORT = BASE = None

def process(row):
    original=row.geometry;rec=dict(key=row.key,source=row.source,anchor=row.anchor,region=row.region,row_id=int(row.row_id),legacy60=bool(row.legacy60),
        status='unprocessed',geometry=original,old_v6_wkb=original.wkb,V6circle_wkb=original.wkb,V7_wkb=original.wkb,
        circle_adopted=False,circle_candidate=False,old_circle=False,hollow=False,possible_small_FP=False,V7_decision='keep_original',V6_changed=False,V7_changed=False)
    if row.key in BASE.index:
        prior=BASE.loc[row.key];rec.update(old_v6_wkb=prior.geometry.wkb,V6circle_wkb=prior.geometry.wkb,
            old_circle=bool(prior.shape_changed),hollow=bool(prior.hollow_cut),V6_changed=not prior.geometry.equals_exact(original,0))
    if row.geometry_conflict or not original.is_valid or original.geom_type!='Polygon':rec['status']='abstain_original_geometry';return rec,{}
    w,why=CTX.window(original,row.anchor)
    if w is None:rec['status']='abstain_context:'+why;return rec,{}
    g=w['g'];t=rasterio.Affine(w['gx'],0,0,0,w['gy'],0);inside=R.burn(g,w['signed'].shape,t)
    if g.area<4 or g.area>20000 or max(w['signed'].shape)>1000:rec['status']='abstain_scale';return rec,{}
    if not inside.any() or w['rgb_valid'][inside].mean()<.98:rec['status']='abstain_incomplete_object_context';return rec,{}
    bounds=S.to_world(box(0,0,w['signed'].shape[1]*w['gx'],w['signed'].shape[0]*w['gy']),w)
    near=COHORT.iloc[COHORT.sindex.query(bounds,predicate='intersects')]
    neighbors=unary_union([CTX.local(r.geometry,w) for r in near.itertuples() if r.key!=row.key and r.geometry.is_valid])
    owner=G.ownership(inside,R.burn(neighbors,w['signed'].shape,t),w['gx'],w['gy'])
    def raw(th):return E.largest(R.geom(R.component((w['prob'][1]>=th)&owner&w['rgb_valid'],inside),t).difference(neighbors))
    model=raw(.5);alts=[raw(.35),raw(.65)]
    # An abstention never discards the published V6 comparator, when available.
    old=BASE.loc[row.key] if row.key in BASE.index else None
    if old is not None:rec['old_v6_wkb']=old.geometry.wkb;rec['V6circle_wkb']=old.geometry.wkb;rec['old_circle']=bool(old.shape_changed);rec['hollow']=bool(old.hollow_cut)
    if model.is_empty:rec['status']='abstain_model_component';return rec,{}
    measured,_=R.measure(w,G.parts(neighbors))
    if old is None and any(a.is_empty for a in alts):
        b6=g
    elif old is None:
        b5,info=E.baseline(g,model,alts,neighbors,w,t,measured)
        b6,detail=N.refine(g,b5,model,alts,neighbors,w,t,measured,protected_shape=info['shape'])
        rec['old_circle']=bool(info['shape']);rec['hollow']=bool(info['hollow'] or any(e['type']=='structural_hollow' for e in detail['events']))
    else:b6=CTX.local(old.geometry,w)
    if not b6.is_valid or b6.geom_type!='Polygon' or b6.intersection(neighbors).area>g.intersection(neighbors).area+.001:b6=g;rec['old_circle']=False;rec['hollow']=False
    exact_published=old is not None and b6.equals_exact(CTX.local(old.geometry,w),0)
    circle,circle_audit=(None,dict(candidate=False,accepted=False,reason='preserve_existing_qualified_primitive')) if rec['old_circle'] else Q.circles(g,model,alts,neighbors,w,t)
    upgraded=circle if circle is not None else b6
    # Accepted old/new hollows are not filled by a newly fitted exterior.
    if circle is not None:upgraded=upgraded.difference(G.holes_of(b6))
    v7,local=Q.retain_details(g,upgraded,model,alts,neighbors,w,t,protected_shape=rec['old_circle'] or circle is not None)
    for name,q in [('old_v6',b6),('V6circle',upgraded),('V7',v7)]:
        if not q.is_valid or q.geom_type!='Polygon' or q.intersection(neighbors).area>g.intersection(neighbors).area+.001:q=g
        world=original if q.symmetric_difference(g).area<=.001 else S.to_world(q,w)
        if exact_published and (name=='old_v6' or (name=='V6circle' and circle is None) or (name=='V7' and q.equals_exact(b6,0))):world=old.geometry
        assert world.is_valid
        rec[name+'_wkb']=world.wkb
    rec.update(status='processed',circle_adopted=circle is not None,circle_candidate=bool(circle_audit.get('candidate')),
        possible_small_FP=bool(g.area<20 and any(m['family']=='small_fp' and 'S' in m['levels'] for m in measured)),
        V7_decision=local['decision'],V6_changed=rec['V6circle_wkb']!=original.wkb,V7_changed=rec['V7_wkb']!=original.wkb)
    attempts=circle_audit.pop('attempts',[])
    circle_audit['attempt_count']=len(attempts)
    circle_audit['top_attempts']=sorted(attempts,key=lambda a:(a['accepted'],a['arc']['sector_fraction'],a['arc']['rgb_score']),reverse=True)[:4]
    return rec,dict(key=row.key,circle=circle_audit,V7=local)
