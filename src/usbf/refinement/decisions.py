import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from . import walls as W, adaptive as V, guided as G, raster_rules as R

def polygon(g):
    return max(G.parts(g),key=lambda p:p.area) if G.parts(g) else Polygon()

def local_mean(g,w,t,band):
    mask=R.burn(g,w['signed'].shape,t)
    return float(np.nanmean(w['prob'][band][mask])) if mask.any() else None

def decide(g,q,raw,alts,neighbors,w,t,fit_failed):
    base=V.contour_score(g,w,t);score=V.contour_score(q,w,t)
    evidence=dict(GBA_score=base,contour_score=score,large_additions=[],large_removals=[])
    veto=[];cautions=[]
    if fit_failed:veto.append('no_safe_supported_wall_fit')
    if not q.is_valid or q.geom_type!='Polygon':veto.append('invalid_candidate')
    if q.intersection(neighbors).area>g.intersection(neighbors).area+.01:veto.append('neighbor_instance_conflict')
    # A score margin by itself is not a reason to keep GBA. Both observed RGB
    # and probability boundary evidence must clearly favor it.
    if base['rgb_edge']>score['rgb_edge']+.06 and base['probability_edge']>score['probability_edge']+.10:
        veto.append('GBA_clearly_stronger_RGB_and_probability_boundary')
    if score['model_inside']<.60:veto.append('weak_model_support_inside_candidate')
    for region in G.parts(q.difference(g.buffer(2))):
        if region.area<30:continue
        state,facts=V.support(region,q,g,w,t)
        m1=local_mean(region,w,t,1);m0=local_mean(region,w,t,0)
        core_means=[]
        for radius in (1.,2.):
            core=region.buffer(-radius)
            if R.burn(core,w['signed'].shape,t).sum()>=4:
                core_means.append(dict(radius_m=radius,B1=local_mean(core,w,t,1),B0=local_mean(core,w,t,0)))
        core_positive=(bool(core_means) and all(v['B1']>=.75 and v['B0']>=.70 for v in core_means) and
                       m1 is not None and m0 is not None and min(m1,m0)>=.55 and
                       facts.get('height_high_scenario_fraction',0)>=.8 and facts.get('green_fraction',1)<.25)
        if state=='uncertain' and core_positive:
            state='supported';facts['reason']='stable_high_core_boundary_mixture_tolerated'
        facts['core_probability_checks']=core_means
        facts.update(state=state,B1_mean=m1,B0_mean=m0)
        evidence['large_additions'].append(facts)
        if state=='opposed' or (m1 is not None and m0 is not None and m1<.55 and m0<.55):
            veto.append('large_addition_locally_opposed')
        elif state!='supported':
            cautions.append('large_addition_height_or_temporal_uncertainty')
            if m1 is None or m0 is None or m1<.70 or m0<.60:
                veto.append('large_addition_insufficient_positive_model_evidence')
    for region in G.parts(g.difference(q.buffer(2))):
        if region.area<30:continue
        m1=local_mean(region,w,t,1);m0=local_mean(region,w,t,0)
        evidence['large_removals'].append(dict(area_m2=region.area,B1_mean=m1,B0_mean=m0))
        if m1 is None or m0 is None or (m1>.60 and m0>.60):
            veto.append('large_removal_would_erase_model_supported_roof')
        elif g.difference(q).area/g.area>.40:
            veto.append('major_GBA_contraction_requires_independent_evidence')
    stability=min((G.poly_iou(raw,a) for a in alts if not a.is_empty),default=0.)
    evidence['raw_threshold_min_iou']=stability
    if stability<.7:cautions.append('threshold_unstable')
    if q.difference(raw).area/q.area>.20:veto.append('regularizer_invented_large_area')
    return not veto,dict(**evidence,veto=sorted(set(veto)),cautions=sorted(set(cautions)))
