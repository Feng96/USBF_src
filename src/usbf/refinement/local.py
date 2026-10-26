import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely import set_precision
from . import walls as W, adaptive as V, guided as G, raster_rules as R

RULES=dict(version='v6_local_additive',wing_min_area_m2=30.,wing_context_m=3.,
          local_wall_min_length_m=4.,local_stable_wall_length_m=14.,local_min_stable_walls=2,
          local_threshold_angle_max_deg=10.,local_RGB_group_coherence_min=.60,
          local_probability_gain_min=.04,local_joint_evidence_gain_min=.04,
          structural_void_stability_min=.9,structural_void_dual_iou_min=.9,
          original_S_rules_unchanged=True,ordinary_default='exact_v5',independent_accuracy=False)

def structural_void(f):
    """Separate evidence tier: all S physical gates, stronger model stability.

    Rectangularity is not a universal courtyard requirement. Its relaxed
    geometry floor is compensated by stronger stability/dual-void requirements.
    """
    altered=dict(f,rectangularity=max(.85,f['rectangularity']))
    return ('S' in R.classify('hollow',altered) and f['rectangularity']>=.65 and
            f['stability']>=.9 and f['dual_void_iou']>=.9)

def clean_overlay_seam(q):
    """Collapse sub-micrometre overlay spikes, not real narrow structures.

    Pixel-scale closing/simplification is deliberately not used. A long
    zero-width slit can have large Hausdorff distance but negligible area.
    """
    if q.geom_type!='Polygon' or not q.is_valid:return q
    fixed=set_precision(q,1e-6)
    if (fixed.geom_type=='Polygon' and fixed.is_valid and
        len(fixed.interiors)==len(q.interiors) and
        fixed.symmetric_difference(q).area<=.001):return fixed
    return q

def local_wall(a,b,raw,gs):
    u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]]);length=np.linalg.norm(b-a)
    pts=W.dense(raw);s=(pts-a)@u;d=(pts-a)@n
    take=(s>=.1*length)&(s<=.9*length)&(np.abs(d)<=2*gs)
    if take.sum()<6:return dict(ok=False)
    ss,dd=s[take],d[take];bounds=np.linspace(.1*length,.9*length,9)
    bucket=np.digitize(ss,bounds[1:-1]);x=[];y=[]
    for i in range(8):
        m=bucket==i
        if m.any():x.append(float(np.median(ss[m])));y.append(float(np.median(dd[m])))
    if len(x)<6:return dict(ok=False)
    slope,offset=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]
    error=float(np.sqrt(np.mean((np.asarray(y)-np.asarray(x)*slope-offset)**2)))
    angle=float(np.arctan2(u[1]+slope*n[1],u[0]+slope*n[0]))
    return dict(ok=bool(error<=.7*gs),angle=angle,error_m=error,offset_m=float(offset),coverage=len(x)/8)

def boundary_evidence(q,w,roi):
    p=w['prob'][1];rgb=w['rgb'].mean(axis=0);pe=[];re=[]
    for poly in G.parts(q):
        xy=W.dense(poly,.5);take=np.array([roi.contains(Point(x)) for x in xy]);xy=xy[take]
        for pt in xy:
            s=poly.exterior.project(Point(pt));a=np.array(poly.exterior.interpolate(max(0,s-.5)).coords[0]);b=np.array(poly.exterior.interpolate(min(poly.exterior.length,s+.5)).coords[0])
            u=b-a;n=np.array([-u[1],u[0]])/max(np.linalg.norm(u),1e-9)
            if not poly.contains(Point(pt+.3*n)):n=-n
            inn=pt+1.5*n;out=pt-1.5*n
            val=V.sample(p,[inn,out],w['gx'],w['gy']);col=V.sample(rgb,[inn,out],w['gx'],w['gy'])
            pe.append(val[0]-val[1]);re.append(min(abs(col[0]-col[1]),.35)/.35)
    return dict(probability=float(np.nanmean(pe)) if pe else -1.,rgb=float(np.nanmean(re)) if re else -1.,samples=len(pe),
                RGB_texture_without_probability_edge_fraction=float(((np.asarray(pe)<.35)&(np.asarray(re)>.35)).mean()) if pe else 0.)

def refine(g,baseline,raw,alternatives,neighbors,w,t,measurements,protected_shape=False):
    """Default is baseline exactly; rejected local proposals never reset to GBA."""
    final=baseline;events=[];diagnostics=[];gs=max(w['gx'],w['gy'])
    # Preserve and union accepted void geometry independently of outer walls.
    accepted_holes=G.holes_of(baseline)
    for m in measurements:
        if m['family']!='hollow':continue
        if not ('S' in m['levels'] or structural_void(m['metrics'])):continue
        hole=m['region'];trial=final.difference(hole)
        if trial.geom_type=='Polygon' and trial.is_valid and hole.difference(Polygon(final.exterior)).area<.001:
            if final.intersection(hole).area>.01:
                final=trial;events.append(dict(type='structural_hollow',area_m2=hole.area,metrics=m['metrics']))
            accepted_holes=accepted_holes.union(hole)
    if protected_shape:return final,dict(events=events,wing_attempts=[],protected_shape=True)
    shell=Polygon(raw.exterior)
    # The large local added footprint includes both versions, so a wrongly
    # rotated v5 protrusion is inside the correction ROI, not left behind.
    outside=shell.union(Polygon(baseline.exterior)).difference(g.buffer(1.5))
    large=[p for p in G.parts(outside) if p.area>=30 and p.buffer(-1.5).area>=3]
    rois=G.parts(unary_union([p.buffer(3) for p in large])) if large else []
    # Group raster stair fragments before proposing wall intervals. Every
    # proposed interval is still checked against the UNSMOOTHED raw contours.
    seed=V.dejag(shell)[0].simplify(.8,preserve_topology=True) # Never exported.
    xy=np.asarray(seed.exterior.coords)
    axis=V.robust_axis(Polygon(baseline.exterior))[0]
    for roi in rois:
        mismatch=final.symmetric_difference(shell).intersection(roi).area
        if mismatch<max(12,9*w['gx']*w['gy']):continue
        walls=[];facts=[]
        for a,b in zip(xy[:-1],xy[1:]):
            line=LineString([a,b]);length=line.length
            if length<max(4,4*gs) or line.intersection(roi).length/length<.75:continue
            obs=local_wall(a,b,shell,gs);alts=[local_wall(a,b,q,gs) for q in alternatives]
            stable=obs['ok'] and len(alts)==2 and all(o['ok'] and
                 abs((o['angle']-obs['angle']+np.pi/2)%np.pi-np.pi/2)<=np.deg2rad(10) and
                 abs(o['offset_m']-obs['offset_m'])<=1.5*gs for o in alts)
            image=W.image_wall_evidence(a,b,w,axis)
            facts.append(dict(length_m=length,stable=bool(stable),raw=obs,alternatives=alts,RGB=image))
            if stable:walls.append(dict(a=a,b=b,angle=obs['angle'],length=length,image=image))
        rec=dict(roi_area_m2=roi.area,mismatch_m2=mismatch,walls=facts,accepted=False)
        total=sum(x['length'] for x in walls)
        # A solar-panel/shadow gradient may dominate a weak exterior edge.
        # Require TWO visible walls, not a high RGB average across every wall;
        # other walls still need raw-threshold stability and a supported roof core.
        visible=[x for x in walls if x['image']['rgb_oblique_coherence']>=.55 and x['image']['rgb_valid_edge_fraction']>=.75]
        visible_length=sum(x['length'] for x in visible)
        coherence=sum(x['length']*x['image']['rgb_oblique_coherence'] for x in visible)/max(visible_length,1e-9)
        core=R.burn(shell.intersection(roi).buffer(-1),w['signed'].shape,t)
        height=R.stats(core,w['signed'],w['height_valid'])
        core_ok=(core.sum()>=9 and height['valid_fraction']>=.9 and height['p25'] is not None and height['p25']>=2 and
                 float(np.nanmean(w['prob'][0][core]))>=.7 and float(np.nanmean(w['prob'][1][core]))>=.8)
        rec.update(stable_wall_length_m=total,visible_wall_length_m=visible_length,RGB_group_coherence=coherence,local_core_supported=bool(core_ok))
        if len(visible)<2 or visible_length<14 or coherence<.60 or not core_ok:
            rec['reason']='insufficient_group_structure';diagnostics.append(rec);continue
        # Whole raw contour is fitted, but ONLY its ROI replacement can be used.
        options=[];before=boundary_evidence(final,w,roi);rec['fit_attempts']=[]
        for phase in range(4):
            fit,info=W.segment_walls(shell,axis,walls,gs,phase)
            attempt=dict(phase=phase,fit=info);rec['fit_attempts'].append(attempt)
            if fit is None:continue
            trial=final.difference(roi).union(fit.intersection(roi)).difference(accepted_holes)
            trial=clean_overlay_seam(trial)
            if trial.geom_type!='Polygon' or not trial.is_valid:attempt['blocked']='union_topology';continue
            change=trial.symmetric_difference(final)
            if change.difference(roi).area>.001:attempt['blocked']='outside_ROI';continue
            if trial.intersection(neighbors).area>final.intersection(neighbors).area+.001:attempt['blocked']='neighbor';continue
            if trial.difference(shell.buffer(.75*gs)).area>final.difference(shell.buffer(.75*gs)).area+3:attempt['blocked']='invention';continue
            after=boundary_evidence(trial,w,roi)
            local_gain=(final.symmetric_difference(shell).intersection(roi).area-trial.symmetric_difference(shell).intersection(roi).area)/max(roi.intersection(shell).area,1)
            attempt.update(before=before,after=after,local_fit_gain=local_gain)
            joint_gain=.6*(after['probability']-before['probability'])+.4*(after['rgb']-before['rgb'])
            attempt['joint_evidence_gain']=joint_gain
            # RGB contrast magnitude can reward solar-panel/shadow cuts. The
            # separately checked visible wall DIRECTIONS remain mandatory.
            if after['probability']<before['probability']+.04 or joint_gain<.04 or local_gain<.04:continue
            options.append((after['probability']+.3*after['rgb'],trial,dict(before=before,after=after,local_fit_gain=local_gain,fit=info)))
        if options:
            _,trial,detail=max(options,key=lambda x:x[0]);delta=trial.symmetric_difference(final).area
            final=trial;events.append(dict(type='local_multi_direction_wing',changed_area_m2=delta));rec.update(accepted=True,reason='local_structure_and_boundary_gain',**detail)
        else:rec['reason']='no_safe_local_evidence_gain'
        diagnostics.append(rec)
    return final,dict(events=events,wing_attempts=diagnostics,protected_shape=False)
