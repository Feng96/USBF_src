from . import detail_rules as D
from .detail_rules import *
from scipy.ndimage import gaussian_filter

RULES=dict(D.RULES,revision='observed_curvature_v002',minimum_minor_diameter_pixels=8,
    gradient_alignment_cosine=.866,diagonal_arc_support=.70,diagonal_quadrants_required=3,
    rectangle_competitor='best_original_raw_and_candidate_envelopes',
    new_primitive_requires_RGB_even_if_models_agree=True)

def curvature(p,w,gradients):
    cx,cy,a,b,theta=p;ang=np.arange(72)*2*np.pi/72
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy=np.column_stack([a*np.cos(ang),b*np.sin(ang)])@rot.T+[cx,cy]
    normal=np.column_stack([np.cos(ang)/a,np.sin(ang)/b])@rot.T
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    grad=np.column_stack([sample(gradients[0],xy,w),sample(gradients[1],xy,w)])
    strength=np.linalg.norm(grad,axis=1)
    alignment=np.abs(np.sum(grad*normal,axis=1))/np.maximum(strength,1e-12)
    valid=sample(w['rgb_valid'].astype(float),xy,w)>.99
    good=valid&(strength*max(w['gx'],w['gy'])>=.025)&(alignment>=.866)
    # The middle of each quadrant distinguishes a curve from four straight
    # walls. Axis-adjacent samples alone cannot establish circularity.
    diagonal=np.abs(np.sin(2*ang))>=np.sqrt(.5)-1e-9
    quarters=[float(good[diagonal&(np.arange(72)//18==k)].mean()) for k in range(4)]
    return dict(diagonal_support=float(good[diagonal].mean()),
        quadrant_support=quarters,quadrants_supported=sum(v>=.60 for v in quarters),
        whole_support=float(good.mean()),median_normal_alignment=float(np.median(alignment)))

def circles(g,raw,alts,neighbors,w,t):
    _,audit=D.circles(g,raw,alts,neighbors,w,t)
    audit['reason']='no_observed_curvature_support';audit['accepted']=False
    audit.pop('selected',None)
    attempts=audit.get('attempts',[])
    if not any(r['accepted'] for r in attempts):return None,audit
    gray=gaussian_filter(w['rgb'].mean(0).astype(float),.65)
    dy,dx=np.gradient(gray,w['gy'],w['gx']);gradients=(dx,dy)
    fixed_rectangles=[g.minimum_rotated_rectangle,raw.minimum_rotated_rectangle]
    fixed=max(boundary(rect,w)['rgb'] for rect in fixed_rectangles)
    best=None
    for r in attempts:
        r['first_pass_accepted']=r['accepted'];r['accepted']=False
        if not r['first_pass_accepted']:continue
        p=np.array(r['parameters']);q=primitive(p);cv=curvature(p,w,gradients)
        competitor=max(fixed,boundary(q.minimum_rotated_rectangle,w)['rgb'])
        reasons=[]
        if 2*p[3]/max(w['gx'],w['gy'])<8:reasons.append('curvature_below_reliable_pixel_scale')
        if cv['diagonal_support']<.70 or cv['quadrants_supported']<3:reasons.append('missing_multiquadrant_curved_edge_directions')
        if r['arc']['rgb_score']<competitor+.025:reasons.append('rectangle_evidence_not_outperformed')
        if r['arc']['sector_fraction']<.70:reasons.append('incomplete_visible_arc')
        r.update(curvature=cv,best_rectangle_RGB=competitor,curvature_rejection=reasons,accepted=not reasons)
        score=r['arc']['rgb_score']+cv['diagonal_support']*.10+max(0,r['model_iou']-r['rectangle_model_iou'])
        if not reasons and (best is None or score>best[0]):best=(score,q,r)
    if best:
        q=best[1].difference(G.holes_of(g))
        if q.is_valid and q.geom_type=='Polygon':
            audit.update(accepted=True,reason='observed_curvature_'+best[2]['kind'],selected=best[2]);return q,audit
    return None,audit
