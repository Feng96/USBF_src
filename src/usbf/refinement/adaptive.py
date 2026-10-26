import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon, Point
from shapely import affinity
from shapely.ops import unary_union
from . import guided as G, raster_rules as R

def clean_ring(coords, corridor=1.25):
    """Collapse repeated alternating short steps, not all short edges/corners.

    A diagonal raster staircase has alternating signed turns, short edges, and
    stays in a narrow corridor around one long edge. Real large bays fail this
    corridor test; a genuine L corner fails the alternating-turn test.
    """
    xy = np.asarray(coords, float)[:-1]
    removed_runs = 0
    for _ in range(12):
        changed = False
        n = len(xy)
        if n < 6:
            break
        for start in range(n):
            rotated = np.roll(xy, -start, axis=0)
            best = None
            for end in range(4, min(len(rotated)-2, 64)):
                points = rotated[:end+1]
                vec = np.diff(points, axis=0)
                lengths = np.linalg.norm(vec, axis=1)
                if (lengths <= 2.5).mean() < .75:
                    continue
                turn = vec[:-1, 0]*vec[1:, 1] - vec[:-1, 1]*vec[1:, 0]
                turn = np.sign(turn[np.abs(turn) > 1e-6])
                if len(turn) < 3 or (turn[1:]*turn[:-1] < 0).mean() < .75:
                    continue
                chord = points[-1]-points[0]
                length = np.linalg.norm(chord)
                if length < 4:
                    continue
                distances = np.abs((points[:,0]-points[0,0])*chord[1] -
                                   (points[:,1]-points[0,1])*chord[0]) / length
                if distances.max() <= corridor and lengths.sum()/length >= 1.08:
                    best = end
            if best is not None:
                trial = np.vstack([rotated[0], rotated[best:]])
                p = Polygon(trial)
                if p.is_valid and not p.is_empty:
                    xy, changed = trial, True
                    removed_runs += 1
                    break
        if not changed:
            break
    return np.vstack([xy, xy[0]]), removed_runs

def dejag(g):
    if g.geom_type != 'Polygon':
        return g, 0
    xy, runs = clean_ring(g.exterior.coords)
    result = Polygon(xy, [r.coords for r in g.interiors])
    if not result.is_valid or G.poly_iou(result, g) < .85:
        return g, 0
    return result, runs

def angle_distance(a, b):
    return abs((a-b+np.pi/4) % (np.pi/2)-np.pi/4)

def robust_axis(g):
    """Long contour segments after several metric simplifications; no pixel vote."""
    votes, weights = [], []
    clean, _ = dejag(g)
    for tolerance in (.8, 1.25, 1.75):
        q = clean.simplify(tolerance, preserve_topology=True)
        if q.geom_type != 'Polygon':
            continue
        v = np.diff(np.asarray(q.exterior.coords), axis=0)
        length = np.linalg.norm(v, axis=1)
        take = length >= max(4., .15*length.max())
        votes.extend(np.arctan2(v[take,1], v[take,0]))
        weights.extend(length[take])
    if not votes:
        return 0., 0.
    vector = np.sum(np.asarray(weights)*np.exp(4j*np.asarray(votes)))
    return float((np.angle(vector)/4) % (np.pi/2)), float(abs(vector)/sum(weights))

def fit_axis(g, axis):
    """Same run-removal and ordered-line fit for every orientation hypothesis."""
    clean, runs = dejag(g)
    seed = clean.simplify(.8, preserve_topology=True)
    xy = np.asarray(seed.exterior.coords)
    segments = [G.BoundarySegment(np.linspace(a,b,max(3,int(np.linalg.norm(b-a)/.5)+1)))
                for a,b in zip(xy[:-1],xy[1:]) if np.linalg.norm(b-a) > .1]
    segments = G.regularize_segments(segments, [axis, (axis+np.pi/2) % np.pi], max_error=1.5)
    segments = G.merge_segments(segments, angle_epsilon=np.deg2rad(5),max_distance=1.25,max_error=1.5)
    if len(segments)<3:
        return clean, dict(runs=runs, fallback=True)
    result = Polygon(G.compute_intersections(segments, perp_dist_weight=3))
    if (not result.is_valid or result.is_empty or G.poly_iou(result, Polygon(g.exterior)) < .70 or
            result.boundary.hausdorff_distance(Polygon(g.exterior).boundary) > 3):
        return clean, dict(runs=runs, fallback=True)
    result = result.difference(G.holes_of(g))
    if result.geom_type != 'Polygon':
        return clean, dict(runs=runs, fallback=True)
    return result, dict(runs=runs, fallback=False)

def sample(array, xy, gx, gy):
    xy = np.asarray(xy)
    return ndi.map_coordinates(array, [xy[:,1]/gy-.5, xy[:,0]/gx-.5], order=1,
                               mode='constant',cval=np.nan)

def contour_score(g, w, t):
    """Compare on original RGB/probabilities, never on GBA IoU or vertex count."""
    p = w['prob'][1]
    mask = R.burn(g,p.shape,t)
    if not mask.any():
        return dict(score=-1., rgb_edge=0., probability_edge=0., model_inside=0.)
    rgb = np.mean(w['rgb'],axis=0)
    values, contrasts = [], []
    for poly in G.parts(g):
        n = max(32,min(512,int(poly.exterior.length/.5)))
        xy = G.sampled_ring(poly,n)
        tangent = np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0)
        normal = np.column_stack([-tangent[:,1],tangent[:,0]])
        normal /= np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
        sign = np.array([1 if poly.contains(Point(q)) else -1 for q in xy+.3*normal])
        inward = normal*sign[:,None]
        inner, outer = xy+1.5*inward, xy-1.5*inward
        values.extend(sample(p,inner,w['gx'],w['gy'])-sample(p,outer,w['gx'],w['gy']))
        contrasts.extend(np.abs(sample(rgb,inner,w['gx'],w['gy'])-sample(rgb,outer,w['gx'],w['gy'])))
    pe = float(np.nanmean(values)); re = float(np.nanmean(np.clip(contrasts,0,.35))/.35)
    coverage = float(np.nanmean(p[mask]))
    score = .50*pe+.30*re+.20*coverage
    return dict(score=score if np.isfinite(score) else -1., rgb_edge=re,
                probability_edge=pe, model_inside=coverage)

def shift_probe(g, w, t, signal):
    """Anchor-level shift vote; evaluation in source raster, no interpolation write."""
    out = []
    for dy in (-2.,-1.,0.,1.,2.):
        for dx in (-2.,-1.,0.,1.,2.):
            q = affinity.translate(g,dx,dy)
            inside = R.burn(q.buffer(-1),signal.shape,t)
            outside = R.burn(q.buffer(2).difference(q),signal.shape,t)
            vi, vo = signal[inside],signal[outside]
            if len(vi)<10 or len(vo)<10 or np.isfinite(vi).mean()<.95 or np.isfinite(vo).mean()<.95:
                continue
            score = float(np.nanmean(vi)-np.nanmean(vo))
            out.append((score,dx,dy))
    if len(out)<20:
        return None
    zero = next(s for s,x,y in out if x==0 and y==0)
    best = max(out,key=lambda v:(v[0],-abs(v[1])-abs(v[2])))
    # Flat scores mean unidentifiable position, not an arbitrary winning shift.
    if best[0]-zero < .025:
        return (0.,0.) if zero>.35 else None
    return (best[1],best[2])

def aggregate_shifts(votes):
    votes = [v for v in votes if v is not None]
    if len(votes)<5:
        return dict(reliable=False,anchors=len(votes),dx=0.,dy=0.,reason='insufficient_anchors')
    v=np.asarray(votes); median=np.median(v,axis=0)
    dist=np.linalg.norm(v-median,axis=1)
    reliable = float(np.median(dist))<=.75 and float((dist<=1).mean())>=.7 and np.max(np.abs(median))<2
    return dict(reliable=bool(reliable),anchors=len(votes),dx=float(median[0]) if reliable else 0.,
                dy=float(median[1]) if reliable else 0., MAD_m=float(np.median(dist)),
                reason='consistent_local_shift' if reliable else 'inconsistent_or_search_edge')

def support(q, contour, gba, w, t, height_shift=(0.,0.)):
    """Three-state local support, eroded core and nine fixed shift scenarios.

    A coreless strip can be UNKNOWN, never rejected just for being unresolved.
    Whole-main-roof height never substitutes for added-region evidence.
    """
    full=R.burn(q,w['signed'].shape,t)
    p0,p1=w['prob']
    if not full.any():
        return 'uncertain',dict(reason='underresolved')
    local_model=float((p1[full]>=.5).mean())
    high_model=float((p1[full]>=.65).mean())
    dual=float((p0[full]>=.5).mean())
    model_stable=local_model>=.85 and high_model>=.65 and dual>=.7
    tests=[]
    for radius in (1.,2.):
        core=q.buffer(-radius)
        mask=R.burn(core,w['signed'].shape,t)
        if mask.sum()<4:
            tests.append(dict(radius_m=radius,verdict='unresolved',pixels=int(mask.sum())))
            continue
        for dx,dy in ((0,0),(-radius,0),(radius,0),(0,-radius),(0,radius)):
            shifted=affinity.translate(core,dx+height_shift[0],dy+height_shift[1])
            m=R.burn(shifted,w['signed'].shape,t)
            s=R.stats(m,w['signed'],w['height_valid'])
            # Require usable local data; interpolation validity is not point density.
            valid=s['valid_fraction']>=.9 and s['p25'] is not None and s['negative_fraction']<=.05
            verdict='high' if valid and s['p25']>=2 else ('low' if valid and s['p95']<1 else 'unknown')
            tests.append(dict(radius_m=radius,dx=dx,dy=dy,verdict=verdict,
                              pixels=s['n'],valid=s['valid_fraction'],p25=s['p25'],p95=s['p95']))
    verdicts=[x['verdict'] for x in tests]
    high=verdicts.count('high')/len(verdicts); low=verdicts.count('low')/len(verdicts)
    # Green is a caution, NOT a tree classifier or universal negative gate.
    rgb=w['rgb']; green=(rgb[1]>1.15*rgb[0])&(rgb[1]>1.1*rgb[2])
    vegetation=float(green[full].mean())
    facts=dict(area_m2=float(q.area),B1_p50=local_model,B1_p65=high_model,B0_p50=dual,
        green_fraction=vegetation,height_high_scenario_fraction=high,height_low_scenario_fraction=low,
        scenarios=tests,native_height_resolution_verified=False)
    if local_model<.35 and dual<.35:
        return 'opposed',dict(**facts,reason='both_models_disagree_locally')
    if low==1 and not model_stable:
        return 'opposed',dict(**facts,reason='low_height_all_scenarios_and_weak_model')
    if model_stable and high>=.8 and vegetation<.25:
        return 'supported',dict(**facts,reason='stable_local_model_and_height')
    reason='height_boundary_or_alignment_uncertain'
    if vegetation>=.25: reason='vegetation_ambiguous_not_proven_tree'
    if low==1 and model_stable: reason='RGB_model_height_conflict_or_temporal_change'
    return 'uncertain',dict(**facts,reason=reason)
