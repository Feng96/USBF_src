import numpy as np
from shapely.geometry import Polygon, LineString, Point
from scipy import ndimage as ndi
from . import guided as G, adaptive as V

RULES = dict(oblique_min_length_m=8., oblique_min_angle_deg=10.,
    oblique_min_lateral_drift_m=3., oblique_coverage_min=.75,
    oblique_line_vs_step_error_ratio_max=.85, oblique_thresholds=[.35,.65],
    oblique_RGB_coherence_min=.75, oblique_RGB_aligned_fraction_min=.65,
    wall_sample_spacing_m=.6, wall_line_penalty=4., shallow_offset_pixels=1.5,
    GBA_retention='explicit contrary evidence or geometry/context failure, not a positive gain margin',
    primitive_policy='adopt all previously qualified circle/ellipse and strict hollow proposals',
    independent_GT=False, production_edits=False)

def dense(g, spacing=.4):
    return G.sampled_ring(Polygon(g.exterior), max(24, min(1600,int(g.exterior.length/spacing))))

def image_wall_evidence(a,b,w,axis):
    """Local RGB gradient directions, not another threshold of the same model."""
    u=(b-a)/np.linalg.norm(b-a);n=np.array([-u[1],u[0]])
    angle=float(np.arctan2(u[1],u[0]));gs=max(w['gx'],w['gy'])
    gray=np.mean(w['rgb'],axis=0)
    gy,gx=np.gradient(ndi.gaussian_filter(gray,.65),w['gy'],w['gx'])
    xy=a+np.linspace(.1,.9,max(12,int(np.linalg.norm(b-a)/gs)))[:,None]*(b-a)
    offsets=np.linspace(-1.5*gs,1.5*gs,7)
    locations=xy[:,None,:]+offsets[None,:,None]*n
    dx=V.sample(gx,locations.reshape(-1,2),w['gx'],w['gy']).reshape(len(xy),-1)
    dy=V.sample(gy,locations.reshape(-1,2),w['gx'],w['gy']).reshape(len(xy),-1)
    magnitude=np.hypot(dx,dy);tangent=np.arctan2(dy,dx)+np.pi/2
    # Exclude perpendicular gradients: squared cos alone would admit both.
    agree=np.maximum(0,np.cos(2*(tangent-angle)))
    primary=np.maximum(0,np.cos(4*(tangent-axis)))
    best=np.argmax(np.nan_to_num(magnitude,nan=-1.),axis=1);idx=np.arange(len(xy))
    strength=magnitude[idx,best];coherence=agree[idx,best]
    available=np.isfinite(strength)&(strength>.015/gs)
    return dict(rgb_valid_edge_fraction=float(available.mean()),
                rgb_oblique_coherence=float(np.average(coherence[available],weights=strength[available])) if available.any() else 0.,
                rgb_axis_coherence=float(np.average(primary[idx,best][available],weights=strength[available])) if available.any() else 0.,
                rgb_direction_aligned_fraction=float((coherence[available]>.75).mean()) if available.any() else 0.)

def wall_observation(a,b,raw,axis,gs):
    """Compare a ramp to one offset step on the original, raster-staircase wall.

    An endpoint chord over a shallow offset is not a straight-wall observation.
    Bin medians suppress pixel quantization, not metre-scale building features.
    """
    length=float(np.linalg.norm(b-a));u=(b-a)/max(length,1e-9);n=np.array([-u[1],u[0]])
    pts=dense(raw);along=(pts-a)@u;off=(pts-a)@n
    take=(along>=.05*length)&(along<=.95*length)&(np.abs(off)<=2.25*gs)
    pts=pts[take]
    if len(pts)<8:return dict(ok=False,reason='insufficient_raw_wall',coverage=0.)
    # Regress transverse position against the nearest main building axis.
    k=round((np.arctan2(u[1],u[0])-axis)/(np.pi/2))
    main=np.array([np.cos(axis+k*np.pi/2),np.sin(axis+k*np.pi/2)])
    normal=np.array([-main[1],main[0]])
    x=(pts-a)@main;y=(pts-a)@normal
    bounds=np.linspace(float(x.min()),float(x.max()),13)
    bins=np.digitize(x,bounds[1:-1]);xx=[];yy=[]
    for i in range(12):
        if (bins==i).sum()>=1:
            xx.append(float(np.median(x[bins==i])));yy.append(float(np.median(y[bins==i])))
    coverage=len(xx)/12
    if len(xx)<8:return dict(ok=False,reason='fragmented_raw_wall',coverage=coverage)
    x=np.asarray(xx);y=np.asarray(yy);design=np.column_stack([x,np.ones(len(x))])
    slope,intercept=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y-design@np.array([slope,intercept]);mse=float(np.mean(residual**2))
    step=min(float(np.mean(np.r_[y[:i]-np.mean(y[:i]),y[i:]-np.mean(y[i:])]**2)) for i in range(2,len(y)-1))
    ratio=mse/max(step,1e-6)
    drift=abs(float(slope*(x.max()-x.min())))
    theta=float(np.arctan2(main[1]+slope*normal[1],main[0]+slope*normal[0]))
    ok=coverage>=.75 and np.sqrt(mse)<=.70*gs and ratio<=.85 and drift>=2.4*gs
    return dict(ok=bool(ok),reason='continuous_ramp' if ok else 'step_or_unstable_wall',
        coverage=coverage,line_rmse_m=float(np.sqrt(mse)),line_step_ratio=ratio,
        observed_drift_m=drift,fitted_angle_deg=float(np.rad2deg(theta)),angle=theta)

def oblique_evidence(seed,raw,alternatives,axis,gs,w=None):
    approved=[];records=[]
    xy=np.asarray(seed.exterior.coords)
    for a,b in zip(xy[:-1],xy[1:]):
        length=float(np.linalg.norm(b-a));angle=float(np.arctan2(*(b-a)[::-1]))
        deviation=V.angle_distance(angle,axis)
        if deviation<np.deg2rad(10):continue
        rec=dict(a=a.tolist(),b=b.tolist(),length_m=length,axis_deviation_deg=float(np.rad2deg(deviation)))
        if length<max(8.,8*gs) or length*np.sin(deviation)<max(3.,3*gs):
            rec.update(approved=False,reason='short_span_or_noise_scale_drift');records.append(rec);continue
        observed=wall_observation(a,b,raw,axis,gs)
        checks=[wall_observation(a,b,q,axis,gs) for q in alternatives if not q.is_empty]
        stable=len(checks)==2 and all(q['ok'] and V.angle_distance(q['angle'],observed.get('angle',angle))<=np.deg2rad(8) for q in checks)
        image=image_wall_evidence(a,b,w,axis) if w is not None else None
        image_good=(image is None or (image['rgb_valid_edge_fraction']>=.75 and
                    image['rgb_oblique_coherence']>=.75 and image['rgb_direction_aligned_fraction']>=.65))
        good=observed['ok'] and stable and image_good
        rec.update(approved=bool(good),reason='raw_long_wall_threshold_stability_RGB_direction' if good else 'raw_threshold_or_RGB_direction_not_supported',
                   RGB=image,
                   raw=observed,thresholds=checks)
        if good:approved.append(dict(a=a,b=b,angle=observed['angle']))
        records.append(rec)
    return approved,records

def line_intersections(lines):
    pts=[]
    for old,new in zip(lines[-1:]+lines[:-1],lines):
        mat=np.stack([old['normal'],new['normal']]);det=np.linalg.det(mat)
        if abs(det)<1e-5:return None
        pts.append(np.linalg.solve(mat,np.array([old['offset'],new['offset']])))
    return np.asarray(pts)

def merge_shallow(lines,gs):
    """Merge a shallow jog/U-notch into its surrounding parallel wall.

    Every original observation in the interval must remain in a 1.5-pixel wall
    band. Large recesses and wings fail this test irrespective of vertex count.
    """
    removed=0
    for _ in range(30):
        changed=False
        for i in range(len(lines)):
            seq=lines[i:]+lines[:i]
            for span in (3,5):
                if len(seq)-span+1<4:continue
                group=seq[:span]
                if abs(np.dot(group[0]['normal'],group[-1]['normal']))<.9999:continue
                if any(q.get('oblique',False) for q in group):continue
                n=group[0]['normal'];points=np.vstack([q['points'] for q in group]);d=points@n
                # Pixel staircase itself consumes ~one pixel of this band.
                if np.quantile(d,.98)-np.quantile(d,.02)>3.*gs:continue
                start=float(np.median(group[0]['points']@n));end=float(np.median(group[-1]['points']@n))
                if abs(start-end)>1.5*gs:continue
                mids=np.concatenate([q['points']@n for q in group[1:-1]])
                if np.max(np.abs(mids-(start+end)/2))>2.5*gs:continue
                u=np.array([-n[1],n[0]])
                if np.ptp(points@u)<4*gs:continue
                merged=dict(normal=n,offset=float(np.mean(d)),points=points,oblique=False)
                trial=[merged]+seq[span:];p=line_intersections(trial)
                if p is None or not Polygon(p).is_valid:continue
                lines=trial;removed+=span-1;changed=True;break
            if changed:break
        if not changed:break
    return lines,removed

def segment_walls(raw,axis,approved,gs,phase=0):
    """Dynamic programming on ordered raw samples with constrained line normals."""
    spacing=max(.6,raw.length/700)
    points=dense(raw,spacing)
    # Start at an extreme in the building frame, avoiding arbitrary raster start.
    direction=np.array([np.cos(axis+np.pi/4+phase*np.pi/2),np.sin(axis+np.pi/4+phase*np.pi/2)])
    start=int(np.argmax(points@direction));points=np.roll(points,-start,axis=0)
    points=np.vstack([points,points[0]]);N=len(points)
    dirs=[(axis,False,None),(axis+np.pi/2,False,None)]+[(q['angle'],True,q) for q in approved]
    projections=[]
    for theta,oblique,region in dirs:
        normal=np.array([-np.sin(theta),np.cos(theta)]);z=points@normal
        sums=np.r_[0,np.cumsum(z)];squares=np.r_[0,np.cumsum(z*z)]
        allowed=np.ones(N,dtype=bool)
        if oblique:
            a,b=region['a'],region['b'];u=(b-a)/np.linalg.norm(b-a)
            allowed=(((points-a)@u)>=-2*gs)&(((points-b)@u)<=2*gs)&(np.abs((points-a)@np.array([-u[1],u[0]]))<=3*gs)
        bad=np.r_[0,np.cumsum(~allowed)]
        projections.append((normal,sums,squares,bad,oblique))
    dp=np.full(N+1,np.inf);dp[0]=0.;previous=np.full(N+1,-1,int);which=np.full(N+1,-1,int)
    penalty=4.*gs**3
    for end in range(3,N+1):
        starts=np.arange(max(0,end-450),end-2)
        count=end-starts
        for k,(normal,sums,squares,bad,oblique) in enumerate(projections):
            s=sums[end]-sums[starts];ss=squares[end]-squares[starts]
            error=np.maximum(0,ss-s*s/count)
            ok=(error/count<=.85**2*gs**2)&(bad[end]-bad[starts]==0)
            # Exclude an entire loop or a folded run with negligible chord span.
            delta=points[end-1]-points[starts]
            chord=np.linalg.norm(delta,axis=1)
            ok&=chord>=.5*(count-1)*spacing
            cost=dp[starts]+error*spacing+penalty
            cost[~ok]=np.inf
            j=int(np.argmin(cost))
            if cost[j]<dp[end]:dp[end]=cost[j];previous[end]=starts[j];which[end]=k
    if not np.isfinite(dp[N]):return None,dict(reason='no_ordered_partition')
    segments=[];end=N
    while end>0:
        begin=previous[end];normal,_,_,_,oblique=projections[which[end]]
        pts=points[begin:end]
        segments.append(dict(normal=normal,offset=float(np.mean(pts@normal)),points=pts,oblique=oblique))
        end=begin
    lines=list(reversed(segments))
    # Adjacent parallel runs on the same wall are one line, including closure.
    for _ in range(len(lines)):
        changed=False
        for i in range(len(lines)):
            j=(i+1)%len(lines)
            if abs(np.dot(lines[i]['normal'],lines[j]['normal']))>.9999:
                pts=np.vstack([lines[i]['points'],lines[j]['points']]);n=lines[i]['normal']
                q=dict(normal=n,offset=float(np.mean(pts@n)),points=pts,oblique=lines[i]['oblique'] or lines[j]['oblique'])
                if j==0:lines=[q]+lines[1:i]
                else:lines=lines[:i]+[q]+lines[j+1:]
                changed=True;break
        if not changed:break
    before=len(lines);lines,removed=merge_shallow(lines,gs)
    xy=line_intersections(lines) if len(lines)>=3 else None
    if xy is None:return None,dict(reason='parallel_intersection')
    result=Polygon(xy,[r.coords for r in raw.interiors])
    if not result.is_valid or result.is_empty:return None,dict(reason='invalid_intersection')
    shell=Polygon(raw.exterior)
    iou=G.poly_iou(Polygon(result.exterior),shell);hd=result.exterior.hausdorff_distance(raw.exterior)
    boundary_dist=np.array([raw.exterior.distance(Point(p)) for p in dense(result,.6)])
    p95=float(np.quantile(boundary_dist,.95));invented=float(result.difference(shell).area/result.area)
    # A re-intersected wall corner can be farther than its observed straight
    # walls. Bound it separately, without discarding all fits for one corner.
    safe_corner=(hd<=5*gs and p95<=2.25*gs and invented<=.15 and iou>=.75)
    if iou<.70 or (hd>max(3.,3*gs) and not safe_corner):
        return None,dict(reason='excessive_wall_displacement',raw_iou=iou,raw_hausdorff_m=hd,
                         boundary_p95_m=p95,invented_fraction=invented,
                         diagnostic_coordinates=xy.tolist(),segments_before_merge=before,segments_final=len(lines))
    return result,dict(reason='ordered_supported_walls',segments_before_merge=before,
                       shallow_segments_removed=removed,segments_final=len(lines),partition_cost=float(dp[N]),
                       oblique_segments_final=sum(q['oblique'] for q in lines),raw_iou=iou,raw_hausdorff_m=hd,
                       boundary_p95_m=p95,invented_fraction=invented)

def regularize(seed,raw,alternatives,axis,gs,w=None,t=None):
    approved,evidence=oblique_evidence(seed,raw,alternatives,axis,gs,w)
    options=[];attempts=[]
    for phase in range(4):
        q,info=segment_walls(raw,axis,approved,gs,phase)
        attempts.append(info)
        if q is not None:
            score=V.contour_score(q,w,t)['score'] if w is not None else G.poly_iou(q,raw)
            options.append((score,q,info))
    if not options:
        return seed,dict(fallback=True,oblique_evidence=evidence,attempts=attempts,reason='no_safe_wall_fit')
    _,q,info=max(options,key=lambda x:x[0])
    return q,dict(fallback=False,oblique_evidence=evidence,attempts=attempts,**info)

def special_shapes(base,circle,holes):
    """No generic score or GBA similarity veto of already qualified primitives."""
    q=circle if circle is not None else base
    for hole in holes:q=q.difference(hole)
    return q,dict(shape_changed=circle is not None,hollow_cut=bool(holes))
