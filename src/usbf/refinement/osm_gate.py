from decimal import Decimal
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import affinity
from skimage.measure import EllipseModel, CircleModel

RULES=dict(version='OSM_circle_confirmation_pilot_v1_20260917',
    sample_points=256,min_osm_fit_iou=.90,min_fit_vs_rectangle_gain=.06,
    max_normalized_radial_p95=.06,max_ellipse_axis_ratio=2.5,
    min_candidate_osm_iou=.85,min_iou_gain_over_original=.02,
    candidate_osm_area_ratio=[.80,1.25],max_center_offset_minor_fraction=.12,
    boundary_tolerance_m=[1.,2.],max_boundary_minor_fraction=.12,
    spatial_min_iou=.50,spatial_unique_iou_margin=.15,
    no_translation_fit=True,missing_or_ambiguous_reference='abstain_revert_circle_edit',
    osm_is_not_independent_gt=True,production_enabled=False)

def osm_key(identifier):
    value=Decimal(str(identifier));assert value.is_finite() and value==value.to_integral_value()
    n=int(value);return ('relation' if n<0 else 'way',str(abs(n)))

def iou(a,b):
    u=a.union(b).area
    return float(a.intersection(b).area/u) if u>0 else 0.

def sample_boundary(g,n=256):
    ring=g.exterior
    return np.array([ring.interpolate(i/n,normalized=True).coords[0] for i in range(n)])

def osm_shape(g):
    if g.geom_type=='MultiPolygon' and len(g.geoms)==1:g=g.geoms[0]
    if g.is_empty or not g.is_valid or g.geom_type!='Polygon':
        return dict(valid=False,round_supported=False,reason='unsupported_osm_geometry')
    shell=Polygon(g.exterior);xy=sample_boundary(shell,RULES['sample_points']);origin=xy.mean(0);xy=xy-origin
    rectangle_iou=iou(shell,shell.minimum_rotated_rectangle);fits=[]
    for kind,model in [('circle',CircleModel()),('ellipse',EllipseModel())]:
        if not model.estimate(xy):continue
        if kind=='circle':cx,cy,a=model.params;b=a;theta=0.
        else:cx,cy,a,b,theta=model.params
        if not np.isfinite([cx,cy,a,b,theta]).all() or min(a,b)<=0:continue
        if a<b:a,b,theta=b,a,theta+np.pi/2
        if a/b>RULES['max_ellipse_axis_ratio']:continue
        delta=xy-[cx,cy];ct,st=np.cos(theta),np.sin(theta)
        local=np.column_stack([delta[:,0]*ct+delta[:,1]*st,-delta[:,0]*st+delta[:,1]*ct])
        radial=np.sqrt((local[:,0]/a)**2+(local[:,1]/b)**2)
        radial_p95=float(np.quantile(np.abs(radial-1),.95))
        primitive=affinity.translate(affinity.rotate(affinity.scale(Point(0,0).buffer(1,quad_segs=64),a,b),theta,use_radians=True,origin=(0,0)),cx+origin[0],cy+origin[1])
        fit_iou=iou(shell,primitive);gain=fit_iou-rectangle_iou
        supported=fit_iou>=RULES['min_osm_fit_iou'] and gain>=RULES['min_fit_vs_rectangle_gain'] and radial_p95<=RULES['max_normalized_radial_p95']
        fits.append(dict(kind=kind,fit_iou=fit_iou,rectangle_iou=rectangle_iou,fit_vs_rectangle_gain=gain,
            radial_p95=radial_p95,minor_radius_m=float(b),major_radius_m=float(a),round_supported=bool(supported)))
    if not fits:return dict(valid=False,round_supported=False,reason='no_stable_osm_fit')
    best=max(fits,key=lambda f:(f['round_supported'],f['fit_iou']))
    return dict(valid=True,reason='supported' if best['round_supported'] else 'osm_not_round',**best)

def gate(original,candidate,reference):
    if reference is None:return dict(accepted=False,reason='osm_unconfirmed',osm_shape=None)
    shape=osm_shape(reference)
    result=dict(accepted=False,reason=shape['reason'],osm_shape=shape)
    if not shape['valid']:return result
    ci=iou(candidate,reference);oi=iou(original,reference);ratio=float(candidate.area/reference.area)
    center=float(candidate.centroid.distance(reference.centroid));minor=shape['minor_radius_m']
    tol=max(RULES['boundary_tolerance_m'][0],min(RULES['boundary_tolerance_m'][1],RULES['max_boundary_minor_fraction']*minor))
    def distances(a,b):
        boundary=a.boundary
        return [boundary.interpolate(i/RULES['sample_points'],normalized=True).distance(b.boundary) for i in range(RULES['sample_points'])]
    bp95=float(np.quantile(distances(candidate,reference)+distances(reference,candidate),.95))
    result.update(candidate_osm_iou=ci,original_osm_iou=oi,iou_gain=ci-oi,candidate_osm_area_ratio=ratio,
        center_offset_m=center,boundary_distance_p95_m=bp95,boundary_tolerance_m=tol)
    if not shape['round_supported']:return result
    checks=[('candidate_scale_mismatch',RULES['candidate_osm_area_ratio'][0]<=ratio<=RULES['candidate_osm_area_ratio'][1]),
        ('candidate_position_mismatch',center<=max(1.,RULES['max_center_offset_minor_fraction']*minor)),
        ('candidate_boundary_mismatch',bp95<=tol),('candidate_osm_overlap_low',ci>=RULES['min_candidate_osm_iou']),
        ('no_improvement_over_original_osm',ci-oi>=RULES['min_iou_gain_over_original'])]
    failed=[name for name,ok in checks if not ok]
    result.update(accepted=not failed,reason=failed[0] if failed else 'osm_shape_confirmed',failed_checks=failed)
    return result
