import numpy as np
import shapely
from shapely.geometry import Polygon

def parts(g):return list(g.geoms) if g.geom_type=='MultiPolygon' else [g]

def holes(g):return shapely.union_all([Polygon(r) for p in parts(g) for r in p.interiors])

def axis(g):
    coords=np.asarray(g.minimum_rotated_rectangle.exterior.coords)[:4];edges=np.roll(coords,-1,axis=0)-coords
    lengths=np.linalg.norm(edges,axis=1);i=int(np.argmax(lengths));aspect=float(lengths.max()/max(lengths.min(),1e-12))
    return float(np.degrees(np.arctan2(edges[i,1],edges[i,0]))%180),aspect

def orthogonality(g):
    values=[];weights=[]
    for p in parts(g):
        c=np.asarray(p.exterior.coords);e=np.diff(c,axis=0);l=np.linalg.norm(e,axis=1);ok=l>1e-8
        values.extend(np.exp(4j*np.arctan2(e[ok,1],e[ok,0])));weights.extend(l[ok])
    return float(abs(np.sum(np.array(values)*weights))/sum(weights)) if weights else np.nan

def classify(f,angle=5,area=.05):
    if not f['modified']:return 'unchanged'
    if f['circle_excluded']:return 'circle_ellipse_fit'
    if f['hole_count_before']!=f['hole_count_after'] or f['hole_symmetric_difference_m2']>=5:return 'hole_geometry_change'
    if f['components_before']!=f['components_after']:return 'component_topology_change'
    if f['axis_reliable'] and f['long_axis_change_deg']>=angle:return 'long_axis_change'
    if f['fractional_area_change']>=area:return 'net_area_expansion'
    if f['fractional_area_change']<=-area:return 'net_area_contraction'
    return 'near_constant_area_edit'

def features(key,a,b,meta):
    ta,ea=axis(a);tb,eb=axis(b);ha,hb=holes(a),holes(b)
    f=dict(key=key,source=meta.source,parent=meta.parent,patch_id=meta.patch_id,modified=bool(meta.modified),circle_excluded=bool(meta.circle_excluded),
      pipeline_hollow=bool(meta.pipeline_hollow),pipeline_small_FP=bool(meta.pipeline_small_FP),pipeline_decision=meta.pipeline_decision,
      area_before_m2=a.area,area_after_m2=b.area,fractional_area_change=b.area/a.area-1,
      added_area_m2=b.difference(a).area,removed_area_m2=a.difference(b).area,
      centroid_displacement_m=a.centroid.distance(b.centroid),long_axis_before_deg=ta,long_axis_after_deg=tb,
      long_axis_change_deg=abs((tb-ta+90)%180-90),elongation_before=ea,elongation_after=eb,axis_reliable=bool(min(ea,eb)>=1.3),
      rectangularity_before=a.area/a.minimum_rotated_rectangle.area,
      hole_count_before=sum(len(p.interiors) for p in parts(a)),hole_count_after=sum(len(p.interiors) for p in parts(b)),
      hole_symmetric_difference_m2=ha.symmetric_difference(hb).area,components_before=len(parts(a)),components_after=len(parts(b)),
      exterior_vertices_before=sum(len(p.exterior.coords)-1 for p in parts(a)),exterior_vertices_after=sum(len(p.exterior.coords)-1 for p in parts(b)),
      orthogonality_before=orthogonality(a),orthogonality_after=orthogonality(b))
    f['primary_class']=classify(f)
    f['size_bin']= '<50' if a.area<50 else '50–100' if a.area<100 else '100–250' if a.area<250 else '250–500' if a.area<500 else '≥500'
    f['shape_bin']='near_rectangular' if f['rectangularity_before']>=.85 else 'complex_outline'
    f['elongation_bin']='compact_<1.5' if ea<1.5 else 'moderate_1.5–3' if ea<3 else 'elongated_≥3'
    move=f['centroid_displacement_m'];f['centroid_shift_bin']='<1m' if move<1 else '1–2m' if move<2 else '≥2m'
    f['orientation_bin']='axis_ambiguous' if not f['axis_reliable'] else '<5°' if f['long_axis_change_deg']<5 else '5–15°' if f['long_axis_change_deg']<15 else '≥15°'
    f['net_area_bin']='expansion_≥5%' if f['fractional_area_change']>=.05 else 'contraction_≥5%' if f['fractional_area_change']<=-.05 else 'within_±5%'
    f['orientation_tag']=bool(f['modified'] and f['axis_reliable'] and f['long_axis_change_deg']>=5)
    f['addition_tag']=bool(f['modified'] and f['added_area_m2']>=max(10,.05*a.area))
    f['removal_tag']=bool(f['modified'] and f['removed_area_m2']>=max(10,.05*a.area))
    return f
