"""Compare newly generated geometries with the exact historical example."""
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio,shapely
from usbf.io.common import read,save

# This is a numerical comparison tolerance, not a generation/evaluation rule.
# 0.1 micrometre accommodates projected-coordinate floating-point roundoff
# across libm/PROJ builds while remaining 5,000,000 times finer than scoring.
ROUND_OFF_METRES = 1e-7

def geometry_check(am,bm):
    """Require matching vertices/rings and sub-micrometre coordinate error."""
    area=float(am.symmetric_difference(bm).area)
    distance=float(am.hausdorff_distance(bm))
    area_limit=ROUND_OFF_METRES*(am.length+bm.length)
    same_structure=am.geom_type==bm.geom_type and am.equals_exact(bm,ROUND_OFF_METRES)
    accepted=bool(am.is_valid and bm.is_valid and not am.is_empty and not bm.is_empty
                  and same_structure and distance<=ROUND_OFF_METRES and area<=area_limit)
    return dict(numerically_equivalent=accepted,symmetric_difference_m2=area,
                hausdorff_distance_m=distance,area_roundoff_limit_m2=float(area_limit))

def compare(data_dir,product,output):
    data_dir=Path(data_dir);product=Path(product)
    actual=gpd.read_parquet(product).set_index('key')
    expected=gpd.read_parquet(data_dir/'example/regression/expected_final.parquet')
    if 'key' in expected:expected=expected.set_index('key')
    if not actual.index.is_unique or not expected.index.is_unique or set(actual.index)!=set(expected.index):
        raise ValueError('Regression requires exactly the complete fixed example identifiers')
    if actual.crs!=expected.crs:raise ValueError('Regression source CRS differs')
    expected=expected.loc[actual.index];rows=[]
    for key,r in actual.iterrows():
        a,b=r.geometry,expected.loc[key].geometry
        am,bm=gpd.GeoSeries([a,b],crs=actual.crs).to_crs(5070)
        expected_modified=not b.equals_exact(r.gba_geometry,0)
        rows.append(dict(key=key,wkb_exact=a.wkb==b.wkb,geometry_exact=a.equals_exact(b,0),
            **geometry_check(am,bm),modification_status_matches=bool(r.modified)==expected_modified,
            modified=bool(r.modified),circle_gate_status=r.circle_gate_status))
    probabilities=[]
    folder=product.parent/'probabilities'
    for p in sorted(folder.glob('*.tif')):
        old=data_dir/'example/regression/probabilities'/p.name
        with rasterio.open(p) as a,rasterio.open(old) as b:
            x,y=a.read(),b.read();valid=a.dataset_mask()>0;error=np.abs(x[:,valid]-y[:,valid])
            probabilities.append(dict(frame=p.stem,max_absolute_difference=float(error.max()),
                mean_absolute_difference=float(error.mean()),threshold_crossings={str(t):int(((x>=t)!=(y>=t))[:,valid].sum()) for t in (.25,.35,.45,.5,.65)}))
    result=dict(status='passed' if all(r['numerically_equivalent'] and r['modification_status_matches'] for r in rows) else 'geometry_regression_difference',
        comparison='same real inputs, frozen weights and rule definitions; historical GPU/software outputs versus selected execution backend',
        objects=rows,probability_comparisons=probabilities,maximum_symmetric_difference_m2=max(r['symmetric_difference_m2'] for r in rows),
        maximum_hausdorff_distance_m=max(r['hausdorff_distance_m'] for r in rows),
        all_geometry_exact=all(r['geometry_exact'] for r in rows),
        geometry_comparison_tolerance_m=ROUND_OFF_METRES,
        comparison_policy='exactness reported separately; same vertex/ring structure and modification status required; numerical roundoff only',
        scientific_generation_or_scoring_thresholds_changed=False)
    save(output,result);return result
