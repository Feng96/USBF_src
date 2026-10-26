"""Synthetic controls only: these tests are not reference-validation samples."""
import numpy as np
import pytest
from shapely.geometry import Point,box
from usbf.refinement.osm_gate import gate,osm_key
from usbf.data.sensors import load_sensor
from usbf.io.common import sha,safe_path

def test_zero_height_is_valid_measurement(tmp_path):
    p=tmp_path/'sensor.npz';np.savez(p,rgb=np.zeros((3,8,8),'float32'),signed=np.zeros((8,8),'float32'),valid=np.ones((8,8)),rgb_valid=np.ones((8,8),bool))
    r=dict(path=str(p),sha256=sha(p),crs='EPSG:32611',transform=[1,0,0,0,-1,8])
    a=load_sensor(r,True);assert a['height_valid'].all();assert np.array_equal(a['signed'],np.zeros((8,8)))

def test_regression_distinguishes_nanometre_roundoff_from_real_displacement():
    from shapely.affinity import translate
    from usbf.evaluation.regression import geometry_check
    a=box(500000,4000000,500020,4000020)
    assert geometry_check(a,translate(a,xoff=5e-9))['numerically_equivalent']
    assert not geometry_check(a,translate(a,xoff=.001))['numerically_equivalent']

def test_regression_rejects_changed_ring_structure_even_for_tiny_holes():
    from shapely.geometry import Polygon
    from usbf.evaluation.regression import geometry_check
    a=box(500000,4000000,500020,4000020)
    b=Polygon(a.exterior,[box(500010,4000010,500010.000001,4000010.000001).exterior])
    assert not geometry_check(a,b)['numerically_equivalent']

def test_regression_refuses_incomplete_fixed_sample(tmp_path):
    import geopandas as gpd
    from usbf.evaluation.regression import compare
    a=gpd.GeoDataFrame(dict(key=['a','b']),geometry=[box(0,0,1,1),box(2,2,3,3)],crs=4326).set_index('key')
    folder=tmp_path/'example/regression';folder.mkdir(parents=True)
    a.to_parquet(folder/'expected_final.parquet')
    a.iloc[:1].reset_index().to_parquet(tmp_path/'product.parquet')
    with pytest.raises(ValueError,match='complete fixed example'):
        compare(tmp_path,tmp_path/'product.parquet',tmp_path/'report.json')

def test_metric_roundoff_preserves_exact_boundary_counts_and_rejects_real_changes():
    import pandas as pd
    from usbf.evaluation.numerical import compare_metric_columns
    expected=pd.DataFrame({'x_intersection_m2':[10.], 'x_union_m2':[25.],
        'x_iou':[.4], 'x_pred_n':[100], 'x_bf1_1m':[.5]})
    actual=expected.copy();actual['x_intersection_m2']+=5e-8
    actual['x_iou']=actual.x_intersection_m2/actual.x_union_m2
    assert compare_metric_columns(actual,expected)['status']=='passed'
    changed=actual.copy();changed['x_pred_n']+=1
    assert compare_metric_columns(changed,expected)['failing_columns']=={'x_pred_n':1}
    changed=actual.copy();changed['x_intersection_m2']+=.001
    assert 'x_intersection_m2' in compare_metric_columns(changed,expected)['failing_columns']
    changed=actual.copy();changed['x_iou']+=.001
    assert 'x_iou' in compare_metric_columns(changed,expected)['failing_columns']

def test_corrupt_input_refused(tmp_path):
    p=tmp_path/'bad.npz';p.write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):load_sensor(dict(path=p,sha256='0'*64),True)

def test_circle_confirmation_positive_is_synthetic():
    circle=Point(0,0).buffer(20,quad_segs=96)
    assert gate(box(-20,-20,20,20),circle,circle)['accepted']

def test_rectangle_reference_rejects_synthetic_circle():
    assert not gate(box(-20,-20,20,20),Point(0,0).buffer(20,quad_segs=96),box(-20,-20,20,20))['accepted']

def test_no_reference_not_confirmed():
    assert not gate(box(-20,-20,20,20),Point(0,0).buffer(20),None)['accepted']

def test_exact_identifier_rules():
    assert osm_key('-123')==('relation','123')
    assert osm_key('123.0')==('way','123')
    with pytest.raises(AssertionError):osm_key('12.5')

def test_asset_path_escape_refused(tmp_path):
    with pytest.raises(ValueError):safe_path(tmp_path,'../outside')

def test_missing_weight_fails_clearly(tmp_path):
    from usbf.models.inference import load_models
    with pytest.raises(FileNotFoundError,match='frozen'):load_models(tmp_path,'cpu')

def test_import_has_no_scheduler_dependency():
    import sys
    assert not any(n.startswith(('pyslurm','scripts.hpc')) for n in sys.modules)

def test_no_matches_has_no_fabricated_zero_scores():
    import pandas as pd,geopandas as gpd
    from usbf.evaluation.replay import score_geometries
    g=gpd.GeoDataFrame(geometry=[],crs=32611)
    result=score_geometries(g,{},g,pd.DataFrame())
    assert result.empty and len(result.columns)==0

def test_hollow_requires_real_low_height_not_mask_dropout():
    from usbf.refinement.raster_rules import measure
    g=box(10,10,90,90);p=np.zeros((100,100),'float32');p[10:90,10:90]=.95;p[40:60,40:60]=.05
    h=np.zeros_like(p);h[10:90,10:90]=8;h[40:60,40:60]=0
    w=dict(g=g,rgb=np.full((3,100,100),.5,'float32'),signed=h,prob=np.stack([p,p]),
           rgb_valid=np.ones_like(p,bool),height_valid=np.ones_like(p,bool),gx=1.,gy=1.)
    a,_=measure(w);assert any(r['family']=='hollow' and 'S' in r['levels'] for r in a)
    w['height_valid'][40:60,40:60]=False
    b,_=measure(w);assert not any(r['family']=='hollow' and 'S' in r['levels'] for r in b)

def test_exact_reference_missing_ambiguous_and_ms_cannot_confirm():
    from usbf.refinement.osm_policy_frozen import References
    refs=object.__new__(References);refs.byid={};refs.ambiguous=set()
    a=box(-115.21,36.20,-115.20,36.21)
    assert refs.assess('osm','123',a,a)['reason']=='no_verified_local_OSM_reference'
    refs.byid[('way','123')]=a;refs.ambiguous.add(('way','123'))
    assert refs.assess('osm','123',a,a)['reason']=='ambiguous_OSM_snapshot_geometry'
    assert refs.assess('ms','123',a,a)['reason']=='not_OSM_source_exact_match_unavailable'

def test_hole_and_original_geometry_survive_geoparquet_readback(tmp_path):
    import geopandas as gpd,shapely
    from shapely.geometry import Polygon
    a=box(0,0,20,20);b=Polygon(a.exterior,[box(5,5,10,10).exterior])
    f=gpd.GeoDataFrame(dict(key=['osm:123'],gba_geometry=gpd.GeoSeries([a],crs=32611)),geometry=[b],crs=32611)
    p=tmp_path/'with-hole.parquet';f.to_parquet(p,index=False);r=gpd.read_parquet(p)
    assert len(r.geometry.iloc[0].interiors)==1 and r.geometry.iloc[0].wkb==b.wkb
    assert r.gba_geometry.iloc[0].wkb==a.wkb

def test_offline_full_inputs_are_actually_verified(tmp_path):
    from usbf.data.download import fetch
    from usbf.io.common import save
    save(tmp_path/'FILES.json',{})
    with pytest.raises(FileNotFoundError):fetch(tmp_path,offline=True,full_inputs=True)
    save(tmp_path/'full/FULL_FILES.json',{})
    assert fetch(tmp_path,offline=True,full_inputs=True)['full']['status']=='passed'

def test_download_extract_verify_and_offline_cache(tmp_path):
    """Local file transport control, not evidence of public download availability."""
    import tarfile
    from usbf.data.download import fetch
    from usbf.io.common import save
    source=tmp_path/'source';source.mkdir();(source/'real-name.txt').write_text('transport fixture')
    spec={'sha256':sha(source/'real-name.txt'),'bytes':(source/'real-name.txt').stat().st_size}
    save(source/'FILES.json',{'real-name.txt':spec})
    archive=tmp_path/'fixture.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(source.iterdir()):t.add(p,arcname=p.name)
    registry=tmp_path/'registry.json'
    save(registry,dict(assets=[dict(name=archive.name,url=archive.as_uri(),destination='downloads/'+archive.name,
        sha256=sha(archive),bytes=archive.stat().st_size,archive='tar.gz')]))
    target=tmp_path/'cache';assert fetch(target,registry)['verification']['files']==1
    assert fetch(target,offline=True)['files']==1
    (target/'real-name.txt').write_text('corrupted')
    with pytest.raises(ValueError,match='checksum'):fetch(target,offline=True)

def test_archive_traversal_is_rejected(tmp_path):
    import tarfile,io
    from usbf.data.download import fetch
    from usbf.io.common import save
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        member=tarfile.TarInfo('../escaped');member.size=1;t.addfile(member,io.BytesIO(b'x'))
    registry=tmp_path/'registry.json'
    save(registry,dict(assets=[dict(name=archive.name,url=archive.as_uri(),destination='downloads/bad.tar.gz',
        sha256=sha(archive),bytes=archive.stat().st_size,archive='tar.gz')]))
    with pytest.raises(ValueError,match='escapes'):fetch(tmp_path/'cache',registry)
    assert not (tmp_path/'escaped').exists()

def test_metrics_use_selected_geometry_column_for_before_confirmation():
    """Inactive final geometry must not replace the selected pre-gate geometry."""
    import geopandas as gpd,pandas as pd
    from usbf.evaluation.replay import score_geometries
    source=gpd.GeoDataFrame(dict(key=['k']),geometry=[box(500000,4000000,500002,4000002)],crs=32611).set_index('key')
    reference=source.copy();reference.index=['r']
    product=source.copy();product['pre_confirmation_geometry']=source.geometry.copy()
    product['geometry']=[box(500000,4000000,500008,4000008)]
    old=product.set_geometry('pre_confirmation_geometry')
    left,bottom,right,top=product.to_crs(5070).total_bounds
    pairs=pd.DataFrame([dict(key='k',reference_id='r',shift_east_m=0.,shift_north_m=0.,
        roi_xmin=left-5,roi_ymin=bottom-5,roi_xmax=right+5,roi_ymax=top+5)])
    result=score_geometries(source,{'before':old,'after':product},reference,pairs)
    assert result.loc['k','raw_reference_before_iou']==1.
    assert result.loc['k','raw_reference_after_iou']<.1
