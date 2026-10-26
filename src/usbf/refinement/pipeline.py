"""Local adapter around the unchanged manuscript rule definitions."""
from pathlib import Path
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from . import context as C,object_engine as E,osm_policy_frozen as O
from usbf.io.common import read,save,sha
from usbf.data.sensors import load_sensor
from usbf.models.inference import run as infer

SCIENCE='GBA_V6circle_V7_reference_night_v003_20260911'

def joint_guard(original,candidates,neighbors,keys):
    """Same three-pass simultaneous/original-neighbor guard as the fixed replay."""
    om=gpd.GeoSeries(original,crs=4326).to_crs(5070).to_numpy()
    metric=gpd.GeoSeries(candidates,crs=4326).to_crs(5070).to_numpy()
    valid_neighbors=neighbors[neighbors.geometry.is_valid].to_crs(5070)
    ng=valid_neighbors.geometry.to_numpy();nk=valid_neighbors.key.to_numpy()
    ntree=shapely.STRtree(ng);changed=~shapely.equals_exact(candidates,original,0);guards=[]
    for _ in range(3):
        bad=set();ii,jj=shapely.STRtree(metric).query(metric,predicate='intersects');take=ii<jj;ii,jj=ii[take],jj[take]
        if len(ii):
            increase=shapely.area(shapely.intersection(metric[ii],metric[jj]))-shapely.area(shapely.intersection(om[ii],om[jj]))
            for i,j in zip(ii[increase>.01],jj[increase>.01]):
                if changed[i]:bad.add(int(i))
                if changed[j]:bad.add(int(j))
        for i in np.flatnonzero(changed):
            ix=ntree.query(metric[i],predicate='intersects');ix=ix[nk[ix]!=keys[i]]
            if len(ix):
                increase=shapely.area(shapely.intersection(metric[i],ng[ix]))-shapely.area(shapely.intersection(om[i],ng[ix]))
                if np.any(increase>.01):bad.add(int(i))
        if not bad:break
        for i in bad:
            candidates[i]=original[i];metric[i]=om[i];changed[i]=False
            guards.append(dict(key=keys[i],reason='joint_or_original_neighbor_overlap'))
    if not shapely.is_valid(candidates).all():raise ValueError('Invalid final geometry')
    return candidates,guards

def generate(data_dir,output,device='cpu',threads=2,historical_probabilities=False,missing_height=False,dataset='example'):
    start=time.time();data_dir=Path(data_dir).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new empty output directory')
    if dataset not in ('example','full'):raise ValueError('Dataset must be example or full')
    output.mkdir(parents=True,exist_ok=True);example=data_dir/dataset;state=output/'state';state.mkdir()
    records=read(example/'sensors.json')['records']
    for r in records:r['path']=str(example/r['path']);load_sensor(r,verify=True)
    if missing_height:
        modified=[];folder=state/'missing_height_inputs';folder.mkdir()
        for r in records:
            with np.load(r['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
            a['valid']=np.zeros_like(a['valid']);a['signed']=np.full_like(a['signed'],np.nan)
            p=folder/(r['name']+'.npz');np.savez_compressed(p,**a)
            modified.append(dict(r,path=str(p),sha256=sha(p)))
        records=modified
    if historical_probabilities:
        if missing_height:raise ValueError('Historical probability replay cannot simulate missing height')
        probabilities=[dict(name=r['name'],probability=str(example/'regression/probabilities'/(r['name']+'.tif')),
                            sha256=sha(example/'regression/probabilities'/(r['name']+'.tif'))) for r in records]
        inference=dict(mode='explicit_archived_probability_regression_only',records=probabilities)
    else:
        inference=infer(records,data_dir,output/'probabilities',device,threads)
        probabilities=inference['records']
    save(state/'INVENTORY.json',dict(records=records));save(state/'PROBABILITIES.json',dict(records=probabilities))
    frames=gpd.read_parquet(example/'frames.parquet');frames.to_parquet(state/'frames.parquet',index=False)
    targets=gpd.read_parquet(example/'targets.parquet').sort_values('row_id').reset_index(drop=True)
    if targets.key.duplicated().any():raise ValueError('Source building IDs must be unique')
    if not targets.geometry.is_valid.all():raise ValueError('Invalid original geometry: quarantine outside this small example')
    E.COHORT=gpd.read_parquet(example/'neighbors.parquet');C.OUT=state;E.CTX=C.Context()
    E.BASE=gpd.GeoDataFrame({'key':pd.Series(dtype=str),'shape_changed':pd.Series(dtype=bool),'hollow_cut':pd.Series(dtype=bool)},
        geometry=gpd.GeoSeries([],crs=4326),crs=4326).set_index('key')
    records_out=[];audits=[];coverage={}
    for row in targets.itertuples():
        rec,audit=E.process(row);records_out.append(rec);audits.append(audit)
        w,_=E.CTX.window(row.geometry,row.anchor)
        if w is not None:
            import rasterio
            inside=E.R.burn(w['g'],w['signed'].shape,rasterio.Affine(w['gx'],0,0,0,w['gy'],0))
            coverage[row.key]=float(w['height_valid'][inside].mean()) if inside.any() else 0.
        else:coverage[row.key]=0.
    originals=targets.geometry.to_numpy();candidates=shapely.from_wkb([r['V7_wkb'] for r in records_out])
    candidates,guards=joint_guard(originals,candidates,E.COHORT,targets.key.to_numpy())
    O.OUT=data_dir/'osm';refs=O.References();out=[]
    for row,rec,audit,candidate in zip(targets.itertuples(),records_out,audits,candidates):
        rec['V7_wkb']=candidate.wkb.hex()
        gated=O.apply(rec,row.geometry,row.source,row.id,refs=refs)
        geometry=shapely.from_wkb(bytes.fromhex(gated['V7_wkb']));changed=not geometry.equals_exact(row.geometry,0)
        a,b=gpd.GeoSeries([row.geometry,geometry],crs=4326).to_crs(5070)
        old_holes=sum(len(p.interiors) for p in E.G.parts(row.geometry));new_holes=sum(len(p.interiors) for p in E.G.parts(geometry))
        kind='unchanged' if not changed else 'circular_or_elliptical_replacement' if gated['circle_gate_status']=='confirmed' else 'interior_opening' if new_holes>old_holes else 'local_boundary_revision'
        attrs={k:getattr(row,k) for k in targets.columns if k not in ('geometry','Index')}
        attrs.update(geometry=geometry,gba_geometry=row.geometry,pre_confirmation_geometry=candidate,modified=changed,modification_type=kind,
            area_before_m2=a.area,area_after_m2=b.area,added_area_m2=b.difference(a).area,removed_area_m2=a.difference(b).area,
            height=getattr(row,'height',None),height_inherited=True,height_valid_fraction=coverage[row.key],
            height_missing=coverage[row.key]==0,height_incomplete=coverage[row.key]<1,
            science_version=SCIENCE,circle_gate_policy=O.VERSION,circle_gate_applied=gated['circle_gate_applied'],
            circle_gate_status=gated['circle_gate_status'],circle_gate_reason=gated['circle_gate_reason'],
            circle_candidate=bool(rec['circle_candidate']),circle_proposed_before_gate=gated['circle_proposed_before_gate'],
            processing_status=gated['status'],decision=gated['V7_decision'],holes_before=old_holes,holes_after=new_holes)
        out.append(attrs);audit['OSM_confirmation']=gated.get('evidence',{}).get('OSM_circle_gate',dict(status='not_applicable'))
    product=gpd.GeoDataFrame(out,crs=4326);product['gba_geometry']=gpd.GeoSeries(product.gba_geometry,crs=4326)
    product['pre_confirmation_geometry']=gpd.GeoSeries(product.pre_confirmation_geometry,crs=4326)
    product.to_parquet(output/'USBF.parquet',index=False)
    back=gpd.read_parquet(output/'USBF.parquet')
    if not np.array_equal(shapely.to_wkb(product.geometry),shapely.to_wkb(back.geometry)):raise AssertionError('Final WKB readback differs')
    if not np.array_equal(shapely.to_wkb(product.gba_geometry),shapely.to_wkb(back.gba_geometry)):raise AssertionError('Original WKB readback differs')
    if not np.array_equal(shapely.to_wkb(product.pre_confirmation_geometry),shapely.to_wkb(back.pre_confirmation_geometry)):raise AssertionError('Pre-confirmation WKB readback differs')
    if back.key.tolist()!=targets.key.tolist():raise AssertionError('Source ID order changed during export')
    if back.source.tolist()!=targets.source.tolist() or back.id.tolist()!=targets.id.tolist():raise AssertionError('Source provenance changed during export')
    if not back.geometry.is_valid.all() or not back.key.is_unique:raise AssertionError('Invalid export or duplicate IDs')
    save(output/'AUDIT.json',audits);save(output/'JOINT_GUARDS.json',guards)
    report=dict(status='complete',objects=len(product),modified=int(product.modified.sum()),
        mode='archived_probability_regression_only' if historical_probabilities else 'actual_frozen_model_generation',
        elapsed_seconds=time.time()-start,output_sha256=sha(output/'USBF.parquet'),
        source_geometry_readback_exact=True,final_geometry_readback_exact=True,
        circle_confirmation_counts=product.circle_gate_status.value_counts().to_dict(),
        missing_height_simulation=missing_height,science_version=SCIENCE,policy=O.VERSION,inference=inference)
    save(output/'GENERATION.json',report);return report
