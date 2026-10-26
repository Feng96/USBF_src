"""Re-rasterize fixed geometries. Existing metric CSVs are comparison-only."""
from pathlib import Path
from types import SimpleNamespace
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import box
from shapely import affinity
from . import metrics as M,classification as K
from usbf.io.common import save,read

FRAMES=('raw_reference','reference_in_NAIP_frame')
METRICS=('iou','bf1_1m','bf1_2m')

def score_geometries(gba,products,reference,matches):
    if not gba.index.is_unique or not reference.index.is_unique:raise ValueError('Duplicate source/reference identifiers')
    if matches.empty:return pd.DataFrame(index=pd.Index([],name='key'))
    rows=[];ga=gba.to_crs(5070)
    pp={k:v.to_crs(5070) for k,v in products.items()}
    raw=reference.to_crs(5070)
    for row in matches.itertuples():
        r=reference.geometry.loc[row.reference_id]
        # Reference translation alone, in the original UTM11 metre grid.
        moved=affinity.translate(r,float(row.shift_east_m),float(row.shift_north_m))
        moved=gpd.GeoSeries([moved],crs=reference.crs).to_crs(5070).iloc[0]
        truth={'raw_reference':raw.geometry.loc[row.reference_id],'reference_in_NAIP_frame':moved}
        roi=box(row.roi_xmin,row.roi_ymin,row.roi_xmax,row.roi_ymax);rec=dict(key=row.key)
        for frame in FRAMES:
            gt=truth[frame]
            for version,pred in [('GBA',ga.geometry.loc[row.key])]+[(k,v.geometry.loc[row.key]) for k,v in pp.items()]:
                pre=frame+'_'+version+'_';inter=pred.intersection(gt).area;union=pred.union(gt).area
                rec[pre+'intersection_m2']=inter;rec[pre+'union_m2']=union;rec[pre+'iou']=inter/union
                counts=M.boundary([pred],[gt],roi,pixel=.5);rec.update({pre+k:v for k,v in counts.items()})
                for tol in (1,2):rec[pre+f'bf1_{tol}m']=M.f1(counts[f'pred_hit_{tol}m'],counts[f'gt_hit_{tol}m'],counts['pred_n'],counts['gt_n'])
        rows.append(rec)
    return pd.DataFrame(rows).set_index('key')

def summarize(data,populations,products=('USBF_OSM_gate',)):
    rows=[]
    for name,q in populations.items():
        parents=sorted(q.parent.unique());draw=np.random.default_rng(20261024).integers(0,len(parents),(5000,len(parents))) if len(parents)>1 else None
        for frame in FRAMES:
            for product in products:
                for metric in METRICS:
                    a=q[frame+'_GBA_'+metric];b=q[frame+'_'+product+'_'+metric];delta=b-a
                    low=high=np.nan
                    if draw is not None:
                        groups=q.assign(d=delta).groupby('parent').agg(ds=('d','sum'),n=('d','size')).reindex(parents)
                        boot=groups.ds.to_numpy()[draw].sum(1)/groups.n.to_numpy()[draw].sum(1);low,high=np.quantile(boot,[.025,.975])*100
                    rows.append(dict(population=name,reference_frame=frame,product=product,metric=metric,
                        objects=len(q),parents=len(parents),gba_mean=float(a.mean()),usbf_mean=float(b.mean()),
                        delta_pp=float(delta.mean()*100),ci_low_pp=low,ci_high_pp=high,
                        improved=int((delta>.01).sum()),stable=int((delta.abs()<=.01).sum()),worsened=int((delta<-.01).sum())))
    return pd.DataFrame(rows)

def replay(data_dir,output,example_product=None,generated_product=None):
    start=time.time();data_dir=Path(data_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    folder=data_dir/('example' if example_product else 'primary');pairs=pd.read_csv(folder/'fixed_matches.csv').set_index('key',drop=False)
    if example_product:
        product=gpd.read_parquet(example_product).set_index('key')
        original=gpd.GeoDataFrame(product.drop(columns='geometry'),geometry=product.gba_geometry,crs=4326).to_crs(32611)
        refs=gpd.read_parquet(folder/'reference_raw.parquet')
        products={'USBF_OSM_gate':product};pairs=pairs.loc[product.index].copy()
    else:
        original=gpd.read_parquet(folder/'gba.parquet');refs=gpd.read_parquet(folder/'reference_raw.parquet')
        products={'USBF_old':gpd.read_parquet(folder/'usbf_before_gate.parquet'),
                  'USBF_OSM_gate':gpd.read_parquet(folder/'usbf.parquet')}
        if 'key' in original:original=original.set_index('key')
        for k,v in list(products.items()):
            if 'key' in v:products[k]=v.set_index('key')
        if generated_product:
            full=gpd.read_parquet(generated_product).set_index('key')
            missing=set(pairs.index)-set(full.index)
            if missing:raise ValueError('Generated product is missing fixed primary objects: '+str(sorted(missing)[:5]))
            final=full.loc[pairs.index].copy()
            if 'pre_confirmation_geometry' not in final:raise ValueError('Generated product must retain its pre-confirmation geometry')
            products={'USBF_old':final.set_geometry('pre_confirmation_geometry'),'USBF_OSM_gate':final}
    if 'reference_id' in refs:refs=refs.set_index('reference_id')
    if not pairs.reference_id.is_unique:raise ValueError('Fixed matching is not one-to-one')
    scores=score_geometries(original,products,refs,pairs.reset_index(drop=True))
    data=pairs.join(scores);changed=~shapely.equals_exact(original.loc[pairs.index].geometry.to_numpy(),products['USBF_OSM_gate'].to_crs(original.crs).loc[pairs.index].geometry.to_numpy(),0)
    # A CRS roundtrip may alter floating coordinates. Primary data retain their exact original matched CRS.
    if example_product or generated_product:changed=products['USBF_OSM_gate'].loc[pairs.index,'modified'].to_numpy(bool)
    data['gated_modified']=changed
    data.to_csv(output/'OBJECT_METRICS.csv',index=False)
    populations={'all':data,'modified':data[changed],'unchanged':data[~changed],
                 'fixed180':data[data.original_modified],'reverted9':data[data.original_circle]}
    for group in ('addition_only','removal_only','addition_and_removal'):
        populations[group]=data[changed&data.original_action_class.eq(group)]
    summary=summarize(data,populations,tuple(products));summary.to_csv(output/'Table3_and_frame_sensitivity.csv',index=False)
    verification=dict(actual_geometry_pairs=len(data),geometry_rasterization_executed=True,
                      reference_translated_only=True,matching_refitted=False,score_csv_used_as_input=False)
    if not example_product:
        if len(data)!=2019 or int(changed.sum())!=171:raise AssertionError('Primary sample definition changed')
        counts=data.loc[changed,'original_action_class'].value_counts().to_dict()
        if counts!={'addition_only':144,'removal_only':20,'addition_and_removal':7}:raise AssertionError(counts)
        expected=pd.read_csv(folder/'expected_metrics.csv').set_index('key');errors={}
        for c in scores:
            if c in expected:errors[c]=float(np.max(np.abs(scores[c]-expected.loc[scores.index,c])))
        verification.update(expected_metrics_used_only_after_rescoring=True,maximum_metric_difference=max(errors.values()),
                            individual_column_differences=errors,accepted_operation_counts=counts)
        if max(errors.values())>1e-10:raise AssertionError('Historical geometry-score regression differs: '+str(max(errors.values())))
        q=summary[summary.population.eq('modified')&summary.reference_frame.eq('reference_in_NAIP_frame')&summary['product'].eq('USBF_OSM_gate')]
        i=q[q.metric.eq('iou')].iloc[0];f=q[q.metric.eq('bf1_1m')].iloc[0]
        if (i.improved,i.stable,i.worsened)!=(102,22,47):raise AssertionError('IoU outcome counts changed')
        verification['registered_modified']=dict(iou_before=i.gba_mean,iou_after=i.usbf_mean,bf1_before=f.gba_mean,bf1_after=f.usbf_mean,
            iou_delta_pp=i.delta_pp,bf1_delta_pp=f.delta_pp,iou_ci=[i.ci_low_pp,i.ci_high_pp],bf1_ci=[f.ci_low_pp,f.ci_high_pp])
        from .report import build
        build(data,original,products['USBF_OSM_gate'],pd.read_csv(folder/'registrations.csv'),summary,output)
        from .curved_reference import replay as curved_replay
        verification['separate_curved_reference_replay']=curved_replay(data_dir,output/'curved_reference')
        import shutil
        shutil.copyfile(output/'curved_reference/CURVED_SUMMARY.csv',output/'TableS8_curved_reference.csv')
    verification.update(status='passed',elapsed_seconds=time.time()-start);save(output/'GEOMETRY_REPLAY.json',verification)
    return verification
