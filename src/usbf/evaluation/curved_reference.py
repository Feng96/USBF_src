"""Separate supplementary curved-reference cohort: 9 raw / 7 registered pairs."""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from . import metrics as M
from usbf.io.common import save

def replay(data_dir,output):
    folder=Path(data_dir)/'curved_reference';output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=folder/'All_13_targets_9_matches_GBA_V7_USBF.gpkg'
    metadata=['reference_id','key','paired_eligible','NAIP_frame_available','reference_only_GBA_assessment',
        'visual_class','roi_xmin','roi_ymin','roi_xmax','roi_ymax']
    pairs=pd.read_csv(folder/'ALL_9_GBA_MATCHED_TARGETS.csv',usecols=metadata)
    raw=gpd.read_file(path,layer='Reference').set_index('reference_id').to_crs(5070)
    aligned=gpd.read_file(path,layer='Reference_own_NAIP_frame').set_index('reference_id').to_crs(5070)
    predictions={name:gpd.read_file(path,layer=layer).set_index('reference_id').to_crs(5070) for name,layer in
        [('GBA','GBA'),('USBF_archive','V7_archive'),('USBF_OSM_gate','USBF_Sept18')]}
    records=[]
    for row in pairs.itertuples():
        rec={k:getattr(row,k) for k in metadata};roi=box(row.roi_xmin,row.roi_ymin,row.roi_xmax,row.roi_ymax)
        for frame,reference in [('raw_reference',raw),('reference_in_NAIP_frame',aligned)]:
            available=bool(row.paired_eligible and (frame=='raw_reference' or row.NAIP_frame_available))
            for version,p in predictions.items():
                prefix=frame+'_'+version+'_'
                if not available:
                    for metric in ('iou','bf1_1m','bf1_2m'):rec[prefix+metric]=np.nan
                    continue
                a,b=p.loc[row.reference_id].geometry,reference.loc[row.reference_id].geometry
                rec[prefix+'iou']=M.iou(a,b);counts=M.boundary([a],[b],roi,pixel=.5)
                for tol in (1,2):rec[prefix+f'bf1_{tol}m']=M.f1(counts[f'pred_hit_{tol}m'],counts[f'gt_hit_{tol}m'],counts['pred_n'],counts['gt_n'])
        records.append(rec)
    result=pd.DataFrame(records);result.to_csv(output/'CURVED_9_GEOMETRY_METRICS.csv',index=False,na_rep='NA')
    # The archived metric CSV is consulted only after all new geometry scores exist.
    expected=pd.read_csv(folder/'ALL_9_GBA_MATCHED_TARGETS.csv').set_index('reference_id');errors={}
    actual=result.set_index('reference_id')
    for c in result:
        if c.startswith(('raw_reference_','reference_in_NAIP_frame_')):
            a,b=actual[c],expected.loc[actual.index,c]
            if not np.array_equal(a.isna(),b.isna()):raise AssertionError('Supplementary unavailable scores changed')
            errors[c]=float((a-b).abs().max())
    if max(errors.values())>1e-10:raise AssertionError('Curved-reference geometry-score regression changed')
    targets=pd.read_csv(folder/'ALL_13_FIXED_TARGETS.csv',usecols=['reference_id','key','fixed_GBA_match','paired_eligible','NAIP_frame_available','exclusion_reason'])
    targets=targets.merge(result[['reference_id']+[c for c in result if c.endswith(('_iou','_bf1_1m','_bf1_2m'))]],on='reference_id',how='left')
    targets.to_csv(output/'ALL_13_TARGET_STATUS_AND_SCORES.csv',index=False,na_rep='NA')
    groups=[('All 9 fixed GBA matches',result)]
    groups.extend((name,q) for name,q in result.groupby('reference_only_GBA_assessment'))
    groups.extend(('Visual: '+str(name),q) for name,q in result.groupby('visual_class'))
    rows=[]
    for name,q in groups:
        for frame in ('raw_reference','reference_in_NAIP_frame'):
            qq=q[q.paired_eligible & (True if frame=='raw_reference' else q.NAIP_frame_available)]
            for metric in ('iou','bf1_1m','bf1_2m'):
                for left,right in [('GBA','USBF_archive'),('GBA','USBF_OSM_gate'),('USBF_archive','USBF_OSM_gate')]:
                    a,b=qq[frame+'_'+left+'_'+metric],qq[frame+'_'+right+'_'+metric];delta=b-a
                    rows.append(dict(group=name,reference_frame=frame,metric=metric,fixed_objects=len(q),scored_pairs=len(qq),
                        comparator=left,product=right,comparator_mean=a.mean(),product_mean=b.mean(),delta_pp=100*delta.mean(),
                        improved=int((delta>.01).sum()),stable=int((delta.abs()<=.01).sum()),worsened=int((delta<-.01).sum())))
    pd.DataFrame(rows).to_csv(output/'CURVED_SUMMARY.csv',index=False,na_rep='NA')
    audit=dict(status='passed',fixed_targets=len(targets),matched_pairs=len(result),raw_scored=int(result.paired_eligible.sum()),
        registered_scored=int(result.NAIP_frame_available.sum()),unmatched_targets=int((~targets.fixed_GBA_match).sum()),
        maximum_metric_difference=max(errors.values()),unavailable_scores='NA, never zero',primary_2019_denominator_changed=False,
        candidate_96_screening_records=(folder/'ALL_96_GBA_SHAPE_COVERAGE.csv').is_file())
    if (audit['fixed_targets'],audit['matched_pairs'],audit['raw_scored'],audit['registered_scored'],audit['unmatched_targets'])!=(13,9,9,7,4):raise AssertionError('Supplementary sample scope changed')
    save(output/'CURVED_REPLAY.json',audit);return audit
