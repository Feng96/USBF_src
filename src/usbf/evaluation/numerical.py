"""Unit-aware regression of recomputed scores; never changes their calculation."""
import numpy as np

# One square millimetre, not a building-size or modification threshold.
AREA_ROUNDOFF_M2 = 1e-6

def compare_metric_columns(actual, expected):
    expected=expected.loc[actual.index]
    errors={};limits={};failures={}
    for c in actual:
        if c not in expected:raise ValueError('Missing historical metric: '+c)
        a=actual[c].to_numpy(float);b=expected[c].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite metric in fixed matched sample: '+c)
        if c.endswith(('_intersection_m2','_union_m2')):
            limit=np.full(len(a),AREA_ROUNDOFF_M2)
        elif c.endswith('_iou'):
            union=expected[c[:-4]+'_union_m2'].to_numpy(float)
            if np.any(union<=AREA_ROUNDOFF_M2):raise ValueError('Invalid union area')
            # Propagate both area-error bounds through intersection/union.
            limit=np.maximum(1e-12,2*AREA_ROUNDOFF_M2/(union-AREA_ROUNDOFF_M2))
        elif c.endswith(('_bf1_1m','_bf1_2m')):
            limit=np.full(len(a),1e-12)
        else:
            # Remaining columns are integer boundary-pixel counts.
            limit=np.zeros(len(a))
        difference=np.abs(a-b)
        errors[c]=float(difference.max());limits[c]=float(limit.max())
        if np.any(difference>limit):failures[c]=int((difference>limit).sum())
    return dict(status='passed' if not failures else 'metric_regression_difference',
        individual_column_differences=errors,maximum_allowed_error_by_column=limits,
        failing_columns=failures,area_roundoff_m2=AREA_ROUNDOFF_M2,
        iou_roundoff='propagated from intersection and union area roundoff',
        boundary_pixel_counts_require_exact_match=True,
        maximum_area_difference_m2=max(v for c,v in errors.items() if c.endswith('_m2')),
        maximum_score_difference=max(v for c,v in errors.items() if c.endswith(('_iou','_bf1_1m','_bf1_2m'))),
        maximum_boundary_count_difference=max(v for c,v in errors.items() if not c.endswith(('_m2','_iou','_bf1_1m','_bf1_2m'))))
