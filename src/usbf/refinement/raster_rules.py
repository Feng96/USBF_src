import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage as ndi
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

LEVELS = ('L', 'M', 'S')

RULES = {
    'hollow': {
        'area_min': [12, 25, 36], 'width_min': [3, 5, 6],
        'rect_min': [.60, .75, .85], 'stability_min': [.50, .70, .80],
        'ring_support_min': [.70, .80, .90], 'valid_min': [.75, .90, .95],
        'hole_p75_max': [3, 2, 1.5], 'roof_p25_min': [2.5, 4, 4],
        'contrast_min': [2, 3, 4], 'dual_void_iou_min': [0, .50, .70],
        'hole_low2_fraction_min': [.60, .80, .90],
    },
    'small_fp': {
        'area_max_exclusive': [120, 80, 50], 'prob_p90_max': [.35, .20, .10],
        'valid_min': [.75, .90, .95], 'height_p95_max': [2.5, 1.5, 1],
        'B0_p90_max': [.50, .35, .20],
    },
    'accessory': {
        'coverage_min': [.75, .85, .90], 'add_area_min': [12, 20, 30],
        'add_fraction_min': [.15, .25, .35], 'width_min': [2, 3, 4],
        'largest_fraction_min': [.50, .65, .75],
        'stability_min': [.65, .75, .85], 'dual_iou_min': [.65, .80, .90],
        'valid_min': [.75, .90, .95], 'add_p25_min': [1.5, 2, 2.5],
        'height_difference_max': [4, 3, 2.5],
        'green_fraction_max': [.35, .25, .15], 'contact_length_min': [3, 5, 7],
    },
    'common': dict(context_m=15, rgb_valid_min=.99, max_window_pixels=2048,
                   minimum_object_pixels=4, minimum_object_width_pixels=2,
                   negative_height_fraction_max=.05, no_manual_candidates=True,
                   automatic_edits=False, independent_GT=False,
                   L_M_S_are_nested=True, geometry_only_baseline='G',
                   G_is_not_a_height_supported_candidate=True),
}

def burn(g, hw, t):
    if g.is_empty:
        return np.zeros(hw, bool)
    return rasterize([(g, 1)], out_shape=hw, transform=t, dtype='uint8').astype(bool)

def geom(mask, t):
    return unary_union([shape(g) for g, v in shapes(mask.astype('uint8'), mask=mask,
                       transform=t, connectivity=4) if v == 1])

def size(g):
    r = g.minimum_rotated_rectangle
    if r.geom_type != 'Polygon' or r.area == 0:
        return 0., 0., 0.
    lengths = np.linalg.norm(np.diff(np.asarray(r.exterior.coords), axis=0), axis=1)
    return float(lengths.min()), float(lengths.max()), float(g.area / r.area)

def component(mask, reference):
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros(mask.shape, bool)
    counts = np.bincount(labels[reference].ravel(), minlength=n + 1)
    counts[0] = 0
    return labels == int(counts.argmax()) if counts.max() else np.zeros(mask.shape, bool)

def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))

def stats(mask, height, valid):
    v = height[mask & valid & np.isfinite(height)]
    out = dict(n=int(mask.sum()), valid_fraction=float(len(v) / max(1, mask.sum())),
               p25=None, p50=None, p75=None, p95=None, low2_fraction=None,
               negative_fraction=None)
    if len(v):
        for q in (25, 50, 75, 95):
            out['p' + str(q)] = float(np.percentile(v, q))
        out.update(low2_fraction=float((v < 2).mean()), negative_fraction=float((v < -1).mean()))
    return out

def le(value, threshold):
    return value is not None and value <= threshold

def ge(value, threshold):
    return value is not None and value >= threshold

def classify(family, f):
    """Re-evaluate all frozen levels from measurements, independently of I/O."""
    rule = RULES[family]
    output = []
    for j, level in enumerate(LEVELS):
        r = {k: v[j] for k, v in rule.items()}
        if family == 'hollow':
            passed = (f['area'] >= r['area_min'] and f['width'] >= r['width_min']
                      and f['rectangularity'] >= r['rect_min']
                      and f['stability'] >= r['stability_min']
                      and f['ring_support'] >= r['ring_support_min']
                      and f['height_valid'] >= r['valid_min']
                      and le(f['hole_p75'], r['hole_p75_max'])
                      and ge(f['roof_p25'], r['roof_p25_min'])
                      and ge(f['four_side_min'], r['roof_p25_min'])
                      and ge(f['contrast'], r['contrast_min'])
                      and f['dual_void_iou'] >= r['dual_void_iou_min']
                      and ge(f['hole_low2_fraction'], r['hole_low2_fraction_min'])
                      and le(f['negative_fraction'], .05))
        elif family == 'small_fp':
            passed = (f['area'] < r['area_max_exclusive']
                      and f['B1_p90'] <= r['prob_p90_max']
                      and f['B0_p90'] <= r['B0_p90_max']
                      and f['height_valid'] >= r['valid_min']
                      and le(f['height_p95'], r['height_p95_max'])
                      and le(f['negative_fraction'], .05)
                      and not f['nearby_roof_or_shift'] and not f['rgb_ambiguous'])
        else:
            passed = (f['coverage'] >= r['coverage_min']
                      and f['area'] >= r['add_area_min']
                      and f['add_fraction'] >= r['add_fraction_min']
                      and f['width'] >= r['width_min']
                      and f['largest_fraction'] >= r['largest_fraction_min']
                      and f['stability'] >= r['stability_min']
                      and f['dual_iou'] >= r['dual_iou_min']
                      and f['height_valid'] >= r['valid_min']
                      and ge(f['add_p25'], r['add_p25_min'])
                      and le(f['height_difference'], r['height_difference_max'])
                      and f['green_fraction'] <= r['green_fraction_max']
                      and f['contact_length'] >= r['contact_length_min']
                      and le(f['negative_fraction'], .05)
                      and f['neighbor_overlap_fraction'] <= .05
                      and f['merged_GBA_instances'] == 1)
        if passed:
            output.append(level)
    assert output in ([], ['L'], ['L', 'M'], ['L', 'M', 'S'])
    return output

def measure(w, neighbors=()):
    g, rgb, h = w['g'], w['rgb'], w['signed']
    p0, p = w['prob'][0], w['prob'][1]
    rv, hv, gx, gy = w['rgb_valid'], w['height_valid'], w['gx'], w['gy']
    t, hw = rasterio.Affine(gx, 0, 0, 0, gy, 0), p.shape
    if g.is_empty or not g.is_valid or g.geom_type != 'Polygon':
        return [], 'invalid_or_multipart'
    inside = burn(g, hw, t)
    width, length, rect = size(g)
    if inside.sum() < 4 or width / max(gx, gy) < 2:
        return [], 'micro_or_underresolved'
    zone = burn(g.buffer(5), hw, t)
    if not zone.any() or not rv[zone].all() or not np.isfinite(w['prob'][:, zone]).all():
        return [], 'incomplete_5m_context'
    found = []
    if g.area < 120:
        p90 = float(np.quantile(p[inside], .9))
        if p90 <= .35:
            hs = stats(inside, h, hv)
            dilation = burn(g.buffer(2), hw, t)
            around = dilation & ~inside
            roof = (p >= .5) & around
            nearby = bool(roof.sum() * gx * gy >= max(3, .2 * g.area))
            local_rgb = rgb[:, inside]
            ambiguous = bool(((local_rgb.max(0) < .08).mean() >= .25) or
                             ((2 * local_rgb[1] - local_rgb[0] - local_rgb[2] > .12).mean() >= .25))
            f = dict(area=float(g.area), width=width, B1_p90=p90,
                     B0_p90=float(np.quantile(p0[inside], .9)),
                     height_valid=hs['valid_fraction'], height_p95=hs['p95'],
                     negative_fraction=hs['negative_fraction'],
                     nearby_roof_or_shift=nearby, rgb_ambiguous=ambiguous)
            found.append(dict(family='small_fp', region=g, metrics=f,
                              levels=classify('small_fp', f), baseline='G',
                              unknown_height=hs['valid_fraction'] < .75))
    if g.area >= 100:
        labels, n = ndi.label((p < .35) & inside)
        counts = np.bincount(labels.ravel())
        for j in np.flatnonzero(counts * gx * gy >= 12):
            if j == 0:
                continue
            mask = labels == j
            hole = geom(mask, t)
            if (hole.geom_type != 'Polygon' or hole.area > .6 * g.area or
                    not g.covers(hole) or hole.distance(g.boundary) < 1.5):
                continue
            short, long, rectangularity = size(hole)
            if short < 3 or long / short > 3.5 or rectangularity < .60:
                continue
            stability = min(iou(mask, component((p < th) & inside, mask)) for th in (.25, .45))
            ring = burn(hole.buffer(3).difference(hole).intersection(g), hw, t)
            support = float((p[ring] >= .5).mean()) if ring.any() else 0.
            if stability < .5 or support < .7:
                continue
            core = burn(hole.buffer(-1), hw, t)
            roof = burn(hole.buffer(5).difference(hole.buffer(1)).intersection(g), hw, t)
            hs, rs = stats(core, h, hv), stats(roof, h, hv)
            yy, xx = np.indices(hw)
            dx, dy = (xx + .5) * gx - hole.centroid.x, (yy + .5) * gy - hole.centroid.y
            sides = []
            for sector in (dx >= abs(dy), -dx >= abs(dy), dy > abs(dx), -dy > abs(dx)):
                ss = stats(roof & sector, h, hv)
                sides.append(ss['p50'] if ss['n'] >= 4 and ss['valid_fraction'] >= .90 else None)
            contrast = rs['p50'] - hs['p50'] if rs['p50'] is not None and hs['p50'] is not None else None
            f = dict(area=float(hole.area), width=short, rectangularity=rectangularity,
                     stability=stability, ring_support=support,
                     height_valid=min(hs['valid_fraction'], rs['valid_fraction']) if hs['n'] >= 4 and rs['n'] >= 16 else 0.,
                     hole_p75=hs['p75'], roof_p25=rs['p25'], contrast=contrast,
                     four_side_min=min(sides) if all(v is not None for v in sides) else None,
                     hole_low2_fraction=hs['low2_fraction'], negative_fraction=hs['negative_fraction'],
                     dual_void_iou=iou(mask, component((p0 < .35) & inside, mask)))
            found.append(dict(family='hollow', region=hole, metrics=f,
                              levels=classify('hollow', f), baseline='G',
                              unknown_height=f['height_valid'] < .75))
    if not 20 <= g.area <= 5000:
        return found, 'screened'
    model = component(p >= .5, inside)
    if not model.any():
        return found, 'screened'
    coverage = float((model & inside).sum() / inside.sum())
    add = model & ~inside
    if coverage < .75 or add.sum() * gx * gy < 12 or add.sum() / inside.sum() < .15:
        return found, 'screened'
    missing = inside & ~model
    if model.sum() / inside.sum() < 1.2 or add.sum() < 2 * missing.sum():
        return found, 'balanced_shift_not_accessory_expansion'
    # Unbounded connected components must not be artificially cut into an annex.
    boundary = ndi.binary_dilation(model) & ~model
    if (model[0].any() or model[-1].any() or model[:, 0].any() or model[:, -1].any()
            or not rv[model | boundary].all() or add.sum() / inside.sum() > 2):
        return found, 'accessory_component_truncated_or_overlarge'
    addition = geom(add, t).difference(g)
    if addition.is_empty:
        return found, 'screened'
    labels, n = ndi.label(add)
    counts = np.bincount(labels.ravel())[1:]
    largest_fraction = float(counts.max() / max(1, add.sum())) if n else 0.
    thick = float(2 * ndi.distance_transform_edt(add, sampling=(gy, gx)).max())
    if thick < 2 or largest_fraction < .5:
        return found, 'screened'
    stability = min(iou(model, component(p >= th, inside)) for th in (.35, .65))
    dual = iou(model, component(p0 >= .5, inside))
    if stability < .65 or dual < .65:
        return found, 'screened'
    main_stats, add_stats = stats(inside, h, hv), stats(add, h, hv)
    other = unary_union(neighbors) if neighbors else Polygon()
    neighbor_overlap = float(addition.intersection(other).area / max(1e-9, addition.area))
    model_geom = geom(model, t)
    merged = 1 + sum(model_geom.intersection(n).area > max(2, .2 * n.area) for n in neighbors)
    local_rgb = rgb[:, add]
    green = float((2 * local_rgb[1] - local_rgb[0] - local_rgb[2] > .12).mean())
    difference = abs(main_stats['p50'] - add_stats['p50']) if main_stats['p50'] is not None and add_stats['p50'] is not None else None
    # Boundary intersection at raster precision: 1.2-cell band estimates contact.
    contact = float(g.boundary.intersection(addition.buffer(1.2 * max(gx, gy))).length)
    f = dict(area=float(addition.area), coverage=coverage,
             add_fraction=float(addition.area / g.area), width=thick,
             largest_fraction=largest_fraction, stability=stability, dual_iou=dual,
             height_valid=min(main_stats['valid_fraction'], add_stats['valid_fraction']),
             add_p25=add_stats['p25'], height_difference=difference,
             green_fraction=green, contact_length=contact,
             negative_fraction=add_stats['negative_fraction'],
             neighbor_overlap_fraction=neighbor_overlap, merged_GBA_instances=merged,
             semantic_status='possible_accessory_OR_missing_main_roof; enclosure_unobservable')
    found.append(dict(family='accessory', region=addition, metrics=f,
                      levels=classify('accessory', f), baseline='G',
                      unknown_height=f['height_valid'] < .75))
    return found, 'screened'
