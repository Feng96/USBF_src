import warnings
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely import affinity
from skimage.measure import EllipseModel
from . import raster_rules as R
from usbf._vendor.buildingregulariser.regularization import regularize_single_polygon
from usbf._vendor.building_boundary.core.segment import BoundarySegment
from usbf._vendor.building_boundary.core.regularize import regularize_segments
from usbf._vendor.building_boundary.core.merge import merge_segments
from usbf._vendor.building_boundary.core.intersect import compute_intersections

RULES = dict(area_thresholds_m2=[30, 50, 100], nuisance_max_pixels=9,
             opening_radius_m=1.5, opening_retention_min=.60,
             attachment_length_m=3, B1_p50_fraction_min=.85,
             B1_p65_fraction_min=.65, B0_p50_fraction_min=.70,
             height_valid_min=.90, height_p25_min_m=2,
             negative_height_fraction_max=.05, green_fraction_max=.25,
             GBA_coverage_min=.75, add_to_missing_ratio_min=2,
             regularization_invented_fraction_max=.15,
             external_ownership='nearest GBA raster footprint; exact neighbor exclusion; ties abstain',
             pixel_grid='native; geodesic local metres', automatic_GBA_edits=False)

def parts(g):
    if g.is_empty:
        return []
    return [g] if g.geom_type == 'Polygon' else [p for q in getattr(g, 'geoms', []) for p in parts(q)]

def poly_iou(a, b):
    return a.intersection(b).area / max(a.union(b).area, 1e-12)

def sampled_ring(g, n=256):
    return np.asarray([g.exterior.interpolate(i / n, normalized=True).coords[0]
                       for i in range(n)])

def fit_primitive(raw):
    """Fit on raw boundary, NEVER on regularized/smoothed polygon.

    Uniform arc-length residuals + IoU + solidity + ellipse rectangularity gate.
    Caller additionally checks cross-threshold agreement, context and sensors.
    """
    if raw.geom_type != 'Polygon' or raw.area < 30:
        return None, 'nonprimitive', {}
    shell = Polygon(raw.exterior)
    pts = sampled_ring(shell)
    sol = shell.area / shell.convex_hull.area
    rect = shell.area / shell.minimum_rotated_rectangle.area
    circularity = 4 * np.pi * shell.area / shell.length ** 2
    a = np.column_stack((2 * pts, np.ones(len(pts))))
    cx, cy, k = np.linalg.lstsq(a, np.sum(pts ** 2, axis=1), rcond=None)[0]
    radius = np.sqrt(max(k + cx * cx + cy * cy, 0))
    residual = np.abs(np.linalg.norm(pts - [cx, cy], axis=1) - radius) / max(radius, 1e-9)
    circle = Point(cx, cy).buffer(radius, quad_segs=96)
    short, long, _ = R.size(shell)
    metrics = dict(circle_iou=poly_iou(shell, circle), circle_p95=float(np.quantile(residual, .95)),
                   solidity=float(sol), rectangularity=float(rect), raw_circularity=float(circularity))
    if (metrics['circle_iou'] >= .90 and long / max(short, 1e-9) <= 1.18 and
            metrics['circle_p95'] <= .12 and residual.max() <= .25 and
            np.std(np.linalg.norm(pts - [cx, cy], axis=1)) / max(radius, 1e-9) <= .07 and
            sol >= .94 and circularity >= .55):
        return circle, 'circle', metrics
    model = EllipseModel()
    if not model.estimate(pts):
        return None, 'nonprimitive', metrics
    cx, cy, ax, by, theta = model.params
    if not np.all(np.isfinite(model.params)) or min(ax, by) < 3:
        return None, 'nonprimitive', metrics
    ellipse = affinity.rotate(affinity.scale(Point(0, 0).buffer(1, quad_segs=96), ax, by),
                              theta, origin=(0, 0), use_radians=True)
    ellipse = affinity.translate(ellipse, cx, cy)
    residual = np.asarray([ellipse.boundary.distance(Point(p)) for p in pts]) / min(ax, by)
    metrics.update(ellipse_iou=poly_iou(shell, ellipse), ellipse_p95=float(np.quantile(residual, .95)),
                   ellipse_aspect=float(max(ax, by) / min(ax, by)))
    if (metrics['ellipse_iou'] >= .92 and metrics['ellipse_p95'] <= .10 and
            residual.max() <= .20 and 1.18 < metrics['ellipse_aspect'] <= 3 and
            sol >= .95 and .68 <= rect <= .86 and circularity >= .40):
        return ellipse, 'ellipse', metrics
    return None, 'nonprimitive', metrics

def holes_of(g):
    return unary_union([Polygon(r) for p in parts(g) for r in p.interiors])

def primitive_branch(raw, alternatives):
    fitted, kind, metrics = fit_primitive(raw)
    if fitted is None:
        return None, kind, metrics
    checks = [fit_primitive(g) for g in alternatives]
    stable = all(q is not None and k == kind and poly_iou(fitted, q) >= .85
                 for q, k, _ in checks)
    metrics['cross_threshold_shape_stable'] = stable
    if not stable:
        return None, 'unstable_' + kind, metrics
    return fitted.difference(holes_of(raw)), kind, metrics

def directions(g):
    """Long GBA edges supply multiple axis families, not a single bounding box."""
    xy = np.asarray(g.exterior.coords)
    vec = np.diff(xy, axis=0)
    lengths = np.linalg.norm(vec, axis=1)
    angles = np.arctan2(vec[:, 1], vec[:, 0]) % np.pi
    selected = []
    for i in np.argsort(-lengths):
        if lengths[i] < max(3, .08 * lengths.max()):
            continue
        if all(abs((angles[i] - a + np.pi / 2) % np.pi - np.pi / 2) > np.deg2rad(7)
               for a in selected):
            selected.append(float(angles[i]))
        if len(selected) == 6:
            break
    if not selected:
        selected = [float(angles[np.argmax(lengths)])]
    # Normalize near-perpendicular axes but preserve genuinely different wings.
    base = selected[0]
    selected = [base + round((a - base) / (np.pi / 2)) * np.pi / 2
                if abs((a - base + np.pi / 4) % (np.pi / 2) - np.pi / 4) < np.deg2rad(7)
                else a for a in selected]
    if len(selected) == 1:
        selected.append(base + np.pi / 2)
    return [float(a % np.pi) for a in selected]

def default_regularize(raw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = regularize_single_polygon(raw, parallel_threshold=1., allow_45_degree=True,
                    diagonal_threshold_reduction=15., allow_circles=False,
                    circle_threshold=.9, simplify=True, simplify_tolerance=.5)
    return unary_union([r['geometry'] for r in rows])

def guided_regularize(raw, gba):
    """Ordered-contour DP segmentation + Geodan line regularize/merge/intersect.

    Unlike full point-cloud trace_boundary, no hull rebuild, random RANSAC, or
    automatic rectangle fallback. At least three points avoid legacy 2-point bug.
    """
    seed = raw.simplify(.8, preserve_topology=True)
    xy = np.asarray(seed.exterior.coords)
    segments = [BoundarySegment(np.linspace(a, b, max(3, int(np.linalg.norm(b-a) / .5) + 1)))
                for a, b in zip(xy[:-1], xy[1:]) if np.linalg.norm(b-a) > .1]
    segments = regularize_segments(segments, directions(gba), max_error=1.5)
    segments = merge_segments(segments, angle_epsilon=np.deg2rad(5), max_distance=1., max_error=1.5)
    if len(segments) < 3:
        return raw, 'too_few_lines_fallback'
    result = Polygon(compute_intersections(segments, perp_dist_weight=3))
    if not result.is_valid or result.is_empty or poly_iou(result, Polygon(raw.exterior)) < .65:
        return raw, 'invalid_or_excessive_change_fallback'
    # Holes are not filled by exterior regularization.
    result = result.difference(holes_of(raw).union(holes_of(gba)))
    if result.geom_type != 'Polygon':
        return raw, 'multipart_fallback'
    return result, 'geodan_ordered_lines'

def ownership(inside, other, gx, gy):
    own_dist = ndi.distance_transform_edt(~inside, sampling=(gy, gx))
    if not other.any():
        return np.ones(inside.shape, bool)
    other_dist = ndi.distance_transform_edt(~other, sampling=(gy, gx))
    # Ties are deliberately unowned, never merge neighboring instances.
    return (own_dist < other_dist) & ~other

def meaningful_additions(contour, gba, pixel_area):
    """Remove small pieces AND thin fringes even when attached to a large annex.

    Opening acts on difference only; no GBA/small-FP labels are touched. A broad
    core is required over most of the retained region, not merely max thickness.
    """
    kept, rejected = [], []
    # Existing GBA holes are not exterior roof additions and must not be filled.
    gba_shell = unary_union([Polygon(p.exterior) for p in parts(gba)])
    for original in parts(contour.difference(gba_shell)):
        if original.area <= max(9 * pixel_area, 0):
            rejected.append((original, 'at_most_9_pixels'))
            continue
        opened = original.buffer(-1.5, join_style=2).buffer(1.5, join_style=2).intersection(original)
        for q in parts(opened):
            # Component-local opening avoids rejecting a true block with many
            # long thin appendages; all those appendages are explicitly discarded.
            if q.area >= 30 and q.area / max(q.minimum_rotated_rectangle.area, 1e-9) >= .30:
                if q.buffer(-1.5, join_style=2).buffer(1.5, join_style=2).area / q.area >= .60:
                    kept.append(q)
        remainder = original.difference(unary_union([q for q in kept if q.intersects(original)]))
        if not remainder.is_empty:
            rejected.append((remainder, 'thin_fringe_or_under_30m2'))
    return kept, rejected

def evidence(q, contour, gba, raw, w, t):
    mask = R.burn(q, w['signed'].shape, t)
    n = int(mask.sum())
    height = R.stats(mask, w['signed'], w['height_valid'])
    p0, p1 = w['prob']
    rgb = w['rgb']
    green = (rgb[1] > 1.15 * rgb[0]) & (rgb[1] > 1.1 * rgb[2])
    missing = gba.difference(contour).area
    f = dict(area_m2=float(q.area), pixels=n,
             B1_p50=float(np.mean(p1[mask] >= .5)) if n else 0,
             B1_p65=float(np.mean(p1[mask] >= .65)) if n else 0,
             B0_p50=float(np.mean(p0[mask] >= .5)) if n else 0,
             rgb_valid=float(np.mean(w['rgb_valid'][mask])) if n else 0,
             height_valid=height['valid_fraction'], height_p25=height['p25'],
             negative_height=height['negative_fraction'], green_fraction=float(np.mean(green[mask])) if n else 1,
             GBA_coverage=float(gba.intersection(contour).area / gba.area),
             missing_m2=float(missing), contact_m=float(q.boundary.intersection(gba.buffer(.35)).length),
             invented_fraction=float(q.difference(raw).area / q.area))
    gates = dict(rgb_valid=f['rgb_valid'] >= .99,
                 model_support=f['B1_p50'] >= .85 and f['B1_p65'] >= .65 and f['B0_p50'] >= .70,
                 height_support=f['height_valid'] >= .90 and f['height_p25'] is not None and f['height_p25'] >= 2
                     and f['negative_height'] <= .05,
                 vegetation=f['green_fraction'] <= .25,
                 main_retained=f['GBA_coverage'] >= .75,
                 not_balanced_shift=q.area >= 2 * missing,
                 attached=f['contact_m'] >= 3,
                 not_regularizer_invention=f['invented_fraction'] <= .15)
    f['failed_gates'] = ','.join(k for k, v in gates.items() if not v)
    return all(gates.values()), f
