"""Meter-based raster contour diagnostics, not cadastral/vector GT accuracy.

Each tile's center geodesic pixel spacings are used for anisotropic EDT. The
approximation is recorded; it is not a global degrees-to-meters constant.
Distances exclude unknown-domain/crop neighborhoods and preserve inner rings.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def pixel_spacing(crs, transform, shape):
    from pyproj import Transformer, Geod
    converter = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    h, w = shape
    points = [converter.transform(*(transform * (w / 2 + dx, h / 2 + dy))) for dx, dy in ((0, 0), (1, 0), (0, 1))]
    geod = Geod(ellps='WGS84')
    dx = abs(geod.inv(*points[0], *points[1])[2]); dy = abs(geod.inv(*points[0], *points[2])[2])
    assert .1 < dx < 5 and .1 < dy < 5
    return dy, dx


class MetricContours:
    def __init__(self):
        self.ep = self.et = 0
        self.matches = {1.: [0, 0], 2.: [0, 0]}
        self.distances = []

    def add(self, prediction, target, valid, spacing):
        prediction, target, valid = [np.asarray(a, dtype=bool) for a in (prediction, target, valid)]
        radius = int(np.ceil(2 / min(spacing))) + 2
        domain = binary_erosion(valid, structure=np.ones((2 * radius + 1, 2 * radius + 1)), border_value=0)
        structure = np.ones((3, 3), bool)
        ep = prediction & ~binary_erosion(prediction, structure=structure, border_value=1)
        et = target & ~binary_erosion(target, structure=structure, border_value=1)
        ep &= domain; et &= domain
        self.ep += int(ep.sum()); self.et += int(et.sum())
        cap = float(np.hypot(prediction.shape[0] * spacing[0], prediction.shape[1] * spacing[1]))
        dp = distance_transform_edt(~et, sampling=spacing)[ep] if et.any() else np.full(ep.sum(), cap)
        dt = distance_transform_edt(~ep, sampling=spacing)[et] if ep.any() else np.full(et.sum(), cap)
        for tolerance, counts in self.matches.items():
            counts[0] += int((dp <= tolerance).sum()); counts[1] += int((dt <= tolerance).sum())
        if dp.size: self.distances.append(dp.astype('float32'))
        if dt.size: self.distances.append(dt.astype('float32'))

    def result(self):
        result = {}
        for tolerance, (mp, mt) in self.matches.items():
            p, r = mp / max(1, self.ep), mt / max(1, self.et)
            result['boundary_f1_%dm' % tolerance] = 1. if not self.ep and not self.et else 2 * p * r / max(1e-12, p + r)
        all_distances = np.concatenate(self.distances) if self.distances else np.zeros(1)
        result.update(boundary_p95_m=float(np.percentile(all_distances, 95)),
                      mean_symmetric_boundary_distance_m=float(all_distances.mean()),
                      metric_contract='Raster contour centers; per-tile center geodesic dx/dy; unknown/rim excluded; empty counterpart uses tile diagonal')
        return result
