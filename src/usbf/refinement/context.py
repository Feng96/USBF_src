import json
from collections import OrderedDict
import numpy as np
import geopandas as gpd
import rasterio, shapely
from shapely.ops import transform as transform_geometry
from shapely import affinity
from shapely.geometry import box
from pyproj import Transformer, Geod
from scipy import ndimage as ndi
from usbf.data.sensors import load_sensor
from usbf.io.common import read, sha
OUT = None

def transform(g, source, dest):
    return transform_geometry(Transformer.from_crs(source, dest, always_xy=True).transform, g)

class Context:
    def __init__(self):
        self.rows = {r['name']: r for r in read(OUT / 'INVENTORY.json')['records']}
        self.frames = gpd.read_parquet(OUT / 'frames.parquet')
        self.meta = {r['name']: r for r in self.frames.drop(columns='geometry').to_dict('records')}
        self.probs = {r['name']: r for r in read(OUT / 'PROBABILITIES.json')['records']}
        self.cache = OrderedDict()
        self.source_cache = {}

    def patch(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        p = load_sensor(self.rows[name])
        spec = self.probs[name]
        assert sha(spec['probability']) == spec['sha256']
        with rasterio.open(spec['probability']) as src:
            assert (src.crs, src.transform, src.shape) == (p['crs'], p['transform'], p['signed'].shape)
            p['prob'] = src.read()
            p['prob'][:, ~p['rgb_valid']] = np.nan
        self.cache[name] = p
        if len(self.cache) > 16:
            self.cache.popitem(last=False)
        return p

    def project(self, g, source, dest):
        key = str(source), str(dest)
        if key not in self.source_cache:
            self.source_cache[key] = Transformer.from_crs(source, dest, always_xy=True).transform
        return transform_geometry(self.source_cache[key], g)

    def local(self, g, w):
        native = self.project(g, 4326, w['crs'])
        it = ~w['transform']
        pixel = affinity.affine_transform(native, [it.a, it.b, it.d, it.e, it.c, it.f])
        return affinity.scale(pixel, xfact=w['gx'], yfact=w['gy'], origin=(0, 0))

    def window(self, g, anchor):
        if anchor not in self.probs:
            return None, 'anchor_inference_unavailable'
        p = self.patch(anchor)
        native = self.project(g, 4326, p['crs'])
        t = p['transform']
        center = native.centroid
        lon, lat = self.project(center, p['crs'], 4326).coords[0]
        dx = self.project(shapely.Point(center.x + t.a, center.y), p['crs'], 4326)
        dy = self.project(shapely.Point(center.x, center.y + t.e), p['crs'], 4326)
        geod = Geod(ellps='WGS84')
        gx, gy = abs(geod.inv(lon, lat, dx.x, dx.y)[2]), abs(geod.inv(lon, lat, dy.x, dy.y)[2])
        it = ~t
        pixels = affinity.affine_transform(native, [it.a, it.b, it.d, it.e, it.c, it.f])
        x0, y0, x1, y1 = pixels.bounds
        pad = int(np.ceil(15 / min(gx, gy)))
        left, top, right, bottom = int(np.floor(x0)) - pad, int(np.floor(y0)) - pad, int(np.ceil(x1)) + pad, int(np.ceil(y1)) + pad
        h, w = bottom - top, right - left
        if max(w, h) > 2048:
            return None, 'window_over_2048_pending'
        t = t * rasterio.Affine.translation(left, top)
        window_bounds = box(*rasterio.transform.array_bounds(h, w, t))
        footprint = self.project(window_bounds, p['crs'], 4326)
        ids = self.frames.sindex.query(footprint, predicate='intersects')
        names = [anchor] + sorted(n for n in self.frames.iloc[ids]['name'] if n != anchor)
        rgb, signed, prob = np.zeros((3, h, w), 'float32'), np.full((h, w), np.nan, 'float32'), np.full((2, h, w), np.nan, 'float32')
        rv, hv, occupied = np.zeros((h, w), bool), np.zeros((h, w), bool), np.zeros((h, w), bool)
        used, incompatible = [], []
        for name in names:
            if name not in self.probs:
                continue
            meta = self.meta[name]
            a = rasterio.Affine(*json.loads(meta['transform'])[:6])
            xy = (~t) * (a.c, a.f)
            col, row = int(round(xy[0])), int(round(xy[1]))
            if meta['crs'] != str(p['crs']) or max(abs(xy[0] - col), abs(xy[1] - row)) > .01 or not np.allclose(
                    [a.a, a.b, a.d, a.e], [t.a, t.b, t.d, t.e], rtol=0, atol=1e-12):
                incompatible.append(name)
                continue
            l, u, r, b = max(0, col), max(0, row), min(w, col + meta['width']), min(h, row + meta['height'])
            if l >= r or u >= b:
                continue
            tile = self.patch(name)
            target, source = np.s_[u:b, l:r], np.s_[u-row:b-row, l-col:r-col]
            take = ~occupied[target]
            rgb[:, u:b, l:r][:, take] = tile['rgb'][:, u-row:b-row, l-col:r-col][:, take]
            signed[target][take] = tile['signed'][source][take]
            prob[:, u:b, l:r][:, take] = tile['prob'][:, u-row:b-row, l-col:r-col][:, take]
            rv[target][take], hv[target][take] = tile['rgb_valid'][source][take], tile['height_valid'][source][take]
            occupied[target][take] = True
            used.append(name)
        window = dict(rgb=rgb, signed=signed, prob=prob, rgb_valid=ndi.binary_erosion(rv, iterations=2),
                      height_valid=hv, gx=gx, gy=gy, transform=t, crs=p['crs'],
                      patches=used, incompatible_grid_neighbors=incompatible)
        window['g'] = self.local(g, window)
        return window, 'built'

def to_world(g, w):
    pixels = affinity.scale(g, xfact=1 / w['gx'], yfact=1 / w['gy'], origin=(0, 0))
    t = w['transform']
    native = affinity.affine_transform(pixels, [t.a, t.b, t.d, t.e, t.c, t.f])
    return transform(native, w['crs'], 4326)
