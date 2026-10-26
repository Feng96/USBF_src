from pathlib import Path
import json, copy, hashlib
from functools import lru_cache
from . import osm_gate as Q
VERSION = 'V7_OSM_circle_confirmation_v1_20260918'
OUT = None
from usbf.io import common as U

def transform_to_utm(zone):
    from pyproj import Transformer
    return Transformer.from_crs(4326,32600+zone,always_xy=True).transform

class References:
    def __init__(self):
        import geopandas as gpd
        self.manifest=U.read(OUT/'OSM_REFERENCE_READY.json');assert self.manifest['status']=='passed'
        assert U.sha(OUT/'osm_references.parquet')==self.manifest['parquet_sha256']
        frame=gpd.read_parquet(OUT/'osm_references.parquet')
        self.byid={(r.osm_type,r.osm_id):r.geometry for r in frame.itertuples()}
        self.ambiguous={tuple(x) for x in self.manifest['ambiguous_ids']}
        self.manifest_sha256=U.sha(OUT/'OSM_REFERENCE_READY.json')
    def assess(self,source,identifier,original,candidate):
        from shapely.ops import transform
        key=None;reason='not_OSM_source_exact_match_unavailable'
        if str(source).lower()=='osm':
            try:key=Q.osm_key(identifier)
            except (ValueError,AssertionError,__import__('decimal').InvalidOperation):reason='unparseable_OSM_id'
        if key is not None:
            reason='ambiguous_OSM_snapshot_geometry' if key in self.ambiguous else 'no_verified_local_OSM_reference'
        ref=None if key is None or key in self.ambiguous else self.byid.get(key)
        if ref is None:return dict(accepted=False,status='unconfirmed',reason=reason,osm_type=key[0] if key else '',osm_id=key[1] if key else '')
        zone=max(10,min(19,int((original.centroid.x+180)//6)+1));project=transform_to_utm(zone)
        a,b,c=[transform(project,g) for g in [original,candidate,ref]]
        if not all(g.is_valid and not g.is_empty for g in [a,b,c]):
            return dict(accepted=False,status='unconfirmed',reason='invalid_projected_reference_or_candidate',osm_type=key[0],osm_id=key[1])
        audit=Q.gate(a,b,c)
        return dict(**audit,status='confirmed' if audit['accepted'] else 'rejected',osm_type=key[0],osm_id=key[1],matching='exact_OSM_type_and_id',metric_crs=f'EPSG:{32600+zone}')

def apply(rec,original,source,identifier,refs=None):
    import shapely
    refs=refs or references();rec=copy.deepcopy(rec)
    if rec.get('circle_gate_policy')==VERSION:
        assert rec['circle_reference_manifest_sha256']==refs.manifest_sha256
        return rec
    candidate=shapely.from_wkb(bytes.fromhex(rec['V7_wkb']))
    circle=bool(rec.get('circle_adopted') or rec.get('old_circle')) and not candidate.equals_exact(original,0)
    rec.update(circle_gate_policy=VERSION,circle_reference_manifest_sha256=refs.manifest_sha256,
        circle_gate_applied=circle,circle_proposed_before_gate=circle,circle_gate_status='not_applicable',circle_gate_reason='not_a_changed_circle',
        circle_gate_osm_type='',circle_gate_osm_id='')
    if not circle:return rec
    audit=refs.assess(source,identifier,original,candidate)
    rec.update(circle_gate_status=audit['status'],circle_gate_reason=audit['reason'],circle_gate_osm_type=audit.get('osm_type',''),circle_gate_osm_id=audit.get('osm_id',''))
    rec.setdefault('evidence',{})['OSM_circle_gate']=dict(**audit,policy=VERSION,reference_manifest_sha256=refs.manifest_sha256,
        candidate_wkb=rec['V7_wkb'],original_wkb_sha256=hashlib.sha256(original.wkb).hexdigest(),mixed_hollow_before_gate=bool(rec.get('hollow')))
    if not audit['accepted']:
        assert original.is_valid and not original.is_empty
        rec.update(V7_wkb=original.wkb.hex(),status='osm_circle_gate_keep_GBA',V7_decision='keep_exact_GBA_unconfirmed_circle',
            circle_adopted=False,old_circle=False,hollow=False,V7_changed=False,V6_changed=False)
    return rec

def columns(records):
    keys=['circle_gate_policy','circle_reference_manifest_sha256','circle_gate_applied','circle_proposed_before_gate','circle_gate_status','circle_gate_reason','circle_gate_osm_type','circle_gate_osm_id']
    return {k:[r[k] for r in records] for k in keys}
