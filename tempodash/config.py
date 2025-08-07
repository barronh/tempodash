__all__ = [
    'api', 'default_where', 'clip', 'pandora_clip',
    'tempo_utc_hours', 'tropomi_utc_hours',
    'locconfigs'
]
import copy
import json
from shapely.ops import prep
from shapely import box, unary_union
import pyrsig
import numpy as np
import os

rawcfg = json.load(open('config.json', 'r'))
dataroot = rawcfg.get('dataroot', os.path.realpath('.'))
server = rawcfg.get('server', 'ofmpub.epa.gov')
baddates = sorted(rawcfg['baddates'])
tropomi_utc_hours = sorted(rawcfg['hours']['tropomi'])
tempo_utc_hours = sorted(rawcfg['hours']['tempo'])
locconfigs = rawcfg['locations']
bbox_polys = [box(*c['bbox']) for ck, c in locconfigs.items()]
clip = prep(unary_union(bbox_polys))
pandora_bbox_polys = [
    box(*c['bbox']) for ck, c in locconfigs.items() if c.get('pandora', False)
]
pandora_clip = prep(unary_union(pandora_bbox_polys))
id2cfg = {}
id2key = {}
for k, cfg in locconfigs.items():
    cfg['centroid'] = (
        np.mean(cfg['bbox'][::2]).round(6),
        np.mean(cfg['bbox'][1::2]).round(6)
    )
    cfg.setdefault('pandora', False)
    cfg.setdefault('pandora_hcho', False)
    if cfg['pandora']:
        id = cfg['pandoraid']
        id2cfg[float(id)] = cfg
        id2key[float(id)] = k

regions = copy.deepcopy(rawcfg['regions'])
regalllocs = []
for reg in regions:
    _locs = reg['lockeys']
    regalllocs.extend(_locs)
    reg['lockeys'].extend([
        locconfigs[k].get('pandoraid', 0)
        for k in _locs if locconfigs[k].get('pandora', False)
    ])

_otherlocs = [
    lockey for lockey in list(locconfigs) if lockey not in regalllocs
]
if len(_otherlocs) > 0:
    print('WARN:: Uncategorized location')
    regions.append({'Other': _otherlocs})

bbox = (-150, 15, -50, 65)
api = pyrsig.RsigApi(bbox=bbox, server=server)
api.tempo_kw['api_key'] = open('/home/bhenders/.tempokey', 'r').read().strip()
api.tempo_kw['maximum_cloud_fraction'] = 1
api.tempo_kw['maximum_solar_zenith_angle'] = 90
api.pandora_kw['minimum_quality'] = 'medium'
api.tropomi_kw['minimum_quality'] = 75
api.tropomi_kw['maximum_cloud_fraction'] = 1.0

default_where = 'tempo_sza <= 70 and tempo_cloud_eff < 0.2'
default_pandora_where = ' or '.join([
    (
        '(tempo_lon >= {0} and tempo_lon <= {2}'
        ' and tempo_lat >= {1} and tempo_lat <= {3})'
    ).format(*cfg['bbox'])
    for k, cfg in locconfigs.items()
    if cfg['pandora']
])
default_pandora_where = f'({default_pandora_where}) and ({default_where})'
default_naa_where = ' or '.join([
    (
        '(tempo_lon >= {0} and tempo_lon <= {2}'
        + ' and tempo_lat >= {1} and tempo_lat <= {3})'
    ).format(*cfg['bbox'])
    for k, cfg in locconfigs.items()
    if not cfg['pandora']
])
default_naa_where = f'({default_naa_where}) and ({default_where})'
where = '(default_naa_where)  or '.join(default_pandora_where)
