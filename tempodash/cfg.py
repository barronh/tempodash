import os
import pyrsig
import json
import pandas as pd
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely import box
# Using malloc_trim to prevent memory allocation runaway
# https://github.com/pandas-dev/pandas/issues/2659
from ctypes import cdll, CDLL
cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")


bbox = (-150, 15, -50, 65)
server = 'maple.hesc.epa.gov'
api = pyrsig.RsigApi(bbox=bbox, server=server)
api.tempo_kw['api_key'] = open('/home/bhenders/.tempokey', 'r').read().strip()
api.tempo_kw['maximum_cloud_fraction'] = 1
api.tempo_kw['maximum_solar_zenith_angle'] = 90
api.pandora_kw['minimum_quality'] = 'medium'
api.tropomi_kw['minimum_quality'] = 75
api.tropomi_kw['maximum_cloud_fraction'] = 1.0


# explicit for sorting control
keys = [
    'tempo.l2.no2.vertical_column_troposphere',
    'tempo.l2.no2.solar_zenith_angle',
    'tempo.l2.no2.vertical_column_total',
    'tempo.l2.no2.vertical_column_stratosphere',
    'tempo.l2.no2.eff_cloud_fraction',
    'tempo.l2.hcho.vertical_column',
    'tempo.l2.hcho.eff_cloud_fraction',
    'tempo.l2.hcho.solar_zenith_angle',
    # 'tropomi.nrti.no2.nitrogendioxide_tropospheric_column',
    # 'tropomi.nrti.no2.nitrogendioxide_stratospheric_column',
    # 'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column',
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
    'tropomi.offl.no2.nitrogendioxide_stratospheric_column',
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column',
    'airnow.no2',
    'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount',
    'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount',
]

configs = json.load(open('config.json', 'r'))
configs.pop('comments')
opts = configs.pop('options', {})

pandora_keys = [
    lk for lk, lcfg in configs.items() if lcfg.get('pandora', False)
]

bbox_polys = [box(*c['bbox']) for ck, c in configs.items()]
analysis_mpoly = prep(unary_union(bbox_polys))

reftime = pd.to_datetime('1970-01-01T00Z')
data_dt = pd.to_timedelta('3599s')
data_freq = '1h'
if os.path.exists('baddates.csv'):
    baddatesdf = pd.read_csv('baddates.csv')
else:
    baddatesdf = pd.DataFrame([
        {
            "Date": "2020-01-01",
            "Comment": (
                "Data dropouts (10-18 10-19 10-20 10-21 10-30 11-09) are not"
                + " flagged here"
            )
        },
        {"Date": "2023-10-29", "Comment": "Bad INR"},
        {"Date": "2023-11-08", "Comment": "Bad INR S01-S07"},
    ])
baddates = pd.to_datetime(baddatesdf['Date'].values, utc=True)

allhours = []
for k, cfg in configs.items():
    allhours.extend(cfg['hours'])

allhours = sorted(set(allhours))
tropomihours = []
for k, cfg in configs.items():
    tropomihours.extend(cfg['tropomi_hours'])

tropomihours = sorted(set(tropomihours))
enddate = pd.to_datetime('now', utc=True).floor('d') - pd.to_timedelta('15d')

if os.path.exists('dates.csv'):
    datedf = pd.read_csv('dates.csv')
    datedf['date'] = dates = pd.to_datetime(datedf['date'].values)
    # Remove bad dates or dates that cannot be compared (e.g., night)
    _keepdate = dates.hour.isin(allhours) & ~dates.floor('1d').isin(baddates)
    datedf = datedf.loc[_keepdate]
    dates = dates[_keepdate]
    version_dates = datedf.groupby('version').agg(
        first_date=('date', 'min'), last_date=('date', 'max'),
        count=('date', 'count')
    )
else:
    dates = pd.date_range('2023-08-02T00Z', '2023-08-03T00Z', freq='1h').union(
        pd.date_range('2023-08-09T00Z', '2023-08-10T00Z', freq='1h')
    ).union(
        pd.date_range('2023-08-16T00Z', '2023-08-17T00Z', freq='1h')
    ).union(
        pd.date_range('2023-08-21T00Z', '2023-08-24T00Z', freq='1h')
    ).union(
        pd.date_range('2023-08-25T00Z', '2023-10-16T23Z', freq='1h')
    ).union(
        pd.date_range('2023-10-17T00Z', enddate, freq='1h'),
    )
    _keepdate = dates.hour.isin(allhours) & ~dates.floor('1d').isin(baddates)
    dates = dates[_keepdate]
    datedf = pd.DataFrame(dict(date=dates))
    datedf['version'] = 0.0
    _v1start = pd.to_datetime('2023-10-01T00', utc=True)
    _v2start = pd.to_datetime('2024-02-26T00', utc=True)
    _v3start = pd.to_datetime('2024-05-13T00', utc=True)
    datedf.loc[datedf.date >= _v1start, 'version'] = 1.0
    datedf.loc[datedf.date >= _v2start, 'version'] = 2.0
    datedf.loc[datedf.date >= _v3start, 'version'] = 3.0

v1dates = datedf.query('version == 1.0')['date']
v2dates = datedf.query('version == 2.0')['date']
v3dates = datedf.query('version == 3.0')['date']
v1start = v1dates.min()
v2start = v2dates.min()
v3start = v3dates.min()

pandora_date_range = pd.to_datetime([dates[0], dates[-1] + data_dt])


queries = []
if len(v1dates) > 0:
    queries.append(('v1', 'v1 == True and tempo_cloud_eff < 0.15', 'v1'))
if len(v2dates) > 0:
    queries.append(('v2', 'v2 == True and tempo_cloud_eff < 0.15', 'v2'))
if len(v3dates) > 0:
    queries.append(('v3', 'v3 == True and tempo_cloud_eff < 0.15', 'v3'))
if len(queries) > 1:
    queries.insert(0, (
        'all',
        '(v1 == True or v2 == True or v3 == True) and tempo_cloud_eff < 0.15',
        'v1-3'
    ))

exclude_pandora_hcho_ids = [
    float(cfg.get('pandoraid', [-999])[0])
    for key, cfg in configs.items()
    if cfg.get('pandora_hcho', False) is False and cfg.get('pandora', False)
]
exclude_pandora_hcho_lockeys = [
    key
    for key, cfg in configs.items()
    if cfg.get('pandora_hcho', False) is False and cfg.get('pandora', False)
]
