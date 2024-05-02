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

data_dt = pd.to_timedelta('3599s')
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
baddates = baddatesdf['Date'].values
bbox = (-150, 15, -50, 65)
api = pyrsig.RsigApi(bbox=bbox, server='maple.hesc.epa.gov')
api.tempo_kw['api_key'] = open('/home/bhenders/.tempokey', 'r').read().strip()
api.tempo_kw['maximum_cloud_fraction'] = 1
api.pandora_kw['minimum_quality'] = 'medium'
api.tropomi_kw['minimum_quality']: 75
api.tropomi_kw['maximum_cloud_fraction']: 1.0

keycorners = {
    'tempo.l2.no2.vertical_column_troposphere': 1,
    'tempo.l2.no2.solar_zenith_angle': 0,
    'tempo.l2.no2.vertical_column_total': 0,
    'tempo.l2.no2.vertical_column_stratosphere': 0,
    'tempo.l2.no2.eff_cloud_fraction': 0,
    'tempo.l2.hcho.vertical_column': 1,
    'tempo.l2.hcho.eff_cloud_fraction': 0,
    'tempo.l2.hcho.solar_zenith_angle': 0,
    'tropomi.nrti.no2.nitrogendioxide_tropospheric_column': 1,
    'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column': 1,
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column': 1,
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column': 1,
    'airnow.no2': 0,
    'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount': 0,
    'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount': 0,

}

idkeys = ['Timestamp', 'LONGITUDE', 'LATITUDE']
crnrkeys = [
    'Longitude_SW', 'Latitude_SW',
    'Longitude_SE', 'Latitude_SE',
    'Longitude_NE', 'Latitude_NE',
    'Longitude_NW', 'Latitude_NW',
]
crnrcoordkeys = crnrkeys + [
    'Longitude_SW', 'Latitude_SW',
]

keycols = {
    'tempo.l2.no2.vertical_column_troposphere': idkeys + crnrkeys + [
        'no2_vertical_column_troposphere'
    ],
    'tempo.l2.no2.solar_zenith_angle': idkeys + [
        'solar_zenith_angle'
    ],
    'tempo.l2.no2.vertical_column_total': idkeys + [
        'no2_vertical_column_total'
    ],
    'tempo.l2.no2.vertical_column_stratosphere': idkeys + [
        'no2_vertical_column_stratosphere'
    ],
    'tempo.l2.no2.eff_cloud_fraction': idkeys + [
        'eff_cloud_fraction'
    ],
    'tempo.l2.hcho.vertical_column': idkeys + [
        'vertical_column'
    ],
    'tempo.l2.hcho.eff_cloud_fraction': idkeys + [
        'eff_cloud_fraction'
    ],
    'tempo.l2.hcho.solar_zenith_angle': idkeys + [
        'solar_zenith_angle'
    ],
    'tropomi.nrti.no2.nitrogendioxide_tropospheric_column': (
        idkeys + crnrkeys
        + [
            'nitrogendioxide_tropospheric_column'
        ]
    ),
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column': (
        idkeys + crnrkeys
        + [
            'nitrogendioxide_tropospheric_column'
        ]
    ),
    'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column': (
        idkeys + crnrkeys + [
            'formaldehyde_tropospheric_vertical_column'
        ]
    ),
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column': (
        idkeys + crnrkeys + [
            'formaldehyde_tropospheric_vertical_column'
        ]
    ),
    'airnow.no2': idkeys + [
        'no2'
    ],
    'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount': (
        idkeys + [
            'ELEVATION', 'STATION', 'NOTE',
            'nitrogen_dioxide_vertical_column_amount'
        ]
    ),
    'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount': (
        idkeys + [
            'ELEVATION', 'STATION', 'NOTE',
            'formaldehyde_total_vertical_column_amount'
        ]
    )
}

proddef = {
    'tempo.l2.no2': [
        'tempo.l2.no2.vertical_column_troposphere',
        'tempo.l2.no2.solar_zenith_angle',
        'tempo.l2.no2.vertical_column_total',
        'tempo.l2.no2.vertical_column_stratosphere',
        'tempo.l2.no2.eff_cloud_fraction',
    ],
    'tempo.l2.hcho': [
        'tempo.l2.hcho.vertical_column',
        'tempo.l2.hcho.solar_zenith_angle',
        'tempo.l2.hcho.eff_cloud_fraction',
    ],
    'tropomi.nrti.no2': [
        'tropomi.nrti.no2.nitrogendioxide_tropospheric_column',
    ],
    'tropomi.offl.no2': [
        'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
    ],
    'tropomi.nrti.hcho': [
        'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column',
    ],
    'tropomi.offl.hcho': [
        'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column',
    ],
    'airnow.no2': ['airnow.no2'],
    'pandora.no2': [
        'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
    ],
    'pandora.hcho': [
        'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount'
    ],
}

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
    'tropomi.nrti.no2.nitrogendioxide_tropospheric_column',
    'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column',
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column',
    'airnow.no2',
    'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount',
    'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount',
]

configs = json.load(open('config.json', 'r'))
configs.pop('comments')
allhours = []
for k, cfg in configs.items():
    allhours.extend(cfg['hours'])

allhours = sorted(set(allhours))
tropomihours = []
for k, cfg in configs.items():
    tropomihours.extend(cfg['tropomi_hours'])

tropomihours = sorted(set(tropomihours))
dates = pd.date_range('2023-10-17T00Z', '2024-04-21T00Z', freq='1h')
dates = [
    d for d in dates
    if d.hour in allhours and d.strftime('%Y-%m-%d') not in baddates
]
pandora_date_range = pd.to_datetime([dates[0], dates[-1] + data_dt])
pandora_keys = [
    lk for lk, lcfg in configs.items() if lcfg.get('pandora', False)
]

bbox_polys = [box(*c['bbox']) for ck, c in configs.items()]
analysis_mpoly = prep(unary_union(bbox_polys))
v1start = pd.to_datetime('2023-10-01T00', utc=True)
v2start = pd.to_datetime('2024-02-26T00', utc=True)
v1dates = [d for d in dates if d > v1start and d < v2start]
v2dates = [d for d in dates if d > v2start or d < v1start]
reftime = pd.to_datetime('1970-01-01T00Z')
