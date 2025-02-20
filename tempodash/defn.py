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
    'tropomi.nrti.no2.nitrogendioxide_stratospheric_column': 0,
    'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column': 1,
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column': 1,
    'tropomi.offl.no2.nitrogendioxide_stratospheric_column': 0,
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
    'tropomi.nrti.no2.nitrogendioxide_stratospheric_column': (
        idkeys
        + [
            'nitrogendioxide_stratospheric_column'
        ]
    ),
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column': (
        idkeys + crnrkeys
        + [
            'nitrogendioxide_tropospheric_column'
        ]
    ),
    'tropomi.offl.no2.nitrogendioxide_stratospheric_column': (
        idkeys
        + [
            'nitrogendioxide_stratospheric_column'
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
        'tropomi.nrti.no2.nitrogendioxide_stratospheric_column',
    ],
    'tropomi.offl.no2': [
        'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
        'tropomi.offl.no2.nitrogendioxide_stratospheric_column',
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
