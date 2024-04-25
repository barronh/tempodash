__all__ = ['gettempo', 'get_tempo_chunk']

_coordkeys = [
    'Longitude_SW', 'Latitude_SW',
    'Longitude_SE', 'Latitude_SE',
    'Longitude_NE', 'Latitude_NE',
    'Longitude_NW', 'Latitude_NW',
    'Longitude_SW', 'Latitude_SW',
]


def gettempo(
    lockey, spc, bdate, edate, freq='1h', asgeo=False, parse_dates=False,
    rename=False
):
    import os
    import pyrsig
    import pandas as pd
    from . import get_configs, server
    from ctypes import cdll, CDLL
    import gc

    cdll.LoadLibrary("libc.so.6")
    libc = CDLL("libc.so.6")
    cfg = get_configs()[lockey]
    bbox = cfg['bbox']
    tempo_pw = open(os.path.expanduser('~/.tempokey'), 'r').read().strip()

    bhours = pd.date_range(bdate, edate, freq='1h')
    if cfg.get('airnow', False) or cfg.get('pandora', False):
        hours = cfg['hours']
    elif cfg.get('tropomi', False):
        hours = cfg['tropomi_hours']

    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    dfs = []
    for bhour in bhours:
        if bhour.hour in hours:
            workdir = f'locations/{lockey}/{bhour:%Y-%m-%d}'
            api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
            api.tempo_kw['api_key'] = tempo_pw
            api.tempo_kw['maximum_cloud_fraction'] = 0.3
            ehour = bhour + dt
            try:
                df = get_tempo_chunk(api, spc, bhour, ehour)
                dfs.append(df)
            except pd.errors.EmptyDataError:
                pass
        gc.collect()
        libc.malloc_trim(0)

    df = pd.concat(dfs)
    if parse_dates:
        df['tempo_time'] = pd.to_datetime(df['Timestamp'])
    if spc == 'no2':
        df['no2_vertical_column_sum'] = (
            df['no2_vertical_column_troposphere']
            + df['no2_vertical_column_stratosphere']
        )
    if asgeo:
        import geopandas as gpd
        from shapely import polygons
        geom = polygons(df[_coordkeys].values.reshape(-1, 5, 2))
        df = gpd.GeoDataFrame(df, geometry=geom, crs=4326)

    if rename:
        df.rename(columns=dict(
            LONGITUDE='tempo_lon', LATITUDE='tempo_lat',
            nitrogen_dioxide_vertical_column_amount='pandora_no2',
            no2_vertical_column_troposphere='tempo_no2_trop',
            no2_vertical_column_stratosphere='tempo_no2_strat',
            no2_vertical_column_total='tempo_no2_total',
            no2_vertical_column_sum='tempo_no2_sum',
            vertical_column='tempo_hcho',
            solar_zenith_angle='tempo_sza',
            eff_cloud_fraction='tempo_ecf',
        ), inplace=True)

    return df


def get_tempo_chunk(api, spc, bdate, edate, ltcf=0.3, gtsza=70):
    import functools
    import pandas as pd

    jkeys = ['Timestamp', 'LONGITUDE', 'LATITUDE']
    get = functools.partial(
        api.to_dataframe, bdate=bdate, edate=edate, unit_keys=False,
        verbose=-10
    )
    if spc == 'hcho':
        tedf = get('tempo.l2.hcho.vertical_column')
        tecdf = get('tempo.l2.hcho.eff_cloud_fraction', corners=0)
        keepkeys = jkeys + ['eff_cloud_fraction']
        tedf = pd.merge(tedf, tecdf[keepkeys], on=jkeys).query(
            f'eff_cloud_fraction < {ltcf}'
        )
        tszdf = get('tempo.l2.hcho.solar_zenith_angle', corners=0)
        keepkeys = jkeys + ['solar_zenith_angle']
        tedf = pd.merge(tedf, tszdf[keepkeys], on=jkeys)
    else:
        tentdf = get('tempo.l2.no2.vertical_column_troposphere', corners=1)
        tenvdf = get('tempo.l2.no2.vertical_column_total', corners=0)
        tensdf = get('tempo.l2.no2.vertical_column_stratosphere', corners=0)
        keepkeys = jkeys + ['no2_vertical_column_stratosphere']
        tenvsdf = pd.merge(tenvdf, tensdf[keepkeys], on=jkeys)
        keepkeys = jkeys + [
            'no2_vertical_column_total',
            'no2_vertical_column_stratosphere'
        ]
        tedf = pd.merge(tentdf, tenvsdf[keepkeys], on=jkeys)

        tecdf = get('tempo.l2.no2.eff_cloud_fraction', corners=0)
        tszdf = get('tempo.l2.no2.solar_zenith_angle', corners=0)
        keepkeys = jkeys + ['eff_cloud_fraction']
        tedf = pd.merge(tedf, tecdf[keepkeys], on=jkeys)
        keepkeys = jkeys + ['solar_zenith_angle']
        tedf = pd.merge(tedf, tszdf[keepkeys], on=jkeys)
        tedf = tedf.query(
            f'eff_cloud_fraction < {ltcf}'
            # + ' and solar_zenith_angle > {gtsza}'
        )

    return tedf
