__all__ = ['open_tempo', 'open_tempo_chunk', 'get_tempo_chunk']

_coordkeys = [
    'Longitude_SW', 'Latitude_SW',
    'Longitude_SE', 'Latitude_SE',
    'Longitude_NE', 'Latitude_NE',
    'Longitude_NW', 'Latitude_NW',
    'Longitude_SW', 'Latitude_SW',
]


def open_tempo(
    lockey, spc, bdate, edate, freq='1h', asgeo=False, parse_dates=False,
    rename=False
):
    import pandas as pd
    from ctypes import cdll, CDLL
    import gc
    from .util import get_configs
    cfg = get_configs()[lockey]

    cdll.LoadLibrary("libc.so.6")
    libc = CDLL("libc.so.6")

    bhours = pd.date_range(bdate, edate, freq='1h')
    if cfg.get('airnow', False) and spc == 'no2':
        hours = cfg['hours']
    elif cfg.get('pandora', False):
        hours = cfg['hours']
    elif cfg.get('tropomi', False):
        hours = cfg['tropomi_hours']

    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    dfs = []
    for bhour in bhours:
        if bhour.hour in hours:
            ehour = bhour + dt
            try:
                df = open_tempo_chunk(lockey, spc, bhour, ehour)
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


def open_tempo_chunk(lockey, spc, bdate, edate, ltcf=0.3, gtsza=70, troponly=True):
    import functools
    import pandas as pd
    import pyrsig
    from .util import get_configs, get_tempopw
    from . import server
    cfg = get_configs()[lockey]
    bbox = cfg['bbox']

    workdir = f'locations/{lockey}/{bdate:%Y-%m-%d}'
    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
    api.tempo_kw['api_key'] = get_tempopw()
    api.tempo_kw['maximum_cloud_fraction'] = 1

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


def get_tempo_chunk(lockey, spc, bdate, edate, verbose=-1):
    import functools
    import pandas as pd
    import pyrsig
    import os
    from .util import get_configs, get_tempopw
    from . import server

    cfg = get_configs()[lockey]
    bbox = cfg['bbox']
    # airnow only has no2
    # tropomi only has hcho at overpass hours
    # only retrieve hcho for tropomi_hours at airnow sites
    if spc == 'no2' or cfg['pandora']:
        hours = cfg['hours']
    else:
        hours = cfg['tropomi_hours']
    bdate = pd.to_datetime(bdate)
    if bdate.hour not in hours:
        return
    edate = pd.to_datetime(edate)
    workdir = f'locations/{lockey}/{bdate:%Y-%m-%d}'
    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
    api.tempo_kw['api_key'] = get_tempopw()
    api.tempo_kw['maximum_cloud_fraction'] = 0.3
    get = functools.partial(
        api.get_file, formatstr='ascii', bdate=bdate, edate=edate, bbox=bbox,
        compress=1, verbose=verbose, overwrite=False
    )

    if spc == 'hcho':
        keycorners = [
            ('tempo.l2.hcho.vertical_column', 1),
            ('tempo.l2.hcho.eff_cloud_fraction', 0),
            ('tempo.l2.hcho.solar_zenith_angle', 0),
        ]
        get(key='tempo.l2.hcho.vertical_column')
        get(key='tempo.l2.hcho.eff_cloud_fraction', corners=0)
        get(key='tempo.l2.hcho.solar_zenith_angle', corners=0)
    else:
        keycorners = [
            ('tempo.l2.no2.vertical_column_troposphere', 1),
            ('tempo.l2.no2.solar_zenith_angle', 0),
            ('tempo.l2.no2.vertical_column_total', 0),
            ('tempo.l2.no2.vertical_column_stratosphere', 0),
            ('tempo.l2.no2.eff_cloud_fraction', 0),
        ]
    for key, corners in keycorners:
        outpath = get(key=key, corners=corners)
        if corners == 1:
            stats = os.stat(outpath)
            if stats.st_size == 20:
                return
