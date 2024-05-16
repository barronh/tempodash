from . import cfg


def openchunk(key, bdate, backend='xdr'):
    """
    Arguments
    ---------
    key : str
        RSIG data key
    bdate : datetime
        must support strftime
    backend : str
        'xdr' or 'ascii'

    Returns
    -------
    df : pandas.DataFrame
        loaded from pyrsig.RsigApi.to_dataframe
    """

    source = key.split('.')[0]
    edate = bdate + cfg.data_dt
    cfg.api.workdir = f'data/{source}/{bdate:%Y-%m}'
    df = cfg.api.to_dataframe(
        key, bdate=bdate, edate=edate,
        unit_keys=False, backend=backend, verbose=-1
    )

    if df.shape[1] > 0:
        df = df[cfg.keycols[key]]
    cfg.libc.malloc_trim(0)
    return df


def open_prod(prod, dates=None, verbose=0):
    """
    Arguments
    ---------
    prod : str
        Compound product defined in cfg.proddef
    dates : list
        List of dates to be processed.
    verbose : int
        Level of verbosity

    Returns
    -------
    df : pandas.DataFrame
        Individual keys loaded from pyrsig.RsigApi.to_dataframe
        and combined to make a product level dataframe
    """
    import geopandas as gpd
    import pandas as pd
    from shapely import polygons, points
    import gc
    import time
    crnrkeys = cfg.crnrcoordkeys

    keys = cfg.proddef[prod]
    if dates is None:
        dates = cfg.dates
    dates = pd.to_datetime(dates)
    daydfs = []
    for bdate in dates:
        dfs = []
        for key in keys:
            if verbose > 0:
                print('open', key, bdate, end='...', flush=True)
                t0 = time.time()
            df = openchunk(key, bdate)
            if df.shape[0] > 0:
                pts = points(
                    df[['LONGITUDE', 'LATITUDE']].values.reshape(-1, 2)
                )
                inanalysis = cfg.analysis_mpoly.intersects(pts)
                dfs.append(df.loc[inanalysis])
            if verbose > 0:
                t1 = time.time()
                print(f'{t1 - t0:.1f}')

        if len(dfs) != len(keys):
            continue
        df = dfs[0]
        for i in range(1, len(keys)):
            df = pd.merge(df, dfs[i])
            gc.collect()
        daydfs.append(df)

    if len(daydfs) == 0:
        return gpd.GeoDataFrame()
    df = pd.concat(daydfs, ignore_index=True)
    if 'Longitude_SW' in df.columns:
        geom = polygons(df[crnrkeys].values.reshape(-1, 5, 2))
        df.drop(columns=crnrkeys, inplace=True)
    elif 'LONGITUDE' in df.columns:
        geom = points(df[['LONGITUDE', 'LATITUDE']].values.reshape(-1, 2))
    else:
        raise KeyError('no geom')
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    return gdf


def open_pandora(prod, verbose=1):
    """
    Arguments
    ---------
    prod : str
        Compound product defined in cfg.proddef
    verbose : int
        Level of verbosity

    Returns
    -------
    df : pandas.DataFrame
        Individual keys loaded from pyrsig.RsigApi.to_dataframe
        and combined to make a product level dataframe
    """

    import geopandas as gpd
    import pandas as pd
    from shapely import points

    bdate, edate = cfg.pandora_date_range
    key = cfg.proddef[prod][0]
    source = key.split('.')[0]
    dfs = []
    qa = cfg.api.pandora_kw['minimum_quality']
    for lockey in cfg.pandora_keys:
        if verbose > 0:
            print(lockey, end='.', flush=True)
        lcfg = cfg.configs[lockey]
        pid = lcfg['pandoraid'][0]
        cfg.api.workdir = f'data/{source}/{lockey}/{qa}'
        cfg.api.bbox = lcfg['bbox']
        df = cfg.api.to_dataframe(
            key, bdate=bdate, edate=edate,
            unit_keys=False, backend='xdr', verbose=-1
        )
        if df.shape[0] > 0:
            dfs.append(df.query(f'STATION == {pid}'))

    if verbose > 0:
        print('\nstack pandora', end='.', flush=True)

    if len(dfs) == 0:
        return df

    df = pd.concat(dfs, ignore_index=True)
    if df.shape[1] > 0:
        df = df[cfg.keycols[key]]
    cfg.libc.malloc_trim(0)
    if verbose > 0:
        print('\nadd geometry pandora', end='.', flush=True)
    geom = points(df[['LONGITUDE', 'LATITUDE']].values.reshape(-1, 2))
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    if verbose > 0:
        print()

    return gdf


def makeintx(spc, dates=None, verbose=1):
    """
    Arguments
    ---------
    spc : str
        'no2' or 'hcho'
    dates : list
        List of dates to be processed.
    verbose : int
        Level of verbosity

    Returns
    -------
    None
    """

    import pandas as pd
    import geopandas as gpd
    import time
    import os
    import gc
    from os.path import exists
    import warnings

    if dates is None:
        dates = cfg.dates
    dates = pd.to_datetime(dates)
    if verbose > 0:
        print('open all pandora')
    pdfs = open_pandora(f'pandora.{spc}')
    for bdate in dates:
        gc.collect()
        edate = bdate + cfg.data_dt
        pixpath = (
            f'intx/pandora/{bdate:%Y-%m}/pandora_intx_{spc}'
            + f'_{bdate:%Y-%m-%dT%H}_0pt00.csv.gz'
        )
        pix3path = (
            f'intx/pandora/{bdate:%Y-%m}/pandora_intx_{spc}'
            + f'_{bdate:%Y-%m-%dT%H}_0pt03.csv.gz'
        )
        aixpath = (
            f'intx/airnow/{bdate:%Y-%m}/airnow_intx_{spc}'
            + f'_{bdate:%Y-%m-%dT%H}.csv.gz'
        )
        tnixpath = (
            f'intx/tropomi/{bdate:%Y-%m}/tropomi_nrti_intx_{spc}'
            + f'_{bdate:%Y-%m-%dT%H}.csv.gz'
        )
        toixpath = (
            f'intx/tropomi/{bdate:%Y-%m}/tropomi_offl_intx_{spc}'
            + f'_{bdate:%Y-%m-%dT%H}.csv.gz'
        )
        for ixp in [pixpath, aixpath, tnixpath, toixpath]:
            os.makedirs(os.path.dirname(ixp), exist_ok=True)

        dopandora = not (exists(pixpath) and exists(pixpath))
        dotropomin = not exists(tnixpath) and bdate.hour in cfg.tropomihours
        dotropomio = not exists(toixpath) and bdate.hour in cfg.tropomihours
        doairnow = not exists(aixpath) and spc == 'no2'
        if (
            not dopandora and not doairnow
            and not dotropomin and not dotropomio
        ):
            continue
        if verbose > 0:
            print(bdate, spc, 'open tempo', end='...', flush=True)
            t0 = time.time()
        df = open_prod(f'tempo.l2.{spc}', dates=[bdate])
        if verbose > 0:
            print(f'{time.time() - t0:.1f}s')
            t0 = time.time()
        if df.shape[0] == 0:
            print(bdate, spc, 'no tempo')
            continue
        if dopandora:
            pdf = pdfs.query(
                f'Timestamp >= "{bdate:%Y-%m-%dT%H:%M:%S}Z"'
                + f' and Timestamp <= "{edate:%Y-%m-%dT%H:%M:%S}Z"'
            )
            if pdf.shape[0] == 0:
                print(bdate, spc, 'no pandora')
            else:
                if verbose > 0:
                    print(bdate, spc, 'sjoin pandora', end='...', flush=True)
                t0 = time.time()
                pixdf = gpd.sjoin(pdf, df, lsuffix='1', rsuffix='2')
                renamer(pixdf, 'pandora')
                tt = pd.to_datetime(pixdf['tempo_time'])
                tp = pd.to_datetime(pixdf['pandora_time'])
                dt = (tp - tt).dt.total_seconds()
                pixdf.drop(columns=['geometry', 'index_2'], inplace=True)
                pixdf.loc[dt.abs() < 900].to_csv(pixpath, index=False)
                bpdf = pdf.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    bpdf['geometry'] = pdf['geometry'].buffer(0.03)

                bpixdf = gpd.sjoin(bpdf, df, lsuffix='1', rsuffix='2')
                renamer(bpixdf, 'pandora')
                tt = pd.to_datetime(bpixdf['tempo_time'])
                tp = pd.to_datetime(bpixdf['pandora_time'])
                dt = (tp - tt).dt.total_seconds()
                bpixdf.drop(columns=['geometry', 'index_2'], inplace=True)
                bpixdf.loc[dt.abs() < 900].to_csv(pix3path, index=False)
                t1 = time.time()
                if verbose > 0:
                    print(f'{t1 - t0:.1f}')
        if dotropomin:
            if verbose > 0:
                print(bdate, spc, 'open tropomi nrti', flush=True)
            tdf = open_prod(f'tropomi.nrti.{spc}', dates=[bdate])
            if tdf.shape[0] == 0:
                print(bdate, spc, 'no tropomi nrti')
            else:
                if verbose > 0:
                    print(
                        bdate, spc, 'sjoin TropOMI nrti', end='...', flush=True
                    )
                t0 = time.time()
                tixdf = gpd.sjoin(tdf, df, lsuffix='1', rsuffix='2')
                renamer(tixdf, 'tropomi')
                tixdf.drop(columns=['geometry', 'index_2'], inplace=True)
                tixdf.to_csv(tnixpath, index=False)
                t1 = time.time()
                if verbose > 0:
                    print(f'{t1 - t0:.1f}')
        if dotropomio:
            if verbose > 0:
                print(bdate, spc, 'open tropomi offl', flush=True)
            tdf = open_prod(f'tropomi.offl.{spc}', dates=[bdate])
            if tdf.shape[0] == 0:
                print(bdate, spc, 'no tropomi offl')
            else:
                if verbose > 0:
                    print(
                        bdate, spc, 'sjoin TropOMI offl', end='...', flush=True
                    )
                t0 = time.time()
                tixdf = gpd.sjoin(tdf, df, lsuffix='1', rsuffix='2')
                renamer(tixdf, 'tropomi')
                tixdf.drop(columns=['geometry', 'index_2'], inplace=True)
                tixdf.to_csv(toixpath, index=False)
                t1 = time.time()
                if verbose > 0:
                    print(f'{t1 - t0:.1f}')
        if doairnow:
            adf = open_prod(f'airnow.{spc}', dates=[bdate])
            if adf.shape[0] == 0:
                print(bdate, spc, 'no tropomi')
            else:
                if verbose > 0:
                    print(bdate, spc, 'sjoin AirNow', end='...', flush=True)
                t0 = time.time()
                aixdf = gpd.sjoin(adf, df, lsuffix='1', rsuffix='2')
                renamer(aixdf, 'airnow')
                aixdf.drop(columns=['geometry', 'index_2'], inplace=True)
                aixdf.to_csv(aixpath, index=False)
                t1 = time.time()
                if verbose > 0:
                    print(f'{t1 - t0:.1f}')
        gc.collect()
    return


def renamer(ixdf, source):
    """
    internal function for renaming columns; maybe move to cfg"""
    o2n = {
        'Timestamp_1': f'{source}_time',
        'LONGITUDE_1': f'{source}_lon',
        'LATITUDE_1': f'{source}_lat',
        'Timestamp_2': 'tempo_time',
        'LONGITUDE_2': 'tempo_lon',
        'LATITUDE_2': 'tempo_lat',
        'solar_zenith_angle': 'tempo_sza',
        'eff_cloud_fraction': 'tempo_cloud_eff',
        'no2': 'airnow_no2_sfc',
        'no2_vertical_column_troposphere': 'tempo_no2_trop',
        'no2_vertical_column_total': 'tempo_no2_total',
        'no2_vertical_column_stratosphere': 'tempo_no2_strat',
        'vertical_column': 'tempo_hcho_total',
        'nitrogen_dioxide_vertical_column_amount': 'pandora_no2_total',
        'nitrogendioxide_tropospheric_column': 'tropomi_no2_trop',
        'formaldehyde_total_vertical_column_amount': 'pandora_hcho_total',
        'formaldehyde_tropospheric_vertical_column': 'tropomi_hcho_total',
        'ELEVATION': 'pandora_elevation',
        'STATION': 'pandora_id', 'NOTE': 'pandora_note',
    }
    return ixdf.rename(columns=o2n, inplace=True)


def loadintx(source, spc, psuffix='0pt03', verbose=0, thin=1):
    """
    Arguments
    ---------
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'
    psuffix : str
        Pandora suffix 0pt00 in pixel only or 0pt03 0.03 degree radius.
    verbose : int
        level of verbosity
    thin : int
        If 1, do not thin. reduces data loaded by a factor of ten.

    Returns
    -------
    df : pandas.DataFrame
    """
    import gc
    import glob
    import pandas as pd
    if source == 'pandora':
        pat = f'intx/{source}/????-??/{source}_intx_{spc}_*_{psuffix}.csv.gz'
    elif source == 'airnow':
        pat = f'intx/{source}/????-??/{source}_intx_{spc}_*.csv.gz'
    else:
        sprefix = source.replace('_offl', '').replace('_nrti', '')
        pat = f'intx/{sprefix}/????-??/{source}_intx_{spc}_*.csv.gz'
    paths = sorted(glob.glob(pat))[::thin]
    dfs = []
    npaths = len(paths)
    chunks = max(1, npaths // 200)
    if verbose > 0:
        print(f'Found {npaths}')

    for pi, p in enumerate(paths):
        df = pd.read_csv(p)
        if df.shape[0] > 0:
            for k in df.columns:
                if df[k].dtype.char == 'd':
                    df[k] = df[k].astype('f')
                if k.endswith('_time'):
                    t = pd.to_datetime(df[k], utc=True)
                    df[k] = (t - cfg.reftime).dt.total_seconds()
                    lst = t + pd.to_timedelta(df['tempo_lon'] / 15, unit='h')
                    df['tempo_lst_hour'] = lst.dt.hour
                    df['v1'] = (t > cfg.v1start) & (t < cfg.v2start)
                    df['v2'] = (t > cfg.v2start) | (t < cfg.v1start)

            dfs.append(df.copy())
        del df
        # once a day
        if verbose > 0 and (pi % chunks) == 0:
            print(f'\r{pi / npaths:6.1%}', end='', flush=True)
        gc.collect()
        cfg.libc.malloc_trim(1)
    gc.collect()
    cfg.libc.malloc_trim(1)

    df = pd.concat(dfs, ignore_index=True)
    if verbose > 0:
        print(f'\r{1:6.1%}', flush=True)
    del dfs
    gc.collect()
    cfg.libc.malloc_trim(1)

    if spc == 'no2':
        df.eval(
            'tempo_no2_sum = tempo_no2_trop + tempo_no2_strat', inplace=True
        )

    return df


def getintx(source, spc, refresh=False):
    import pandas as pd
    try:
        if refresh:
            raise IOError()
        df = pd.read_hdf(f'intx/{source}.h5', key=f'{source}_{spc}')
        print('reuse cached')
    except Exception:
        df = loadintx(source, spc, verbose=1, thin=1)
        df.to_hdf(f'intx/{source}.h5', key=f'{source}_{spc}', complevel=1)
    return df


if __name__ == '__main__':
    import tempodash
    import pandas as pd

    after = pd.to_datetime('2024-04-20T00Z')
    dates = [d for d in tempodash.cfg.dates if d > after]
    tempodash.pair.makeintx('no2', dates)
    tempodash.pair.getintx('pandora', 'no2', refresh=True)
    tempodash.pair.getintx('tropomi_offl', 'no2', refresh=True)
    tempodash.pair.makeintx('hcho', dates)
    tempodash.pair.getintx('pandora', 'hcho', refresh=True)
    tempodash.pair.getintx('tropomi_offl', 'hcho', refresh=True)
