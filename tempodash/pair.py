def _write_empty_intx_df(intxpath):
    import pandas as pd
    # knows about one column and no records -- readable and concatenatable to
    # any other file with tempo_time without modifying
    emptydf = pd.DataFrame.from_dict({'tempo_time': {}})
    emptydf.to_csv(intxpath, index=False)


def checkprod(prod, bdate, backend='xdr', verbose=0):
    """
    Check if product file is available for bdate.
    Arguments
    ---------
    prod: str
        Expects tempo.l2.no2, tropomi.nrti.no2, tropomi.offl.no2, airnow.no2
        tempo.l2.hcho, tropomi.nrti.hcho, or tropomi.offl.hcho
    bdate: pandas.Datetime
        Any datetime
    backend: str
        ascii or xdr

    Returns
    -------
    avail : bool
        True if product gz file exists and its size is greater than 20 bytes.
    """
    from . import cfg
    from . import defn
    import os
    source = prod.split('.')[0]
    edate = bdate + cfg.data_dt
    keys = defn.proddef[prod]
    pfx = f'data/{source}/{bdate:%Y-%m}'
    sfx = f'{bdate:%Y-%m-%dT%H%M%SZ}_{edate:%Y-%m-%dT%H%M%SZ}.{backend}.gz'
    if backend == 'ascii':
        backend = 'csv'
    checks = {}
    for key in keys:
        path = f'{pfx}/{key}_{sfx}'
        if os.path.exists(path):
            size = os.stat(path).st_size
            checks[key] = size > 20
        else:
            checks[key] = False

    if verbose:
        print(checks)

    return all(checks.values())


def openchunk(key, bdate, backend='xdr'):
    """
    Open a specific data key downloaded from RSIG.

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
    from . import cfg
    from . import defn

    source = key.split('.')[0]
    edate = bdate + cfg.data_dt
    cfg.api.workdir = f'data/{source}/{bdate:%Y-%m}'
    df = cfg.api.to_dataframe(
        key, bdate=bdate, edate=edate,
        unit_keys=False, backend=backend, verbose=-1
    )

    if df.shape[1] > 0:
        df = df[defn.keycols[key]]
    cfg.libc.malloc_trim(0)
    return df


def open_prod(prod, dates=None, verbose=0):
    """
    Open all data keys downloaded from RSIG associated with a product.

    Arguments
    ---------
    prod : str
        Compound product defined in defn.proddef
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
    from . import cfg
    from . import defn

    crnrkeys = defn.crnrcoordkeys

    keys = defn.proddef[prod]
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
    open_pandora is like open_prod, but for Pandora. Unlike other products,
    Pandora is more efficient to open one site at a time. So, it has its open
    function.

    Arguments
    ---------
    prod : str
        Compound product defined in defn.proddef
    verbose : int
        Level of verbosity

    Returns
    -------
    df : pandas.DataFrame
        Individual keys loaded from pyrsig.RsigApi.to_dataframe
        and combined to make a product level dataframe
    """
    from . import cfg
    from . import defn
    import geopandas as gpd
    import pandas as pd
    from shapely import points

    bdate, edate = cfg.pandora_date_range
    key = defn.proddef[prod][0]
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
        df = df[defn.keycols[key]]
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
    Make all intersections (airnow, tropomi, and pandora) at the same time.
    Opening a full slice of TEMPO is the slowest part, so leveraging the one
    opening to make intersections for all species.

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
    from . import cfg

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

        # Check if we have data for each product, so that we do not open
        # tempo when no data is available for intersecting.
        # perhaps no longer necessary with empty intersections being built.
        havetempo = checkprod(f'tempo.l2.{spc}', bdate)
        if not havetempo:
            print(bdate, spc, 'tempo missing or empty: skip', flush=True)
            continue
        # stop processing NRTI
        # havetropominrti = checkprod(f'tropomi.nrti.{spc}', bdate)
        havetropominrti = False
        havetropomioffl = checkprod(f'tropomi.offl.{spc}', bdate)
        if spc == 'no2':
            haveairnow = checkprod(f'airnow.{spc}', bdate)
        else:
            haveairnow = False

        # istomihour is duplicative of havetropominrt or havetropomioffl
        # because we do not have TROPOMI inputs for hours that are not
        # in cfg.tropomihours
        istomihour = bdate.hour in cfg.tropomihours

        # product specific do booleans.
        dopandora = not (exists(pixpath) and exists(pix3path))
        dotropomin = havetropominrti and not exists(tnixpath) and istomihour
        dotropomio = havetropomioffl and not exists(toixpath) and istomihour
        doairnow = haveairnow and not exists(aixpath) and spc == 'no2'
        if (
            not dopandora and not doairnow
            and not dotropomin and not dotropomio
        ):
            continue

        if verbose > 0:
            print(bdate, spc, 'tempo open:', end=' ', flush=True)
            t0 = time.time()
        df = open_prod(f'tempo.l2.{spc}', dates=[bdate])
        if verbose > 0:
            print(f'{time.time() - t0:.1f}s')
            t0 = time.time()
        if df.shape[0] == 0:
            print(bdate, spc, 'tempo no records: write empty intx')
            _write_empty_intx_df(pixpath)
            _write_empty_intx_df(pix3path)
            _write_empty_intx_df(tnixpath)
            _write_empty_intx_df(toixpath)
            _write_empty_intx_df(aixpath)
            continue
        if dopandora:
            pdf = pdfs.query(
                f'Timestamp >= "{bdate:%Y-%m-%dT%H:%M:%S}Z"'
                + f' and Timestamp <= "{edate:%Y-%m-%dT%H:%M:%S}Z"'
            )
            if pdf.shape[0] == 0:
                print(bdate, spc, 'pandora no records: skip (may exist later)')
            else:
                if verbose > 0:
                    print(bdate, spc, 'pandora sjoin:', end='...', flush=True)
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
                print(bdate, spc, 'tropomi nrti open', flush=True)
            tdf = open_prod(f'tropomi.nrti.{spc}', dates=[bdate])
            if tdf.shape[0] == 0:
                print(bdate, spc, 'tropomi nrti no records: write empty intx')
                _write_empty_intx_df(tnixpath)
            else:
                if verbose > 0:
                    print(
                        bdate, spc, 'tropomi nrti sjoin', end='...',
                        flush=True
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
                print(bdate, spc, 'tropomi offl open', flush=True)
            tdf = open_prod(f'tropomi.offl.{spc}', dates=[bdate])
            if tdf.shape[0] == 0:
                print(bdate, spc, 'tropomi offl no records: write empty intx')
                _write_empty_intx_df(toixpath)
            else:
                if verbose > 0:
                    print(
                        bdate, spc, 'trompomi offl sjoin:', end='...',
                        flush=True
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
                print(bdate, spc, 'airnow no records: write empty intx')
                _write_empty_intx_df(aixpath)
            else:
                if verbose > 0:
                    print(bdate, spc, 'airnow sjoin:', end='...', flush=True)
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
        'nitrogendioxide_stratospheric_column': 'tropomi_no2_strat',
        'formaldehyde_total_vertical_column_amount': 'pandora_hcho_total',
        'formaldehyde_tropospheric_vertical_column': 'tropomi_hcho_total',
        'ELEVATION': 'pandora_elevation',
        'STATION': 'pandora_id', 'NOTE': 'pandora_note',
    }
    return ixdf.rename(columns=o2n, inplace=True)


def loadintx(source, spc, psuffix='0pt03', verbose=0, thin=1, newer=None):
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
        If 1, do not thin. thin=10 reduces data loaded by a factor of ten
        for debugging.

    Returns
    -------
    ctime, df : float, pandas.DataFrame
        ctime is the latest creation time of the loaded intersections
        df is the dataframe of the intersections
    """
    import gc
    import os
    import glob
    import pandas as pd
    from . import cfg
    if source == 'pandora':
        pat = f'intx/{source}/????-??/{source}_intx_{spc}_*_{psuffix}.csv.gz'
    elif source == 'airnow':
        pat = f'intx/{source}/????-??/{source}_intx_{spc}_*.csv.gz'
    else:
        sprefix = source.replace('_offl', '').replace('_nrti', '')
        pat = f'intx/{sprefix}/????-??/{source}_intx_{spc}_*.csv.gz'
    paths = sorted(glob.glob(pat))[::thin]
    ctimes = [os.stat(p).st_ctime for p in paths]
    newestctime = max(ctimes)
    if newer is not None:
        paths = [p for p, c in zip(paths, ctimes) if c > newer]
        npaths = len(paths)
        if npaths == 0:
            return newestctime, pd.DataFrame()
    else:
        npaths = len(paths)

    dfs = []
    chunks = max(1, npaths // 200)
    if verbose > 0:
        print(f'Found {npaths} {source} {spc}')

    for pi, p in enumerate(paths):
        df = pd.read_csv(p)
        if df.shape[0] > 0:
            for k in df.columns:
                if df[k].dtype.char == 'd':
                    df[k] = df[k].astype('f')
                if k.endswith('_time'):
                    t = pd.to_datetime(df[k], utc=True)
                    th = t.dt.floor('1h')
                    df[k] = (t - cfg.reftime).dt.total_seconds()
                    lst = t + pd.to_timedelta(df['tempo_lon'] / 15, unit='h')
                    df['tempo_lst_hour'] = lst.dt.hour
                    # V1 has one run
                    # df['v1'] = (t > cfg.v1start) & (t < cfg.v2start)
                    df['v1'] = th.isin(cfg.v1dates)
                    # V2 is between v1 and v3 or before v1
                    # df['v2'] = (
                    #     ((t > cfg.v2start) & (t < cfg.v3start))
                    #     | (t < cfg.v1start)
                    # )
                    df['v2'] = th.isin(cfg.v2dates)
                    # df['v3'] = (t > cfg.v3start)
                    df['v3'] = th.isin(cfg.v3dates)

            dfs.append(df.copy())
        del df
        # once a day
        if verbose > 0 and (pi % chunks) == 0:
            print(f'\r{pi / npaths:6.1%}', end='', flush=True)
        gc.collect()
        cfg.libc.malloc_trim(1)
    gc.collect()
    cfg.libc.malloc_trim(1)
    if len(dfs) == 0:
        return newestctime, pd.DataFrame()

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
        
    if 'tropomi_no2_strat' in df.columns and 'tropomi_no2_trop' in df.columns:
        df.eval(
            'tropomi_no2_sum = tropomi_no2_trop + tropomi_no2_strat', inplace=True
        )

    return newestctime, df


def getintx_old(source, spc, refresh=False):
    """
    rebuilds the whole h5 dataframe each time
    """
    import pandas as pd
    try:
        if refresh:
            raise IOError()
        df = pd.read_hdf(f'intx/{source}.h5', key=f'{source}_{spc}')
        print('reuse cached')
    except Exception:
        ctime, df = loadintx(source, spc, verbose=1, thin=1)
        df.to_hdf(f'intx/{source}.h5', key=f'{source}_{spc}', complevel=1)
    return df


def getintx(source, spc, refresh=False):
    """
    Appends new intersections to the h5 dataframe and the returns the whole
    dataframe.

    Arguments
    ---------
    source : str
        For example, pandora, airnow, tropomi_offl
    spc : str
        no2 or hcho
    refresh : bool
        If True, add intersections made since the last time.

    Returns
    -------
    df : pandas.DataFrame
        dataframe of all intersections
    """
    import os
    import pandas as pd
    pd.set_option('io.hdf.default_format', 'table')

    hkey = f'{source}_{spc}'
    hpath = f'intx/{hkey}.h5'
    tkey = f'{hkey}_time'
    hexists = os.path.exists(hpath)
    # if file does not exist, refresh is true
    refresh = refresh or ~hexists
    if refresh:
        # how new was the last intersection?
        if hexists:
            dft = pd.read_hdf(hpath, key=tkey)
            newer = dft['timestamp'].max()
        else:
            newer = None

        # Find all intersections after the last batch upload
        lasttime, df = loadintx(source, spc, verbose=1, newer=newer)
        # If there are new intersections, append them and update last
        # upload date
        if df.shape[0] != 0:
            df.to_hdf(hpath, key=hkey, complevel=1, append=True, index=False, data_columns=True)
            dtf = pd.DataFrame(dict(timestamp=[lasttime]))
            dtf.to_hdf(hpath, key=tkey, append=True, index=False)

    # Load the whole dataset
    df = pd.read_hdf(hpath, key=hkey)
    return df


if __name__ == '__main__':
    import tempodash
    import pandas as pd
    import argparse

    prsr = argparse.ArgumentParser()
    prsr.add_argument('after', nargs='?', default=None)
    args = prsr.parse_args()

    if args.after is None:
        after = tempodash.cfg.dates.min() - pd.to_timedelta('1h')
    else:
        after = pd.to_datetime(args.after)

    dates = [d for d in tempodash.cfg.dates if d > after]
    tempodash.pair.makeintx('no2', dates)
    tempodash.pair.getintx('airnow', 'no2', refresh=True)
    tempodash.pair.getintx('pandora', 'no2', refresh=True)
    tempodash.pair.getintx('tropomi_offl', 'no2', refresh=True)
    tempodash.pair.makeintx('hcho', dates)
    tempodash.pair.getintx('pandora', 'hcho', refresh=True)
    tempodash.pair.getintx('tropomi_offl', 'hcho', refresh=True)
