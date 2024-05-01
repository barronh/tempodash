from . import cfg


def openchunk(key, bdate, backend='xdr'):
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
    prod = 'pandora.no2'
    prod = 'pandora.hcho'
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
        dfs.append(df.query(f'STATION == {pid}'))

    if verbose > 0:
        print('\nstack pandoara', end='.', flush=True)

    df = pd.concat(dfs, ignore_index=True)
    if df.shape[1] > 0:
        df = df[cfg.keycols[key]]
    cfg.libc.malloc_trim(0)
    if verbose > 0:
        print('\nadd geometry pandoara', end='.', flush=True)
    geom = points(df[['LONGITUDE', 'LATITUDE']].values.reshape(-1, 2))
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    if verbose > 0:
        print()

    return gdf


def makeintx(spc, dates=None, verbose=1):
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
        tixpath = (
            f'intx/tropomi/{bdate:%Y-%m}/tropomi_intx_{spc}'
            + f'_{bdate:%Y-%m-%dT%H}.csv.gz'
        )
        for ixp in [pixpath, aixpath, tixpath]:
            os.makedirs(os.path.dirname(ixp), exist_ok=True)

        dopandora = not (exists(pixpath) and exists(pixpath))
        dotropomi = (not exists(tixpath)) and (bdate.hour in cfg.tropomihours)
        doairnow = not exists(aixpath) and spc == 'no2'
        if not dopandora and not dotropomi and not doairnow:
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
        if dotropomi:
            if verbose > 0:
                print(bdate, spc, 'open tropomi', flush=True)
            tdf = open_prod(f'tropomi.nrti.{spc}', dates=[bdate])
            if tdf.shape[0] == 0:
                print(bdate, spc, 'no tropomi')
            else:
                if verbose > 0:
                    print(bdate, spc, 'sjoin TropOMI', end='...', flush=True)
                t0 = time.time()
                tixdf = gpd.sjoin(tdf, df, lsuffix='1', rsuffix='2')
                renamer(tixdf, 'tropomi')
                tixdf.drop(columns=['geometry', 'index_2'], inplace=True)
                tixdf.to_csv(tixpath, index=False)
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


if __name__ == '__main__':
    import daydash
    daydash.pair.makeintx('no2')
    daydash.pair.makeintx('hcho')
