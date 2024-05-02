import gc
from . import cfg
import pandas as pd
import glob


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


def getintx(source, spc):
    try:
        df = pd.read_hdf(f'intx/{source}.h5', key=f'{source}_{spc}')
        print('cached')
    except Exception:
        df = loadintx(source, spc, verbose=1, thin=1)
        df.to_hdf(f'intx/{source}.h5', key=f'{source}_{spc}', complevel=1)
