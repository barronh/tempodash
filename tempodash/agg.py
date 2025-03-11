__all__ = [
    'odrfit', 'linregresdf', 'agg', 'byloc',
    'weeklygrouper', 'monthlygrouper', 'lsthgrouper', 'hourlygrouper',
    'szagrouper', 'cldgrouper',
]

import pandas as pd


def ioa(x, y):
    import numpy as np
    xm = x.mean()
    yabsdev = np.abs(y - xm)
    xabsdev = np.abs(x - xm)
    sse = ((y - x)**2).sum()
    den = ((xabsdev + yabsdev)**2).sum()
    return 1 - sse / den


def ioadf(df):
    return ioa(df.iloc[:, 0], df.iloc[:, 1])


def target_function(p, x):
    m, c = p
    return m * x + c


def odrfit(x, y, beta0=None):
    from scipy import odr
    if beta0 is None:
        import scipy.stats.mstats as sms
        lr = sms.linregress(x, y)
        beta0 = [lr.slope, lr.intercept]
        print('beta0 estimate:', beta0)
    #  model fitting.
    odr_model = odr.Model(target_function)

    # Create a RealData object standard deviations to weight samples
    rdata = odr.RealData(
        x, y,
        sx=x.std(), sy=y.std()
    )
    # beta0 = [vdf['vcd'].mean() / vdf['no2'].mean(), vdf['vcd'].min()]
    dr = odr.ODR(rdata, odr_model, beta0=beta0).run()
    return dr


def linregresdf(df, orthogonal=False):
    from scipy.stats.mstats import linregress
    nan = float('nan')
    try:
        lrv = linregress(df)
    except:
        out = {
            'lr_slope': nan, 'lr_intercept': nan,
            'rvalue': nan, 'lr_pvalue': nan,
        }
        if orthogonal:
            out.update(
                dr_slope=nan, dr_intercept=nan,
                dr_slope_std=nan, dr_intercept_std=nan
            )
        return pd.Series(out)
    out = {
        'lr_slope': lrv.slope, 'lr_intercept': lrv.intercept,
        'rvalue': lrv.rvalue, 'lr_pvalue': lrv.pvalue
    }
    if orthogonal:
        beta0 = [lrv.slope, lrv.intercept]
        try:
            dr = odrfit(df.iloc[:, 0], df.iloc[:, 1], beta0=beta0)
            out['dr_slope'] = dr.beta[0]
            out['dr_intercept'] = dr.beta[1]
            out['dr_slope_std'] = dr.sd_beta[0]
            out['dr_intercept_std'] = dr.sd_beta[1]
        except:
            out.update(
                dr_slope=nan, dr_intercept=nan,
                dr_slope_std=nan, dr_intercept_std=nan
            )

    return pd.Series(out)


def q1(ds):
    """
    Return the 25-percentile data from the data

    Arguments
    ---------
    ds : pandas.DataFrame or pandas.Series
        input data

    Returns
    -------
    q :
        the 25-percentile data from the data

    Example
    -------
    >>> from tempodash.agg import q1
    >>> import pandas as pd
    >>> import numpy as np
    >>> q1(pd.Series(np.random.normal(0, 1, size=1000)))
    -0.5852221993474036
    """
    return ds.quantile(0.25)


def q3(ds):
    """
    Return the 75-percentile data from the data

    Arguments
    ---------
    ds : pandas.DataFrame or pandas.Series
        input data

    Returns
    -------
    q :
        the 75-percentile data from the data

    Example
    -------
    >>> from tempodash.agg import q3
    >>> import pandas as pd
    >>> import numpy as np
    >>> q3(pd.Series(np.random.normal(0, 1, size=1000)))
    0.633665815294846
    """
    return ds.quantile(0.75)


def agg(df, grouper, columns=None, xkey=None, ykey=None, orthogonal=False):
    """
    Group and aggregate information adding quantiles and correlation

    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame of data (typically from loadintx)
    grouper : str, Grouper, or function
        str or Grouper are used like typical pandas.
    columns : list
        List of str subsetting which columns to use.
    xkey : str
        Key to use as reference value
    ykey : str
        Key to use as comparison value
    orthogonal : bool
        If True, add the orthogonal regression.

    Returns
    -------
    gdf : pandas.DataFrame
        Dataframe with each key's quantiles and (optionally) the correlation,
        the slope, and the intercept from linear regression.
    Example
    -------
    """
    if columns is None:
        columns = [
            k for k in df.columns
            if not (
                k.endswith('_lon') or k.endswith('_lat')
                or k.endswith('_time') or k.endswith('_id')
            )
        ]
    aggdict = {}
    aggdict['tempo_time_start'] = ('tempo_time', 'min')
    aggdict['tempo_time_end'] = ('tempo_time', 'max')
    for k in columns:
        aggdict.update({
            f'{k}_mean': (k, 'mean'),
            f'{k}_std': (k, 'std'),
            f'{k}_q0': (k, 'min'),
            f'{k}_q1': (k, q1),
            f'{k}_q2': (k, 'median'),
            f'{k}_q3': (k, q3),
            f'{k}_q4': (k, 'max'),
        })
    aggdict['count'] = ('tempo_time', 'count')
    if df.shape[0] == 0:
        return pd.DataFrame(columns=list(aggdict))
    if grouper is None:
        grouper = df['tempo_time'] > 0

    if callable(grouper):
        grouper = grouper(df)
    gby = df.groupby(grouper)
    outdf = gby.agg(**aggdict)
    if xkey is not None:
        ldf = gby[[xkey, ykey]].apply(linregresdf, orthogonal=orthogonal)
        outdf[ldf.columns] = ldf
        # outdf['corr'] = gby[[xkey, ykey]].corr().iloc[::2, -1].values
        outdf['ioa'] = gby[[xkey, ykey]].apply(ioadf)
    return outdf


def byloc(
    df, grouper=None, columns=None, xkey=None, ykey=None, orthogonal=False
):
    from .config import locconfigs
    dfs = []
    locs = sorted(locconfigs)
    for k in locs:
        cfg = locconfigs[k]
        qstr = (
            'tempo_lon >= {0} and tempo_lon <= {2}'
            + ' and tempo_lat >= {1} and tempo_lat <= {3}'
        ).format(*cfg['bbox'])
        ldf = df.query(qstr)
        if ldf.shape[0] == 0:
            continue
        ldf = agg(
            ldf, grouper, columns=columns,
            xkey=xkey, ykey=ykey, orthogonal=orthogonal
        )
        ldf['lockey'] = k
        dfs.append(ldf)
    outcolumns = [k for k in dfs[0].columns if k != 'lockey']
    outdf = pd.concat(dfs, ignore_index=True)[['lockey'] + outcolumns]
    outdf.set_index('lockey', inplace=True)
    return outdf


def weeklygrouper(df, t=None):
    if t is None:
        t = pd.to_datetime(df['tempo_time'], unit='s')
    wkt = t.dt.to_period('W').dt.start_time + pd.to_timedelta(6, unit='d')
    isweekday = t.dt.dayofweek < 5
    wkt.loc[isweekday] -= pd.to_timedelta(2, unit='d')
    return wkt


def monthlygrouper(df, t=None):
    if t is None:
        t = pd.to_datetime(df['tempo_time'], unit='s')
    m = t.dt.to_period('M')
    return m


def szagrouper(df, interval=10):
    return (df['tempo_sza'] // interval) * interval


def cldgrouper(df, interval=0.1):
    return (df['tempo_cloud_eff'] // interval) * interval


def lsthgrouper(df, t=None):
    off = df['tempo_lon'] / 15.
    if t is None:
        t = pd.to_datetime(df['tempo_time'] + off * 3600, unit='s')
    else:
        t = t + pd.to_timedelta(off, unit='h')
    return t.dt.hour


def hourlygrouper(df, t=None):
    if t is None:
        t = pd.to_datetime(df['tempo_time'], unit='s')
    h = t.dt.floor('1h')
    return h


def tempogrouper(df, ykey=None):
    import numpy as np
    if ykey is None:
        for ykey in ['tempo_no2_sum', 'tempo_no2_trop', 'tempo_hcho_total']:
            if ykey in df.columns:
                break
        else:
            raise KeyError('ykey not found')
    edges = [
        0, 2e14, 4e14, 6e14, 8e14,
        1e15, 2e15, 4e15, 6e15, 8e15,
        1e16, 2e16, 4e16, 6e16, 8e16,
        np.inf
    ]
    labels=edges[:-1]
    intv = pd.cut(df[ykey], bins=edges, labels=labels).astype('d')
    #df[ykey].max()
    #intv = (df[ykey] // interval) * interval
    return intv
