import numpy as np
from . import cfg
import pandas as pd
import matplotlib.pyplot as plt


id2label = {
    v.get('pandoraid', [-999])[0]: v.get('label', k)
    for k, v in cfg.configs.items()
    if v.get('pandora')
}
id2label.update({k: v.get('label', k) for k, v in cfg.configs.items()})

pid2key = {
    v.get('pandoraid', [-999])[0]: k
    for k, v in cfg.configs.items()
    if v.get('pandora')
}
pid2label = {
    v.get('pandoraid', [-999])[0]: v.get('label', k.replace('Pandora', ''))
    for k, v in cfg.configs.items()
    if v.get('pandora')
}


def q1(s):
    return s.quantile(.25)


def q3(s):
    return s.quantile(.75)


allfuncs = {
    'pandora_time': ('pandora_time', 'mean'),
    'pandora_time_count': ('pandora_time', 'count'),
    'pandora_time_min': ('pandora_time', 'min'),
    'pandora_time_max': ('pandora_time', 'max'),
    'pandora_lon': ('pandora_lon', 'mean'),
    'pandora_lat': ('pandora_lat', 'mean'),
    'pandora_elevation': ('pandora_elevation', 'mean'),
    'pandora_no2_total': ('pandora_no2_total', 'mean'),
    'pandora_no2_total_q0': ('pandora_no2_total', 'min'),
    'pandora_no2_total_q1': ('pandora_no2_total', q1),
    'pandora_no2_total_q2': ('pandora_no2_total', 'median'),
    'pandora_no2_total_q3': ('pandora_no2_total', q3),
    'pandora_no2_total_q4': ('pandora_no2_total', 'max'),
    'pandora_no2_total_std': ('pandora_no2_total', 'std'),
    'pandora_hcho_total': ('pandora_hcho_total', 'mean'),
    'pandora_hcho_total_q0': ('pandora_hcho_total', 'min'),
    'pandora_hcho_total_q1': ('pandora_hcho_total', q1),
    'pandora_hcho_total_q2': ('pandora_hcho_total', 'median'),
    'pandora_hcho_total_q3': ('pandora_hcho_total', q3),
    'pandora_hcho_total_q4': ('pandora_hcho_total', 'max'),
    'pandora_hcho_total_std': ('pandora_hcho_total', 'std'),
    'airnow_time': ('airnow_time', 'mean'),
    'airnow_time_count': ('airnow_time', 'count'),
    'airnow_time_min': ('airnow_time', 'min'),
    'airnow_time_max': ('airnow_time', 'max'),
    'airnow_lon': ('airnow_lon', 'mean'),
    'airnow_lat': ('airnow_lat', 'mean'),
    'airnow_no2_sfc': ('airnow_no2_sfc', 'mean'),
    'airnow_no2_sfc_std': ('airnow_no2_sfc', 'std'),
    'airnow_no2_sfc_q0': ('airnow_no2_sfc', 'min'),
    'airnow_no2_sfc_q1': ('airnow_no2_sfc', q1),
    'airnow_no2_sfc_q2': ('airnow_no2_sfc', 'median'),
    'airnow_no2_sfc_q3': ('airnow_no2_sfc', q3),
    'airnow_no2_sfc_q4': ('airnow_no2_sfc', 'max'),
    'tropomi_time_count': ('tropomi_time', 'count'),
    'tropomi_time_min': ('tropomi_time', 'min'),
    'tropomi_time_max': ('tropomi_time', 'max'),
    'tropomi_lon': ('tropomi_lon', 'mean'),
    'tropomi_lat': ('tropomi_lat', 'mean'),
    'tropomi_no2_trop': ('tropomi_no2_trop', 'mean'),
    'tropomi_no2_trop_q0': ('tropomi_no2_trop', 'min'),
    'tropomi_no2_trop_q1': ('tropomi_no2_trop', q1),
    'tropomi_no2_trop_q2': ('tropomi_no2_trop', 'median'),
    'tropomi_no2_trop_q3': ('tropomi_no2_trop', q3),
    'tropomi_no2_trop_q4': ('tropomi_no2_trop', 'max'),
    'tropomi_no2_trop_std': ('tropomi_no2_trop', 'std'),
    'tropomi_hcho_total': ('tropomi_hcho_total', 'mean'),
    'tropomi_hcho_total_q0': ('tropomi_hcho_total', 'min'),
    'tropomi_hcho_total_q1': ('tropomi_hcho_total', q1),
    'tropomi_hcho_total_q2': ('tropomi_hcho_total', 'median'),
    'tropomi_hcho_total_q3': ('tropomi_hcho_total', q3),
    'tropomi_hcho_total_q4': ('tropomi_hcho_total', 'max'),
    'tropomi_hcho_total_std': ('tropomi_hcho_total', 'std'),
    'tempo_time': ('tempo_time', 'mean'),
    'tempo_time_count': ('tempo_time', 'count'),
    'tempo_time_min': ('tempo_time', 'min'),
    'tempo_time_max': ('tempo_time', 'max'),
    'tempo_lon': ('tempo_lon', 'mean'),
    'tempo_lat': ('tempo_lat', 'mean'),
    'tempo_no2_sum': ('tempo_no2_sum', 'mean'),
    'tempo_no2_sum_q0': ('tempo_no2_sum', 'min'),
    'tempo_no2_sum_q1': ('tempo_no2_sum', q1),
    'tempo_no2_sum_q2': ('tempo_no2_sum', 'median'),
    'tempo_no2_sum_q3': ('tempo_no2_sum', q3),
    'tempo_no2_sum_q4': ('tempo_no2_sum', 'max'),
    'tempo_no2_sum_std': ('tempo_no2_sum', 'std'),
    'tempo_no2_trop': ('tempo_no2_trop', 'mean'),
    'tempo_no2_trop_q0': ('tempo_no2_trop', 'min'),
    'tempo_no2_trop_q1': ('tempo_no2_trop', q1),
    'tempo_no2_trop_q2': ('tempo_no2_trop', 'median'),
    'tempo_no2_trop_q3': ('tempo_no2_trop', q3),
    'tempo_no2_trop_q4': ('tempo_no2_trop', 'max'),
    'tempo_no2_trop_std': ('tempo_no2_trop', 'std'),
    'tempo_no2_total': ('tempo_no2_total', 'mean'),
    'tempo_no2_total_q0': ('tempo_no2_total', 'min'),
    'tempo_no2_total_q1': ('tempo_no2_total', q1),
    'tempo_no2_total_q2': ('tempo_no2_total', 'median'),
    'tempo_no2_total_q3': ('tempo_no2_total', q3),
    'tempo_no2_total_q4': ('tempo_no2_total', 'max'),
    'tempo_no2_total_std': ('tempo_no2_total', 'std'),
    'tempo_hcho_total': ('tempo_hcho_total', 'mean'),
    'tempo_hcho_total_q0': ('tempo_hcho_total', 'min'),
    'tempo_hcho_total_q1': ('tempo_hcho_total', q1),
    'tempo_hcho_total_q2': ('tempo_hcho_total', 'median'),
    'tempo_hcho_total_q3': ('tempo_hcho_total', q3),
    'tempo_hcho_total_q4': ('tempo_hcho_total', 'max'),
    'tempo_hcho_total_std': ('tempo_hcho_total', 'std'),
    'tempo_no2_strat': ('tempo_no2_strat', 'mean'),
    'tempo_no2_strat_q0': ('tempo_no2_strat', 'min'),
    'tempo_no2_strat_q1': ('tempo_no2_strat', q1),
    'tempo_no2_strat_q2': ('tempo_no2_strat', 'median'),
    'tempo_no2_strat_q3': ('tempo_no2_strat', q3),
    'tempo_no2_strat_q4': ('tempo_no2_strat', 'max'),
    'tempo_no2_strat_std': ('tempo_no2_strat', 'std'),
    'tempo_sza': ('tempo_sza', 'mean'),
    'tempo_cloud_eff': ('tempo_cloud_eff', 'mean'),
    'err': ('err', 'mean'), 'err_std': ('err', 'std'),
    'err_q0': ('err', 'min'),
    'err_q1': ('err', q1), 'err_q2': ('err', 'median'), 'err_q3': ('err', q3),
    'err_q4': ('err', 'max'),
    'nerr': ('nerr', 'mean'), 'nerr_std': ('nerr', 'std'),
    'nerr_q0': ('nerr', 'min'),
    'nerr_q1': ('nerr', q1), 'nerr_q2': ('nerr', 'median'),
    'nerr_q3': ('nerr', q3), 'nerr_q4': ('nerr', 'max'),
}


def agg(df, groupkeys, err=True):
    funcs = {
        k: v for k, v in allfuncs.items()
        if v[0] in df.columns
    }
    if not err:
        funcs = {
            k: v for k, v in funcs.items()
            if not (k.startswith('err_') or k.startswith('nerr_'))
        }
    dfg = df.groupby(groupkeys, observed=False)
    dfm = dfg.agg(**funcs)
    return dfm


def getdfkeys(df):
    if 'pandora_no2_total' in df.columns:
        xkey, ykey = getkeys('pandora', 'no2')
    elif 'pandora_hcho_total' in df.columns:
        xkey, ykey = getkeys('pandora', 'hcho')
    elif 'tropomi_no2_trop' in df.columns:
        xkey, ykey = getkeys('tropomi', 'no2')
    elif 'tropomi_hcho_total' in df.columns:
        xkey, ykey = getkeys('tropomi', 'hcho')
    elif 'airnow_no2_sfc' in df.columns:
        xkey, ykey = getkeys('airnow', 'no2')
    else:
        raise KeyError('Did not find any reference keys')
    return xkey, ykey


def adderr(df):
    xkey, ykey = getdfkeys(df)
    if 'err' not in df.columns:
        df['err'] = df[ykey] - df[xkey]
    if 'nerr' not in df.columns:
        df['nerr'] = df['err'] / df[xkey]


def aggbybox(df):
    from . import cfg
    xkey, ykey = getdfkeys(df)
    adderr(df)

    if 'pandora_id' in df.columns:
        bdf = agg(df, 'pandora_id')
        pids = bdf.index.values
        bdf['lockey'] = [pid2key[pid] for pid in pids]
        bdf = bdf.set_index('lockey')
        bdf['pandora_id'] = pids
    else:
        bdfs = []
        for lockey, lcfg in cfg.configs.items():
            wlon, slat, elon, nlat = lcfg['bbox']
            sdf = df.query(
                f'tempo_lon > {wlon} and tempo_lon < {elon}'
                + f' and tempo_lat > {slat} and tempo_lat < {nlat}'
            ).copy()
            if sdf.shape[0] == 0:
                continue
            sdf['lockey'] = lockey
            bdf = agg(sdf, 'lockey')
            bdfs.append(bdf)
        bdf = pd.concat(bdfs)

    bdf.eval(f'nmb = ({ykey} - {xkey}) / {xkey} * 100', inplace=True)
    return bdf


def get_trange(df):
    if 'tempo_time' not in df.columns:
        return 'unknown', 'unknown'
    tstart = cfg.reftime + pd.to_timedelta(df['tempo_time'].min(), unit='s')
    tend = cfg.reftime + pd.to_timedelta(df['tempo_time'].max(), unit='s')
    tstart = tstart.strftime('%Y-%m-%d')
    tend = tend.strftime('%Y-%m-%d')
    return tstart, tend


def getaxislabels(source, spc, unit=True):
    if source.startswith('tropomi'):
        source = 'tropomi'
    deflabels = (f'{source.title()} {spc.upper()}', f'TEMPO {spc.upper()}')
    xlbl, ylbl = {
        ('pandora', 'no2'): ('Pandora totNO$_2$', 'TEMPO totNO$_2$'),
        ('tropomi', 'no2'): ('TropOMI tropNO$_2$', 'TEMPO tropNO$_2$'),
        ('pandora', 'hcho'): ('Pandora totHCHO', 'TEMPO totHCHO'),
        ('tropomi', 'hcho'): ('TropOMI totHCHO', 'TEMPO totHCHO'),
        ('airnow', 'no2'): ('AirNow sfcNO$_2$', 'TEMPO tropNO$_2$'),
    }.get((source, spc), deflabels)
    if unit:
        ylbl += ' [molec/cm$^3$]'
        if source == 'airnow':
            xlbl += ' [ppb]'
        else:
            xlbl += ' [molec/cm$^3$]'
    return xlbl, ylbl


def getsourcelabel(source):
    deflabel = source.title()
    return {
        'pandora': 'Pandora',
        'tropomi': 'TropOMI',
        'tropomi_offl': 'TropOMI OFFL',
        'tropomi_nrti': 'TropOMI NRTI',
        'tempo': 'TEMPO',
        'airnow': 'AirNow',
    }.get(source, deflabel)


def getkeys(source, spc, std=False):
    tomis = ('tropomi', 'tropomi_offl', 'tropomi_nrti')
    if source == 'pandora' and spc == 'no2':
        y = 'tempo_no2_sum'
        x = 'pandora_no2_total'
    elif source == 'pandora' and spc == 'hcho':
        y = 'tempo_hcho_total'
        x = 'pandora_hcho_total'
    elif source == 'airnow' and spc == 'no2':
        x = 'airnow_no2_sfc'
        y = 'tempo_no2_trop'
    elif source in tomis and spc == 'no2':
        y = 'tempo_no2_trop'
        x = 'tropomi_no2_trop'
    elif source in tomis and spc == 'hcho':
        y = 'tempo_hcho_total'
        x = 'tropomi_hcho_total'
    else:
        raise KeyError(f'{source} with {spc} is unknown')
    if std:
        xy = (x, y, x + '_std', y + '_std')
    else:
        xy = (x, y)
    return xy


def getxy(df, source=None, spc=None, err=False):
    if source is None or spc is None:
        xk, yk = getdfkeys(df)
    else:
        xk, yk = getkeys(source, spc)
    if err:
        x = df[xk + '_q2']
        y = df[yk + '_q2']
        xerr = (x - df[xk + '_q1'], df[xk + '_q3'] - x)
        yerr = (y - df[yk + '_q1'], df[yk + '_q3'] - y)
        return x, y, xerr, yerr
    else:
        return df[xk], df[yk]


def plot_scatter(
    df, source, spc='no2', hexn=1e4, tstart=None, tend=None, reg=None,
    colorby='tempo_sza', ax=None
):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        loaded from getintx
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'
    hexn : int
        If df.shape[0] >= hexn, use hexbin instead of scatter.
        Otherwise, use scatter.
    reg : pandas.DataFrame
        Precalculated regression from util.regressions
    colorby : str
        Field to color markers.
    ax : None or matplotlib.axes.Axes

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    from .util import regressions
    # from scipy.stats.mstats import linregress
    # from numpy.ma import masked_invalid
    xk, yk, xks, yks = getkeys(source, spc, std=True)
    if 'pandora_id' in df.columns and df.index.name != 'lockey':
        gdf = agg(df, ['pandora_id', 'tempo_time'])
        x, y, xs, ys = getxy(gdf, source, spc, err=True)
    elif (
        xks in df.columns
        and xks in df.columns
    ):
        gdf = df
        x, y, xs, ys = getxy(gdf, source, spc, err=True)
    else:
        gdf = df
        x, y = getxy(gdf, source, spc, err=False)
        xs = np.zeros_like(x)
        ys = np.zeros_like(y)

    # tcol = xk.split(spc)[-1][1:]

    c = gdf[colorby]
    units = 'molec/cm$^2$'
    if source == 'airnow':
        units = '1'
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        xs = x * 0
        ys = y * 0

    # lr = linregress(masked_invalid(x.values), masked_invalid(y.values))
    if reg is None:
        reg = regressions(x, y)
    lrstr = 'LR={lr_slope:.2f}x{lr_intercept:+.2e}'.format(**reg)
    lrstr += ' (r={lr_rvalue:.2f})'.format(**reg)
    drstr = 'DR={dr_slope:.2f}x{dr_intercept:+.2e}'.format(**reg)
    xmax = max(x.max(), y.max()) * 1.05
    # tstart, tend = get_trange(df)

    gskw = dict(right=0.925, left=0.15, bottom=0.12)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(4.5, 4), gridspec_kw=gskw, rasterized=True
        )
    else:
        fig = ax.figure
    n = x.size
    if n < hexn:
        ax.errorbar(
            x=x, y=y, yerr=ys, xerr=xs, color='k', linestyle='none', zorder=1
        )
        s = ax.scatter(x=x, y=y, c=c, zorder=2)
        if colorby == 'tempo_sza':
            clabel = 'Solar Zenith Angle [deg]'
        elif colorby == 'tempo_lat':
            clabel = 'Latitude [deg]'
        else:
            clabel = colorby
        fig.colorbar(s, ax=ax, label=clabel)
    else:
        vmax = max(x.max(), y.max())
        lnorm = plt.matplotlib.colors.LogNorm(vmin=1)
        s = ax.hexbin(
            x=x, y=y, gridsize=(20, 20), extent=(0, vmax, 0, vmax),
            mincnt=1, norm=lnorm
        )
        ax.set(facecolor='gainsboro')
        fig.colorbar(s, ax=ax, label='Count [#]')
        # cymax = cb.ax.get_ylim()[1]
        # ordermag = np.floor(np.log10(cymax))
        # cyticks = np.arange(0, cymax, 10**ordermag)
        # cylabels = [f'{cv / (10**ordermag):.0f}' for cv in cyticks]
        # cb.ax.set_yticks(cyticks)
        # cb.ax.set_yticklabels(cylabels)
        # cb.ax.set_ylabel(f'Count [{10**ordermag:.0f}s]')

    ax.axline((0, 0), slope=1, zorder=3, label='1:1', color='grey')
    ax.axline(
        (0, reg['lr_intercept']), slope=reg['lr_slope'], zorder=3,
        label=lrstr, color='k'
    )
    ax.axline(
        (0, reg['dr_intercept']), slope=reg['dr_slope'], zorder=3,
        label=drstr, color='k', linestyle='--'
    )
    xlbl, ylbl = getaxislabels(source, spc, unit=False)
    ax.set(
        aspect=1, xlim=(0, xmax), ylim=(0, xmax),
        xlabel=f'{xlbl} [{units}]',
        ylabel=f'{ylbl} [{units}]',
    )
    ax.legend()
    if tstart is None and 'tempo_time_min' in df.columns:
        tstart = pd.to_datetime(
            df['tempo_time_min'].min(), unit='s'
        ).strftime('%Y-%m-%d')
        tend = pd.to_datetime(
            df['tempo_time_max'].max(), unit='s'
        ).strftime('%Y-%m-%d')

    if tstart is not None:
        fig.text(0.58, 0.01, f'{tstart} to {tend}')
    return ax


def plot_tod_scatter(df, source, spc, tstart=None, tend=None, reghdf=None):
    # from scipy.stats.mstats import linregress
    from .util import regressions
    gskw = dict(bottom=0.05, left=0.05, right=0.92, top=0.90)
    fig, axx = plt.subplots(
        3, 4, figsize=(14, 12), gridspec_kw=gskw, sharex=False, sharey=False
    )
    pdf = df.query('tempo_lst_hour > 5 and tempo_lst_hour < 18')
    x, y = getxy(df, source, spc, err=False)
    # outlier max
    xmax = x.quantile(0.75) + 2 * (x.quantile(0.75) - x.quantile(0.25))
    ymax = y.quantile(0.75) + 2 * (y.quantile(0.75) - y.quantile(0.25))
    vmax1 = max(ymax, xmax)
    vmax2 = max(x.max(), y.max())
    vmax = max(vmax1, vmax2)
    ss = []
    for hi, (h, hdf) in enumerate(pdf.groupby('tempo_lst_hour')):
        hdf = hdf.groupby(['pandora_id', 'tempo_time']).mean(numeric_only=True)
        ax = axx.ravel()[hi]
        ax.text(
            vmax * 0.95, 2e15, f'{h:.0f}-{h+1:.0f}LST', size=16,
            horizontalalignment='right'
        )
        x, y = getxy(hdf, source, spc, err=False)
        if reghdf is None:
            regopts = regressions(x, y)
        else:
            regopts = reghdf.loc[h]
        lr_slope = regopts['lr_slope']
        lr_intercept = regopts['lr_intercept']
        lr_rvalue = regopts['lr_rvalue']
        dr_slope = regopts['dr_slope']
        dr_intercept = regopts['dr_intercept']
        lrstr = f'LR={lr_slope:.2f}x{lr_intercept:+.2e}'
        drstr = f'OR={dr_slope:.2f}x{dr_intercept:+.2e}'
        ax.set(title=f'n={hdf.shape[0]}; r={lr_rvalue:.2f}')
        norm = plt.matplotlib.colors.LogNorm(1, None)
        s = ax.hexbin(
            x, y, mincnt=1, norm=norm, extent=(0, vmax, 0, vmax), gridsize=25
        )
        ss.append(s)
        ax.axline((0, 0), slope=1, label='1:1', color='grey', linestyle='-')
        ax.axline(
            (0, lr_intercept), slope=lr_slope, label=lrstr,
            color='k', linestyle='--'
        )
        ax.axline(
            (0, dr_intercept), slope=dr_slope, label=drstr,
            color='k', linestyle=':'
        )
        ax.legend(loc='upper left')
    cmax = sorted([s.get_array().max() for s in ss])[-2]
    if cmax > 100:
        norm = plt.matplotlib.colors.LogNorm(1, cmax)
    else:
        norm = plt.matplotlib.colors.Normalize(1, cmax)
    for s in ss:
        s.set_norm(norm)
    plt.setp(axx, aspect=1, xlim=(0, vmax), ylim=(0, vmax))
    plt.setp(axx.ravel()[hi+1:], visible=False)
    cax = fig.add_axes([0.93, .1, 0.03, 0.8])
    fig.colorbar(s, cax=cax, label='count')
    start, end = pd.to_datetime(pdf['tempo_time'].quantile([0, 1]), unit='s')
    xlbl, ylbl = getaxislabels(source, spc)
    xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    fig.suptitle(f'{xlblnu} and {ylblnu}: {start:%Y-%m-%d} to {end:%Y-%m-%d}', size=24)
    plt.setp(axx[-1], xlabel=xlbl)
    plt.setp(axx[:, 0], ylabel=ylbl)
    if tstart is None and 'tempo_time_min' in df.columns:
        tstart = pd.to_datetime(
            df['tempo_time_min'].min(), unit='s'
        ).strftime('%Y-%m-%d')
        tend = pd.to_datetime(
            df['tempo_time_max'].min(), unit='s'
        ).strftime('%Y-%m-%d')

    # titlestr = f'{xlblnu} and {ylblnu}'
    # if tstart is not None:
    #     titlestr += f': {tstart} to {tend}'
    # fig.text(0.58, 0.01, titlestr)

    return fig


def plot_month_scatter(df, source, spc, tstart=None, tend=None, reghdf=None):
    # from scipy.stats.mstats import linregress
    from .util import regressions
    gskw = dict(bottom=0.05, left=0.05, right=0.92, top=0.90)
    fig, axx = plt.subplots(
        3, 4, figsize=(14, 12), gridspec_kw=gskw, sharex=False, sharey=False
    )
    if source == 'pandora':
        lockey = 'pandora_id'
        pdf = df.groupby([lockey, 'tempo_time'], as_index=False).mean(
            numeric_only=True
        )
    else:
        pdf = df

    pdf['month'] = pd.to_datetime(pdf['tempo_time'], unit='s').dt.month
    x, y = getxy(df, source, spc, err=False)
    # outlier max
    xmax = x.quantile(0.75) + 2 * (x.quantile(0.75) - x.quantile(0.25))
    ymax = y.quantile(0.75) + 2 * (y.quantile(0.75) - y.quantile(0.25))
    vmax1 = max(ymax, xmax)
    vmax2 = max(x.max(), y.max())
    vmax = max(vmax1, vmax2)
    ss = []
    plt.setp(axx, visible=False)
    for hi, (h, hdf) in enumerate(pdf.groupby('month')):
        ax = axx.ravel()[h - 1]
        ax.set(visible=True)
        m = pd.to_datetime(f'1970-{h:02.0f}-15').strftime('%b')
        ax.text(vmax * 0.95, 2e15, m, size=16, horizontalalignment='right')
        x, y = getxy(hdf, source, spc, err=False)
        if reghdf is None:
            regopts = regressions(x, y)
        else:
            regopts = reghdf.loc[m]
        lr_slope = regopts['lr_slope']
        lr_intercept = regopts['lr_intercept']
        lr_rvalue = regopts['lr_rvalue']
        dr_slope = regopts['dr_slope']
        dr_intercept = regopts['dr_intercept']
        lrstr = f'LR={lr_slope:.2f}x{lr_intercept:+.2e}'
        drstr = f'OR={dr_slope:.2f}x{dr_intercept:+.2e}'
        ax.set(title=f'n={hdf.shape[0]}; r={lr_rvalue:.2f}')
        norm = plt.matplotlib.colors.LogNorm(1, None)
        s = ax.hexbin(
            x, y, mincnt=1, norm=norm, extent=(0, vmax, 0, vmax), gridsize=25
        )
        ss.append(s)
        ax.axline((0, 0), slope=1, label='1:1', color='grey', linestyle='-')
        ax.axline(
            (0, lr_intercept), slope=lr_slope, label=lrstr,
            color='k', linestyle='--'
        )
        ax.axline(
            (0, dr_intercept), slope=dr_slope, label=drstr,
            color='k', linestyle=':'
        )
        ax.legend(loc='upper left')
    cmax = sorted([s.get_array().max() for s in ss])[-2]
    if cmax > 100:
        norm = plt.matplotlib.colors.LogNorm(1, cmax)
    else:
        norm = plt.matplotlib.colors.Normalize(1, cmax)
    for s in ss:
        s.set_norm(norm)
    plt.setp(axx, aspect=1, xlim=(0, vmax), ylim=(0, vmax))
    cax = fig.add_axes([0.93, .1, 0.03, 0.8])
    fig.colorbar(s, cax=cax, label='count')
    start, end = pd.to_datetime(pdf['tempo_time'].quantile([0, 1]), unit='s')
    xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    fig.suptitle(f'{xlblnu} and {ylblnu}: {start:%Y-%m-%d} to {end:%Y-%m-%d}', size=24)

    xlbl, ylbl = getaxislabels(source, spc)
    plt.setp(axx[-1], xlabel=xlbl)
    plt.setp(axx[:, 0], ylabel=ylbl)
    if tstart is None and 'tempo_time_min' in df.columns:
        tstart = pd.to_datetime(
            df['tempo_time_min'].min(), unit='s'
        ).strftime('%Y-%m-%d')
        tend = pd.to_datetime(
            df['tempo_time_max'].min(), unit='s'
        ).strftime('%Y-%m-%d')

    # xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    # titlestr = f'{xlblnu} and {ylblnu}'
    # if tstart is not None:
    #     titlestr += f': {tstart} to {tend}'
    # fig.text(0.58, 0.01, titlestr)

    return fig


def plot_ts(df, source, spc, freq='1h'):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        loaded from getintx
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from .cfg import reftime, v1start, v2start, v3start, queries
    srctkey = source.replace('_nrti', '').replace('_offl', '') + '_time'
    st = reftime + pd.to_timedelta(df[srctkey], unit='s')
    tt = reftime + pd.to_timedelta(df['tempo_time'], unit='s')
    xkey, ykey, xskey, yskey = getkeys(source, spc, std=True)
    tdf = agg(df[[ykey]], tt.dt.floor(freq), err=False)
    sdf = agg(df[[xkey]], st.dt.floor(freq), err=False)
    tv = tdf[ykey]
    sv = sdf[xkey]
    tvs = (
        tdf[ykey + '_q2'] - tdf[ykey + '_q1'],
        tdf[ykey + '_q3'] - tdf[ykey + '_q2'],
    )
    svs = (
        sdf[xkey + '_q2'] - sdf[xkey + '_q1'],
        sdf[xkey + '_q3'] - sdf[xkey + '_q2'],
    )

    # using width instead of offset
    dt = pd.to_timedelta(freq) / 6
    tt = tv.index.values + dt
    st = sv.index.values - dt
    if source == 'airnow':
        right = 0.97
    else:
        right = 0.99
    gskw = dict(left=0.04, right=right)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw, rasterized=True)
    xlbl, ylbl = getaxislabels(source, spc)
    lbl = ylbl.replace('TEMPO ', '')
    if source == 'airnow':
        rax = ax
        tax = ax.twinx()
        ax.set(ylabel=xlbl)
        tax.set(ylabel=ylbl)
    else:
        tax = rax = ax
        ax.set(ylabel=lbl)
    ebs = rax.errorbar(
        st, sv, yerr=svs, color='k', marker='s', linestyle='none', label=source
    )
    ebt = tax.errorbar(
        tt, tv, yerr=tvs, color='r', marker='o', linestyle='none', linewidth=2,
        label='TEMPO'
    )
    ebt[-1][0].set_linestyle('--')
    ebs[-1][0].set_linewidth(ebt[-1][0].get_linewidth() * 1.5)
    if len(queries) > 1:
        ax.axvline(v1start, color='grey', linestyle='--')
        ax.axvline(v2start, color='grey', linestyle='--')
        ax.axvline(v3start, color='grey', linestyle='--')
    ax.legend([ebs, ebt], [getsourcelabel(source), 'TEMPO'])
    ax.set(ylabel=lbl)
    if source == 'airnow':
        rax.set(ylim=(0, None))
        tax.set(ylim=(0, None))
    return ax


def plot_ds(df, source, spc):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        loaded from pair.getintx
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    tstart, tend = get_trange(df)
    gdf = agg(df, 'tempo_lst_hour')
    xk, yk, xks, yks = getkeys(source, spc, std=True)
    sv, tv, svs, tvs = getxy(gdf, source, spc, err=True)
    if source == 'airnow':
        right = 0.9
    else:
        right = 0.98
    gskw = dict(left=0.1, right=right)
    fig, ax = plt.subplots(figsize=(8, 4), gridspec_kw=gskw, rasterized=True)
    xlbl, ylbl = getaxislabels(source, spc)
    if source == 'airnow':
        tax = ax.twinx()
        ax.set(ylabel=xlbl, xlabel='Hour (UTC + LON/15)')
        tax.set(ylabel=ylbl, xlabel='Hour (UTC + LON/15)')
        lbl = ylbl
    else:
        lbl = ylbl.replace('TEMPO ', '')
        tax = ax
        ax.set(ylabel=lbl, xlabel='Hour (UTC + LON/15)')
    ax.errorbar(
        x=sv.index.values - 0.1, y=sv, yerr=svs,
        color='k', marker='o', linestyle='none', label=getsourcelabel(source)
    )
    tax.errorbar(
        x=tv.index.values + 0.1, y=tv, yerr=tvs,
        color='r', marker='+', linestyle='none', label='TEMPO'
    )
    ax.legend()
    fig.text(0.01, 0.01, f'{tstart} to {tend}')
    return ax


def plot_map(pdf, source, spc, tstart=None, tend=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    import numpy as np
    import pycno
    rkey, tkey = getkeys(source, spc)
    rkey = rkey + '_q2'
    tkey = tkey + '_q2'
    pdf.eval(f'nmdnb = ({tkey} - {rkey}) / {rkey} * 100', inplace=True)
    mpdf = pdf.mean()
    lqdf = pdf.quantile(0.25)
    hqdf = pdf.quantile(0.75)
    iqrdf = hqdf - lqdf
    ubdf = mpdf + 2 * iqrdf
    xpdf = pdf.max()
    vmax = min(ubdf[tkey], xpdf[tkey])
    norm = mc.Normalize(vmin=0, vmax=vmax)
    gskw = dict(left=0.0333, right=0.95, wspace=.1)
    fig, axx = plt.subplots(
        1, 3, figsize=(18, 4), gridspec_kw=gskw, sharey=True
    )
    ax = axx[0]
    xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    xlbl, ylbl = getaxislabels(source, spc)
    ckw = dict(label=ylbl)
    lonkey = f'{source}_lon'.replace('_nrti', '').replace('_offl', '')
    latkey = f'{source}_lat'.replace('_nrti', '').replace('_offl', '')
    allopts = dict(norm=norm, zorder=2, colorbar=False, cmap='viridis')
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c=tkey, ax=ax, **allopts)
    fig.colorbar(ax.collections[-1], ax=ax, **ckw)
    tstr = ylblnu
    if tstart is not None and tend is not None:
        tstr += f' {tstart} to {tend}'
    ax.set(title=tstr)
    ax = axx[1]
    ckw = dict(label=xlbl)
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c=rkey, ax=ax, **allopts)
    fig.colorbar(ax.collections[-1], ax=ax, **ckw)
    tstr = xlblnu
    if tstart is not None and tend is not None:
        tstr += f' {tstart} to {tend}'
    ax.set(title=tstr)

    edges = np.array([-100, -60, -45, -30, -15, 15, 30, 45, 60, 100])
    # edges = np.arange(-55, 60, 10)
    ax = axx[2]
    srclabel = getsourcelabel(source)
    ckw = dict(label=f'NMdnB = (TEMPO - {srclabel}) / {srclabel} [%]')
    allopts['cmap'] = 'seismic'
    allopts['norm'] = mc.BoundaryNorm(edges, 256)
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c='nmdnb', ax=ax, **allopts)
    tstr = f'NMdnB {ylblnu.replace("TEMPO ", "")} {tstart} to {tend}'
    ax.set(title=tstr)
    fig.colorbar(ax.collections[-1], ax=ax, **ckw)
    ylim = (17, 51)
    xlim = (-125, -65)
    pycno.cno(ylim=ylim, xlim=xlim).drawstates(resnum=1, zorder=1, ax=axx)
    allopts = dict(
        facecolor='gainsboro',
        xlabel='longitude [deg]', ylabel='latitude [deg]'
    )
    for ax in axx.ravel():
        ax.set(**allopts)
    return axx


def plot_locs(source):
    import pycno
    gskw = dict(left=0.05, bottom=0.5, top=0.97)
    fig, ax = plt.subplots(figsize=(6, 8), gridspec_kw=gskw)
    lockeys = [
        (lcfg['bbox'][0], lockey)
        for lockey, lcfg in cfg.configs.items()
        if (
            (lcfg.get('pandora', False) and source == 'pandora')
            or (
                not lcfg.get('pandora', False) and source != 'pandora'
                and lcfg.get('tropomi', True)
            )
        )
    ]

    lockeys = [k for v, k in sorted(lockeys)]
    for li, lockey in enumerate(lockeys):
        li += 1
        c = plt.get_cmap('tab10')(li % 10)
        lcfg = cfg.configs[lockey]
        wlon, slat, elon, nlat = lcfg['bbox']
        clon = (wlon + elon) / 2
        clat = (slat + nlat) / 2
        if li % 2 == 0:
            offset = 1
        else:
            offset = -1
        ax.annotate(
            f'{li}', (clon, clat), (clon + offset, clat + offset), color=c,
            arrowprops=dict(color=c, arrowstyle='-', shrinkA=0, shrinkB=0)
        )
        coff = (li - 1) % 3
        roff = (li - 1) // 3
        fig.text(
            0.05 + 0.3 * coff, .45 - 0.025 * roff,
            f'{li:02} ' + lcfg.get('label', lockey), color=c
        )
    pycno.cno(ylim=(18, 52), xlim=(-130, -65)).drawstates()
    ax.set_title(f'{source.title()} Locations and Labels')
    # fig.text(1, 1, 'x')
    # fig.text(0, 0, 'x')
    return ax


def plot_summary(pdf, source, spc, vert=True):
    srckey = source.replace('_offl', '').replace('_nrti', '')
    pdf = pdf.sort_values([f'{srckey}_lon'])
    if vert:
        gskw = dict(bottom=0.35, top=0.95, left=0.075, right=0.985)
        figsize = (10, 5)
    else:
        gskw = dict(bottom=0.05, top=0.97, left=0.35, right=0.96)
        figsize = (5, 10)
    fig, ax = plt.subplots(figsize=figsize, gridspec_kw=gskw)
    xkey, ykey, xskey, yskey = getkeys(source, spc, std=True)
    xv, yv, xerr, yerr = getxy(pdf, source, spc, err=True)
    xqkeys = [k for k in pdf if k.startswith(xkey + '_q')]
    yqkeys = [k for k in pdf if k.startswith(ykey + '_q')]
    vlim = (0, pdf[[xqkeys[-2], yqkeys[-2]]].max().max() * 1.05)
    y = np.arange(pdf.shape[0]) - 0.2
    # ax.errorbar(y=y, x=xv, xerr=xerr, linestyle='none', color='k', )
    # ax.plot(xv, y, linestyle='none', color='k', marker='o', label=source)
    xboxes = ax.boxplot(
        pdf[xqkeys].values.T, positions=y, vert=vert, widths=0.2,
        patch_artist=True, whis=1e9
    )
    plt.setp(xboxes['boxes'], facecolor='grey', edgecolor='grey')
    plt.setp(xboxes['medians'], color='red')
    plt.setp(xboxes['whiskers'], linewidth=.25, color='grey')
    plt.setp(xboxes['caps'], linewidth=.25, color='grey')
    y = np.arange(pdf.shape[0]) + 0.2
    # ax.errorbar(y=y, x=yv, xerr=yerr, linestyle='none', color='r', )
    # ax.plot(yv, y, linestyle='none', color='r', marker='+', label='TEMPO')
    yboxes = ax.boxplot(
        pdf[yqkeys].values.T, positions=y, vert=vert, widths=0.2,
        patch_artist=True, whis=1e9
    )
    plt.setp(yboxes['boxes'], facecolor='red', edgecolor='red')
    plt.setp(yboxes['medians'], color='k', )
    plt.setp(yboxes['whiskers'], linewidth=.25, color='red')
    plt.setp(yboxes['caps'], linewidth=.25, color='red')
    y = np.arange(pdf.shape[0])
    xlbl, ylbl = getaxislabels(source, spc)
    lbl = ylbl.replace('TEMPO ', '')
    sfx = ''
    if vert:
        ylim = vlim
        xlim = (-3, y.max() + 3)
        ax.set_xticks(y)
        lbls = [id2label.get(i) for i in pdf.index.values]
        if all([lbl.endswith('-NAA') for lbl in lbls]):
            sfx = ' at NAA'
            lbls = [lbl[:-4] for lbl in lbls]
        else:
            sfx = ' at Pandoras'
        _ = ax.set_xticklabels(lbls, rotation=90)
        ax.set(ylabel=lbl, ylim=ylim, xlim=xlim)
    else:
        ax.set_yticks(y)
        lbls = [id2label.get(i) for i in pdf.index.values]
        if all([lbl.endswith('-NAA') for lbl in lbls]):
            sfx = ' at NAA'
            lbls = [lbl[:-4] for lbl in lbls]
        else:
            sfx = ' at Pandoras'
        _ = ax.set_yticklabels(lbls)
        ylim = (-3, y.max() + 3)
        xlim = vlim
        ax.set(xlabel=lbl, ylim=ylim, xlim=xlim)

    xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    titlestr = f'{xlblnu} and {ylblnu}{sfx}'

    if 'tempo_time_min' in pdf.columns:
        tstart = pd.to_datetime(pdf['tempo_time_min'].min(), unit='s')
        tend = pd.to_datetime(pdf['tempo_time_max'].max(), unit='s')
        titlestr += f': {tstart:%F} to {tend:%F}'
    ax.set_title(titlestr, loc='right')
    trans = ax.transAxes
    if vert:
        ax.text(0.01, .35, 'West', transform=trans, size=18, rotation=90)
        ax.text(0.97, .35, 'East', transform=trans, size=18, rotation=90)
    else:
        ax.text(.35, 0.01, 'West', transform=trans, size=18)
        ax.text(.35, 0.97, 'East', transform=trans, size=18)
    hndl = [xboxes['boxes'][0], yboxes['boxes'][0]]
    lbls = [getsourcelabel(srckey), 'TEMPO']
    ax.legend(hndl, lbls, loc='upper right')
    # fig.text(0, 0, 'x')
    # fig.text(1, 1, 'x')
    return ax


def plot_bias_summary(pdf, source, spc, vert=True):
    srckey = source.replace('_offl', '').replace('_nrti', '')
    pdf = pdf.sort_values([f'{srckey}_lon'])
    if vert:
        gskw = dict(bottom=0.35, top=0.95, left=0.06, right=0.97)
        figsize = (10, 5)
    else:
        gskw = dict(bottom=0.05, top=0.97, left=0.35, right=0.96)
        figsize = (5, 10)
    fig, ax = plt.subplots(figsize=figsize, gridspec_kw=gskw)
    xkey, ykey, xskey, yskey = getkeys(source, spc, std=True)
    xv, yv, xerr, yerr = getxy(pdf, source, spc, err=True)
    qkeys = [k for k in pdf if k.startswith('nerr_q')]
    vlim = (pdf[qkeys[1]].min() * 105, pdf[qkeys[-2]].max() * 105)
    y = np.arange(pdf.shape[0])
    # ax.errorbar(y=y, x=xv, xerr=xerr, linestyle='none', color='k', )
    # ax.plot(xv, y, linestyle='none', color='k', marker='o', label=source)
    xboxes = ax.boxplot(
        pdf[qkeys].values.T * 100, positions=y, vert=vert, widths=0.8,
        patch_artist=True, whis=1e9
    )
    plt.setp(xboxes['boxes'], facecolor='grey')
    plt.setp(xboxes['medians'], color='k')
    srclbl = getsourcelabel(source)
    xlbl, ylbl = getaxislabels(source, spc)
    # lbl = ylbl.replace('TEMPO ', '')
    if vert:
        ylim = vlim
        xlim = (-3, y.max() + 3)
        ax.set_xticks(y)
        _ = ax.set_xticklabels(
            [id2label.get(i) for i in pdf.index.values], rotation=90
        )
        ax.set(
            ylabel=f'(TEMPO - {srclbl}) / {srclbl} [%]', ylim=ylim, xlim=xlim
        )
    else:
        ax.set_yticks(y)
        _ = ax.set_yticklabels([id2label.get(i) for i in pdf.index.values])
        ylim = (-3, y.max() + 3)
        xlim = vlim
        ax.set(
            xlabel=f'(TEMPO - {srclbl}) / {srclbl} [%]', ylim=ylim, xlim=xlim
        )

    xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    titlestr = f'{xlblnu} and {ylblnu}'

    if 'tempo_time_min' in pdf.columns:
        tstart = pd.to_datetime(pdf['tempo_time_min'].min(), unit='s')
        tend = pd.to_datetime(pdf['tempo_time_max'].max(), unit='s')
        titlestr += f': {tstart:%F} to {tend:%F}'
    ax.set_title(titlestr, loc='right')
    opts = dict(transform=ax.transAxes, size=18, bbox=dict(facecolor='white'))
    if vert:
        ax.text(0.01, .35, 'West', rotation=90, **opts)
        ax.text(0.97, .35, 'East', rotation=90, **opts)
        ax.axhline(0, color='k', linestyle='--')
    else:
        ax.text(.35, 0.01, 'West', **opts)
        ax.text(.35, 0.97, 'East', **opts)
        ax.axvline(0, color='k', linestyle='--')
    # fig.text(0, 0, 'x')
    # fig.text(1, 1, 'x')
    return ax


def remakedestfrom(dest, src):
    import os
    if os.path.exists(dest):
        destt = os.stat(dest).st_mtime
        srct = os.stat(src).st_mtime
        remake = destt < srct
    else:
        remake = True
    return remake


def make_plots(source, spc, df=None, debug=False, summary_only=False):
    """
    Arguments
    ---------
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'
    df : pandas.DataFrame
        loaded from makeintx; If None , makeintx is called.
    summary_only : bool
        If True, do not make site-specific plots

    Returns
    -------
    None
    """
    from .pair import getintx
    from .util import regressiondf
    import gc
    import os

    if df is None:
        df = getintx(source, spc)

    adderr(df)
    xlbl, ylbl = getaxislabels(source, spc, unit=False)
    xlblnu, ylblnu = getaxislabels(source, spc, unit=False)
    # lbl = ylbl.replace('TEMPO ', '')
    # lblnu = ylblnu.replace('TEMPO ', '')
    os.makedirs('figs/summary', exist_ok=True)
    if len(cfg.queries) > 1:
        allkey = 'all'
    else:
        allkey = cfg.queries[0][0]
    for qkey, qstr, qlabel in cfg.queries:
        print(source, spc, qkey, flush=True)
        sdf = df.query(qstr)
        if sdf.shape[0] == 0:
            print(source, spc, qkey, 'skipped... no data')
            continue
        regpath = f'csv/{source}_{spc}_{qkey}_regression.csv'
        aggpath = f'csv/{source}_{spc}_{qkey}.csv'
        intxpath = f'intx/{source}_{spc}.h5'
        remakeagg = remakedestfrom(aggpath, intxpath)
        remakereg = remakedestfrom(regpath, intxpath)

        regdf = regressiondf(sdf, outpath=regpath, refresh=remakereg)
        regdef = regdf.iloc[0] * np.nan
        if remakeagg:
            bdf = aggbybox(sdf)
            bdf.to_csv(aggpath)
        else:
            print('reuse aggregation')
            bdf = pd.read_csv(aggpath, index_col=0)
        tstart, tend = get_trange(sdf)
        # Excluding from SDF gets rid of the site-specific
        if source == 'pandora' and spc == 'hcho':
            osdf = sdf.loc[
                ~sdf['pandora_id'].isin(cfg.exclude_pandora_hcho_ids)
            ]
            obdf = bdf.loc[
                ~bdf['pandora_id'].isin(cfg.exclude_pandora_hcho_ids)
            ]
        else:
            osdf = sdf
            obdf = bdf

        # Make summary plots
        ax = plot_ts(sdf, source, spc, freq='3d')
        ax.set(title=f'3-day IQR {xlblnu} and {ylblnu} {qlabel} at All Sites')
        ax.figure.savefig(
            f'figs/summary/{source}_{spc}_{qkey}_ts.png'
        )
        plt.close(ax.figure)

        if source.startswith('tropomi'):
            pdf = obdf.filter(regex='Ozone.*', axis=0)
            ax = plot_summary(pdf, source, spc)
            ax.text(
                0.005, .99, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left', verticalalignment='top'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_summary_ozone.png'
            )
            plt.close(ax.figure)
            ax = plot_bias_summary(pdf, source, spc)
            ax.text(
                0.005, .99, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left', verticalalignment='top'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_bias_summary_ozone.png'
            )
            plt.close(ax.figure)
            pdf = obdf.filter(regex='.*Pandora', axis=0)
            ax = plot_summary(pdf, source, spc)
            ax.text(
                0.005, .99, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left', verticalalignment='top'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_summary_pandora.png'
            )
            plt.close(ax.figure)
            pdf = obdf.filter(regex='.*Pandora', axis=0)
            ax = plot_bias_summary(pdf, source, spc)
            ax.text(
                0.005, .99, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left', verticalalignment='top'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_bias_summary_pandora.png'
            )
            plt.close(ax.figure)
        else:
            ax = plot_summary(obdf, source, spc)
            ax.text(
                0.005, .99, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left', verticalalignment='top'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_summary.png'
            )
            plt.close(ax.figure)
            ax = plot_bias_summary(obdf, source, spc)
            ax.text(
                0.005, .99, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left', verticalalignment='top'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_bias_summary.png'
            )
            plt.close(ax.figure)

        axx = plot_map(obdf, source, spc, tstart=tstart, tend=tend)
        for ax in axx.ravel()[::2]:
            ax.text(
                .025, .025, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left'
            )
        axx[0].figure.savefig(f'figs/summary/{source}_{spc}_{qkey}_map.png')
        plt.close(axx[0].figure)

        # Scatter by site
        ax = plot_scatter(
            obdf, source, spc, tstart=tstart, tend=tend, colorby='tempo_lat'
        )
        ax.set(title=f'All {xlblnu} and {ylblnu}')
        ax.text(
            0.98, 0.02, qlabel, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes, size=24
        )
        ax.figure.savefig(f'figs/summary/{source}_{spc}_{qkey}_all_scat.png')
        plt.close(ax.figure)
        ax = plot_ds(
            osdf.query('tempo_lst_hour > 5 and tempo_lst_hour < 18'),
            source, spc
        )
        ax.set(title=f'All {xlblnu} and {ylblnu} {qlabel}')
        ax.figure.savefig(f'figs/summary/{source}_{spc}_{qkey}_all_ds.png')
        plt.close(ax.figure)
        if source == 'pandora':
            hregpath = f'csv/{source}_{spc}_{qkey}_regression_tod.csv'
            remakehreg = remakedestfrom(hregpath, intxpath)
            reghdf = regressiondf(
                osdf, outpath=hregpath, refresh=remakehreg,
                keys='tempo_lst_hour'
            )
            fig = plot_tod_scatter(osdf, source, spc, reghdf=reghdf)
            fig.text(0.05, 0.92, qlabel, size=24)
            fig.savefig(f'figs/summary/{source}_{spc}_{qkey}_tod_scat.png')
            plt.close(fig)

        # mregpath = f'csv/{source}_{spc}_{qkey}_regression_tod.csv'
        # remakemreg = remakedestfrom(mregpath, intxpath)
        # regmdf = regressiondf(
        #     osdf, outpath=mregpath, refresh=remakemreg, keys='month'
        # )
        fig = plot_month_scatter(osdf, source, spc, reghdf=None)
        fig.text(0.05, 0.92, qlabel, size=24)
        fig.savefig(f'figs/summary/{source}_{spc}_{qkey}_month_scat.png')
        plt.close(fig)

        if summary_only:
            gc.collect()
            continue
        # Make site plots
        if source == 'pandora':
            for pid, pdf in sdf.groupby('pandora_id'):
                lockey = pid2key[pid]
                loclabel = pid2label[pid]
                os.makedirs(f'figs/{lockey}', exist_ok=True)
                print(source, spc, qkey, lockey, flush=True)
                pname = pdf.iloc[0]['pandora_note'].split(';')[-1].strip()
                if pname == '':
                    pname = loclabel
                if lockey in regdf.index:
                    reg = regdf.loc[lockey]
                else:
                    reg = regdef
                ax = plot_scatter(pdf, source, spc, hexn=1, reg=reg)
                ax.set_title(f'{qlabel} at {pname} ({pid:.0f})')
                ax.figure.savefig(
                    f'figs/{lockey}/pandora_{spc}_{qkey}_{lockey}_scat.png'
                )
                plt.close(ax.figure)
                ax = plot_ds(pdf, source, spc)
                ax.set_title(f'{qlabel} at {pname} ({pid:.0f})')
                ax.figure.savefig(
                    f'figs/{lockey}/pandora_{spc}_{qkey}_{lockey}_ds.png'
                )
                plt.close(ax.figure)
                if qkey == allkey:
                    ax = plot_ts(pdf, source, spc)
                    ax.set_title(f'Hourly IQR {qlabel} at {pname} ({pid:.0f})')
                    ax.figure.savefig(
                        f'figs/{lockey}/pandora_{spc}_{qkey}_{lockey}_ts.png'
                    )
                    plt.close(ax.figure)
                if debug:
                    break
        else:
            for lockey, lcfg in cfg.configs.items():
                shortsrc = source.replace('_offl', '').replace('_nrti', '')
                if not lcfg.get(shortsrc, False):
                    continue
                os.makedirs(f'figs/{lockey}', exist_ok=True)
                print(source, spc, qkey, lockey, end='', flush=True)
                loclabel = lockey.replace('Pandora', '')
                loclabel = loclabel.replace('Ozone_8-hr.2015.', '')
                loclabel = lcfg.get('label', loclabel)
                wlon, slat, elon, nlat = lcfg['bbox']
                pdf = sdf.query(
                    f'tempo_lon > {wlon} and tempo_lon < {elon}'
                    f'and tempo_lat > {slat} and tempo_lat < {nlat}'
                )
                print(end='.', flush=True)
                if pdf.shape[0] == 0:
                    print(flush=True)
                    continue
                if lockey in regdf.index:
                    reg = regdf.loc[lockey]
                else:
                    reg = regdef
                ax = plot_scatter(pdf, source, spc, hexn=1, reg=reg)
                ax.set_title(f'{qlabel} at {loclabel}')
                ax.figure.savefig(
                    f'figs/{lockey}/{source}_{spc}_{qkey}_{lockey}_scat.png'
                )
                plt.close(ax.figure)
                print(end='.', flush=True)
                ax = plot_ds(pdf, source, spc)
                ax.set(title=f'{qlabel} at {loclabel}')
                ax.figure.savefig(
                    f'figs/{lockey}/{source}_{spc}_{qkey}_{lockey}_ds.png'
                )
                plt.close(ax.figure)
                print(end='.', flush=True)
                if qkey == allkey:
                    ax = plot_ts(pdf, source, spc)
                    ax.set(title=f'Hourly IQR {qlabel} at {loclabel}')
                    ax.figure.savefig(
                        f'figs/{lockey}/{source}_{spc}_{qkey}_{lockey}_ts.png'
                    )
                    plt.close(ax.figure)
                print(flush=True)
                if debug:
                    break
        if debug:
            break
        del sdf, bdf
        gc.collect()
    del df
    gc.collect()


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    prsr = argparse.ArgumentParser()
    prsr.add_argument(
        '-d', '--dpi', default=100.0, type=float
    )
    prsr.add_argument(
        '-s', '--summary-only', default=False, action='store_true'
    )
    prsr.add_argument(
        '--source', type=lambda x: x.split(','),
        default=['airnow', 'pandora', 'tropomi_offl']
    )
    prsr.add_argument(
        '--spc', type=lambda x: x.split(','), default=['no2', 'hcho']
    )
    args = prsr.parse_args()
    plt.rcParams['figure.dpi'] = args.dpi
    for source in args.source:
        for spc in args.spc:
            if not (source == 'airnow' and spc == 'hcho'):
                make_plots(source, spc, summary_only=args.summary_only)
