from . import cfg
import pandas as pd
import matplotlib.pyplot as plt


pid2name = {
    v.get('pandoraid', [-999])[0]: k.replace('Pandora', '')
    for k, v in cfg.configs.items()
    if v.get('pandora')
}

allfuncs = {
    'pandora_time': ('pandora_time', 'mean'),
    'pandora_lon': ('pandora_lon', 'mean'),
    'pandora_lat': ('pandora_lat', 'mean'),
    'pandora_elevation': ('pandora_elevation', 'mean'),
    'pandora_no2_total': ('pandora_no2_total', 'mean'),
    'pandora_no2_total_std': ('pandora_no2_total', 'std'),
    'pandora_hcho_total': ('pandora_hcho_total', 'mean'),
    'pandora_hcho_total_std': ('pandora_hcho_total', 'std'),
    'airnow_time': ('airnow_time', 'mean'),
    'airnow_lon': ('airnow_lon', 'mean'),
    'airnow_lat': ('airnow_lat', 'mean'),
    'airnow_no2_sfc': ('airnow_no2_sfc', 'mean'),
    'airnow_no2_sfc_std': ('airnow_no2_sfc', 'std'),
    'tropomi_lon': ('tropomi_lon', 'mean'),
    'tropomi_lat': ('tropomi_lat', 'mean'),
    'tropomi_no2_trop': ('tropomi_no2_trop', 'mean'),
    'tropomi_no2_trop_std': ('tropomi_no2_trop', 'std'),
    'tropomi_hcho_total': ('tropomi_hcho_total', 'mean'),
    'tropomi_hcho_total_std': ('tropomi_hcho_total', 'std'),
    'tempo_time': ('tempo_time', 'mean'),
    'tempo_lon': ('tempo_lon', 'mean'),
    'tempo_lat': ('tempo_lat', 'mean'),
    'tempo_no2_sum': ('tempo_no2_sum', 'mean'),
    'tempo_no2_sum_std': ('tempo_no2_sum', 'std'),
    'tempo_no2_trop': ('tempo_no2_trop', 'mean'),
    'tempo_no2_trop_std': ('tempo_no2_trop', 'std'),
    'tempo_no2_total': ('tempo_no2_total', 'mean'),
    'tempo_no2_total_std': ('tempo_no2_total', 'std'),
    'tempo_hcho_total': ('tempo_hcho_total', 'mean'),
    'tempo_hcho_total_std': ('tempo_hcho_total', 'std'),
    'tempo_no2_strat': ('tempo_no2_strat', 'mean'),
    'tempo_no2_strat_std': ('tempo_no2_strat', 'std'),
    'tempo_sza': ('tempo_sza', 'mean'),
    'tempo_cloud_eff': ('tempo_cloud_eff', 'mean'),
}


def agg(df, groupkeys):
    funcs = {
        k: v for k, v in allfuncs.items()
        if v[0] in df.columns
    }
    dfg = df.groupby(groupkeys)
    dfm = dfg.agg(**funcs)
    return dfm


def aggbybox(df):
    from . import cfg
    pdfs = []
    for lockey, lcfg in cfg.configs.items():
        wlon, slat, elon, nlat = lcfg['bbox']
        pdf = df.query(
            f'tempo_lon > {wlon} and tempo_lon < {elon}'
            + f' and tempo_lat > {slat} and tempo_lat < {nlat}'
        ).copy()
        if pdf.shape[0] == 0:
            continue
        pdf['lockey'] = lockey
        pdf = agg(pdf, 'lockey')
        pdfs.append(pdf)
    pdf = pd.concat(pdfs)
    return pdf


def get_trange(df):
    if 'tempo_time' not in df.columns:
        return 'unknown', 'unknown'
    tstart = cfg.reftime + pd.to_timedelta(df['tempo_time'].min(), unit='s')
    tend = cfg.reftime + pd.to_timedelta(df['tempo_time'].max(), unit='s')
    tstart = tstart.strftime('%Y-%m-%d')
    tend = tend.strftime('%Y-%m-%d')
    return tstart, tend


def getkeys(source, spc, std=False):
    tomis = ('tropomi', 'tropomi_offl', 'tropomi_nrti')
    if source == 'pandora' and spc == 'no2':
        y = 'tempo_no2_sum'
        x = 'pandora_no2_total'
    elif source == 'pandora' and spc == 'no2':
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


def plot_scatter(df, source, spc='no2', tcol='sum'):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        loaded from makeintx
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'
    tcol : str
        'sum' or 'trop' or 'total' or 'strat'

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    from scipy.stats.mstats import linregress
    from numpy.ma import masked_invalid

    if source == 'pandora':
        gkeys = ['pandora_id', 'tempo_time']
    else:
        gkeys = ['tempo_time', 'tempo_lon', 'tempo_lat']
    xk, yk, xks, yks = getkeys(source, spc, std=True)
    if xks in df.columns:
        gdf = df
    else:
        gdf = agg(df, gkeys)

    x = gdf[xk]
    xs = gdf[xks]
    y = gdf[yk]
    ys = gdf[yks]
    print(xk, yk, xks, yks)
    c = gdf['tempo_sza']
    units = 'molec/cm**2'
    if source == 'airnow':
        units = '1'
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        xs = xs * 0
        ys = ys * 0

    lr = linregress(masked_invalid(x.values), masked_invalid(y.values))
    lrstr = f'y = {lr.slope:.2f}x{lr.intercept:+.2e} (r={lr.rvalue:.2f})'
    xmax = max(x.max(), y.max()) * 1.05
    tstart, tend = get_trange(df)

    gskw = dict(right=0.925)
    fig, ax = plt.subplots(figsize=(4.5, 4), gridspec_kw=gskw, rasterized=True)
    n = x.size
    if n < 1000:
        ax.errorbar(
            x=x, y=y, yerr=ys, xerr=xs, color='k', linestyle='none', zorder=1
        )
        s = ax.scatter(x=x, y=y, c=c, zorder=2)
        fig.colorbar(s, ax=ax, label='Solar Zenith Angle [deg]')
    else:
        vmax = max(x.max(), y.max())
        s = ax.hexbin(
            x=x, y=y, gridsize=(20, 20), extent=(0, vmax, 0, vmax),
            mincnt=1
        )
        ax.set(facecolor='gainsboro')
        fig.colorbar(s, ax=ax, label='Count [#]')

    ax.axline((0, 0), slope=1, zorder=3, label='1:1', color='grey')
    ax.axline(
        (0, lr.intercept), slope=lr.slope, zorder=3, label=lrstr, color='k'
    )
    ax.set(
        aspect=1, xlim=(0, xmax), ylim=(0, xmax),
        xlabel=f'{source.upper()} [{units}]',
        ylabel=f'TEMPO {tcol} [{units}]',
    )
    ax.legend()
    fig.text(0.58, 0.01, f'{tstart} to {tend}')
    return ax


def plot_ts(df, source, spc):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        loaded from makeintx
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    srctkey = source.replace('_nrti', '').replace('_offl', '') + '_time'
    st = pd.to_datetime(df[srctkey])
    tt = pd.to_datetime(df['tempo_time'])
    rkey, tkey = getkeys(source, spc)

    tv = df[tkey]
    sv = df[rkey]
    gskw = dict(left=0.03, right=0.99)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw, rasterized=True)
    ax.plot(st, sv, color='k', marker='o', linestyle='none')
    ax.plot(tt, tv, color='r', marker='+', linestyle='none')
    return ax


def plot_ds(df, source, spc):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        loaded from makeintx
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
    tv = gdf[yk]
    tvs = gdf[yks]
    sv = gdf[xk]
    svs = gdf[xks]
    if source == 'airnow':
        right = 0.9
    else:
        right = 0.98
    gskw = dict(left=0.1, right=right)
    fig, ax = plt.subplots(figsize=(8, 4), gridspec_kw=gskw, rasterized=True)
    if source == 'airnow':
        tax = ax.twinx()
    else:
        tax = ax
    ax.errorbar(
        x=sv.index.values - 0.1, y=sv, yerr=svs,
        color='k', marker='o', linestyle='none', label=source
    )
    tax.errorbar(
        x=tv.index.values + 0.1, y=tv, yerr=tvs,
        color='r', marker='+', linestyle='none', label='TEMPO'
    )
    ax.set(ylabel='molec/cm**2', xlabel='Hour (UTC + LON/15)')
    ax.legend()
    fig.text(0.01, 0.01, f'{tstart} to {tend}')
    return ax


def make_map(df, source, spc):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    import numpy as np
    import pycno
    if source == 'pandora':
        pdf = agg(df, 'pandora_id')
    else:
        pdf = aggbybox(df)

    if 'tempo_hcho_total' in pdf.columns:
        spc = 'hcho'
    else:
        spc = 'no2'
    rkey, tkey = getkeys(source, spc)

    pdf.eval(f'nmb = ({tkey} - {rkey}) / {rkey} * 100', inplace=True)
    tstart, tend = get_trange(df)
    mpdf = pdf.mean()
    lqdf = pdf.quantile(0.25)
    hqdf = pdf.quantile(0.75)
    iqrdf = hqdf - lqdf
    ubdf = mpdf + 2 * iqrdf
    xpdf = pdf.max()
    vmax = min(ubdf[tkey], xpdf[tkey])
    norm = mc.Normalize(vmin=0, vmax=vmax)
    fig, axx = plt.subplots(1, 3, figsize=(18, 4))
    ax = axx[0]
    ckw = dict(label='TEMPO [#/cm**2]')
    lonkey = f'{source}_lon'.replace('_nrti', '').replace('_offl', '')
    latkey = f'{source}_lat'.replace('_nrti', '').replace('_offl', '')
    allopts = dict(norm=norm, zorder=2, colorbar=False, cmap='viridis')
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c=tkey, ax=ax, **allopts)
    fig.colorbar(ax.collections[-1], ax=ax, **ckw)
    tstr = f'{spc.upper()} TEMPO {tstart} to {tend}'
    ax.set(title=tstr)
    ax = axx[1]
    ckw = dict(label=f'{source} [#/cm**2]')
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c=rkey, ax=ax, **allopts)
    fig.colorbar(ax.collections[-1], ax=ax, **ckw)
    tstr = f'{spc.upper()} {source} {tstart} to {tend}'
    ax.set(title=tstr)

    edges = np.arange(-50, 60, 20)
    # edges = np.arange(-55, 60, 10)
    ax = axx[2]
    ckw = dict(label=f'NMB = (TEMPO - {source}) / {source} [%]')
    allopts['cmap'] = 'seismic'
    allopts['norm'] = mc.BoundaryNorm(edges, 256)
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c='nmb', ax=ax, **allopts)
    tstr = f'NMB {spc.upper()} {source} {tstart} to {tend}'
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
    return ax


def makeplots(source, spc, df=None):
    """
    Arguments
    ---------
    source : str
        'pandora' or 'airnow' or 'tropomi_nrti' or 'tropomi_offl'
    spc : str
        'no2' or 'hcho'
    df : pandas.DataFrame
        loaded from makeintx; If None , makeintx is called.

    Returns
    -------
    None
    """
    from .intx import getintx
    if df is None:
        df = getintx(source, spc)

    ax = plot_scatter(df, source, spc)
    ax.set(title=f'All {source}')
    ax.figure.savefig(f'figs/{source}_{spc}_all_scat.png')
    ax = plot_ds(df, source, spc)
    ax.set(title=f'All {source}')
    ax.figure.savefig(f'figs/{source}_{spc}_all_ds.png')
    if source == 'pandora':
        for pid, pdf in df.groupby('pandora_id'):
            lockey = pid2name[pid]
            pname = pdf.iloc[0]['pandora_note'].split(';')[-1].strip()
            if pname == '':
                pname = lockey
            ax = plot_scatter(pdf, source, spc)
            ax.set_title(f'{pname} ({pid})')
            ax.figure.savefig(f'figs/pandora_{spc}_{lockey}_scat.png')
            plt.close(ax.figure)
            ax = plot_ds(pdf, source, spc)
            ax.set_title(f'{pname} ({pid})')
            ax.figure.savefig(f'figs/pandora_{spc}_{lockey}_ds.png')
            plt.close(ax.figure)
    else:
        for lockey, lcfg in cfg.configs.items():
            lockey = lockey.replace('Pandora', '')
            lockey = lockey.replace('Ozone_8-hr.2015.', '')
            wlon, slat, elon, nlat = lcfg['bbox']
            pdf = df.query(
                f'tempo_lon > {wlon} and tempo_lon < {elon}'
                f'and tempo_lat > {slat} and tempo_lat < {nlat}'
            )
            if pdf.shape[0] == 0:
                continue
            ax = plot_scatter(pdf, source, spc)
            ax.set_title(f'{lockey}')
            ax.figure.savefig(f'figs/{source}_{spc}_{lockey}_scat.png')
            plt.close(ax.figure)
            ax = plot_ds(pdf, source, spc)
            ax.set(title=f'{lockey}')
            ax.figure.savefig(f'figs/{source}_{spc}_{lockey}_ds.png')
            plt.close(ax.figure)


if __name__ == '__main__':
    makeplots('airnow', 'no2')
    makeplots('pandora', 'no2')
    makeplots('pandora', 'hcho')
    makeplots('tropomi_nrti', 'no2')
    makeplots('tropomi_nrti', 'hcho')
    makeplots('tropomi_offl', 'no2')
    makeplots('tropomi_offl', 'hcho')
