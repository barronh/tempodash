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
    if 'pandora_id' in df.columns:
        pdf = agg(df, 'pandora_id')
        return pdf
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


def plot_scatter(df, source, spc='no2', hexn=1000, tstart=None, tend=None):
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

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    from scipy.stats.mstats import linregress
    from numpy.ma import masked_invalid
    if (
        'tempo_no2_sum_std' not in df.columns
        and 'tempo_no2_trop_std' not in df.columns
        and 'tempo_hcho_total_std' not in df.columns
    ):
        if 'pandora_id' in df.columns:
            gdf = agg(df, ['pandora_id', 'tempo_time'])
        elif 'tempo_time' in df.columns:
            gdf = agg(df, ['tempo_time', 'tempo_lon', 'tempo_lat'])
    else:
        gdf = df
    xk, yk, xks, yks = getkeys(source, spc, std=True)
    tcol = xk.split(spc)[-1][1:]
    x = gdf[xk]
    xs = gdf[xks]
    y = gdf[yk]
    ys = gdf[yks]
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
    # tstart, tend = get_trange(df)

    gskw = dict(right=0.925, left=0.125)
    fig, ax = plt.subplots(figsize=(4.5, 4), gridspec_kw=gskw, rasterized=True)
    n = x.size
    if n < hexn:
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
    if tstart is not None:
        fig.text(0.58, 0.01, f'{tstart} to {tend}')
    return ax


def plot_ts(df, source, spc):
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


def plot_map(pdf, source, spc, tstart=None, tend=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mc
    import numpy as np
    import pycno
    rkey, tkey = getkeys(source, spc)

    pdf.eval(f'nmb = ({tkey} - {rkey}) / {rkey} * 100', inplace=True)
    mpdf = pdf.mean()
    lqdf = pdf.quantile(0.25)
    hqdf = pdf.quantile(0.75)
    iqrdf = hqdf - lqdf
    ubdf = mpdf + 2 * iqrdf
    xpdf = pdf.max()
    vmax = min(ubdf[tkey], xpdf[tkey])
    norm = mc.Normalize(vmin=0, vmax=vmax)
    gskw = dict(left=0.0333, right=0.95)
    fig, axx = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw=gskw)
    ax = axx[0]
    ckw = dict(label='TEMPO [#/cm**2]')
    lonkey = f'{source}_lon'.replace('_nrti', '').replace('_offl', '')
    latkey = f'{source}_lat'.replace('_nrti', '').replace('_offl', '')
    allopts = dict(norm=norm, zorder=2, colorbar=False, cmap='viridis')
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c=tkey, ax=ax, **allopts)
    fig.colorbar(ax.collections[-1], ax=ax, **ckw)
    tstr = f'{spc.upper()} TEMPO'
    if tstart is not None and tend is not None:
        tstr += f' {tstart} to {tend}'
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
        fig.text(0.05 + 0.3 * coff, .45 - 0.025 * roff, f'{li:02} ' + lcfg.get('label', lockey), color=c)
    pycno.cno(ylim=(18, 52), xlim=(-130, -65)).drawstates()
    ax.set_title(f'{source.title()} Locations and Labels')
    # fig.text(1, 1, 'x')
    # fig.text(0, 0, 'x')
    return ax


def plot_summary(pdf, source, spc, tstart=None, tend=None):
    srckey = source.replace('_offl', '').replace('_nrti', '')
    pdf = pdf.sort_values([f'{srckey}_lon'])
    gskw = dict(bottom=0.05, top=0.97, left=0.35, right=0.96)
    fig, ax = plt.subplots(figsize=(5, 10), gridspec_kw=gskw)
    xkey, ykey, xskey, yskey = getkeys(source, spc, std=True)
    y = np.arange(pdf.shape[0]) - 0.2
    ax.errorbar(y=y, x=pdf[xkey], xerr=pdf[xskey], linestyle='none', color='k', )
    ax.plot(pdf[xkey], y, linestyle='none', color='k', marker='o')
    y = np.arange(pdf.shape[0]) + 0.2
    ax.errorbar(y=y, x=pdf[ykey], xerr=pdf[yskey], linestyle='none', color='r', )
    ax.plot(pdf[ykey], y, linestyle='none', color='r', marker='+')
    y = np.arange(pdf.shape[0])
    ax.set_yticks(y)
    _ = ax.set_yticklabels([id2label.get(i) for i in pdf.index.values])
    ylim = (-3, y.max() + 3)
    ax.set(xlabel='NO2 molec/cm**2', ylim=ylim)
    titlestr = f'{source.title()} {spc.upper()}'
    if tstart is not None and tend is not None:
        titlestr += f': {tstart}-{tend}'
    ax.set_title(titlestr, loc='right')
    ax.text(.35, 0.01, 'West', transform=ax.transAxes, size=18)
    ax.text(.35, 0.97, 'East', transform=ax.transAxes, size=18)
    # fig.text(0, 0, 'x')
    # fig.text(1, 1, 'x')
    return ax


def make_plots(source, spc, df=None):
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
    from .pair import getintx
    import gc
    if df is None:
        df = getintx(source, spc)
    qs = [
        ('all', 'v1 == True or v2 == True', 'v1-v2'),
        ('v1', 'v1 == True', 'v1'),
        ('v2', 'v2 == True', 'v2')
    ]
    for qkey, qstr, qlabel in qs:
        sdf = df.query(qstr)
        bdf = aggbybox(sdf)
        tstart, tend = get_trange(sdf)
        # Make summary plots
        if source.startswith('tropomi'):
            pdf = bdf.filter(regex='Ozone.*', axis=0)
            ax = plot_summary(pdf, source, spc, tstart=tstart, tend=tend)
            ax.text(.95, .05, qlabel, transform=ax.transAxes, size=24, horizontalalignment='right')
            titlestr = f'{source.title()} {spc.upper()}: {tstart} - {tend}'
            ax.set_title(titlestr, loc='right')
            ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_summary_ozone.png')
            plt.close(ax.figure)
            pdf = bdf.filter(regex='.*Pandora', axis=0)
            ax = plot_summary(pdf, source, spc, tstart=tstart, tend=tend)
            ax.text(.95, .05, qlabel, transform=ax.transAxes, size=24, horizontalalignment='right')
            titlestr = f'{source.title()} {spc.upper()}: {tstart} - {tend}'
            ax.set_title(titlestr, loc='right')
            ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_summary_pandora.png')
            plt.close(ax.figure)
            ax = plot_map(bdf, source, spc, tstart=tstart, tend=tend)
            ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_map.png')
            plt.close(ax.figure)
        else:
            ax = plot_summary(bdf, source, spc, tstart=tstart, tend=tend)
            ax.text(.95, .05, qlabel, transform=ax.transAxes, size=24, horizontalalignment='right')
            titlestr = f'{source.title()} {spc.upper()}: {tstart} - {tend}'
            ax.set_title(titlestr, loc='right')
            ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_summary.png')
            plt.close(ax.figure)
            ax = plot_map(bdf, source, spc, tstart=tstart, tend=tend)
            ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_map.png')
            plt.close(ax.figure)

        # Scatter by site
        ax = plot_scatter(bdf, source, spc, tstart=tstart, tend=tend)
        ax.set(title=f'All {source}')
        ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_all_scat.png')
        plt.close(ax.figure)
        ax = plot_ds(sdf, source, spc)
        ax.set(title=f'All {source}')
        ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_all_ds.png')
        plt.close(ax.figure)
        # Make site plots
        if source == 'pandora':
            for pid, pdf in sdf.groupby('pandora_id'):
                lockey = pid2name[pid]
                pname = pdf.iloc[0]['pandora_note'].split(';')[-1].strip()
                if pname == '':
                    pname = lockey
                ax = plot_scatter(pdf, source, spc, tstart=tstart, tend=tend)
                ax.set_title(f'{pname} ({pid})')
                ax.figure.savefig(f'figs/pandora_{spc}_{qkey}_{lockey}_scat.png')
                plt.close(ax.figure)
                ax = plot_ds(pdf, source, spc)
                ax.set_title(f'{pname} ({pid})')
                ax.figure.savefig(f'figs/pandora_{spc}_{qkey}_{lockey}_ds.png')
                plt.close(ax.figure)
        else:
            for lockey, lcfg in cfg.configs.items():
                lockey = lockey.replace('Pandora', '')
                lockey = lockey.replace('Ozone_8-hr.2015.', '')
                wlon, slat, elon, nlat = lcfg['bbox']
                pdf = sdf.query(
                    f'tempo_lon > {wlon} and tempo_lon < {elon}'
                    f'and tempo_lat > {slat} and tempo_lat < {nlat}'
                )
                if pdf.shape[0] == 0:
                    continue
                ax = plot_scatter(pdf, source, spc, tstart=tstart, tend=tend)
                ax.set_title(f'{lockey}')
                ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_{lockey}_scat.png')
                plt.close(ax.figure)
                ax = plot_ds(pdf, source, spc)
                ax.set(title=f'{lockey}')
                ax.figure.savefig(f'figs/{source}_{spc}_{qkey}_{lockey}_ds.png')
                plt.close(ax.figure)
        del sdf, bdf
        gc.collect()


if __name__ == '__main__':
    make_plots('airnow', 'no2')
    make_plots('pandora', 'no2')
    make_plots('pandora', 'hcho')
    make_plots('tropomi_nrti', 'no2')
    make_plots('tropomi_nrti', 'hcho')
    make_plots('tropomi_offl', 'no2')
    make_plots('tropomi_offl', 'hcho')
