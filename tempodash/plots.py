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
    'airnow_lon': ('airnow_lon', 'mean'),
    'airnow_lat': ('airnow_lat', 'mean'),
    'airnow_no2_sfc': ('airnow_no2_sfc', 'mean'),
    'airnow_no2_sfc_std': ('airnow_no2_sfc', 'std'),
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
    dfg = df.groupby(groupkeys)
    dfm = dfg.agg(**funcs)
    return dfm


def aggbybox(df):
    from . import cfg
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

    if 'err' not in df.columns:
        df.eval(f'err = {ykey} - {xkey}', inplace=True)
    if 'nerr' not in df.columns:
        df.eval(f'nerr = err / {xkey}', inplace=True)

    if 'pandora_id' in df.columns:
        bdf = agg(df, 'pandora_id')
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


def getxy(df, source, spc, err=False):
    xk, yk = getkeys(source, spc)
    if err:
        x = df[xk + '_q2']
        y = df[yk + '_q2']
        xerr = (x - df[xk + '_q1'], df[xk + '_q3'] - x)
        yerr = (y - df[yk + '_q1'], df[yk + '_q3'] - y)
        return x, y, xerr, yerr
    else:
        return df[xk], df[yk]


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
    xk, yk, xks, yks = getkeys(source, spc, std=True)
    if 'pandora_id' in df.columns:
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

    tcol = xk.split(spc)[-1][1:]

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

    gskw = dict(right=0.925, left=0.125, bottom=0.12)
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
    if tstart is None and 'tempo_time_min' in df.columns:
        tstart = cfg.reftime + pd.to_timedelta(df['tempo_time_min'].min())
        tend = cfg.reftime + pd.to_timedelta(df['tempo_time_max'].max())

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
    from .cfg import reftime, v1start, v2start
    srctkey = source.replace('_nrti', '').replace('_offl', '') + '_time'
    st = reftime + pd.to_timedelta(df[srctkey], unit='s')
    tt = reftime + pd.to_timedelta(df['tempo_time'], unit='s')
    xkey, ykey, xskey, yskey = getkeys(source, spc, std=True)

    tdf = agg(df[[ykey]], tt.dt.floor('h'))
    sdf = agg(df[[xkey]], st.dt.floor('h'))
    tv = tdf[ykey]
    sv = sdf[xkey]
    tvs = tdf[yskey]
    svs = sdf[xskey]
    tt = tv.index.values
    st = sv.index.values
    gskw = dict(left=0.03, right=0.99)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw, rasterized=True)
    ax.errorbar(st, sv, yerr=svs, color='k', marker='o', linestyle='none')
    ax.errorbar(tt, tv, yerr=tvs, color='r', marker='+', linestyle='none')
    ax.axvline(v1start, color='grey', linestyle='--')
    ax.axvline(v2start, color='grey', linestyle='--')
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

    edges = np.array([-100, -60, -45, -30, -15, 15, 30, 45, 60, 100])
    # edges = np.arange(-55, 60, 10)
    ax = axx[2]
    ckw = dict(label=f'NMdnB = (TEMPO - {source}) / {source} [%]')
    allopts['cmap'] = 'seismic'
    allopts['norm'] = mc.BoundaryNorm(edges, 256)
    ax = pdf.plot.scatter(x=lonkey, y=latkey, c='nmdnb', ax=ax, **allopts)
    tstr = f'NMdnB {spc.upper()} {source} {tstart} to {tend}'
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


def plot_summary(pdf, source, spc, vert=False):
    srckey = source.replace('_offl', '').replace('_nrti', '')
    pdf = pdf.sort_values([f'{srckey}_lon'])
    if vert:
        gskw = dict(bottom=0.35, top=0.96, left=0.05, right=0.97)
        figsize = (10, 5)
    else:
        gskw = dict(bottom=0.05, top=0.97, left=0.35, right=0.96)
        figsize = (5, 10)
    fig, ax = plt.subplots(figsize=figsize, dpi=144, gridspec_kw=gskw)
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
    plt.setp(xboxes['boxes'], facecolor='grey')
    plt.setp(xboxes['medians'], color='k')
    y = np.arange(pdf.shape[0]) + 0.2
    # ax.errorbar(y=y, x=yv, xerr=yerr, linestyle='none', color='r', )
    # ax.plot(yv, y, linestyle='none', color='r', marker='+', label='TEMPO')
    yboxes = ax.boxplot(
        pdf[yqkeys].values.T, positions=y, vert=vert, widths=0.2,
        patch_artist=True, whis=1e9
    )
    plt.setp(yboxes['boxes'], facecolor='red')
    plt.setp(yboxes['medians'], color='k')
    y = np.arange(pdf.shape[0])
    if vert:
        ylim = vlim
        xlim = (-3, y.max() + 3)
        ax.set_xticks(y)
        _ = ax.set_xticklabels(
            [id2label.get(i) for i in pdf.index.values], rotation=90
        )
        ax.set(ylabel=f'{spc.upper()} molec/cm**2', ylim=ylim, xlim=xlim)
    else:
        ax.set_yticks(y)
        _ = ax.set_yticklabels([id2label.get(i) for i in pdf.index.values])
        ylim = (-3, y.max() + 3)
        xlim = vlim
        ax.set(xlabel=f'{spc.upper()} molec/cm**2', ylim=ylim, xlim=xlim)
    titlestr = f'{source.title()} {spc.upper()}'

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
    lbls = [srckey, 'tempo']
    ax.legend(hndl, lbls, loc='upper right')
    # fig.text(0, 0, 'x')
    # fig.text(1, 1, 'x')
    return ax


def plot_bias_summary(pdf, source, spc, vert=False):
    srckey = source.replace('_offl', '').replace('_nrti', '')
    pdf = pdf.sort_values([f'{srckey}_lon'])
    if vert:
        gskw = dict(bottom=0.35, top=0.96, left=0.05, right=0.97)
        figsize = (10, 5)
    else:
        gskw = dict(bottom=0.05, top=0.97, left=0.35, right=0.96)
        figsize = (5, 10)
    fig, ax = plt.subplots(figsize=figsize, dpi=144, gridspec_kw=gskw)
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
    if vert:
        ylim = vlim
        xlim = (-3, y.max() + 3)
        ax.set_xticks(y)
        _ = ax.set_xticklabels(
            [id2label.get(i) for i in pdf.index.values], rotation=90
        )
        ax.set(
            ylabel=f'(tempo - {source}) / {source} [%]', ylim=ylim, xlim=xlim
        )
    else:
        ax.set_yticks(y)
        _ = ax.set_yticklabels([id2label.get(i) for i in pdf.index.values])
        ylim = (-3, y.max() + 3)
        xlim = vlim
        ax.set(
            xlabel=f'(tempo - {source}) / {source} [%]', ylim=ylim, xlim=xlim
        )
    titlestr = f'{source.title()} {spc.upper()}'

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
    import os

    if df is None:
        df = getintx(source, spc)
    qs = [
        ('all', 'v1 == True or v2 == True', 'v1-v2'),
        ('v1', 'v1 == True', 'v1'),
        ('v2', 'v2 == True', 'v2')
    ]
    os.makedirs('figs/summary', exist_ok=True)

    for qkey, qstr, qlabel in qs:
        print(source, spc, qkey, flush=True)
        if qkey == 'all':
            sdf = df
        else:
            sdf = df.query(qstr)
        bdf = aggbybox(sdf)
        bdf.to_csv(f'csv/{source}_{spc}_{qkey}.csv')
        tstart, tend = get_trange(sdf)
        # Make summary plots
        if source.startswith('tropomi'):
            pdf = bdf.filter(regex='Ozone.*', axis=0)
            ax = plot_summary(pdf, source, spc)
            ax.text(
                .95, .01, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='right'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_summary_ozone.png'
            )
            plt.close(ax.figure)
            ax = plot_bias_summary(pdf, source, spc)
            ax.text(
                .95, .01, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='right'
            )
            ax.figure.savefig(
                f'figs/{source}_{spc}_{qkey}_bias_summary_ozone.png'
            )
            plt.close(ax.figure)
            pdf = bdf.filter(regex='.*Pandora', axis=0)
            ax = plot_summary(pdf, source, spc)
            ax.text(
                .95, .01, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='right'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_summary_pandora.png'
            )
            plt.close(ax.figure)
            pdf = bdf.filter(regex='.*Pandora', axis=0)
            ax = plot_bias_summary(pdf, source, spc)
            ax.text(
                .95, .01, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='right'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_bias_summary_pandora.png'
            )
            plt.close(ax.figure)
        else:
            ax = plot_summary(bdf, source, spc)
            ax.text(
                .95, .01, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='right'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_summary.png'
            )
            plt.close(ax.figure)
            ax = plot_bias_summary(bdf, source, spc)
            ax.text(
                .95, .01, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='right'
            )
            ax.figure.savefig(
                f'figs/summary/{source}_{spc}_{qkey}_bias_summary.png'
            )
            plt.close(ax.figure)

        axx = plot_map(bdf, source, spc, tstart=tstart, tend=tend)
        for ax in axx.ravel():
            ax.text(
                .025, .025, qlabel, transform=ax.transAxes, size=24,
                horizontalalignment='left'
            )
        axx[0].figure.savefig(f'figs/summary/{source}_{spc}_{qkey}_map.png')
        plt.close(axx[0].figure)

        # Scatter by site
        ax = plot_scatter(bdf, source, spc, tstart=tstart, tend=tend)
        ax.set(title=f'All {source} {qlabel}')
        ax.figure.savefig(f'figs/summary/{source}_{spc}_{qkey}_all_scat.png')
        plt.close(ax.figure)
        ax = plot_ds(sdf, source, spc)
        ax.set(title=f'All {source} {qlabel}')
        ax.figure.savefig(f'figs/summary/{source}_{spc}_{qkey}_all_ds.png')
        plt.close(ax.figure)
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
                ax = plot_scatter(pdf, source, spc, tstart=tstart, tend=tend)
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
                if qkey == 'all':
                    ax = plot_ts(pdf, source, spc)
                    ax.set_title(f'{qlabel} at {pname} ({pid:.0f})')
                    ax.figure.savefig(
                        f'figs/{lockey}/pandora_{spc}_{qkey}_{lockey}_ts.png'
                    )
                    plt.close(ax.figure)
        else:
            for lockey, lcfg in cfg.configs.items():
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
                ax = plot_scatter(pdf, source, spc, tstart=tstart, tend=tend)
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
                if qkey == 'all':
                    ax = plot_ts(pdf, source, spc)
                    ax.set(title=f'{qlabel} at {loclabel}')
                    ax.figure.savefig(
                        f'figs/{lockey}/{source}_{spc}_{qkey}_{lockey}_ts.png'
                    )
                    plt.close(ax.figure)
                print(flush=True)

        del sdf, bdf
        gc.collect()
    del df
    gc.collect()


if __name__ == '__main__':
    # make_plots('airnow', 'no2')
    # make_plots('pandora', 'no2')
    # make_plots('pandora', 'hcho')
    # make_plots('tropomi_nrti', 'no2')
    # make_plots('tropomi_nrti', 'hcho')
    make_plots('tropomi_offl', 'no2')
    # make_plots('tropomi_offl', 'hcho')
