import gc
from . import cfg
import pandas as pd
import glob
import matplotlib.pyplot as plt


def loadintx(source, spc):
    """
    source = 'pandora'
    spc = 'no2'
    """
    paths = sorted(glob.glob(f'intx/{source}/????-??/*.csv.gz'))[::20]
    dfs = []
    for pi, p in enumerate(paths):
        df = pd.read_csv(p)
        if df.shape[0] > 0:
            dfs.append(df.copy())
        del df
        # once a day
        if (pi % 15) == 0:
            gc.collect()
            cfg.libc.malloc_trim(1)
    gc.collect()
    cfg.libc.malloc_trim(1)

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    cfg.libc.malloc_trim(1)

    if spc == 'no2':
        df.eval(
            'tempo_no2_sum = tempo_no2_trop + tempo_no2_strat', inplace=True
        )
    t = pd.to_datetime(df['tempo_time']).dt.tz_convert(None)
    lst = t + pd.to_timedelta(df['tempo_lon'] / 15, unit='h')
    df['tempo_lst_hour'] = lst.dt.hour

    df['v1'] = (t > cfg.v1start) & (t < cfg.v2start)
    df['v2'] = (t > cfg.v2start) | (t < cfg.v1start)
    return df


allfuncs = {
    'pandora_lon': ('pandora_lon', 'mean'),
    'pandora_lat': ('pandora_lat', 'mean'),
    'pandora_elevation': ('pandora_elevation', 'mean'),
    'airnow_lon': ('airnow_lon', 'mean'),
    'airnow_lat': ('airnow_lat', 'mean'),
    'tempo_lon': ('tempo_lon', 'mean'),
    'tempo_lat': ('tempo_lat', 'mean'),
    'tropomi_lon': ('tropomi_lon', 'mean'),
    'tropomi_lat': ('airnow_lat', 'mean'),
    'pandora_no2': ('pandora_no2_total', 'mean'),
    'pandora_no2_std': ('pandora_no2_total', 'std'),
    'pandora_hcho': ('pandora_hcho_total', 'mean'),
    'pandora_hcho_std': ('pandora_hcho_total', 'std'),
    'tropomi_no2': ('tropomi_no2_trop', 'mean'),
    'tropomi_no2_std': ('tropomi_no2_trop', 'std'),
    'airnow_no2': ('airnow_no2_sfc', 'mean'),
    'airnow_no2_std': ('airnow_no2_sfc', 'std'),
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

pid2name = {
    v.get('pandoraid', [-999])[0]: k.replace('Pandora', '')
    for k, v in cfg.configs.items()
    if v.get('pandora')
}


def plot_scatter(df, source, spc='no2', tcol='sum'):
    from scipy.stats.mstats import linregress
    from numpy.ma import masked_invalid
    funcs = {
        k: v for k, v in allfuncs.items()
        if v[0] in df.columns
    }
    if source == 'pandora':
        gdf = df.groupby(['pandora_id', 'tempo_time']).agg(**funcs)
    else:
        gdf = df.groupby(['tempo_time', 'tempo_lon', 'tempo_lat']).agg(**funcs)
    x = gdf[f'{source}_{spc}']
    xs = gdf[f'{source}_{spc}_std']
    y = gdf[f'tempo_{spc}_{tcol}']
    ys = gdf[f'tempo_{spc}_{tcol}_std']
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
    tstart = str(df['tempo_time'].min()).split('T')[0]
    tend = str(df['tempo_time'].max()).split('T')[0]

    gskw = dict(right=0.925)
    fig, ax = plt.subplots(figsize=(4.5, 4), gridspec_kw=gskw)
    ax.errorbar(
        x=x, y=y, yerr=ys, xerr=xs, color='k', linestyle='none', zorder=1
    )
    s = ax.scatter(x=x, y=y, c=c, zorder=2)
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
    fig.colorbar(s, ax=ax, label='Solar Zenith Angle [deg]')
    fig.text(0.58, 0.01, f'{tstart} to {tend}')
    return ax


def plot_ts(df, source, spc):
    st = pd.to_datetime(df[f'{source}_time'])
    tt = pd.to_datetime(df['tempo_time'])
    if source == 'pandora':
        tv = df['tempo_no2_sum']
        sv = df['pandora_no2']
    elif source == 'airnow':
        tv = df['tempo_no2_trop']
        sv = df['airnow_no2']
    elif source == 'tropomi':
        tv = df['tempo_no2_trop']
        sv = df['tropomi_no2']
    gskw = dict(left=0.03, right=0.99)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw, rasterized=True)
    ax.plot(st, sv, color='k', marker='o', linestyle='none')
    ax.plot(tt, tv, color='r', marker='+', linestyle='none')
    return ax


def plot_ds(df, source, spc):
    funcs = {
        k: v for k, v in allfuncs.items()
        if v[0] in df.columns
    }
    tstart = str(df['tempo_time'].min()).split('T')[0]
    tend = str(df['tempo_time'].max()).split('T')[0]
    gdf = df.groupby('tempo_lst_hour').agg(**funcs)
    if source == 'pandora':
        if spc == 'no2':
            tv = gdf['tempo_no2_sum']
            tvs = gdf['tempo_no2_sum_std']
        else:
            tv = gdf['tempo_hcho_total']
            tvs = gdf['tempo_hcho_total_std']
        sv = gdf[f'pandora_{spc}']
        svs = gdf[f'pandora_{spc}_std']
    elif source == 'airnow' and spc == 'no2':
        tv = gdf['tempo_no2_trop']
        tvs = gdf['tempo_no2_trop_std']
        sv = gdf['airnow_no2']
        svs = gdf['airnow_no2_std']
    elif source == 'tropomi' and spc == 'no2':
        tv = gdf['tempo_no2_trop']
        tvs = gdf['tempo_no2_trop_std']
        sv = gdf['tropomi_no2']
        svs = gdf['tropomi_no2_std']
    elif source == 'tropomi' and spc == 'hcho':
        tv = gdf['tempo_hcho_total']
        tvs = gdf['tempo_hcho_total_std']
        sv = gdf['tropomi_hcho']
        svs = gdf['tropomi_hcho_std']
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
        x=sv.index.values - 0.2, y=sv, yerr=svs,
        color='k', marker='o', linestyle='none', label=source
    )
    tax.errorbar(
        x=tv.index.values + 0.2, y=tv, yerr=tvs,
        color='r', marker='+', linestyle='none', label='TEMPO'
    )
    ax.set(ylabel='molec/cm**2', xlabel='Hour (UTC + LON/15)')
    fig.text(0.01, 0.01, f'{tstart} to {tend}')
    return ax


def makeplots(source, spc, df=None):
    if df is None:
        df = loadintx(source, spc)

    ax = plot_scatter(df, source, spc)
    ax.set(title=f'All {source}')
    ax.figure.savefig(f'figs/{source}_all_scat.png')
    ax = plot_ds(df, source, spc)
    ax.set(title=f'All {source}')
    ax.figure.savefig(f'figs/{source}_all_ds.png')
    if source == 'pandora':
        for pid, pdf in df.groupby('pandora_id'):
            lockey = pid2name[pid]
            pname = pdf.iloc[0]['pandora_note'].split(';')[-1].strip()
            if pname == '':
                pname = lockey
            ax = plot_scatter(pdf, source, spc)
            ax.set_title(f'{pname} ({pid})')
            ax.figure.savefig(f'figs/pandora_{lockey}_scat.png')
            plt.close(ax.figure)
            ax = plot_ds(pdf, source, spc)
            ax.set_title(f'{pname} ({pid})')
            ax.figure.savefig(f'figs/pandora_{lockey}_ds.png')
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
            ax.figure.savefig(f'figs/{source}_{lockey}_scat.png')
            plt.close(ax.figure)
            ax = plot_ds(pdf, source, spc)
            ax.set(title=f'{lockey}')
            ax.figure.savefig(f'figs/{source}_{lockey}_ds.png')
            plt.close(ax.figure)


if __name__ == '__main__':
    makeplots('airnow', 'no2')
    makeplots('pandora', 'no2')
    # makeplots('pandora', 'hcho')
    # makeplots('tropomi', 'no2')
    # makeplots('tropomi', 'hcho')
