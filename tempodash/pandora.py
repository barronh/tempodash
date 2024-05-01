__all__ = ['open_pandora', 'open_pandora_chunk', 'pair_pandora', 'make_intx']


def make_intx(
    lockey, spc, bdate, edate, buffer=0.03, withins=900, pdf=None
):
    from .util import get_configs
    import pandas as pd
    import os

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)
    cfg = get_configs()[lockey]
    pid = cfg['pandoraid'][0]
    ixpath = f'locations/{lockey}/store/pandora_{spc}_{buffer:.2f}_v2.csv'
    metapath = (
        f'locations/{lockey}/store/'
        + f'pandora_{spc}_{buffer:.2f}_v2_meta.csv'
    )

    if os.path.exists(metapath) and os.path.exists(ixpath):
        metadf = pd.read_csv(metapath)
        ixdf = pd.read_csv(ixpath)
        pre_bdate = bdate
        pre_edate = pd.to_datetime(metadf['bdate'].min(), utc=True)
        post_bdate = pd.to_datetime(metadf['edate'].max(), utc=True)
        post_edate = edate
        ixdfs = [ixdf]
    else:
        metadf = pd.DataFrame(dict(
            bdate=[], edate=[], updated=[], action=[]
        ))
        pre_bdate = bdate
        pre_edate = edate
        post_bdate = edate
        post_edate = edate
        ixdfs = []

    updated = False
    if pre_bdate < pre_edate:
        opts = dict(lockey=lockey, spc=spc, bdate=pre_bdate, edate=pre_edate)
        try:
            ixdf = pair_pandora(
                **opts, buffer=buffer, withins=withins, pdf=pdf
            ).query(f'pandora_station == {pid}')
            ixdfs.insert(0, ixdf)
            updated = True
        except ValueError:
            pass

    if post_bdate < post_edate:
        opts = dict(lockey=lockey, spc=spc, bdate=post_bdate, edate=post_edate)
        try:
            ixdf = pair_pandora(
                **opts, buffer=buffer, withins=withins, pdf=pdf
            ).query(f'pandora_station == {pid}')
            ixdfs.append(ixdf)
            updated = True
        except ValueError:
            pass

    ixdf = pd.concat(ixdfs)
    if updated:
        now = pd.to_datetime('now', utc=True)
        metadf = pd.concat([
            metadf,
            pd.DataFrame(dict(
                bdate=[bdate], edate=[edate], updated=[now]
            ))
        ], ignore_index=True)
        metadf.to_csv(metapath, index=False)
        ixdf.to_csv(ixpath, index=False)

    ixdf['tempo_time'] = pd.to_datetime(ixdf['tempo_time'])
    ixdf['pandora_time'] = pd.to_datetime(ixdf['pandora_time'])
    ixdf = ixdf.query(
        f'tempo_time >= "{bdate:%Y-%m-%dT%H:%M:%S%z}"'
        + f' and tempo_time <= "{edate:%Y-%m-%dT%H:%M:%S%z}"'
    )
    return ixdf


def make_plots(lockey, spc, bdate, edate, buffer=0.03, withins=900):
    from .util import get_configs
    import pandas as pd

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)
    cfg = get_configs()[lockey]
    pid = cfg['pandoraid'][0]
    opts = dict(lockey=lockey, spc=spc, bdate=bdate, edate=edate)
    pdf = open_pandora(**opts, rename=True, parse_dates=True).query(
        f'pandora_station == {pid}'
    )
    ixdf = make_intx(**opts, buffer=buffer, withins=withins, pdf=pdf)
    tkey = f'tempo_{spc}'
    if spc == 'no2':
        tkey += '_sum'
    agfuncs = {
        'pandora_lon': ('pandora_lon', 'mean'),
        'pandora_lat': ('pandora_lat', 'mean'),
        'pandora': (f'pandora_{spc}', 'mean'),
        'pandora_std': (f'pandora_{spc}', 'std'),
        'tempo_lon': ('tempo_lon', 'mean'),
        'tempo_lat': ('tempo_lat', 'mean'),
        'tempo_sza': ('tempo_sza', 'mean'),
        'tempo': (tkey, 'mean'),
        'tempo_std': (tkey, 'std'),
    }
    pairdf = ixdf.groupby(['tempo_time'], as_index=False).agg(**agfuncs)
    plot_ts(pdf, ixdf, spc, lockey)
    plot_ds(pairdf, spc, lockey)
    plot_scat(pairdf, spc, lockey)


def plot_ts(pdf, ixdf, spc, lockey):
    import matplotlib.pyplot as plt
    gskw = dict(right=0.975, left=0.05)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw)
    ptkey = 'pandora_time'
    pvkey = f'pandora_{spc}'
    ttkey = 'tempo_time'
    tvkey = f'tempo_{spc}'
    if spc == 'no2':
        tvkey = f'tempo_{spc}_sum'
    ax.scatter(pdf[ptkey], pdf[pvkey], color='grey', zorder=1)
    ax.scatter(
        ixdf[ptkey], ixdf[pvkey], color='k', zorder=2, label='Pandora Total'
    )
    ax.scatter(
        ixdf[ttkey], ixdf[tvkey], color='r', zorder=3,
        label='TEMPO Trop+Strat', marker='+'
    )
    ax.set(ylabel=spc + ' [#/cm$^2$]', xlabel='Time [UTC]', title=lockey)
    ax.legend()
    fig.savefig(f'locations/{lockey}/figs/pandora/{lockey}_{spc}_ts.png')


def plot_ds(pairdf, spc, lockey):
    import matplotlib.pyplot as plt
    import pandas as pd
    pkey = 'pandora'
    tkey = 'tempo'
    utc = pairdf['tempo_time']
    offset = pd.to_timedelta(pairdf['pandora_lon'] / 15., unit='h')
    offset = offset.mean()
    lst = utc + offset
    lst.name = 'lst'
    lstg = pairdf.groupby(lst.dt.hour)
    fig, ax = plt.subplots(figsize=(12, 4))
    for lsth, lstdf in lstg:
        x = lsth - 0.15
        ax.plot([x], [lstdf[pkey].mean()], c='b', marker='o', linestyle='none')
        ys = [lstdf[pkey].min(), lstdf[pkey].max()]
        ax.plot([x, x], ys, color='b', linestyle='-')
        x = lsth + 0.15
        ax.plot([x], [lstdf[tkey].mean()], c='r', marker='o', linestyle='none')
        ys = [lstdf[tkey].min(), lstdf[tkey].max()]
        ax.plot([x, x], ys, color='r', linestyle='-')

    xlim = 6, 18
    offseth = offset.total_seconds() / 3600.
    ax.set(
        title=lockey, xlim=xlim, ylim=(0, None),
        ylabel=spc + ' [#/cm$^2$]', xlabel=f'Hour [LST={offseth:.2f}]',
    )
    fig.savefig(f'locations/{lockey}/figs/pandora/{lockey}_{spc}_ds.png')


def plot_scat(pairdf, spc, lockey):
    from scipy.stats.mstats import linregress
    from .odrfit import odrfit
    import matplotlib.pyplot as plt
    from .util import get_configs
    import pycno

    cno = pycno.cno()

    pkey = 'pandora'
    pskey = 'pandora_std'
    tkey = 'tempo'
    tskey = 'tempo_std'
    vmax = pairdf[[tkey, pkey]].max().max() * 1.05
    n = pairdf.shape[0]
    bbox = get_configs()[lockey]['bbox']
    xlim = bbox[::2]
    ylim = bbox[1::2]

    fig, axx = plt.subplots(1, 2, figsize=(12, 4))
    ax = axx[0]
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    ax.plot(x, y, marker='+', linestyle='--', color='k')
    opt = dict(marker='+', linestyle='none', color='r', label='intersections')
    ax.plot(pairdf['tempo_lon'], pairdf['tempo_lat'], **opt)
    dx = 3
    ax.set(
        title=lockey, xlim=(xlim[0] - dx, xlim[1] + dx),
        ylim=(ylim[0] - dx, ylim[1] + dx)
    )
    cno.drawstates(resnum=1, ax=ax, label='States')

    ax = axx[1]
    s = ax.scatter(pairdf[pkey], pairdf[tkey], c=pairdf['tempo_sza'], zorder=2)
    fig.colorbar(s, label='SZA', ax=ax)
    ax.errorbar(
        pairdf[pkey], pairdf[tkey], yerr=pairdf[tskey], xerr=pairdf[pskey],
        linestyle='none', zorder=1, color='k'
    )

    lr = linregress(pairdf[pkey], pairdf[tkey])
    m = lr.slope
    b = lr.intercept
    label = f'$LR$={m:.2f}$p${b:+.2e}'
    ax.axline((0, b), slope=m, color='k', linestyle='--', label=label)
    try:
        beta0 = [m, b]
        dr = odrfit(pairdf[pkey], pairdf[tkey], beta0=beta0)
        m = dr.beta[0]
        b = dr.beta[1]
        label = f'$DR$={m:.2f}$p${b:+.2e}'
        ax.axline((0, b), slope=m, color='k', linestyle=':', label=label)
    except Exception:
        pass

    ax.axline((0, 0), slope=1, color='grey', linestyle='-')
    ax.set(
        xlim=(0, vmax), ylim=(0, vmax), title=f'(r={lr.rvalue:.2f}; n={n})',
        xlabel='Pandora [#/cm$^2$]', ylabel='TEMPO [#/cm$^2$]'
    )
    ax.legend()
    fig.savefig(f'locations/{lockey}/figs/pandora/{lockey}_{spc}_scat.png')


def pair_pandora(
    lockey, spc, bdate, edate, freq='1h', withins=900, buffer=0.03, pdf=None
):
    import pandas as pd
    import geopandas as gpd
    from .util import get_configs
    from .tempo import open_tempo
    hours = get_configs()[lockey]['hours']
    if pdf is None:
        pdf = open_pandora(
            lockey, spc, bdate, edate, freq='1h', asgeo=True, rename=True
        )
    pdf['pandora_time'] = pd.to_datetime(pdf['Timestamp'])
    if buffer is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pdf['geometry'] = pdf['geometry'].buffer(buffer)
    bhours = sorted(pdf['pandora_time'].dt.floor(freq).unique())
    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    withindt = pd.to_timedelta(withins, unit='s')
    ixdfs = []
    for bhour in bhours:
        if bhour.hour not in hours:
            continue
        ehour = bhour + dt
        lbt = bhour - withindt
        ubt = ehour + withindt
        pchnk = pdf.query(
            f'pandora_time >= "{lbt:%Y-%m-%dT%H:%M:%S%z}"'
            + f' and pandora_time <= "{ubt:%Y-%m-%dT%H:%M:%S%z}"'
        )
        if pchnk.shape[0] == 0:
            continue
        try:
            tdf = open_tempo(
                lockey, spc, bhour, ehour, asgeo=True, rename=True,
                parse_dates=True
            )
        except Exception:
            # usually means no data
            continue
        if tdf.shape[0] == 0:
            continue
        ixdf = gpd.overlay(pchnk, tdf)
        if ixdf.shape[0] == 0:
            continue
        ixdf['pandora_dt'] = (
            ixdf['tempo_time'] - ixdf['pandora_time']
        ).dt.total_seconds()
        ixdf['adt'] = ixdf['pandora_dt'].abs()
        ixdfs.append(ixdf.query(f'adt < {withins}'))
    ixdf = pd.concat(ixdfs)
    keepcols = [
        k for k in ixdf.columns
        if k.startswith('tempo_') or k.startswith('pandora_')
    ]
    return ixdf[keepcols]


def open_pandora(
    lockey, spc, bdate, edate, freq='1h', asgeo=False, parse_dates=False,
    rename=False, minq='medium'
):
    import pyrsig
    import pandas as pd
    from .util import get_configs
    from . import server
    cfg = get_configs()[lockey]
    bbox = cfg['bbox']
    if spc == 'no2':
        pkey = 'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
    elif spc == 'hcho':
        pkey = 'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount'
    else:
        raise KeyError(f'spc must be no2 or hcho: got {spc}')

    workdir = f'locations/{lockey}/{minq}'
    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
    api.pandora_kw['minimum_quality'] = minq
    df = api.to_dataframe(
        pkey, bdate=bdate, edate=edate, backend='xdr', unit_keys=False
    )
    if parse_dates:
        df['pandora_time'] = pd.to_datetime(df['Timestamp'])
    if asgeo:
        import geopandas as gpd
        geom = gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE'])
        df = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    if rename:
        df.rename(columns=dict(
            LONGITUDE='pandora_lon', LATITUDE='pandora_lat',
            ELEVATION='pandora_alt', STATION='pandora_station',
            nitrogen_dioxide_vertical_column_amount='pandora_no2',
            formaldehyde_total_vertical_column_amount='pandora_hcho',
        ), inplace=True)

    return df


def open_pandora_chunk(api, spc, bdate, edate, verbose=-1):
    if spc == 'no2':
        pkey = 'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
    elif spc == 'hcho':
        pkey = 'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount'
    else:
        raise KeyError(f'spc must be no2 or hcho: got {spc}')

    # Try to get either/both
    pdf = api.to_dataframe(
        pkey, bdate=bdate, edate=edate, unit_keys=False, verbose=verbose
    )
    return pdf
