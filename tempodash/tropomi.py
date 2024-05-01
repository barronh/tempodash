__all__ = [
    'open_tropomi', 'open_tropomi_chunk', 'get_tropomi_chunk', 'pair_tropomi',
    'make_intx'
]

_coordkeys = [
    'Longitude_SW', 'Latitude_SW',
    'Longitude_SE', 'Latitude_SE',
    'Longitude_NE', 'Latitude_NE',
    'Longitude_NW', 'Latitude_NW',
    'Longitude_SW', 'Latitude_SW',
]


def make_intx(lockey, spc, bdate, edate):
    import pandas as pd
    import os

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)
    ixpath = f'locations/{lockey}/store/tropomi_{spc}_v2.csv'
    metapath = f'locations/{lockey}/store/tropomi_{spc}_v2_meta.csv'

    if os.path.exists(metapath) and os.path.exists(ixpath):
        metadf = pd.read_csv(metapath)
        pre_bdate = bdate
        pre_edate = pd.to_datetime(metadf['bdate'].min(), utc=True)
        post_bdate = pd.to_datetime(metadf['edate'].max(), utc=True)
        post_edate = edate
        ixdfs = [pd.read_csv(ixpath)]
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
            ixdf = pair_tropomi(**opts)
            ixdfs.insert(0, ixdf)
            updated = True
        except ValueError:
            pass

    if post_bdate < post_edate:
        opts = dict(lockey=lockey, spc=spc, bdate=post_bdate, edate=post_edate)
        try:
            ixdf = pair_tropomi(**opts)
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
    ixdf['tropomi_time'] = pd.to_datetime(ixdf['tropomi_time'])
    ixdf = ixdf.query(
        f'tempo_time >= "{bdate:%Y-%m-%dT%H:%M:%S%z}"'
        + f' and tempo_time <= "{edate:%Y-%m-%dT%H:%M:%S%z}"'
    )
    return ixdf


def make_plots(lockey, spc, bdate, edate):
    import pandas as pd

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)
    opts = dict(lockey=lockey, spc=spc, bdate=bdate, edate=edate)
    pdf = open_tropomi(**opts, rename=True, parse_dates=True)
    ixdf = make_intx(**opts)
    tkey = f'tempo_{spc}'
    if spc == 'no2':
        tkey = f'tempo_{spc}_sum'

    aggfuncs = {
        'tropomi_lon': ('tropomi_lon', 'mean'),
        'tropomi_lat': ('tropomi_lat', 'mean'),
        'tropomi': (f'tropomi_{spc}', 'mean'),
        'tropomi_std': (f'tropomi_{spc}', 'std'),
        'tempo': (tkey, 'mean'),
        'tempo_std': (tkey, 'std'),
        'tempo_sza': ('tempo_sza', 'mean')
    }
    pairdf = ixdf.groupby(['tempo_time'], as_index=False).agg(**aggfuncs)
    plot_ts(pdf, ixdf, spc, lockey)
    plot_scat(pairdf, spc, lockey)


def plot_ts(pdf, ixdf, spc, lockey):
    import matplotlib.pyplot as plt
    gskw = dict(right=0.975, left=0.05)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw)
    ptkey = 'tropomi_time'
    pvkey = f'tropomi_{spc}'
    ttkey = 'tempo_time'
    tvkey = f'tempo_{spc}'
    if spc == 'no2':
        tvkey = f'tempo_{spc}_sum'
    ax.scatter(pdf[ptkey], pdf[pvkey], color='grey', zorder=1)
    ax.scatter(
        ixdf[ptkey], ixdf[pvkey], color='k', zorder=2, label='tropomi Total'
    )
    ax.scatter(
        ixdf[ttkey], ixdf[tvkey], color='r', zorder=3,
        label='TEMPO Trop+Strat', marker='+'
    )
    ax.set(ylabel=spc + ' [#/cm$^2$]', xlabel='Time [UTC]', title=lockey)
    ax.legend()
    fig.savefig(f'locations/{lockey}/figs/tropomi/{lockey}_{spc}_ts.png')


def plot_scat(pairdf, spc, lockey):
    from scipy.stats.mstats import linregress
    import matplotlib.pyplot as plt
    from .odrfit import odrfit
    from .util import get_configs
    import pycno

    cno = pycno.cno()

    bbox = get_configs()[lockey]['bbox']
    xlim = bbox[::2]
    ylim = bbox[1::2]
    rkey = 'tropomi'
    tkey = 'tempo'
    rskey = 'tropomi_std'
    tskey = 'tempo_std'
    vmax = pairdf[[tkey, rkey]].max().max() * 1.05
    n = pairdf.shape[0]
    fig, axx = plt.subplots(1, 2, figsize=(12, 4))
    ax = axx[0]
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    ax.plot(x, y, marker='+', linestyle='--', color='k')
    opt = dict(marker='+', linestyle='none', color='r', label='intersections')
    ax.plot(pairdf['tropomi_lon'], pairdf['tropomi_lat'], **opt)
    dx = 3
    ax.set(
        title=lockey, xlim=(xlim[0] - dx, xlim[1] + dx),
        ylim=(ylim[0] - dx, ylim[1] + dx)
    )
    cno.drawstates(resnum=1, ax=ax, label='States')
    ax = axx[1]
    s = ax.scatter(pairdf[rkey], pairdf[tkey], c=pairdf['tempo_sza'], zorder=2)
    fig.colorbar(s, label='SZA', ax=ax)
    ax.errorbar(
        pairdf[rkey], pairdf[tkey], yerr=pairdf[tskey], xerr=pairdf[rskey],
        linestyle='none', zorder=1, color='k'
    )

    lr = linregress(pairdf[rkey], pairdf[tkey])
    m = lr.slope
    b = lr.intercept
    label = f'$LR$={m:.2f}$p${b:+.2e}'
    ax.axline((0, b), slope=m, color='k', linestyle='--', label=label)
    try:
        beta0 = [m, b]
        dr = odrfit(pairdf[rkey], pairdf[tkey], beta0=beta0)
        m = dr.beta[0]
        b = dr.beta[1]
        label = f'$DR$={m:.2f}$p${b:+.2e}'
        ax.axline((0, b), slope=m, color='k', linestyle=':', label=label)
    except Exception as e:
        print(e)
        pass

    ax.axline((0, 0), slope=1, color='grey', linestyle='-')
    ax.set(
        xlim=(0, vmax), ylim=(0, vmax), title=f'(r={lr.rvalue:.2f}; n={n})',
        xlabel='TropOMI [#/cm$^2$]', ylabel='TEMPO [#/cm$^2$]'
    )
    ax.legend()
    fig.savefig(f'locations/{lockey}/figs/tropomi/{lockey}_{spc}_scat.png')


def pair_tropomi(lockey, spc, bdate, edate, freq='1h'):
    import pandas as pd
    import geopandas as gpd
    from .tempo import open_tempo
    pdf = open_tropomi(
        lockey, spc, bdate, edate, freq='1h', asgeo=True, rename=True,
        parse_dates=True
    )
    bhours = sorted(pdf['tropomi_time'].dt.floor(freq).unique())
    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    ixdfs = []
    for bhour in bhours:
        ehour = bhour + dt
        try:
            tdf = open_tempo(
                lockey, spc, bhour, ehour, asgeo=True, rename=True,
                parse_dates=True
            )
        except ValueError:
            continue
        tchnk = pdf.query(
            f'tropomi_time >= "{bhour:%Y-%m-%dT%H:%M:%S%z}"'
            + f' and tropomi_time <= "{ehour:%Y-%m-%dT%H:%M:%S%z}"'
        )
        ixdf = gpd.overlay(tchnk, tdf)
        ixdfs.append(ixdf)
    ixdf = pd.concat(ixdfs)
    keepcols = [
        k for k in ixdf.columns
        if k.startswith('tempo_') or k.startswith('tropomi_')
    ]
    return ixdf[keepcols]


def open_tropomi(
    lockey, spc, bdate, edate, freq='1h', asgeo=False, parse_dates=False,
    rename=False
):
    import pandas as pd
    from .util import get_configs
    cfg = get_configs()[lockey]

    bhours = pd.date_range(bdate, edate, freq='1h')
    hours = cfg['tropomi_hours']

    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    dfs = []
    for bhour in bhours:
        if bhour.hour in hours:
            ehour = bhour + dt
            try:
                df = open_tropomi_chunk(lockey, spc, bhour, ehour)
                dfs.append(df)
            except pd.errors.EmptyDataError:
                pass

    df = pd.concat(dfs)
    if parse_dates:
        df['tropomi_time'] = pd.to_datetime(df['Timestamp'])

    if asgeo:
        import geopandas as gpd
        from shapely import polygons
        geom = polygons(df[_coordkeys].values.reshape(-1, 5, 2))
        df = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    if rename:
        df.rename(columns=dict(
            LONGITUDE='tropomi_lon',
            LATITUDE='tropomi_lat',
            nitrogendioxide_tropospheric_column='tropomi_no2',
            formaldehyde_tropospheric_vertical_column='tropomi_hcho',
        ), inplace=True)
    return df


def open_tropomi_chunk(lockey, spc, bdate, edate, verbose=-1):
    import pyrsig
    import pandas as pd
    from .util import get_configs
    from . import server

    cfg = get_configs()[lockey]
    bbox = cfg['bbox']
    bdate = pd.to_datetime(bdate)
    edate = pd.to_datetime(edate)
    workdir = f'locations/{lockey}/{bdate:%Y-%m-%d}'
    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
    if spc == 'no2':
        tomikey = 'tropomi.nrti.no2.nitrogendioxide_tropospheric_column'
    elif spc == 'hcho':
        tomikey = 'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column'
    else:
        raise KeyError(f'spc must be no2 or hcho: got {spc}')

    # Try to get either/both
    tdf = api.to_dataframe(
        tomikey, bdate=bdate, edate=edate, unit_keys=False, verbose=verbose
    )
    return tdf


def get_tropomi_chunk(lockey, spc, bdate, edate, verbose=-1):
    import pyrsig
    import pandas as pd
    from .util import get_configs
    from . import server

    cfg = get_configs()[lockey]
    bbox = cfg['bbox']
    hours = cfg['tropomi_hours']
    bdate = pd.to_datetime(bdate)
    if bdate.hour not in hours:
        return
    edate = pd.to_datetime(edate)
    workdir = f'locations/{lockey}/{bdate:%Y-%m-%d}'
    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
    if spc == 'no2':
        tomikey = 'tropomi.nrti.no2.nitrogendioxide_tropospheric_column'
    elif spc == 'hcho':
        tomikey = 'tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column'
    else:
        raise KeyError(f'spc must be no2 or hcho: got {spc}')

    # Try to get either/both
    api.get_file(
        'ascii', key=tomikey, bdate=bdate, edate=edate, bbox=bbox, corners=1,
        compress=1, verbose=verbose, overwrite=False
    )
