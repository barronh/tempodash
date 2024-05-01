__all__ = [
    'open_airnow', 'open_airnow_chunk', 'get_airnow_chunk', 'pair_airnow',
    'make_intx'
]


def make_intx(lockey, spc, bdate, edate, buffer=0.03):
    import pandas as pd
    import os

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)

    ixpath = f'locations/{lockey}/store/airnow_{spc}_{buffer:.2f}_v2.csv'
    metapath = (
        f'locations/{lockey}/store/'
        + f'airnow_{spc}_{buffer:.2f}_v2_meta.csv'
    )

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
            ixdf = pair_airnow(**opts, buffer=buffer)
            ixdfs.insert(0, ixdf)
            updated = True
        except ValueError:
            pass

    if post_bdate < post_edate:
        opts = dict(lockey=lockey, spc=spc, bdate=post_bdate, edate=post_edate)
        try:
            ixdf = pair_airnow(**opts, buffer=buffer)
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
    ixdf['airnow_time'] = pd.to_datetime(ixdf['airnow_time'])
    ixdf = ixdf.query(
        f'tempo_time >= "{bdate:%Y-%m-%dT%H:%M:%S%z}"'
        + f' and tempo_time <= "{edate:%Y-%m-%dT%H:%M:%S%z}"'
    )
    return ixdf


def makeplots(lockey, spc, bdate, edate, buffer=0.03, update=False):
    import pandas as pd

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)
    opts = dict(lockey=lockey, spc=spc, bdate=bdate, edate=edate)
    pdf = open_airnow(**opts, rename=True, parse_dates=True)
    ixdf = make_intx(**opts, buffer=buffer, update=update)
    pairdf = ixdf.groupby(['tempo_time'], as_index=False).agg(
        airnow_lon=('airnow_lon', 'mean'),
        airnow_no2=('airnow_no2', 'mean'),
        airnow_no2_std=('airnow_no2', 'std'),
        tempo_no2_sum=('tempo_no2_sum', 'mean'),
        tempo_sza=('tempo_sza', 'mean')
    )
    plot_ts(pdf, ixdf, spc, lockey)
    plot_ds(pairdf, spc, lockey)
    plot_scat(pairdf, spc, lockey)


def plot_ts(pdf, ixdf, spc, lockey):
    import matplotlib.pyplot as plt
    gskw = dict(right=0.975, left=0.05)
    fig, ax = plt.subplots(figsize=(18, 4), gridspec_kw=gskw)
    ptkey = 'airnow_time'
    pvkey = f'airnow_{spc}'
    ttkey = 'tempo_time'
    tvkey = f'tempo_{spc}'
    if spc == 'no2':
        tvkey = f'tempo_{spc}_sum'
    ax.scatter(pdf[ptkey], pdf[pvkey], color='grey', zorder=1)
    ax.scatter(
        ixdf[ptkey], ixdf[pvkey], color='k', zorder=2, label='airnow Total'
    )
    ax.scatter(
        ixdf[ttkey], ixdf[tvkey], color='r', zorder=3,
        label='TEMPO Trop+Strat', marker='+'
    )
    ax.set(ylabel=spc + ' [#/cm$^2$]', xlabel='Time [UTC]', title=lockey)
    ax.legend()
    fig.savefig(f'locations/{lockey}/figs/airnow/{lockey}_{spc}_ts.png')


def plot_ds(pairdf, spc, lockey):
    import matplotlib.pyplot as plt
    import pandas as pd
    pkey = f'airnow_{spc}'
    tkey = f'tempo_{spc}'
    if spc == 'no2':
        tkey = f'tempo_{spc}_sum'
    utc = pairdf['tempo_time']
    offset = pd.to_timedelta(pairdf['airnow_lon'] / 15., unit='h')
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
    fig.savefig(f'locations/{lockey}/figs/airnow/{lockey}_{spc}_ds.png')


def plot_scat(pairdf, spc, lockey):
    from scipy.stats.mstats import linregress
    from .odrfit import odrfit
    import matplotlib.pyplot as plt

    pkey = f'airnow_{spc}'
    if spc == 'no2':
        tkey = 'tempo_no2_sum'
    else:
        tkey = f'tempo_{spc}'
    vmax = pairdf[[tkey, pkey]].max().max() * 1.05
    n = pairdf.shape[0]
    lr = linregress(pairdf[pkey], pairdf[tkey])
    fig, ax = plt.subplots()
    s = ax.scatter(pairdf[pkey], pairdf[tkey], c=pairdf['tempo_sza'])
    fig.colorbar(s, label='SZA')
    m = lr.slope
    b = lr.intercept
    label = f'$LR$={m:.2f}$p${b:+.2e} (r={lr.rvalue:.2f}; n={n})'
    ax.axline((0, b), slope=m, color='k', linestyle='--', label=label)
    try:
        dr = odrfit(pairdf[pkey], pairdf[tkey])
        m = dr.beta[0]
        b = dr.beta[1]
        label = f'$DR$={m:.2f}$p${b:+.2e}'
        ax.axline((0, b), slope=m, color='k', linestyle=':', label=label)
    except Exception:
        pass

    ax.axline((0, 0), slope=1, color='grey', linestyle='-')
    ax.set(
        xlim=(0, vmax), ylim=(0, vmax),
        xlabel='airnow [#/cm$^2$]', ylabel='Tempo [#/cm$^2$]'
    )
    ax.legend()
    fig.savefig(f'locations/{lockey}/figs/airnow/{lockey}_{spc}_scat.png')


def pair_airnow(lockey, spc, bdate, edate, freq='1h', buffer=0.03):
    import pandas as pd
    import geopandas as gpd
    from .tempo import open_tempo
    adf = open_airnow(
        lockey, spc, bdate, edate, freq='1h', asgeo=True, rename=True
    )
    adf['airnow_time'] = pd.to_datetime(adf['Timestamp'])
    if buffer is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            adf['geometry'] = adf['geometry'].buffer(buffer)

    bhours = sorted(adf['airnow_time'].dt.floor(freq).unique())
    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    ixdfs = []
    for bhour in bhours:
        ehour = bhour + dt
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
        achnk = adf.query(
            f'airnow_time >= "{bhour:%Y-%m-%dT%H:%M:%S%z}"'
            + f' and airnow_time <= "{ehour:%Y-%m-%dT%H:%M:%S%z}"'
        )
        ixdf = gpd.overlay(achnk, tdf)
        if ixdf.shape[0] == 0:
            continue

        ixdfs.append(ixdf)
    ixdf = pd.concat(ixdfs)
    keepcols = [
        k for k in ixdf.columns
        if k.startswith('tempo_') or k.startswith('airnow_')
    ]
    return ixdf[keepcols]


def open_airnow(
    lockey, spc, bdate, edate, freq='1h', asgeo=False, parse_dates=False,
    rename=False
):
    import pandas as pd
    from .util import get_configs
    cfg = get_configs()[lockey]

    bhours = pd.date_range(bdate, edate, freq=freq)
    hours = cfg['hours']

    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    dfs = []
    for bhour in bhours:
        if bhour.hour in hours:
            ehour = bhour + dt
            try:
                df = open_airnow_chunk(lockey, spc, bhour, ehour)
                dfs.append(df)
            except pd.errors.EmptyDataError:
                pass

    df = pd.concat(dfs)
    if parse_dates:
        df['airnow_time'] = pd.to_datetime(df['Timestamp'])
    if asgeo:
        import geopandas as gpd
        geom = gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE'])
        df = gpd.GeoDataFrame(df, geometry=geom, crs=4326)
    if rename:
        df.rename(columns=dict(
            LONGITUDE='airnow_lon', LATITUDE='airnow_lat',
            ELEVATION='airnow_alt', STATION='airnow_station',
            no2='airnow_no2',
        ), inplace=True)

    return df


def open_airnow_chunk(lockey, spc, bdate, edate, verbose=-10):
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
        akey = 'airnow.no2'
    else:
        raise KeyError(f'spc must be no2: got {spc}')

    # Try to get either/both
    adf = api.to_dataframe(
        akey, bdate=bdate, edate=edate, unit_keys=False, verbose=verbose
    )
    return adf


def get_airnow_chunk(lockey, spc, bdate, edate, verbose=-10):
    import pyrsig
    import pandas as pd
    from .util import get_configs
    from . import server
    cfg = get_configs()[lockey]
    bbox = cfg['bbox']
    hours = cfg['hours']
    bdate = pd.to_datetime(bdate)
    if bdate.hour not in hours:
        return

    edate = pd.to_datetime(edate)
    workdir = f'locations/{lockey}/{bdate:%Y-%m-%d}'
    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)

    if spc == 'no2':
        akey = 'airnow.no2'
    else:
        raise KeyError(f'spc must be no2: got {spc}')

    # Try to get either/both
    api.get_file(
        'ascii', key=akey, bdate=bdate, edate=edate, bbox=bbox, corners=1,
        compress=1, verbose=verbose, overwrite=False
    )
