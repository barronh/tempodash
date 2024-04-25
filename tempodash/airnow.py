__all__ = ['getairnow', 'get_airnow_chunk', 'pairairnow']


def makeplots(lockey, spc, bdate, edate, buffer=0.03, update=False):
    import os
    import pandas as pd

    bdate = pd.to_datetime(bdate, utc=True)
    edate = pd.to_datetime(edate, utc=True)
    opts = dict(lockey=lockey, spc=spc, bdate=bdate, edate=edate)
    pdf = getairnow(**opts, rename=True, parse_dates=True)
    ixpath = f'locations/{lockey}/store/airnow_{spc}_{buffer:.02f}_v2.csv'
    update = update or not os.path.exists(ixpath)
    if update:
        print(f'updating {ixpath}')
        ixdf = pairairnow(**opts, buffer=buffer)
        ixdf.to_csv(ixpath, index=False)
    ixdf = pd.read_csv(ixpath)
    ixdf['tempo_time'] = pd.to_datetime(ixdf['tempo_time'])
    ixdf['airnow_time'] = pd.to_datetime(ixdf['airnow_time'])
    ixdf = ixdf.query(
        f'tempo_time >= "{bdate:%Y-%m-%dT%H:%M:%S%z}"'
        + f' and tempo_time <= "{edate:%Y-%m-%dT%H:%M:%S%z}"'
    )

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


def pairairnow(lockey, spc, bdate, edate, freq='1h', buffer=0.03):
    import pandas as pd
    import geopandas as gpd
    from .tempo import gettempo
    adf = getairnow(
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
            tdf = gettempo(
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


def getairnow(
    lockey, spc, bdate, edate, freq='1h', asgeo=False, parse_dates=False,
    rename=False
):
    import pyrsig
    import pandas as pd
    from . import get_configs, server
    cfg = get_configs()[lockey]
    bbox = cfg['bbox']

    bhours = pd.date_range(bdate, edate, freq=freq)
    hours = cfg['hours']

    dt = pd.to_timedelta(freq) - pd.to_timedelta('1s')
    dfs = []
    for bhour in bhours:
        if bhour.hour in hours:
            workdir = f'locations/{lockey}/{bhour:%Y-%m-%d}'
            api = pyrsig.RsigApi(bbox=bbox, workdir=workdir, server=server)
            ehour = bhour + dt
            try:
                df = get_airnow_chunk(api, spc, bhour, ehour)
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


def get_airnow_chunk(api, spc, bdate, edate, verbose=-10):
    if spc == 'no2':
        akey = 'airnow.no2'
    else:
        raise KeyError(f'spc must be no2: got {spc}')

    # Try to get either/both
    adf = api.to_dataframe(
        akey, bdate=bdate, edate=edate, unit_keys=False, verbose=verbose
    )
    return adf
