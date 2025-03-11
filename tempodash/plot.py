def getdatetag(df):
    import pandas as pd
    s = pd.to_datetime(df['tempo_time_start'].min(), unit='s')
    e = pd.to_datetime(df['tempo_time_end'].max(), unit='s')
    return f'{s:%F} to {e:%F}'


def plot_summary_bars(bdf, xkey, subplot_kw=None, ax_kw=None, axx=None, x=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if x is None:
        x = bdf.index.values
    if not np.issubdtype(x.dtype, np.number):
        x = np.arange(x.size)
    dx = np.diff(x).mean()
    assert (np.diff(x) > 0).all()
    gw = dx * .6
    x = x + dx / 2
    xtl = [f'{i - dx / 2:.0f}-{i+dx / 2:.0f}' for i in x]
    if ax_kw is None:
        ax_kw = {}
    if subplot_kw is None:
        gskw = dict(left=0.075, right=0.99, bottom=0.125)
        subplot_kw = dict(
            figsize=(12, 6), gridspec_kw=gskw,
            sharex=True
        )
    if axx is not None:
        fig = axx.ravel()[0].figure
    else:
        fig, axx = plt.subplots(2, 2, **subplot_kw)
    ax = axx[0, 0]
    off = gw / 4
    w = off * 0.9
    ck = bdf['count'] / 1e3
    ax.bar(x=x, width=dx, height=ck, color='lightblue', label='N')
    ax.set(ylabel='Count [x1000]')
    ax = axx[0, 1]
    ax.bar(x=x - off, width=w, height=bdf['rvalue'], color='b', label='R')
    ax.bar(x=x + off, width=w, height=bdf['ioa'], color='g', label='IOA')
    ax.set(ylim=(0, 1))
    off = gw / 3
    w = off * 0.9
    ax = axx[1, 0]
    mab = bdf['err_mean']
    mabp = mab / bdf[f'{xkey}_mean'] * 100
    mb = bdf['bias_mean']
    mbp = mb / bdf[f'{xkey}_mean'] * 100
    rmse = bdf['sqerr_mean']**.5
    rmsep = rmse / bdf[f'{xkey}_mean'] * 100
    ax.bar(x=x - off, width=w, height=mabp, color='b', label='MABP')
    ax.bar(x=x + 0, width=w, height=mbp, color='purple', label='MBP')
    ax.bar(x=x + off, width=w, height=rmsep, color='g', label='RMSEP')
    _ = ax.set(ylabel='Percentage Error [%]')
    if ax.get_ylim()[1] < 100:
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(10))
    ax = axx[1, 1]
    ax.bar(x=x - off, width=w, height=mab / 1e15, color='b', label='MAB')
    ax.bar(x=x + 0, width=w, height=mb / 1e15, color='purple', label='MB')
    ax.bar(x=x + off, width=w, height=rmse / 1e15, color='g', label='RMSE')
    _ = ax.set(ylabel='Error [#/m$^2$ x 10$^{15}$]')
    for ax in axx.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(xtl, rotation=90)
    for ax in axx.ravel()[:]:
        ax.legend(bbox_to_anchor=(1, 1), loc='lower right', ncol=3)
        ax.set(**ax_kw)
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return fig


def plot_scatter(
    scdf, ykey, xkey, c=None, source=None, ax_kw=None, cbar_kw=None, ax=None
):
    import matplotlib.pyplot as plt
    from .agg import odrfit
    from scipy.stats import linregress
    if source is None:
        print(f'WARN:: source inferred from {xkey}/{ykey} for plot_scatter')
        source = xkey.split('_')[0].title()
    if ax_kw is None:
        ax_kw = {}
    if cbar_kw is None:
        cbar_kw = {}
    y = scdf[f'{ykey}_q2']
    yerrl = y - scdf[f'{ykey}_q1']
    yerrh = scdf[f'{ykey}_q3'] - y
    x = scdf[f'{xkey}_q2']
    if c is None:
        c = 'b'
    else:
        c = scdf[c]
    lr = linregress(x, y)
    dr = odrfit(x, y, beta0=[lr.slope, lr.intercept])
    m, b = dr.beta
    fitlabel = f'y={m:.2f}x+{b:.2e} (r={lr.rvalue:.2})'
    xerrl = x - scdf[f'{xkey}_q1']
    xerrh = scdf[f'{xkey}_q3'] - x
    gskw = dict(left=0.15, bottom=0.15, right=0.9)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4), gridspec_kw=gskw)
    else:
        fig = ax.figure
    ax.errorbar(
        x, y, xerr=(xerrl, xerrh), yerr=(yerrl, yerrh),
        linestyle='none', color='k', zorder=1
    )
    ax.axline((0, b), slope=m, label=fitlabel, color='k', linestyle='-')
    ax.axline((0, 0), slope=1, label='1:1', color='gray', linestyle=':')
    s = ax.scatter(x, y, c=c, zorder=3)
    xlim = x.quantile([0, 1])
    ylim = y.quantile([0, 1])
    vlim = min([xlim[0], ylim[0]]), max(xlim[1], ylim[1]) * 1.05
    ax.set(xlim=vlim, ylim=vlim, **ax_kw)
    ax.legend()
    fig.colorbar(s, **cbar_kw)
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return ax


def plot_weekdayweekend(wmdf, ykey, xkey, source=None, ax_kw=None, subplots_kw=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    if source is None:
        print(f'WARN:: source inferred from {xkey}/{ykey} for plot_weekdayweekend')
        source = xkey.split('_')[0].title()
    if ax_kw is None:
        ax_kw = {}
    if subplots_kw is None:
        gskw = dict(left=0.025, right=0.99)
        subplots_kw = dict(figsize=(24, 3), gridspec_kw=gskw)
    wdmdf = wmdf.query('tempo_time.dt.dayofweek < 5')
    wemdf = wmdf.query('tempo_time.dt.dayofweek >= 5')
    xd = wdmdf.index.astype('i8') / 3600e9 / 24
    dx = 0.6
    xe = wemdf.index.astype('i8') / 3600e9 / 24
    sd, ed = pd.to_datetime([
        min(xd.min(), xe.min()), max(xd.max(), xe.max())
    ], unit='d')
    xt = pd.date_range(sd, ed, freq='MS')
    fig, ax = plt.subplots(**subplots_kw)
    tmpopts = dict(x=xd, dx=dx, ax=ax, iqr=True, source=f'{source} Weekday')
    bdx, bdy = plot_boxs(wdmdf, ykey, xkey, **tmpopts)
    bdy['boxes'][0].set_label('TEMPO Weekday')
    tmpopts = dict(x=xe, dx=dx, ax=ax, source=f'{source} Weekend', iqr=True)
    bex, bey = plot_boxs(wemdf, ykey, xkey, **tmpopts)
    bey['boxes'][0].set_label('TEMPO Weekend')
    for patch in bey['boxes']:
        patch.set_facecolor('pink')
    for patch in bex['boxes']:
        patch.set_facecolor('gainsboro')
    ax.set(**ax_kw)
    ax.set_xticks(xt.astype('i8') / 3600e9 / 24)
    ax.set_xticklabels(xt.strftime('%Y-%m'), rotation=90)
    ax.legend(ncol=2, loc='upper left')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return ax


def plot_ts(
    tmdf, ykey, xkey, t=None, dt=None, source=None,
    subplot_kw=None, ax_kw=None, ax=None
):
    import pandas as pd
    import matplotlib.pyplot as plt
    if source is None:
        print(f'WARN:: source inferred from {xkey}/{ykey} for plot_ts')
        source = xkey.split('_')[0].title()
    if ax_kw is None:
        ax_kw = {}
    if t is None:
        t = tmdf.index.get_level_values('tempo_time')
    if dt is None:
        dt = t.diff().median()

    gskw = dict(left=0.025, right=0.99)
    if subplot_kw is None:
        subplot_kw = dict(figsize=(24, 3), gridspec_kw=gskw)
    if ax is None:
        fig, ax = plt.subplots(**subplot_kw)
    else:
        fig = ax.figure

    dt = pd.to_timedelta(dt) / 4
    plot_iqr(tmdf, xkey, x=t - dt, color='k')
    plot_iqr(tmdf, ykey, x=t, color='r', label='TEMPO')
    ax.set(**ax_kw)
    ax.legend(ncol=2, loc='upper left')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return ax


def plot_iqrs(
    df, ykey, xkey, x=None, dx=None, source=None,
    subplot_kw=None, ax_kw=None, ax=None, ycolor='r', xcolor='k'
):
    import numpy as np
    import matplotlib.pyplot as plt
    if source is None:
        print(f'WARN:: source inferred from {xkey}/{ykey} for plot_iqrs')
        source = xkey.split('_')[0].title()
    if ax_kw is None:
        ax_kw = {}
    if x is None:
        x = df.index.values
    elif isinstance(x, str):
        x = df[x]
    if dx is None:
        dx = np.median(np.diff(x)) / 4
    gskw = dict(left=0.025, right=0.99)
    if subplot_kw is None:
        subplot_kw = dict(figsize=(24, 3), gridspec_kw=gskw)
    if ax is None:
        fig, ax = plt.subplots(**subplot_kw)
    else:
        fig = ax.figure
    ex, lx = plot_iqr(df, xkey, x=x - dx, color=xcolor, label=source)
    ey, ly = plot_iqr(df, ykey, x=x, color=ycolor, label='TEMPO')
    ax.set(**ax_kw)
    ax.legend(ncol=2, loc='upper left')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return ax


def plot_iqr(df, ykey, x=None, err_kw=None, minmax=False, ax=None, **plot_kw):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = df.index.values
    elif isinstance(x, str):
        x = df[x]

    plot_kw.setdefault('marker', 's')
    if err_kw is None:
        err_kw = {'color': plot_kw.get('color', None)}
    err_kw.setdefault('elinewidth', plt.rcParams['lines.markersize'] / 2)
    out = []
    y = df[f'{ykey}_q2']
    if minmax:
        xerr_kw = err_kw.copy()
        xerr_kw['elinewidth'] = err_kw['elinewidth'] / 2.
        yerrn = y - df[f'{ykey}_q0']
        yerrx = df[f'{ykey}_q4'] - y
        e = ax.errorbar(x, y, yerr=(yerrn, yerrx), linestyle='none', **xerr_kw)
        out.append(e)
    yerrl = y - df[f'{ykey}_q1']
    yerrh = df[f'{ykey}_q3'] - y
    e = ax.errorbar(x, y, yerr=(yerrl, yerrh), linestyle='none', **err_kw)
    out = [e]
    line = ax.plot(x, y, linestyle='none', **plot_kw)
    out.append(line)
    return out


def plot_box(
    df, ykey, x=None, ax=None, color=None, mediancolor=None, label=None,
    iqr=False, **kw
):
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = df.index.values
    elif isinstance(x, str):
        x = df[x]
    kw = kw.copy()
    kw.setdefault('widths', np.median(np.diff(x)) / 4 * 0.9)
    kw.setdefault('capprops', {'color': color})
    kw.setdefault('boxprops', {'color': color, 'facecolor': color})
    kw.setdefault('whiskerprops', {'color': color})
    kw.setdefault('flierprops', {'color': color})
    kw.setdefault('medianprops', {'color': mediancolor})
    kw.setdefault('meanprops', {'color': mediancolor})
    keys = [f'{ykey}_{sfx}' for sfx in ['q0', 'q1', 'q2', 'q3', 'q4']]
    kw.setdefault('whis',  {True: 0, False: np.inf}[iqr])
    kw.setdefault('showfliers', kw.get('whis'))
    y = df[keys]
    b = ax.boxplot(y.T, positions=x, patch_artist=True, **kw)

    if label is not None:
        b['boxes'][0].set_label(label)
    return b


def plot_boxs(df, ykey, xkey, x=None, source=None, ax=None, iqr=False, **kw):
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = df.index.values
    elif isinstance(x, str):
        x = df[x]
    if source is None:
        print(f'WARN:: source inferred from {xkey}/{ykey} for plot_boxs')
        source = xkey.split('_')[0].title()
    dx = kw.pop('dx', np.median(np.diff(x)) / 4)
    widths = kw.get('widths', dx * 2 * 0.9)
    cmnopts = dict(iqr=iqr, widths=widths, mediancolor='k', ax=ax)
    bx = plot_box(df, xkey, x=x - dx, label=source, color='grey', **cmnopts)
    by = plot_box(df, ykey, x=x + dx, label='TEMPO', color='r', **cmnopts)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    return bx, by
