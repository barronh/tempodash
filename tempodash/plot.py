def getdatetag(df, timekey=None):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame with data having tempo_time_start and tempo_time_end
    Returns
    -------
    tag : str
        Start to end 'YYYY-MM-DD to YYYY-MM-DD'
    """
    import pandas as pd
    from .util import getfirstkey
    timekey = getfirstkey(df, '_time_start').replace('_start', '')
    s = pd.to_datetime(df[f'{timekey}_start'].min(), unit='s')
    e = pd.to_datetime(df[f'{timekey}_end'].max(), unit='s')
    return f'{s:%F} to {e:%F}'


def plot_summary_bars(
    bdf, xkey, subplot_kw=None, ax_kw=None, axx=None, x=None
):
    """
    Plot panels on axx (2, 2)
    - (0, 0) : count,
    - (0, 1) : R/IOA,
    - (1, 0) : MB/MAB/RSME,
    - (1, 1) : MBP/MABP/RSMEP,

    Arguments
    ---------
    bdf : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    xkey : str
        Name of element from DataFrame
    ax_kw : dict
        Keywords to Axes
    cbar_kw : dict
        Keywords to colrobar
    axx : array
        Array of matplotlib.axes.Axes object to plot on
    x : array-like
        Location for bars

    Returns
    -------
    ax : Axes
        matplotlib.axes.Axes
    """
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
    ax = axx[1, 1]
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
    ax = axx[1, 0]
    ax.bar(x=x - off, width=w, height=mab / 1e15, color='b', label='MAB')
    ax.bar(x=x + 0, width=w, height=mb / 1e15, color='purple', label='MB')
    ax.bar(x=x + off, width=w, height=rmse / 1e15, color='g', label='RMSE')
    _ = ax.set(ylabel='Error [#/cm$^2$ x 10$^{15}$]')
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
    scdf, ykey, xkey, c=None, ax_kw=None, cbar_kw=None, ax=None
):
    """
    Create a scatter plot with color and statistics

    Arguments
    ---------
    scdf : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    ykey : str
        Name of element from DataFrame
    xkey : str
        Name of element from DataFrame
    c : str
        Name of element from DataFrame
    ax_kw : dict
        Keywords to Axes
    cbar_kw : dict
        Keywords to colrobar
    ax : matplotlib.axes.Axes
        Axes object to plot on

    Returns
    -------
    ax : Axes
        matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from .agg import odrfit
    from scipy.stats import linregress
    from .util import getsource, getlabel
    if ax_kw is None:
        ax_kw = {}
    ax_kw.setdefault('ylabel', f'{getsource(ykey)} {getlabel(ykey)}')
    ax_kw.setdefault('xlabel', f'{getsource(xkey)} {getlabel(xkey)}')
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


def plot_weekdayweekend(
    wmdf, ykey, xkey, ax_kw=None, subplots_kw=None, ysource=None
):
    """
    Plot boxplots with weekdays as one pair and weekends as another.

    Arguments
    ---------
    wmdf : pandas.DataFrame
        DataFrame with ykey and xkey column
    ykey : str
        Name of element from DataFrame
    xkey : str
        Name of element from DataFrame
    ax_kw : dict
        Keywords to Axes
    subplot_kw : dict
        Keywords to subplot

    Returns
    -------
    ax : Axes
        matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from .util import getfirstkey, getsource
    source = getsource(xkey)
    if ax_kw is None:
        ax_kw = {}
    if subplots_kw is None:
        gskw = dict(left=0.025, right=0.99)
        subplots_kw = dict(figsize=(24, 3), gridspec_kw=gskw)
    timekey = getfirstkey(wmdf, '_time')
    wdmdf = wmdf.query(f'{timekey}.dt.dayofweek < 5')
    wemdf = wmdf.query(f'{timekey}.dt.dayofweek >= 5')
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
    ysrc = getsource(ykey)
    bdy['boxes'][0].set_label(f'{ysrc} Weekday')
    tmpopts = dict(x=xe, dx=dx, ax=ax, source=f'{source} Weekend', iqr=True)
    bex, bey = plot_boxs(wemdf, ykey, xkey, **tmpopts)
    bey['boxes'][0].set_label(f'{ysrc} Weekend')
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
    tmdf, ykey, xkey, t=None, dt=None,
    subplot_kw=None, ax_kw=None, ax=None
):
    """
    Plot a time-series of error bars

    Arguments
    ---------
    tmdf : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    ykey : str
        Name of element from DataFrame
    xkey : str
        Name of element from DataFrame
    t : list, str, or None
        x-locations for the boxes or a named column (default index)
    dt : int or None
        defaults to median(x) / 4
    subplot_kw : dict
        Keywords to subplot
    ax_kw : dict
        Keywords to Axes
    ax : matplotlib.axes.Axes
        axes to plot on

    Returns
    -------
    ax : Axes
        matplotlib.axes.Axes
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from .util import getsource
    if ax_kw is None:
        ax_kw = {}
    if t is None:
        for tk in tmdf.index.names:
            if tk.endswith('_time'):
                break
        else:
            raise KeyError('time not found')
        t = tmdf.index.get_level_values(tk)
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
    plot_iqr(tmdf, xkey, x=t - dt, color='k', label=getsource(xkey))
    plot_iqr(tmdf, ykey, x=t, color='r', label=getsource(ykey))
    ax.set(**ax_kw)
    ax.legend(ncol=2, loc='upper left')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return ax


def plot_iqrs(
    df, ykey, xkey, x=None, dx=None,
    subplot_kw=None, ax_kw=None, ax=None, ycolor='r', xcolor='k'
):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    ykey : str
        Name of element from DataFrame
    xkey : str
        Name of element from DataFrame
    x : list, str, or None
        x-locations for the boxes or a named column (default index)
    dx : int or None
        defaults to median(x) / 4
    subplot_kw : dict
        Keywords to subplot
    ax_kw : dict
        Keywords to Axes
    ycolor : str
        Color for y elements
    xcolor : str
        Color for x elements

    Returns
    -------
    ax : Axes
        matplotlib.axes.Axes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .util import getsource
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
    ex, lx = plot_iqr(df, xkey, x=x - dx, color=xcolor, label=getsource(xkey))
    ey, ly = plot_iqr(df, ykey, x=x, color=ycolor, label=getsource(ykey))
    ax.set(**ax_kw)
    ax.legend(ncol=2, loc='upper left')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    return ax


def plot_iqr(df, ykey, x=None, err_kw=None, minmax=False, ax=None, **plot_kw):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    ykey : str
        Name of element from DataFrame
    x : list, str, or None
        x-locations for the boxes or a named column (default index)
    err_kw : dict
        Options for errorbar command
    minmax : bool
        Show min/max not just interquartile
    ax : Axes
        matplotlib.axes.Axes object
    plot_kw : dict
        Options for plot commands

    Returns
    -------
    lines : list
        Elements returned from errorbar and plot
    """
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
    """
    Plot a series of boxplots

    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    ykey : str
        Name of element from DataFrame
    x : list, str, or None
        x-locations for the boxes or a named column (default index)
    ax : Axes
        matplotlib.axes.Axes object
    color : str
        matplotlib named color or hex
    mediancolor : str
        matplotlib named color or hex
    label : str
        Name for boxes

    Returns
    -------
    boxdict : dict
        Elements returned from boxplot
    """
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


def plot_boxs(df, ykey, xkey, x=None, ax=None, iqr=False, source=None, **kw):
    """
    Plot a series of boxplots for ykey and xkey centered around x

    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame with ykey and, if provided, named x column
    ykey : str
        Name of element from DataFrame
    xkey : str
        Name of element from DataFrame
    x : list, str, or None
        x-locations for the boxes or a named column (default index)
    ax : Axes
        matplotlib.axes.Axes object
    iqr : bool
        Show only the inter-quartile range (no whis)
    source : str
        Defaults to source of xkey
    kw : dict
        plot_box options

    Returns
    -------
    boxx, boxy : dict, dict
        Elements returned from boxplot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from .util import getsource
    source = source or getsource(xkey)
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = df.index.values
    elif isinstance(x, str):
        x = df[x]
    dx = kw.pop('dx', np.median(np.diff(x)) / 4)
    widths = kw.get('widths', dx * 2 * 0.9)
    cmnopts = dict(iqr=iqr, widths=widths, mediancolor='k', ax=ax)
    xsrc = getsource(xkey)
    ysrc = getsource(ykey)
    bx = plot_box(df, xkey, x=x - dx, label=xsrc, color='grey', **cmnopts)
    by = plot_box(df, ykey, x=x + dx, label=ysrc, color='r', **cmnopts)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    return bx, by


def plot_map(
    df, ckey, xkey=None, ykey=None, title=None, clabel=None,
    sortkey=None, ax=None, **kw
):
    """
    Arguments
    ---------
    df : pandas.DataFrame
        Data frame with ckey, xkey, and ykey
    ckey : str
        Key to color by
    xkey : str
        Defaults to tempo_lon
    ykey : str
        Defaults to tempo_lat
    title : str
        Defaults to ckey
    clabel : str
        Defaults to ckey
    ax : matplotlib.axes.Axes
        Axes to plot on (defaults to result from subplots)
    Returns
    -------
    """
    import pycno
    import matplotlib.pyplot as plt
    from .util import getfirstkey
    if ax is None:
        gskw = dict(left=0.05, right=0.95, top=0.925, bottom=0.1)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300, gridspec_kw=gskw)
    else:
        fig = ax.figure
    if sortkey is None:
        sortkey = ckey
    xkey = xkey or getfirstkey(df, '_lon')
    ykey = ykey or getfirstkey(df, '_lat')
    ax.set(facecolor='gainsboro')
    plotdf = df.copy()
    plotdf = plotdf.sort_values(by=sortkey)
    c = plotdf[ckey]
    x = plotdf[xkey]
    y = plotdf[ykey]
    title = title or ckey
    clabel = clabel or ckey
    ax.set(title=title)
    pycno.cno().drawstates(zorder=1)
    s = ax.scatter(x, y, c=c, zorder=2, **kw)
    ax.set(xlim=(-125, -65), ylim=(17, 52))
    tag = getdatetag(df)
    ax.figure.text(0.8, 0, tag, ha='right', )
    fig.colorbar(s, label=clabel)
    ax.figure.text(0, 0, 'x', ha='right', va='top')
    ax.figure.text(1, 1, 'x')
    ax.figure.text(0, 1, 'x', ha='right')
    return ax
