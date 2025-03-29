#!/usr/bin/env python
import pandas as pd
import os
import matplotlib.pyplot as plt
from .config import locconfigs, id2cfg, id2key, regions
from . import plot as tplot
import numpy as np


def makeplots(xkey, ykey, lockey):
    if lockey == 'summary':
        loclabel = 'All Sites'
    else:
        loclabel = locconfigs[lockey]['label']

    os.makedirs(f'figs/{lockey}', exist_ok=True)
    qname = {
        'tempo_no2_sum': 'totNO$_2$ [molec/cm$^2$]',
        'tempo_no2_trop': 'tropNO$_2$ [molec/cm$^2$]',
        'tempo_hcho_total': 'totHCHO [molec/cm$^2$]',
    }[ykey]
    if xkey.startswith('pandora_'):
        srclabel = 'Pandora'
    elif xkey.startswith('tropomi_offl_'):
        srclabel = 'TropOMI'
    else:
        srclabel = xkey.split('_')[0]
    q1keys = [xkey + '_q1', ykey + '_q1']
    q3keys = [xkey + '_q3', ykey + '_q3']

    def getrange(df, buff=0.05):
        vmin = df[q1keys].min().min() * (1 - buff)
        vmax = df[q3keys].max().max() * (1 + buff)
        return vmin, vmax

    zmdf = pd.read_csv(
        f'csv/{lockey}/{ykey}_vs_{xkey}_bysza.csv', index_col='tempo_sza'
    ).query('count > 1')
    hmdf = pd.read_csv(
        f'csv/{lockey}/{ykey}_vs_{xkey}_bylsth.csv', index_col=0
    )
    mmdf = pd.read_csv(
        f'csv/{lockey}/{ykey}_vs_{xkey}_bymonth.csv', index_col='tempo_time',
        parse_dates=['tempo_time']
    )
    wmdf = pd.read_csv(
        f'csv/{lockey}/{ykey}_vs_{xkey}_byweek.csv', index_col='tempo_time',
        parse_dates=['tempo_time']
    )
    cmdf = pd.read_csv(f'csv/{lockey}/{ykey}_vs_{xkey}_bycld.csv')
    tmdf = pd.read_csv(f'csv/{lockey}/{ykey}_vs_{xkey}_bytempo.csv')
    cmdf['tempo_cloud_eff'] *= 100
    cmdf.set_index('tempo_cloud_eff', inplace=True)
    # if lockey == 'summary':
    #     tmdf = tmdf.query('count >= 100').copy()
    # tmdf[ykey] /= 1e15
    tmdf = tmdf.set_index(ykey)
    wdmdf = wmdf.query('tempo_time.dt.dayofweek < 5')
    wemdf = wmdf.query('tempo_time.dt.dayofweek >= 5')

    # %%
    # By TEMPO stat plots are not combined.
    # -------------------------------------
    datetag = tplot.getdatetag(tmdf)
    gskw = dict(left=0.075, right=0.99, bottom=0.2)
    tfigc, ax1 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    tfigri, ax2 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    tfigpe, ax3 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    tfige, ax4 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    axx = np.array([[ax1, ax2], [ax3, ax4]])
    x = np.arange(tmdf.shape[0])
    xv = tmdf.index.values / 1e15
    xtl = [f'{lb:.2g}-{ub:.2g}' for lb, ub in zip(xv[:-1], xv[1:])]
    xtl.append(f'>{xv[-1]:.0f}')
    tplot.plot_summary_bars(tmdf, xkey, axx=axx, x=x)
    for ax in axx.ravel():
        ax.text(0, 1.02, f'{loclabel} {datetag}', transform=ax.transAxes)
        ax.set(xlabel=qname + 'x10$^{15}$')
        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(xtl, rotation=90)

    tfigc.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bytempo_stats_pdf.png')
    tfigri.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bytempo_stats_rioa.png')
    tfigpe.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bytempo_stats_pcterr.png')
    tfige.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bytempo_stats_err.png')
    plt.close(tfigc)
    plt.close(tfigri)
    plt.close(tfigpe)
    plt.close(tfige)

    # %%
    # By SZA stats plots are created as a 4-part panel
    # ------------------------------------------------
    datetag = tplot.getdatetag(zmdf)
    zfig = tplot.plot_summary_bars(zmdf, xkey)
    ax = zfig.axes[0]
    ax.text(0, 1.02, f'{loclabel} {qname} {datetag}', transform=ax.transAxes)
    plt.setp(zfig.axes[2:], xlabel='Solar Zenith Angle')
    zfig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bysza_stats.png')
    plt.close(zfig)

    # %%
    # By Local Hour of Day stats plots are created as a 4-part panel
    # --------------------------------------------------------------
    datetag = tplot.getdatetag(hmdf)
    hfig = tplot.plot_summary_bars(hmdf, xkey)
    plt.setp(hfig.axes[2:], xlabel='Local Hour')
    ax = hfig.axes[0]
    ax.text(0, 1.02, f'{loclabel} {qname} {datetag}', transform=ax.transAxes)
    hfig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bylsth_stats.png')
    plt.close(hfig)

    # %%
    # By effective cloud fraction stats plots are created as a 4-part panel
    # ---------------------------------------------------------------------
    datetag = tplot.getdatetag(cmdf)
    cfig = tplot.plot_summary_bars(cmdf, xkey)
    plt.setp(cfig.axes[2:], xlabel='Effective Cloud')
    ax = cfig.axes[0]
    ax.text(0, 1.02, f'{loclabel} {qname} {datetag}', transform=ax.transAxes)
    cfig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bycld_stats.png')
    plt.close(cfig)

    # %%
    # By month stats plots are created as a 4-part panel
    # --------------------------------------------------
    datetag = tplot.getdatetag(mmdf)
    mfig = tplot.plot_summary_bars(mmdf, xkey)
    plt.setp(mfig.axes[1], xticklabels=mmdf.index.strftime('%Y-%m'))
    ax = mfig.axes[0]
    ax.text(0, 1.02, f'{loclabel} {qname} {datetag}', transform=ax.transAxes)
    mfig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bymonth_stats.png')
    plt.close(mfig)

    # %%
    # By weekday as a series of IQR boxes
    # -----------------------------------
    datetag = tplot.getdatetag(wmdf)
    title = '25%, 50%, 75% for weekdays and weekend'
    ax_kw = dict(
        title=f'{loclabel} {title}: {datetag}',
        ylabel=qname, ylim=getrange(wmdf)
    )
    gskw = dict(left=0.025, right=0.99, bottom=0.25)
    spkw = dict(figsize=(24, 3), gridspec_kw=gskw)
    ax = tplot.plot_weekdayweekend(
        wmdf, ykey=ykey, xkey=xkey, ax_kw=ax_kw, subplots_kw=spkw,
        source=srclabel
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byweek_ts.png')
    ax.figure.text(0, 0, 'x', ha='right', va='top')
    ax.figure.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(ax.figure)

    # %%
    # By Local Hour of Day as a series of IQR boxes
    # ---------------------------------------------
    datetag = tplot.getdatetag(hmdf)
    title = '25%, 50%, 75% for LST Hour'
    ax_kw = dict(
        title=f'{loclabel} {title}: {datetag}',
        ylabel=qname, ylim=getrange(hmdf), xlabel='Hour of day [LST]'
    )
    gskw = dict(left=0.05, right=0.99)
    fig, ax = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    b1, b2 = tplot.plot_boxs(
        hmdf, ykey=ykey, xkey=xkey, ax=ax, dx=0.2, widths=0.35, source=srclabel
    )
    ax.set(**ax_kw)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    fig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bylsth_ts.png')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(fig)

    # %%
    # By Solar Zenith Angle as a series of IQR boxes
    # ----------------------------------------------
    datetag = tplot.getdatetag(zmdf)
    ax_kw = dict(
        title=f'{lockey} 25%, 50%, 75% for SZA: ' + datetag,
        ylabel=qname, ylim=getrange(zmdf), xlabel='Solar Zenith Angle'
    )
    gskw = dict(left=0.06, right=0.99)
    fig, ax = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    b1, b2 = tplot.plot_boxs(
        zmdf, ykey=ykey, xkey=xkey, ax=ax, dx=2, widths=3.5, source=srclabel
    )
    ax.set(**ax_kw)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    fig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bysza_ts.png')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(fig)

    # %%
    # By Effective Cloud as a series of IQR boxes
    # -------------------------------------------
    datetag = tplot.getdatetag(zmdf)
    ax_kw = dict(
        title=f'{lockey} 25%, 50%, 75% for Eff Cloud: {datetag}',
        ylabel=qname, ylim=getrange(cmdf), xlabel='Cloud'
    )
    gskw = dict(left=0.06, right=0.99)
    fig, ax = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    b1, b2 = tplot.plot_boxs(
        cmdf, ykey=ykey, xkey=xkey, ax=ax, dx=2, widths=3.5, source=srclabel
    )
    ax.set(**ax_kw)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    fig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bycld_ts.png')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(fig)

    # %%
    # By Weekday as a series of IQR boxes
    # -----------------------------------
    datetag = tplot.getdatetag(wmdf)
    ax_kw = dict(
        title=f'{loclabel} 25%, 50%, 75% for weekdays: {datetag}',
        ylabel=qname, ylim=getrange(wdmdf)
    )
    gskw = dict(left=0.04, right=0.99, bottom=0.2)
    fig, ax = plt.subplots(figsize=(24, 4), gridspec_kw=gskw)
    x = wdmdf.index.astype('i8') / 3600e9 / 24
    tplot.plot_boxs(
        wdmdf, ykey=ykey, xkey=xkey, ax=ax, x=x, dx=1, source=srclabel
    )
    ax.set(**ax_kw)
    xt = pd.date_range('2023-08-01', '2025-02-28', freq='MS')
    ax.set_xticks(xt.astype('i8') / 3600e9 / 24)
    ax.set_xticklabels(xt.strftime('%Y-%m'), rotation=90)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byweek_tsweekday.png')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(fig)

    # %%
    # By Weekend as a series of IQR boxes
    # -----------------------------------
    gskw = dict(left=0.04, right=0.99, bottom=0.2)
    fig, ax = plt.subplots(figsize=(24, 4), gridspec_kw=gskw)
    datetag = tplot.getdatetag(wmdf)
    ax_kw = dict(
        title=f'{loclabel} 25%, 50%, 75% for weekends: {datetag}',
        ylabel=qname, ylim=getrange(wemdf)
    )
    x = wemdf.index.astype('i8') / 3600e9 / 24
    tplot.plot_boxs(
        wemdf, ykey=ykey, xkey=xkey, ax=ax, x=x, dx=1, source=srclabel
    )
    ax.set(**ax_kw)
    ax.set_xticks(xt.astype('i8') / 3600e9 / 24)
    ax.set_xticklabels(xt.strftime('%Y-%m'), rotation=90)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    fig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byweek_tsweekend.png')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(fig)

    # %%
    # By month as a series of IQR boxes
    # ---------------------------------
    datetag = tplot.getdatetag(mmdf)
    ax_kw = dict(
        title=f'{loclabel} 25%, 50%, 75% by month: {datetag}',
        ylabel=qname, ylim=getrange(mmdf)
    )
    gskw = dict(left=0.075, right=0.99, bottom=0.2)
    spkw = dict(figsize=(12, 4), gridspec_kw=gskw)
    fig, ax = plt.subplots(**spkw)
    x = mmdf.index.values.astype('i8') / 3600e9 / 24
    b1, b2 = tplot.plot_boxs(
        mmdf, ykey=ykey, xkey=xkey, x=x, ax=ax, dx=5, widths=7,
        source=srclabel
    )
    ax.set(**ax_kw, xticks=x)
    _ = ax.set_xticklabels(mmdf.index.strftime('%Y-%m'), rotation=90)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    fig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bymonth_ts.png')
    fig.text(0, 0, 'x', ha='right', va='top')
    fig.text(1, 1, 'x', ha='left', va='bottom')
    plt.close(fig)

    # %%
    # By weekday as a scatter plot
    # ----------------------------
    datetag = tplot.getdatetag(wmdf)
    ax_kw = dict(ylabel=f'TEMPO {qname}', xlabel=f'{srclabel} {qname}')
    ax_kw['title'] = f'By Week: {datetag}'
    ax = tplot.plot_scatter(
        wmdf, ykey=ykey, xkey=xkey, c='rvalue', source=srclabel,
        ax_kw=ax_kw, cbar_kw=dict(label='Correlation [1]')
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byweek_scat.png')

    datetag = tplot.getdatetag(wdmdf)
    ax_kw = dict(ylabel=f'TEMPO {qname}', xlabel=f'{srclabel} {qname}')
    ax_kw['title'] = f'By Week (M-F only): {datetag}'
    ax = tplot.plot_scatter(
        wdmdf, ykey=ykey, xkey=xkey, c='rvalue', source=srclabel,
        ax_kw=ax_kw, cbar_kw=dict(label='Correlation [1]')
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byweek_scatweekday.png')

    datetag = tplot.getdatetag(wemdf)
    ax_kw = dict(ylabel=f'TEMPO {qname}', xlabel=f'{srclabel} {qname}')
    ax_kw['title'] = f'By Week (S-S only): {datetag}'
    ax = tplot.plot_scatter(
        wemdf, ykey=ykey, xkey=xkey, c='rvalue', source=srclabel,
        ax_kw=ax_kw, cbar_kw=dict(label='Correlation [1]')
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byweek_scatweekend.png')

    # %%
    # By month as a scatter plot
    # --------------------------
    datetag = tplot.getdatetag(mmdf)
    ax_kw['title'] = f'By Month: {datetag}'
    ax = tplot.plot_scatter(
        mmdf, ykey=ykey, xkey=xkey, c='rvalue', source=srclabel,
        ax_kw=ax_kw, cbar_kw=dict(label='Correlation [1]')
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bymonth_scat.png')
    plt.close(ax.figure)

    # %%
    # By sza as a scatter plot
    # --------------------------
    datetag = tplot.getdatetag(zmdf)
    ax_kw['title'] = f'By Solar Zenith Angle: {datetag}'
    ax = tplot.plot_scatter(
        zmdf, ykey=ykey, xkey=xkey, c='rvalue', source=srclabel,
        ax_kw=ax_kw, cbar_kw=dict(label='Correlation [1]')
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_bysza_scat.png')
    plt.close(ax.figure)
    for i in plt.get_fignums():
        plt.close(i)


def makebylocplots(xkey, ykey, lockey):
    import pandas as pd
    from functools import reduce
    # %%
    # By Location as a series of IQR boxes
    # ------------------------------------
    lmdf = pd.read_csv(f'csv/{lockey}/{ykey}_vs_{xkey}_byloc.csv')
    if 'pandora_id' in lmdf.columns:
        lmdf['label'] = [
            id2cfg[i]['label'] for i in lmdf.pandora_id
        ]
        lmdf['lockey'] = [id2key[i] for i in lmdf.pandora_id]
        lmdf.set_index('lockey', inplace=True, drop=False)
    else:
        lmdf['label'] = [
            locconfigs[k]['label']
            for k in lmdf['lockey']
        ]
        lmdf.set_index('lockey', inplace=True, drop=False)
    sorted_idx = reduce(
        list.__add__, [reg['lockeys'] for reg in regions]
    )
    sorted_idx = [i for i in sorted_idx if i in lmdf.index.values]
    assert lmdf.index.isin(sorted_idx).all()
    slmdf = lmdf.loc[sorted_idx].copy()
    if 'pandora' in xkey:
        plotbyloc(xkey, ykey, lockey, slmdf)
    else:
        plotbyloc(
            xkey, ykey, lockey, slmdf.filter(regex='NAA_', axis=0), sfx='naa'
        )
        plotbyloc(
            xkey, ykey, lockey, slmdf.filter(regex='Pandora', axis=0),
            sfx='pandora'
        )


def plotbyloc(xkey, ykey, lockey, slmdf, sfx=''):
    qname = {
        'tempo_no2_sum': 'totNO$_2$ [molec/cm$^2$]',
        'tempo_no2_trop': 'tropNO$_2$ [molec/cm$^2$]',
        'tempo_hcho_total': 'totHCHO [molec/cm$^2$]',
    }[ykey]
    if xkey.startswith('pandora_'):
        srclabel = 'Pandora'
    elif xkey.startswith('tropomi_offl_'):
        srclabel = 'TropOMI'
    else:
        srclabel = xkey.split('_')[0]

    q1keys = [xkey + '_q1', ykey + '_q1']
    q3keys = [xkey + '_q3', ykey + '_q3']

    def getrange(df, buff=0.05):
        vmin = df[q1keys].min().min() * (1 - buff)
        vmax = df[q3keys].max().max() * (1 + buff)
        return vmin, vmax

    # %%
    # Create Scatter Plot by Location
    # -------------------------------
    datetag = tplot.getdatetag(slmdf)
    ax_kw = dict(ylabel=f'TEMPO {qname}', xlabel=f'{srclabel} {qname}')
    ax = tplot.plot_scatter(
        slmdf, ykey=ykey, xkey=xkey, c='rvalue', source=srclabel,
        ax_kw=ax_kw, cbar_kw=dict(label='Correlation [1]')
    )
    ax.figure.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byloc_scat{sfx}.png')

    # %%
    # Create Stats Plot by Location
    # -----------------------------
    regcount = []
    for reg in tempodash.config.regions:
        regcnt = sum([1 for i in slmdf.index.values if i in reg['lockeys']])
        if regcnt > 0:
            regcount.append((reg['label'], regcnt))
    bdf = slmdf
    gskw = dict(left=0.07, right=0.99, bottom=0.4)
    lfigc, ax1 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    lfigri, ax2 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    lfigpe, ax3 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    lfige, ax4 = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    axx = np.array([[ax1, ax2], [ax3, ax4]])
    tplot.plot_summary_bars(bdf, xkey, axx=axx)
    for ax in axx.ravel():
        ax.set_xticklabels(bdf['label'], rotation=90)
        yt = ax.get_ylim()[1]
        rleft = 0
        for rlbl, rcnt in regcount:
            ax.axvline(rleft + rcnt, color='k', linestyle='-', linewidth=2)
            ax.text(rleft + rcnt / 2, yt, rlbl, ha='center', va='top')
            rleft += rcnt
        ax.set(xlim=(0, rleft))
    lfigc.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byloc_stats_pdf{sfx}.png')
    lfigri.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byloc_stats_rioa{sfx}.png')
    lfigpe.savefig(
        f'figs/{lockey}/{ykey}_vs_{xkey}_byloc_stats_pcterr{sfx}.png'
    )
    lfige.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byloc_stats_err{sfx}.png')
    plt.close(lfigc)
    plt.close(lfigri)
    plt.close(lfigpe)
    plt.close(lfige)

    # %%
    # Create Boxes Plot by Location
    # -----------------------------
    pdf = slmdf
    x = np.arange(pdf.shape[0]) + 0.5
    gskw = dict(left=0.06, bottom=.4, right=0.99)
    fig, ax = plt.subplots(figsize=(12, 4), gridspec_kw=gskw)
    bx, by = tplot.plot_boxs(
        pdf, ykey=ykey, xkey=xkey, x=x, dx=.15, widths=.25, iqr=False,
        source=srclabel
    )
    # ymax = pdf[[xkey + '_q2', ykey + '_q2']].max().max() * 1.1
    ylim = getrange(slmdf)
    yt = ylim[1] * 0.98
    rleft = 0
    for rlbl, rcnt in regcount:
        ax.axvline(rleft + rcnt, color='k', linestyle='-', linewidth=2)
        xt = rleft + rcnt / 2
        ax.text(xt, yt, rlbl, ha='center', va='top', backgroundcolor='w')
        rleft += rcnt
    ax.set_xticklabels(pdf['label'], rotation=90)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 1))
    ax.set(ylabel=qname, xlim=(0, rleft), ylim=ylim)
    fig.suptitle('25%, 50%, 75% for Locations: ' + datetag)
    fig.savefig(f'figs/{lockey}/{ykey}_vs_{xkey}_byloc_ts{sfx}.png')
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    import tempodash.util
    from joblib import Parallel, delayed

    xychoices = [
        'pandora_no2_total,tempo_no2_sum',
        'tropomi_offl_no2_trop,tempo_no2_trop',
        'pandora_hcho_total,tempo_hcho_total',
        'tropomi_offl_no2_sum,tempo_no2_sum',
        'tropomi_offl_hcho_total,tempo_hcho_total',
    ]
    prsr = argparse.ArgumentParser()
    prsr.add_argument('--jobs', default=1, type=int, help='n-jobs')
    prsr.add_argument('--dpi', default=100.0, type=float)
    prsr.add_argument('--debug', action='store_true', default=False)
    prsr.add_argument(
        '--xykeys', action='append', default=[],
        choices=xychoices
    )
    args = prsr.parse_args()
    if len(args.xykeys) == 0:
        args.xykeys = xychoices
    args.xykeys = [xykey.split(',') for xykey in args.xykeys]
    plt.rcParams['figure.dpi'] = args.dpi
    for xkey, ykey in args.xykeys:
        print(xkey, ykey, 'summary', flush=True)
        # Quick check if this needs to be done
        # if any csv is older than any store; remake
        spc = ykey.split('_')[1]
        src = xkey.split('_' + spc)[0]
        remake = tempodash.util.depends(
            f'figs/*/{ykey}_vs_{xkey}*.png', f'csv/*/{ykey}_vs_{xkey}*.csv', 1
        )
        if not remake:
            continue

        makebylocplots(xkey, ykey, 'summary')
        makeplots(xkey, ykey, 'summary')
        actions = []
        for lockey, cfg in locconfigs.items():
            print(xkey, ykey, lockey, '...', flush=True)
            if 'pandora_no2_total' == xkey and not cfg.get('pandora', False):
                continue
            elif (
                'pandora_hcho_total' == xkey
                and not cfg.get('pandora_hcho', False)
            ):
                continue
            actions.append(delayed(makeplots)(xkey, ykey, lockey))
            if args.debug:
                print('WARN:: running only first location')
                break

        with Parallel(n_jobs=args.jobs, verbose=10) as par:
            outpaths = par(actions)
        print(len(outpaths), outpaths[0], '...', outpaths[-1])
