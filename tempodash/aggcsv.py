"""
aggcsv.py rewritten to read data from disk only once.
"""
import os
import pandas as pd
from . import loadintx
from .agg import agg, szagrouper, lsthgrouper, monthlygrouper, byloc
from .agg import weeklygrouper, cldgrouper, tempogrouper, allgrouper
from .config import default_where, locconfigs, dataroot
from joblib import Parallel, delayed


def csvs(cdf, lockey, overwrite=False, **grpopts):
    xkey = grpopts['xkey']
    ykey = grpopts['ykey']
    # csv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_paired.csv'
    acsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_byall.csv'
    lcsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_byloc.csv'
    zcsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_bysza.csv'
    hcsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_bylsth.csv'
    mcsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_bymonth.csv'
    wcsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_byweek.csv'
    ccsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_bycld.csv'
    tcsv = f'{dataroot}/csv/{lockey}/{ykey}_vs_{xkey}_bytempo.csv'
    outpaths = [zcsv, hcsv, mcsv, wcsv, ccsv, tcsv, acsv]
    if not overwrite:
        if all([os.path.exists(op) for op in outpaths]):
            print('keep cached')
            return

    os.makedirs(os.path.dirname(tcsv), exist_ok=True)
    if 'tempo_cloud_eff' in cdf.columns:
        print('- by CLOUD...', flush=True)
        cmdf = agg(cdf, cldgrouper, **grpopts)
        cmdf.to_csv(ccsv)
    
        print('query good cloud...', flush=True)
        mdf = cdf.query(default_where)
    else:
        mdf = cdf
    # print('- by PIXEL...', flush=True)
    # save out the aggregated 1-per tempo pixel
    # mdf.to_csv(csv, index=False)
    print('- by all (overall)')
    amdf = agg(mdf, allgrouper, **grpopts)
    amdf.to_csv(acsv)
    if lockey == 'summary':
        if 'pandora_id' in mdf.columns:
            print('- by PANDORA ID...', flush=True)
            lmdf = agg(mdf, 'pandora_id', **grpopts)
        else:
            print('- by Location...', flush=True)
            lmdf = byloc(mdf, None, **grpopts)
        lmdf.to_csv(lcsv)
    # aggregate by site, by sza, by lsth, by month, by week-weekday
    print('- by TEMPO...', flush=True)
    tdf = agg(mdf, tempogrouper, **grpopts)
    tdf.to_csv(tcsv)
    if 'tempo_sza' in mdf.columns:
        print('- by SZA...', flush=True)
        zmdf = agg(mdf, szagrouper, **grpopts)
        zmdf.to_csv(zcsv)
    if 'tempo_time' in mdf.columns:
        print('- by HOUR...', flush=True)
        hmdf = agg(mdf, lsthgrouper, **grpopts)
        hmdf.to_csv(hcsv)
    print('- by MONTH...', flush=True)
    mmdf = agg(mdf, monthlygrouper, **grpopts)
    mmdf.to_csv(mcsv)
    print('- by Weekday...', flush=True)
    wmdf = agg(mdf, weeklygrouper, **grpopts)
    wmdf.to_csv(wcsv)
    return outpaths


if __name__ == '__main__':
    import argparse
    import tempodash.util
    xychoices = [
        'pandora_no2_total,tropomi_offl_no2_sum',
        'pandora_hcho_total,tropomi_offl_hcho_total',
        'pandora_no2_total,tempo_no2_sum',
        'tropomi_offl_no2_trop,tempo_no2_trop',
        'pandora_hcho_total,tempo_hcho_total',
        'tropomi_offl_no2_sum,tempo_no2_sum',
        'tropomi_offl_hcho_total,tempo_hcho_total',
    ]
    prsr = argparse.ArgumentParser()
    prsr.add_argument('-j', '--jobs', default=1, help='Jobs', type=int)
    prsr.add_argument(
        '--xykeys', action='append', default=[],
        choices=xychoices
    )
    args = prsr.parse_args()
    if len(args.xykeys) == 0:
        args.xykeys = xychoices
    args.xykeys = [xykey.split(',') for xykey in args.xykeys]
    start_date = '2023-08-01'
    end_date = (pd.to_datetime('now') - pd.to_timedelta('14d'))
    end_date = end_date.strftime('%Y-%m-%d')
    overwrite = True
    # end_date = '2025-02-15'
    for xkey, ykey in args.xykeys:
        spc = ykey.split('_')[-2]
        xsrc = xkey.split('_' + spc)[0]
        ysrc = ykey.split('_' + spc)[0]
        src = {
            'pandora_no2_total': 'pandora', 'pandora_hcho_total': 'pandora',
            'tropomi_offl_no2_sum': 'tropomi_offl',
            'tropomi_offl_no2_trop': 'tropomi_offl',
            'tropomi_offl_hcho_total': 'tropomi_offl',
        }[xkey]
        print('summary', src, xkey, ykey)
        if 'tropomi_offl' in ykey:
            default_cldwhere = None
            grouper = ['tropomi_offl_time', 'tropomi_offl_lon', 'tropomi_offl_lat', 'pandora_id']
        else:
            default_cldwhere = 'tempo_sza <= 70'
            if xkey.startswith('pandora_'):
                grouper = ['tempo_time', 'tempo_lon', 'tempo_lat', 'pandora_id']
            else:
                grouper = ['tempo_time', 'tempo_lon', 'tempo_lat']

        print(ysrc)
        # Quick check if this needs to be done
        # if any csv is older than any store; remake
        remake = tempodash.util.depends(
            f'{dataroot}/csv/*/{ykey}_vs_{xkey}*.csv',
            f'{dataroot}/intx/stores/{ysrc}_{xsrc}_{spc}_????-??.h5', 1
        )
        if not remake and not overwrite:
            continue

        getcols = grouper + [f'{ysrc}_sza', f'{ysrc}_cloud_eff', ykey, xkey]
        sumcols = [xkey, ykey, 'bias', 'err', 'sqerr']
        opts = dict(
            start_date=start_date, end_date=end_date, columns=getcols, ysource=ysrc
        )
        grpopts = dict(
            columns=sumcols, xkey=xkey, ykey=ykey, orthogonal=True
        )
        print('load cloud...', flush=True)
        # the default cldwhere is a super set of default_where
        # so, instead of loading twice, teh cldwhere df will
        # be used as the data store.
        cdf = loadintx(xsrc, spc, where=default_cldwhere, **opts)
        cmdf = cdf.groupby(grouper, as_index=False).mean()
        cmdf['bias'] = cmdf[ykey] - cmdf[xkey]
        cmdf['err'] = cmdf['bias'].abs()
        cmdf['sqerr'] = cmdf['err']**2

        csvs(cmdf, 'summary', overwrite=overwrite, **grpopts)
        actions = []
        for lockey, cfg in locconfigs.items():
            print(lockey, spc, xkey, ykey)
            bbox = cfg['bbox']
            if xkey == 'pandora_no2_total' and not cfg.get('pandora', False):
                continue
            elif (
                xkey == 'pandora_hcho_total'
                and not cfg.get('pandora_hcho', False)
            ):
                continue
            if xsrc == 'pandora':
                pid = float(cfg['pandoraid'])
                lwhere = f'pandora_id == {pid}'
            else:
                lwhere = (
                    '{0}_lon >= {1} and {0}_lon <= {3} '.format(ysrc, *bbox)
                    + 'and {0}_lat >= {2} and {0}_lat <= {4}'.format(ysrc, *bbox)
                )
            lmcdf = cmdf.query(lwhere)
            actions.append(
                delayed(csvs)(lmcdf, lockey, overwrite=overwrite, **grpopts)
            )
        with Parallel(n_jobs=args.jobs, verbose=10) as par:
            outpaths = par(actions)
        print(len(outpaths), outpaths[0], '...', outpaths[-1])
