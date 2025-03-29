"""
aggcsv.py rewritten to read data from disk only once.
"""
import os
import pandas as pd
from . import loadintx
from .agg import agg, szagrouper, lsthgrouper, monthlygrouper, byloc
from .agg import weeklygrouper, cldgrouper, tempogrouper
from .config import default_where, locconfigs
from joblib import Parallel, delayed


def csvs(cdf, lockey, overwrite=False, **grpopts):
    xkey = grpopts['xkey']
    ykey = grpopts['ykey']
    # csv = f'csv/{lockey}/{ykey}_vs_{xkey}_paired.csv'
    lcsv = f'csv/{lockey}/{ykey}_vs_{xkey}_byloc.csv'
    zcsv = f'csv/{lockey}/{ykey}_vs_{xkey}_bysza.csv'
    hcsv = f'csv/{lockey}/{ykey}_vs_{xkey}_bylsth.csv'
    mcsv = f'csv/{lockey}/{ykey}_vs_{xkey}_bymonth.csv'
    wcsv = f'csv/{lockey}/{ykey}_vs_{xkey}_byweek.csv'
    ccsv = f'csv/{lockey}/{ykey}_vs_{xkey}_bycld.csv'
    tcsv = f'csv/{lockey}/{ykey}_vs_{xkey}_bytempo.csv'
    outpaths = [zcsv, hcsv, mcsv, wcsv, ccsv, tcsv]
    if not overwrite:
        if all([os.path.exists(op) for op in outpaths]):
            print('keep cached')
            return

    os.makedirs(os.path.dirname(tcsv), exist_ok=True)
    print('- by CLOUD...', flush=True)
    cmdf = agg(cdf, cldgrouper, **grpopts)
    cmdf.to_csv(ccsv)

    print('query good cloud...', flush=True)
    mdf = cdf.query(default_where)
    print('- by PIXEL...', flush=True)
    # save out the aggregated 1-per tempo pixel
    # mdf.to_csv(csv, index=False)
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
    print('- by SZA...', flush=True)
    zmdf = agg(mdf, szagrouper, **grpopts)
    zmdf.to_csv(zcsv)
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
    default_cldwhere = 'tempo_sza <= 70'
    for xkey, ykey in args.xykeys:
        src = {
            'pandora_no2_total': 'pandora', 'pandora_hcho_total': 'pandora',
            'tropomi_offl_no2_sum': 'tropomi_offl',
            'tropomi_offl_no2_trop': 'tropomi_offl',
            'tropomi_offl_hcho_total': 'tropomi_offl',
        }[xkey]
        print('summary', src, xkey, ykey)
        if xkey.startswith('pandora_'):
            grouper = ['tempo_time', 'tempo_lon', 'tempo_lat', 'pandora_id']
        else:
            grouper = ['tempo_time', 'tempo_lon', 'tempo_lat']

        spc = ykey.split('_')[1]
        src = xkey.split('_' + spc)[0]
        # Quick check if this needs to be done
        # if any csv is older than any store; remake
        remake = tempodash.util.depends(
            f'csv/*/{ykey}_vs_{xkey}*.csv',
            f'intx/stores/tempo_{src}_{spc}_????-??.h5', 1
        )
        if not remake and not overwrite:
            continue

        getcols = grouper + ['tempo_sza', 'tempo_cloud_eff', ykey, xkey]
        sumcols = [xkey, ykey, 'bias', 'err', 'sqerr']
        opts = dict(
            start_date=start_date, end_date=end_date, columns=getcols
        )
        grpopts = dict(
            columns=sumcols, xkey=xkey, ykey=ykey, orthogonal=True
        )
        print('load cloud...', flush=True)
        # the default cldwhere is a super set of default_where
        # so, instead of loading twice, teh cldwhere df will
        # be used as the data store.
        cdf = loadintx(src, spc, where=default_cldwhere, **opts)
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
            if src == 'pandora':
                pid = float(cfg['pandoraid'])
                lwhere = f'pandora_id == {pid}'
            else:
                lwhere = (
                    'tempo_lon >= {0} and tempo_lon <= {2} '.format(*bbox)
                    + 'and tempo_lat >= {1} and tempo_lat <= {3}'.format(*bbox)
                )
            lmcdf = cmdf.query(lwhere)
            actions.append(
                delayed(csvs)(lmcdf, lockey, overwrite=overwrite, **grpopts)
            )
        with Parallel(n_jobs=args.jobs, verbose=10) as par:
            outpaths = par(actions)
        print(len(outpaths), outpaths[0], '...', outpaths[-1])
