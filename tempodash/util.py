from scipy import odr
# Define a linear function (y = m*x + b)


def target_function(p, x):
    m, c = p
    return m * x + c


def odrfit(x, y, beta0=None):
    if beta0 is None:
        import scipy.stats.mstats as sms
        lr = sms.linregress(x, y)
        beta0 = [lr.slope, lr.intercept]
        print('beta0 estimate:', beta0)
    #  model fitting.
    odr_model = odr.Model(target_function)

    # Create a RealData object standard deviations to weight samples
    rdata = odr.RealData(
        x, y,
        sx=x.std(), sy=y.std()
    )
    # beta0 = [vdf['vcd'].mean() / vdf['no2'].mean(), vdf['vcd'].min()]
    dr = odr.ODR(rdata, odr_model, beta0=beta0).run()
    return dr


def iterbox(sdf, keys=None):
    """
    Arguments
    ---------
    sdf : pandas.DataFrame
        Must have tempo_lon and tempo_lat
    """
    from .cfg import configs
    from .plots import agg, pid2key

    if keys == 'tempo_lst_hour':
        for hkey, hdf in sdf.groupby(keys):
            yield hkey, hdf
    elif 'pandora_id' in sdf.columns:
        if keys is None:
            keys = list(configs)
        for pid, pdf in sdf.groupby('pandora_id'):
            lockey = pid2key[pid]
            if lockey in keys:
                yield lockey, pdf
    else:
        if keys is None:
            keys = list(configs)
        for lockey in keys:
            lcfg = configs[lockey]
            wlon, slat, elon, nlat = lcfg['bbox']
            pdf = sdf.query(
                f'tempo_lon > {wlon} and tempo_lon < {elon}'
                f'and tempo_lat > {slat} and tempo_lat < {nlat}'
            )
            yield lockey, pdf


def regressions(x, y):
    from scipy.stats.mstats import linregress
    import numpy as np
    lrpks = [
        'slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr'
    ]
    drpks = [
        'slope', 'intercept', 'slope_std', 'intercept_std'
    ]
    if x.size == 0:
        regrec = {'lr_' + k: np.nan for k in lrpks}
        regrec.update({'dr_' + k: np.nan for k in drpks})
        return regrec

    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    valid = ~(x.mask | y.mask)
    lr = linregress(x[valid], y[valid])
    dr = odrfit(x[valid], y[valid], beta0=[lr.slope, lr.intercept])
    regrec = {
        'lr_' + pk: getattr(lr, pk)
        for pk in lrpks
    }
    regrec['dr_slope'] = dr.beta[0]
    regrec['dr_intercept'] = dr.beta[1]
    regrec['dr_slope_std'] = dr.sd_beta[0]
    regrec['dr_intercept_std'] = dr.sd_beta[1]

    return regrec


def regressiondf(df, outpath=None, verbose=0, refresh=False, keys=None):
    import gc
    import os
    import pandas as pd
    from .plots import getxy, agg
    from .cfg import configs
    if outpath is not None and os.path.exists(outpath) and not refresh:
        df = pd.read_csv(outpath, index_col=0)
        return df
    regrecs = []
    for lockey, pdf in iterbox(df, keys=keys):
        if verbose > 0:
            print(lockey)
        if pdf.shape[0] == 0:
            continue
        if 'pandora_id' in pdf.columns:
            # pandora within 15min should be compared by median
            pdf = agg(pdf, ['pandora_id', 'tempo_time'])
            # err=True returns q2 for x and y
            x, y, xs, ys = getxy(pdf, err=True)
        else:
            x, y = getxy(pdf)
        regrec = regressions(x, y)
        regrec['lockey'] = lockey
        regrecs.append(regrec)
        del pdf
        gc.collect()

    regdf = pd.DataFrame.from_records(regrecs).set_index('lockey')
    if outpath is not None:
        regdf.to_csv(outpath)

    return regdf
