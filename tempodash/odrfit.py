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
    sx = x.std()
    sy = y.std()
    rdata = odr.RealData(x, y, sx=sx, sy=sy)
    dr = odr.ODR(rdata, odr_model, beta0=beta0).run()
    # if dr.info > 9999:
    #     raise ValueError('Fatal results')
    return dr
