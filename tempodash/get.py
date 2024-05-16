__all__ = ['get']


def get(dates=None):
    """
    Gets all data in cfg.keys for all cfg.dates
    """
    import time
    import os
    from . import cfg
    if dates is None:
        dates = cfg.dates
    for bdate in dates:
        edate = bdate + cfg.data_dt
        opts = dict(bdate=bdate, edate=edate, compress=1, verbose=-10)
        for key in cfg.keys:
            corners = cfg.keycorners[key]
            print(bdate, key, end='... ', flush=True)
            source = key.split('.')[0]
            if source == 'pandora':
                print('skip')
                continue
            cfg.api.workdir = f'data/{source}/{bdate:%Y-%m}/'
            os.makedirs(cfg.api.workdir, exist_ok=True)
            t0 = time.time()
            cfg.api.get_file('xdr', key=key, corners=corners, **opts)
            t1 = time.time()
            print(f'{t1 - t0:.1f} seconds')

    bdate, edate = cfg.pandora_date_range

    qa = cfg.api.pandora_kw['minimum_quality']
    opts = dict(bdate=bdate, edate=edate, compress=1, verbose=-10)
    for key in cfg.keys:
        if key.startswith('pandora'):
            for lockey in cfg.pandora_keys:
                lcfg = cfg.configs[lockey]
                print(bdate, key, lockey, end='... ', flush=True)
                cfg.api.workdir = f'data/pandora/{lockey}/{qa}'
                cfg.api.bbox = lcfg['bbox']
                t0 = time.time()
                cfg.api.get_file('xdr', key=key, **opts)
                t1 = time.time()
                print(f'{t1 - t0:.1f} seconds')


if __name__ == '__main__':
    get()
