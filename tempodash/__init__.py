__all__ = [
    'tempo', 'tropomi', 'airnow', 'pandora',
    'config', 'dates', 'intx', 'agg', 'makeintx', 'loadintx'
]
__version__ = '0.0.0'

from .core import tempo, tropomi, airnow, pandora
from . import config
from . import dates
from . import intx
from . import agg
import time as _time
loadintx = intx.loadintx


def makeintx(spc, start_date, end_date, verbose=0, hourdf=None):
    if hourdf is None:
        hourdf = dates.hourframe(spc, start_date, end_date)
    tomi_store = intx.intxstore('tempo', 'tropomi_offl', spc)
    an_store = intx.intxstore('tempo', 'airnow', spc)
    pan_store = intx.intxstore('tempo', 'pandora', spc)
    minhour = hourdf.reset_index()['time'].min()
    if tomi_store.mostrecent() < minhour:
        print(f'WARN:: Most recent TropOMI intersection is before {start_date}')
    if pan_store.mostrecent() < minhour:
        print(f'WARN:: Most recent Pandora intersection is before {start_date}')
    if spc == 'no2' and an_store.mostrecent() < minhour:
        print(f'WARN:: Most recent AirNow intersection is before {start_date}')
    if verbose > 1:
        print(hourdf.reset_index().astype('i8').describe().to_markdown())
    if verbose > 0:
        print('Loading Pandora')
    p = pandora(spc, start_date, end_date, verbose=verbose)
    if verbose > 1:
        print(hourdf.to_markdown())

    opts = dict(check=False, clip=config.clip, verbose=0)
    popts = dict(check=False, clip=config.pandora_clip, verbose=0)
    for date, row in hourdf.iterrows():
        if verbose > 0:
            print(date.strftime('%Y-%m-%dT%HZ'), end='.', flush=True)
        dotropomi = row['tropomi'] and not tomi_store.hasdate(date)
        doairnow = row['airnow'] and not an_store.hasdate(date)
        dopandora = row['pandora'] and not pan_store.hasdate(date)
        if not any([dotropomi, doairnow, dopandora]):
            if verbose > 0:
                print(' - all done')
            continue

        t = tempo(spc, date)
        if t.get().shape[0] == 0:
            if verbose > 0:
                print(' - no valid tempo')
            continue
        if dotropomi:
            tr = tropomi(spc, date)
            if verbose > 0:
                t0 = _time.time()
                print('tropomi...', end='', flush=True)
            t.intx_append(tr, store=tomi_store, **opts)
            if verbose > 0:
                t1 = _time.time()
                print(f'{t1 - t0:.0f}s', end='.', flush=True)
        if doairnow:
            an = airnow(spc, date)
            if verbose > 0:
                t0 = _time.time()
                print('airnow...', end='', flush=True)
            t.intx_append(an, store=an_store, **opts)
            if verbose > 0:
                t1 = _time.time()
                print(f'{t1 - t0:.0f}s', end='.', flush=True)
        if dopandora:
            if verbose > 0:
                t0 = _time.time()
                print('pandora...', end='', flush=True)
            p.set_bdate(date)
            t.intx_append(p, store=pan_store, dt=900, **popts)
            if verbose > 0:
                t1 = _time.time()
                print(f'{t1 - t0:.0f}s', end='.', flush=True)
        if verbose > 0:
            print()
