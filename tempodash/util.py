__all__ = ['get_configs', 'get_tempopw']

import gc
# Using malloc_trim to prevent memory allocation runaway
# https://github.com/pandas-dev/pandas/issues/2659
from ctypes import cdll, CDLL
cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")

_configs = {}
_tempo_pw = None
_sites = {}


def get_tempopw(pwpath='~/.tempokey'):
    global _tempo_pw
    import os
    from getpass import getpass

    if _tempo_pw is None:
        _pwpath = os.path.expanduser(pwpath)
        if os.path.exists(_pwpath):
            _tempo_pw = open(_pwpath, 'r').read().strip()
        else:
            _tempo_pw = getpass('Enter RSIG TEMPO PW:\n')

    return _tempo_pw


def get_configs(cfgpath='config.json'):
    import json

    if cfgpath not in _configs:
        _configs[cfgpath] = json.load(open(cfgpath, 'r'))
    return _configs[cfgpath]


def sort_westeast(key, cfgs):
    return cfgs.get(key, {}).get('bbox', [-97])[0]


def retrieve(bdate, edate):
    from .tempo import get_tempo_chunk
    from .tropomi import get_tropomi_chunk
    from .airnow import get_airnow_chunk
    import pandas as pd
    bdate = pd.to_datetime(bdate)
    edate = pd.to_datetime(edate)
    cfgs = get_configs()
    pansites = [k for k, cfg in cfgs.items() if cfg.get('pandora', False)]
    tropomisites = [k for k, cfg in cfgs.items() if cfg.get('tropomi', False)]
    airnowsites = [k for k, cfg in cfgs.items() if cfg.get('airnow', False)]
    temposites = list(set(pansites + tropomisites + airnowsites))
    temposites = list(set(pansites))
    # Sort satellite sites West to East, which should mean more subsequent
    # retrievals are to the same granule
    temposites = sorted(temposites, key=lambda x: sort_westeast(x, cfgs))
    tropomisites = sorted(tropomisites, key=lambda x: sort_westeast(x, cfgs))

    allhours = []
    for k, cfg in cfgs.items():
        if k in temposites:
            allhours.extend(cfg['hours'])
    allhours = set(allhours)
    alltropomihours = []
    for k, cfg in cfgs.items():
        if k in tropomisites:
            alltropomihours.extend(cfg['tropomi_hours'])

    alltropomihours = set(alltropomihours)

    # process hcho and no2 for TEMPO and TropOMI
    # process only no2 for AirNow
    # pandora is acquired as needed rather than up front
    processors = [
        ('tempo', 'no2', temposites, get_tempo_chunk),
        ('tropomi', 'no2', tropomisites, get_tropomi_chunk),
        #('airnow', 'no2', airnowsites, get_airnow_chunk),
        ('tempo', 'hcho', temposites, get_tempo_chunk),
        ('tropomi', 'hcho', tropomisites, get_tropomi_chunk),
    ]
    # Go hour-by-hour to reuse granules
    bdates = pd.date_range(bdate, edate, freq='1h')
    for bdate in bdates:
        edate = bdate + pd.to_timedelta('3599s')
        if bdate.hour not in allhours:
            continue
        for (source, spc, sites, getter) in processors:
            if source == 'tropomi' and bdate.hour not in alltropomihours:
                continue
            print(source, spc, bdate, len(sites))
            for lockey in sites:
                print(lockey, end=',', flush=True)
                try:
                    getter(lockey=lockey, spc=spc, bdate=bdate, edate=edate)
                except Exception:
                    pass
            print()
            gc.collect()
            libc.malloc_trim(0)


def intx(bdate, edate):
    from .pandora import make_intx as make_pandora
    from .tropomi import make_intx as make_tropomi
    from .airnow import make_intx as make_airnow
    cfgs = get_configs()
    pansites = [k for k, cfg in cfgs.items() if cfg.get('pandora', False)]
    tropomisites = [k for k, cfg in cfgs.items() if cfg.get('tropomi', False)]
    airnowsites = [k for k, cfg in cfgs.items() if cfg.get('airnow', False)]
    print('Intx Pandora', end=':')
    for lockey in pansites:
        print(lockey, end=',', flush=True)
        try:
            make_pandora(lockey, 'no2', bdate, edate)
        except:
            print('failed', end=',', flush=True)
        try:
            make_pandora(lockey, 'hcho', bdate, edate)
        except:
            print('failed', end=',', flush=True)
    print('\nIntx TropOMI', end=':')
    for lockey in tropomisites:
        print(lockey, 'no2', end=',', flush=True)
        try:
            make_tropomi(lockey, 'no2', bdate, edate)
        except Exception as e:
            print(str(e), end=',', flush=True)
        print(lockey, 'hcho', end=',', flush=True)
        try:
            make_tropomi(lockey, 'hcho', bdate, edate)
        except Exception as e:
            print(str(e), end=',', flush=True)
    print('\nAirNow TropOMI', end=':')
    for lockey in airnowsites:
        print(lockey, end=',', flush=True)
        try:
            make_airnow(lockey, 'no2', bdate, edate)
        except:
            print('failed', end=',', flush=True)


def make_plots(bdate, edate):
    from .airnow import make_plots as make_airnow_plots
    from .tropomi import make_plots as make_tropomi_plots
    from .pandora import make_plots as make_pandora_plots

    cfgs = get_configs()
    pansites = [k for k, cfg in cfgs.items() if cfg.get('pandora', False)]
    tropomisites = [k for k, cfg in cfgs.items() if cfg.get('tropomi', False)]
    airnowsites = [k for k, cfg in cfgs.items() if cfg.get('airnow', False)]

    # Now make plots for each site
    for lockey in pansites:
        make_pandora_plots(lockey=lockey, bdate=bdate, edate=edate)
        gc.collect()
        libc.malloc_trim(0)

    for lockey in tropomisites:
        make_tropomi_plots(lockey=lockey, bdate=bdate, edate=edate)
        gc.collect()
        libc.malloc_trim(0)

    for lockey in airnowsites:
        make_airnow_plots(lockey=lockey, bdate=bdate, edate=edate)
        gc.collect()
        libc.malloc_trim(0)


def make_report(antypes):
    from .word import from_antype

    # Now make word reports
    for antype in antypes:
        print(f'{antype}...')
        from_antype(antype, 'no2')
        if antype != 'airnow':
            from_antype(antype, 'hcho')
