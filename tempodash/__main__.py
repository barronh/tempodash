import gc
import pandas as pd
from . import get_configs
from .tempo import gettempo
from .pandora import makeplots as makepandoraplots
from .tropomi import gettropomi, makeplots as maketropomiplots
from .airnow import getairnow, makeplots as makeairnowplots
from argparse import ArgumentParser
# Using malloc_trim to prevent memory allocation runaway
# https://github.com/pandas-dev/pandas/issues/2659
from ctypes import cdll, CDLL
cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")

prsr = ArgumentParser()
prsr.add_argument('bdate', default='2023-10-17T00Z', nargs='?')
prsr.add_argument('edate', default=None, nargs='?')
args = prsr.parse_args()

start_date = pd.to_datetime(args.bdate)
if args.edate is None:
    dt = pd.to_timedelta('1d')
    end_date = pd.to_datetime('now', utc=True).floor('1d') - dt
else:
    end_date = args.edate

cfgs = get_configs()
temposites = [
    k for k, cfg in cfgs.items()
    if (
        cfg.get('pandora', False) or cfg.get('tropomi', False)
        or cfg.get('airnow', False)
    )
]
pandorasites = [k for k, cfg in cfgs.items() if cfg.get('pandora', False)]
tropomisites = [k for k, cfg in cfgs.items() if cfg.get('tropomi', False)]
airnowsites = [k for k, cfg in cfgs.items() if cfg.get('airnow', False)]
bdates = pd.date_range(start_date, end_date, freq='1d')
processors = [
    ('no2', 'TEMPO', temposites, gettempo),
    ('no2', 'TropOMI', tropomisites, gettropomi),
    ('no2', 'AirNow', airnowsites, getairnow),
    ('hcho', 'TEMPO', temposites, gettempo),
    ('hcho', 'TropOMI', tropomisites, gettropomi),
]
for bdate in bdates:
    edate = bdate + pd.to_timedelta('23h')
    for (spc, source, sites, getter) in processors:
        print(spc, source, bdate, len(sites))
        for lockey in sites:
            print(lockey, end=',', flush=True)
            try:
                getter(lockey=lockey, spc=spc, bdate=bdate, edate=edate)
            except Exception:
                pass
        print()
        gc.collect()
        libc.malloc_trim(0)


for lockey in pandorasites:
    makepandoraplots(lockey=lockey, bdate=start_date, edate=end_date)
    gc.collect()
    libc.malloc_trim(0)

for lockey in tropomisites:
    maketropomiplots(lockey=lockey, bdate=start_date, edate=end_date)
    gc.collect()
    libc.malloc_trim(0)

for lockey in airnowsites:
    makeairnowplots(lockey=lockey, bdate=start_date, edate=end_date)
    gc.collect()
    libc.malloc_trim(0)
