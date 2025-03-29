from . import makeintx
import argparse
import pandas as pd

twoweeks = pd.to_timedelta('14d')
oneseconds = pd.to_timedelta('1s')
today = pd.to_datetime('now').floor('1d')
refdate = today - twoweeks - twoweeks

prsr = argparse.ArgumentParser()


prsr.add_argument('--verbose', action='count', default=0)
prsr.add_argument('--spc', default='no2', choices={'no2', 'hcho'})
prsr.add_argument('start_date', nargs='?', default=None, type=pd.to_datetime)
prsr.add_argument('end_date', nargs='?', default=None, type=pd.to_datetime)
args = prsr.parse_args()

# run for two weeks starting four weeks ago
if args.start_date is None:
    args.start_date = refdate

if args.end_date is None:
    end_date = args.start_date + pd.offsets.MonthBegin() - oneseconds
    args.end_date = min(end_date, today - twoweeks - oneseconds)

if args.end_date <= args.start_date:
    prsr.exit(status=0, message='WARN:: no data to process')
makeintx(**vars(args))
