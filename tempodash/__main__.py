from . import makeintx
import argparse
import pandas as pd


prsr = argparse.ArgumentParser()

prsr.add_argument('--verbose', action='count', default=0)
prsr.add_argument('--spc', default='no2', choices={'no2', 'hcho'})
prsr.add_argument('start_date', nargs='?', default=None, type=pd.to_datetime)
prsr.add_argument('end_date', nargs='?', default=None, type=pd.to_datetime)
args = prsr.parse_args()

makeintx(**vars(args))
