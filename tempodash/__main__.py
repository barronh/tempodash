if __name__ == "__main__":
    from argparse import ArgumentParser
    import pandas as pd
    from .util import retrieve
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
    retrieve(start_date, end_date)
    # make_plots(start_date, end_date)
    # make_report(['pandora', 'tropomi', 'airnow'])
