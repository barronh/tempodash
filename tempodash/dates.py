__all__ = ['hourframe', 'date2num', 'num2date']


def hourframe(spc, start_date, end_date):
    """
    Get a list of hourly dates and whether particular products should be
    enabled. pandora and airnow are available any time tempo is, but tropomi
    is only available during overpass hours.

    Arguments
    ---------
    spc : str
        'no2' or 'hcho'
    start_date : date-like
        Passed to pandas.date_range
    end_date : date-like
        Passed to pandas.date_range

    Returns
    -------
    hourdf : pandas.DataFrame
        Has time, airnow, pandora and tropomi set to true where an intersection
        may exist for all hours where tempo may exist.

    Example
    -------
    >>> hourframe('no2', '2023-08-01T00', '2023-08-02T23')
                         pandora  airnow  tropomi
    time
    2023-08-01 00:00:00     True    True    False
    ...
    2023-08-01 15:00:00     True    True    False
    2023-08-01 16:00:00     True    True     True
    ...
    2023-08-01 23:00:00     True    True     True
    2023-08-02 00:00:00     True    True    False
    ...
    2023-08-02 15:00:00     True    True    False
    ...
    2023-08-02 23:00:00     True    True     True
    """
    import pandas as pd
    from .config import baddates, tempo_utc_hours, tropomi_utc_hours
    dates = pd.date_range(start_date, end_date, freq='1h')
    isbad = dates.strftime('%F').isin(baddates)
    # use all daylight hours (aka tempo hours)
    istempo = dates.hour.isin(tempo_utc_hours)
    istropomi = dates.hour.isin(tropomi_utc_hours)
    hourdf = pd.DataFrame(dict(
        time=dates, pandora=istempo, airnow=istempo, tropomi=istropomi
    )).loc[~isbad].query('pandora == True').set_index('time')
    if spc == 'hcho':
        hourdf['airnow'] = False
    return hourdf


def date2num(date):
    import pandas as pd
    num = pd.to_datetime(date).to_numpy().astype('i8') / 1e9
    return num


def num2date(num):
    import pandas as pd
    date = pd.to_datetime(num, unit='s')
    return date
