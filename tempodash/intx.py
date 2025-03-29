import pandas as pd
import os


class intxstore:
    def __init__(self, source1, source2, spc):
        """
        Arguments
        ---------
        source1 : str
            'tempo', 'tropomi_offl', 'pandora', 'airnow'
            (source1 typically 'tempo')
        source2 : str
            'tempo', 'tropomi_offl', 'pandora', 'airnow'
            (source2 typically not 'tempo')
        spc : str
            'no2' or 'hcho'

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash.intx import intxstore
        >>> store = intxstore('tempo', 'pandora', 'no2')
        >>> store.hasdate('2023-08-01')
        False
        """
        self.source1 = source1
        self.source2 = source2
        self.spc = spc
        self.datekey = 'time'
        self.datakey = f'{self.source1}_{self.source2}_{self.spc}'
        self.month = None

    def load(self, dates, where=None, columns=None):
        """
        Arguments
        ---------
        dates : pands.Series
            Series of data-like values
        where : str or list of str
            conditions statements
        columns : str or list of str
            columns to retrieve

        Returns
        -------
        df : pandas.DataFrame
            All data available within dates

        Example
        -------
        >>> from tempodash.intx import intxstore
        >>> import pandas as pd
        >>> store = intxstore('tempo', 'pandora', 'no2')
        >>> dates = pd.date_range('2023-08-01', '2024-05-01')
        >>> cols = ['tempo_time', 'tempo_no2_sum', 'pandora_no2_total']
        >>> condstr = 'tempo_lat > 40 and tempo_lon > -80'
        >>> df = store.load(dates, where=condstr, columns=cols)
        >>> df
                  tempo_time  tempo_no2_sum  pandora_no2_total
        0       1.711975e+09   8.406275e+15       6.521979e+15
        1       1.711975e+09   8.248625e+15       6.521979e+15
        ...              ...            ...                ...
        403768  1.714926e+09   1.251497e+16       1.038579e+16
        403769  1.714926e+09   1.336576e+16       1.038579e+16
        """
        dates = pd.to_datetime(dates)
        mdates = pd.to_datetime(sorted(set(dates.strftime('%Y-%m-01'))))
        dfs = []
        for date in mdates:
            self.set_month(date)
            df = pd.read_hdf(
                self.hdfpath, key=self.datakey, where=where, columns=columns
            )
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def set_month(self, bdate, force=False):
        """
        Arguments
        ---------
        bdate : date-like
            Beginning of hour date (used just for month)
        force : bool
            If True, reread rather than using cached times.

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash.intx import intxstore
        >>> store = intxstore('tempo', 'pandora', 'no2')
        >>> store.set_month('2024-06-01')
        """
        month = pd.to_datetime(bdate).strftime('%Y-%m')
        if self.month != month or force:
            self.month = month
            self.hdfpath = f'intx/stores/{self.datakey}_{month}.h5'
            if os.path.exists(self.hdfpath):
                self._timedf = pd.read_hdf(self.hdfpath, key=self.datekey)
            else:
                self._timedf = pd.DataFrame({self.datekey: [], 'nrows': []})

    def hasdate(self, date):
        """
        Arguments
        ---------
        date : date-like
            Beginning of hour date (used just for month)

        Returns
        -------
        found : bool
            True if found in file; False otherwise

        Example
        -------
        >>> from tempodash.intx import intxstore
        >>> store = intxstore('tempo', 'pandora', 'no2')
        >>> store.hasdate('2024-01-01T17')
        True
        >>> store.hasdate('2017-01-01T16')
        False
        """
        from .dates import date2num
        self.set_month(date)
        if os.path.exists(self.hdfpath):
            ndate = date2num(date)
            return ndate in self._timedf[self.datekey].values
        else:
            return False

    def append(self, intxdf, check=True, verbose=0, bdate=None):
        """
        Arguments
        ---------
        intxdf : pandas.DataFrame
            Dataframe to be appended to the store.
        check : bool
            If True, only add if not already available.
        verbose : int
            Level of verbsity.
        bdate : date-like
            Start date. If not specified, infer from intxdf.tempo_time

        Returns
        -------
        nrows : int
            Rows added (intxdf.shape[0] or 0)

        Example
        -------
        >>> from tempodash.intx import intxstore
        >>> import pandas as pd
        >>> t = [0.0, 1800.0]
        >>> oth = [0.0, 5.0]
        >>> data = dict(tempo_time=t, o_time=t, tempo_no2_sum=oth, o_no2=oth)
        >>> intxdf = pd.DataFrame(data)
        >>> store = intxstore('tempo', 'other', 'no2')
        >>> store.append(intxdf)
        2
        >>> store.append(intxdf)
        0
        """
        from .dates import date2num
        nrows = intxdf.shape[0]
        if nrows == 0 and bdate is None:
            return 0
        if verbose > 0:
            print(nrows, intxdf.dtypes)
        if bdate is None:
            bdate = pd.to_datetime(intxdf['tempo_time'].min(), unit='s')
            bdate = bdate.floor('1h')

        self.set_month(bdate)
        force = not os.path.exists(self.hdfpath)
        os.makedirs(os.path.dirname(self.hdfpath), exist_ok=True)
        if check:
            if self.hasdate(bdate):
                if verbose > 0:
                    print(f'WARN:: cached {bdate} for {self.datakey}')
                return 0

        ndate = date2num(bdate)

        if verbose > 0:
            print(f'INFO:: adding {bdate} for {self.datakey}')
        timedf = pd.DataFrame(
            {self.datekey: [ndate], 'nrows': [nrows]}
        )
        opts = dict(append=True, data_columns=True, index=False)
        intxdf.to_hdf(self.hdfpath, key=self.datakey, **opts)
        timedf.to_hdf(self.hdfpath, key=self.datekey, **opts)
        self.set_month(bdate, force=force)

        return nrows

    def mostrecent(self, to_datetime=True):
        """
        What is the most recent hour of in all stores for this spc?
        """
        import glob
        import pandas as pd
        hdfpat = f'intx/stores/{self.datakey}_????-??.h5'
        print(hdfpat)
        hdfpaths = sorted(glob.glob(hdfpat))
        hdfpath = hdfpaths[-1]
        tdf = pd.read_hdf(hdfpath, key=self.datekey)
        time = tdf['time'].max()
        if to_datetime:
            time = pd.to_datetime(time, unit='s')
        return time


def loadintx(
    source, spc, start_date, end_date, where=None, columns=None,
    bbox=None, lockey=None
):
    """
    Load intersected data easily.

    Arguments
    ---------
    source : str
        'tropomi_offl', 'pandora', 'airnow'
    spc : str
        'no2' or 'hcho'
    start_date : date-like
        Used to construct a date range to load
    end_date : date-like
        Used to construct a date range to load
    where : str or list of str
        conditions statements
    columns : str or list of str
        columns to retrieve
    bbox : list
        Bounding box swlon, swlat, nelon, nelat to create
        a tempo_lon/tempo_lat query (>=, <=)
    lockey : str
        Lookup bbox from config file

    Returns
    -------
    df : pandas.DataFrame
        All data available within dates

    Example
    -------
    >>> from tempodash import loadintx
    >>> cols = ['tempo_time', 'tempo_no2_sum', 'pandora_no2_total']
    >>> sdt = '2023-08-01'
    >>> edt = '2024-05-01'
    >>> src = 'pandora'
    >>> spc = 'no2'
    >>> bbox = (-80, 40, -60, 55)
    >>> df = loadintx(src, spc, sdt, edt, columns=cols, bbox=bbox)
    >>> df
    >>> df
               tempo_time  tempo_no2_sum  pandora_no2_total
    0        1.690990e+09   4.839697e+15       4.360452e+15
    1        1.690990e+09   4.839697e+15       4.159072e+15
    ...               ...            ...                ...
    2978805  1.714510e+09   8.803951e+15       5.897724e+15
    2978806  1.714510e+09   8.803951e+15       5.977277e+15

    [2978807 rows x 3 columns]
    """
    import pandas as pd
    import warnings
    if lockey is not None:
        from .config import locconfigs
        bbox = locconfigs[lockey]['bbox']
    if bbox is not None:
        if where is not None:
            if isinstance(where, str):
                where = [where]
        else:
            where = []
        where.append((
            'tempo_lon >= {0} and tempo_lon <= {2}'
            ' and tempo_lat >= {1} and tempo_lat <= {3}'
        ).format(*bbox))

    dates = pd.date_range(start_date, end_date)
    store = intxstore('tempo', source, spc)
    df = store.load(dates, where=where, columns=columns)
    if 'tempo_time' not in df.columns:
        warnings.warn('loadintx returned all times in files')
    nstart = dates[0].to_numpy().astype('i8') / 1e9
    nend = dates[-1].to_numpy().astype('i8') / 1e9 + 3599.
    df = df.query(f'tempo_time >= {nstart} and tempo_time <= {nend}')
    return df
