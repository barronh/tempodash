import pandas as pd
_crnrkeys = [
    'Longitude_SW', 'Latitude_SW',
    'Longitude_SE', 'Latitude_SE',
    'Longitude_NE', 'Latitude_NE',
    'Longitude_NW', 'Latitude_NW',
    'Longitude_SW', 'Latitude_SW',
]


class dataloader:
    def _renamer(self, k):
        """
        Generic renamer that fits all date in tempo, tropomi, and pandora

        Arguments
        ---------
        k : str
            Input key to be renamed

        Returns
        -------
        ok : str
            Usually {self.source}_{shortendk}

        Example
        -------
        an = dataloader('airnow', 'no2', '2023-09-01T17Z', keys=['airnow.no2'])
        print(an._renamer('Timestamp'))
        # 'airnow_time'
        """
        ok = {
            'Timestamp': 'time',
            'LONGITUDE': 'lon',
            'Longitude_SW': 'lon_sw',
            'Latitude_SW': 'lat_sw',
            'Longitude_SE': 'lon_se',
            'Latitude_SE': 'lat_se',
            'Longitude_NW': 'lon_nw',
            'Latitude_NW': 'lat_nw',
            'Longitude_NE': 'lon_ne',
            'Latitude_NE': 'lat_ne',
            'LATITUDE': 'lat',
            'solar_zenith_angle': 'sza',
            'eff_cloud_fraction': 'cloud_eff',
            'no2': 'no2_sfc',  # airnow
            'no2_vertical_column_troposphere': 'no2_trop',  # tempo
            'no2_vertical_column_total': 'no2_total',  # tempo
            'no2_vertical_column_stratosphere': 'no2_strat',  # tempo
            'vertical_column': 'hcho_total',  # tempo
            'nitrogen_dioxide_vertical_column_amount': 'no2_total',  # tomi
            'nitrogendioxide_tropospheric_column': 'no2_trop',  # tomi
            'nitrogendioxide_stratospheric_column': 'no2_strat',  # tomi
            'formaldehyde_total_vertical_column_amount': 'hcho_total',  # tomi
            'formaldehyde_tropospheric_vertical_column': 'hcho_total',  # tomi
            'ELEVATION': 'elevation',
            'STATION': 'id', 'NOTE': 'note',
        }.get(k, k)
        if ok != 'geometry' and not ok.startswith(f'{self.source}_'):
            ok = f'{self.source}_{ok}'
        return ok

    def _geom(self, df=None, pointbuffer=0.03):
        """
        Generic function to return geometry with polygons for tempo and tropomi
        and points for airnow and Pandora

        Arguments
        ---------
        df : pandas.DataFrame
            Must have:
            'LONGITUDE' and 'LATITUDE', or
            '<source>_lon' and '<source>_lat', or
            all corners as
              'Longitude_SW', 'Latitude_SW',
              'Longitude_SE', 'Latitude_SE',
              'Longitude_NW', 'Latitude_NW',
              'Longitude_NE', and 'Latitude_NE'
            or all corners as:
              '{source}_lon_sw', '{source}_lat_sw',
              '{source}_lon_se', '{source}_lat_se',
              '{source}_lon_nw', '{source}_lat_nw',
              '{source}_lon_ne', '{source}_lat_ne'

        Returns
        -------
        geom : shapely.array
            Array of points or polygons

        Example
        -------
        >>> keys = ['airnow.no2']
        >>> an = dataloader('airnow', 'no2', '2023-09-01T17Z', keys=keys)
        >>> print(an._geom())
        [<POLYGON ((-66.616 45.957, -66.616 45.951, ...>
         <POLYGON ((-65.978 45.309, -65.979 45.303, ...>
         ...
         <POLYGON ((-109.76 41.751, -109.76 41.745, ...>
         <POLYGON ((-110.801 44.373, -110.801 44.367, ...>]
        """
        from shapely import points, polygons, buffer
        lonlatkeys = ['LONGITUDE', 'LATITUDE']
        rlonlatkeys = [self._renamer(k) for k in lonlatkeys]
        rcrnrkeys = [self._renamer(k) for k in _crnrkeys]
        if df is None:
            df = self.get(geo=False)
        if df.shape[0] == 0:
            return []
        if all([k in df.columns for k in _crnrkeys]):
            geom = polygons(df[_crnrkeys].values.reshape(-1, 5, 2))
        elif all([k in df.columns for k in rcrnrkeys]):
            geom = polygons(df[_crnrkeys].values.reshape(-1, 5, 2))
        elif all([k in df.columns for k in lonlatkeys]):
            geom = points(df[lonlatkeys].values.reshape(-1, 2))
            geom = buffer(geom, pointbuffer)
        elif all([k in df.columns for k in rlonlatkeys]):
            geom = points(df[rlonlatkeys].values.reshape(-1, 2))
            geom = buffer(geom, pointbuffer)
        else:
            raise KeyError('unknown geometry')
        return geom

    def __init__(self, source, spc, date, keys):
        """
        Arguments
        ---------
        source : str
            'airnow', 'pandora', 'tempo', 'tropomi_offl', or 'tropomi_nrti'
        spc : str
            'no2' or 'hcho'
        date : str or date-like
            Passed to pandas.to_datetime for retreiving data.

        Returns
        -------
        None

        Example
        -------
        >>> keys = ['airnow.no2']
        >>> an = dataloader('airnow', 'no2', '2023-09-01T17Z', keys=keys)
        >>> an
        <tempodash.core.dataloader object at 0x1479185c0550>
        """
        self.source = source
        self.spc = spc
        self.keys = keys
        self.bdate = pd.to_datetime(date)
        self.edate = self.bdate + pd.to_timedelta('3599s')
        self.bbox = (-150, 15, -50, 65)
        self._df = None
        self._geodf = None

    def get(self, geo=False, verbose=0):
        """
        Retrieve and open pandas.DataFrame using pyrsig

        Arguments
        ---------
        geo : bool
            If True, add geometry (see _geom) to output
        Returns
        -------
        df : pandas.DataFrame
            Results optionally with geometry

        Example
        -------
        >>> keys = ['airnow.no2']
        >>> an = dataloader('airnow', 'no2', '2023-09-01T17Z', keys=keys)
        >>> df = an.get()
        >>> df
              airnow_time  airnow_lon  airnow_lat  ...  airnow_no2_sfc  ...
        0    1.693588e+09  -66.645599   45.957298  ...           0.659  ...
        1    1.693588e+09  -66.008301   45.308800  ...           2.802  ...
        ..            ...         ...         ...  ...             ...  ...
        404  1.693588e+09 -109.789703   41.750599  ...           1.000  ...
        405  1.693588e+09 -110.830803   44.373100  ...           0.700  ...

        [406 rows x 6 columns]
        """
        from .config import api
        if self._df is None:
            api.workdir = f'data/{self.source}/{self.bdate:%Y-%m}'
            api.bbox = self.bbox
            api.bdate = self.bdate
            api.edate = self.edate
            df = None
            for key in self.keys:
                corners = int(key == self.keys[0])
                try:
                    tmpdf = api.to_dataframe(
                        key, backend='xdr', corners=corners, unit_keys=False,
                        bdate=self.bdate, edate=self.edate, verbose=verbose - 1
                    )
                except Exception as e:
                    print(e)
                    tmpdf = pd.DataFrame()
                if tmpdf.shape == (0, 0):
                    df = tmpdf
                    break
                if df is None:
                    df = tmpdf
                else:
                    df = pd.merge(df, tmpdf)
            timekey = 'Timestamp'  # f'{self.source}_time'
            if df.shape[0] > 0:
                timeo = pd.to_datetime(df[timekey]).dt.tz_convert(None)
                df[timekey] = timeo.astype('i8') / 1e9

            for k, v in list(df.dtypes.items()):
                if v.char == 'f':
                    df[k] = df[k].astype('f8')
                elif v.char == 'i':
                    df[k] = df[k].astype('i8')

            self._df = df
        outdf = self._df
        if geo:
            if self._geodf is None:
                import geopandas as gpd
                geom = self._geom(self._df)
                gdf = gpd.GeoDataFrame(self._df, geometry=geom, crs=4326)
                self._geodf = gdf
            outdf = self._geodf
        crnrkeys = list(set([k for k in _crnrkeys if k in outdf.columns]))
        return outdf.drop(crnrkeys, axis=1).rename(columns=self._renamer)

    def intx(
        self, oth, dt=None, clip=None, returndf=True, persist=False, verbose=0
    ):
        """
        Intersect this data store's dataframe with another using their
        geometries.

        Arguments
        ---------
        oth : datastore
            Other datastore to intersect spatially.
        dt : float
            Seconds apart allowed for valid intersections.
        clip : shapley.geometry
            First subset both dataframes for only the points within clip
        returndf : bool
            If True (default), return the result
        persist : bool
            If True (default), save intersection to disk
        verbose : int
            Level of verbosity

        Returns
        -------
        intxdf : pandas.DataFrame
            Intersection result with geometry removed.

        Example
        -------
        >>> akeys = ['airnow.no2']
        >>> an = dataloader('airnow', 'no2', '2023-09-01T17Z', keys=akeys)
        >>> pkeys = [
        >>>   'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
        >>> ]
        >>> pn = dataloader('pandora', 'no2', '2023-09-01T17Z', keys=pkeys)
        >>> clip = box(-75, 40, -60, 50)
        >>> intxdf = an.intx(pn, clip=clip)
        >>> intxdf.describe()
                airnow_time  airnow_lon  ...  pandora_no2_total
        count  5.770000e+02  577.000000  ...       5.770000e+02
        mean   1.693588e+09  -73.293135  ...       1.295700e+16
        std    0.000000e+00    0.983900  ...       8.988108e+15
        min    1.693588e+09  -74.429398  ...       5.151219e+15
        25%    1.693588e+09  -74.126099  ...       5.677795e+15
        50%    1.693588e+09  -73.336899  ...       8.510490e+15
        75%    1.693588e+09  -72.902702  ...       2.170199e+16
        max    1.693588e+09  -71.361702  ...       3.308324e+16
        """
        import os
        assert (self.spc == oth.spc)
        pfx = f'{self.source}_{oth.source}_{self.spc}'
        outpath = (
            f'intx/{self.bdate:%Y-%m}/{pfx}'
            f'_{self.bdate:%Y-%m-%dT%H%M%S}Z'
            f'_{self.edate:%Y-%m-%dT%H%M%S}Z.csv.gz'
        )
        if not os.path.exists(outpath):
            import geopandas as gpd
            medf = self.get(geo=True)
            othdf = oth.get(geo=True)
            if clip is not None:
                meclip = clip.intersects(medf['geometry'])
                medf = medf.loc[meclip]
                othclip = clip.intersects(othdf['geometry'])
                othdf = othdf.loc[othclip]
            if medf.shape[0] == 0 or othdf.shape[0] == 0:
                df = pd.DataFrame(columns=['tempo_time'])
            else:
                df = gpd.sjoin(medf, othdf, lsuffix='1', rsuffix='2')
                if 'index_2' in df:
                    df.drop('index_2', axis=1, inplace=True)
                df.drop('geometry', axis=1, inplace=True)
                if dt is not None:
                    met = df[f'{self.source}_time']
                    otht = df[f'{oth.source}_time']
                    adt = (met - otht).abs()
                    df = df.loc[adt < dt]
            if persist:
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                df.to_csv(outpath, index=False)
            if returndf:
                return df
        else:
            if verbose > 0:
                print(f'WARN:: Using cached {outpath}')
            if returndf:
                df = pd.read_csv(outpath)
                return df

    def intx_append(
        self, oth, dt=None, clip=None, check=True, store=None, verbose=0
    ):
        """
        Arguments
        ---------
        oth : datastore
            Other datastore to intersect spatially.
        dt : float
            Seconds apart allowed for valid intersections.
        clip : shapley.geometry
            First subset both dataframes for only the points within clip
        check : bool
            If True (default), check if it already exists in store and, if so,
            do not add.
        store : intx.intxstore
            Object that implements append and hasdate
        verbose : int
            Level of verbosity

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash.core import dataloader
        >>> from tempodash.intx import intxstore
        >>> store = intxstore('airnow', 'pandora', 'no2')
        >>> akeys = ['airnow.no2']
        >>> an = dataloader('airnow', 'no2', '2023-09-01T17Z', keys=akeys)
        >>> pkeys = [
        >>>   'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
        >>> ]
        >>> pn = dataloader('pandora', 'no2', '2023-09-01T17Z', keys=pkeys)
        >>> clip = box(-75, 40, -60, 50)
        >>> an.intx_append(pn, clip=clip, store=store)
        """
        from .intx import intxstore
        assert self.spc == oth.spc
        bdate = self.bdate
        if store is None:
            store = intxstore(self.source, oth.source, self.spc)
        if check:
            isinfile = store.hasdate(self.bdate)
            if verbose > 0 and isinfile:
                print(
                    f'WARN:: store({self.source}, {oth.source}, {oth.spc})'
                    f'already has {bdate}'
                )
            return

        intxdf = self.intx(oth, dt=dt, clip=clip, verbose=verbose)
        out = store.append(intxdf, check=False, verbose=verbose, bdate=bdate)
        return out


class tempo(dataloader):
    def __init__(self, spc, date):
        """
        dataloader for tempo that is aware of keys and how to construct
        the tempo_no2_sum from tempo_no2_trop and tempo_no2_strat. Otherwise,
        just a dataloader.

        Arguments
        ---------
        spc : str
            'no2' or 'hcho'
        date : str or date-like
            Passed to pandas.to_datetime for retreiving data.

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash import tempo
        >>> te = tempo('no2', '2023-09-01T17Z')
        >>> te
        <tempodash.core.tempo object at 0x14b3b06cc430>
        """
        keys = {
            'no2': [
                'tempo.l2.no2.vertical_column_troposphere',
                'tempo.l2.no2.solar_zenith_angle',
                'tempo.l2.no2.vertical_column_total',
                'tempo.l2.no2.vertical_column_stratosphere',
                'tempo.l2.no2.eff_cloud_fraction',
            ],
            'hcho': [
                'tempo.l2.hcho.vertical_column',
                'tempo.l2.hcho.solar_zenith_angle',
                'tempo.l2.hcho.eff_cloud_fraction',
            ],
        }[spc]
        dataloader.__init__(self, 'tempo', spc, date, keys)

    def get(self, geo=False):
        """
        Retrieve and open pandas.DataFrame using pyrsig

        Arguments
        ---------
        geo : bool
            If True, add geometry (see _geom) to output
        Returns
        -------
        df : pandas.DataFrame
            Results optionally with geometry

        Example
        -------
        >>> from tempodash import tempo
        >>> te = tempo('no2', '2023-09-01T17Z')
        >>> df = te.get()
        >>> df
                   tempo_time   tempo_lon  tempo_lat  ...  tempo_no2_sum
        0        1.693588e+09 -115.477547  58.377090  ...   3.257515e+15
        1        1.693588e+09 -115.576996  58.385010  ...   5.263524e+15
        ...               ...         ...        ...  ...            ...
        1371630  1.693591e+09  -81.591011  17.380074  ...   3.009454e+15
        1371631  1.693591e+09  -81.763535  17.378859  ...   3.299410e+15

        [1371632 rows x 9 columns]
        """
        df = dataloader.get(self, geo=geo)
        if self.spc == 'no2' and 'tempo_no2_trop' in df.columns:
            df.eval(
                'tempo_no2_sum = tempo_no2_trop + tempo_no2_strat',
                inplace=True
            )
        return df


class tropomi(dataloader):
    def __init__(self, spc, date, kind='offl'):
        """
        dataloader for tempo that is aware of keys and how to construct
        the tropomi_no2_sum from tropomi_no2_trop and tropomi_no2_strat.
        Otherwise, just a dataloader.

        Arguments
        ---------
        spc : str
            'no2' or 'hcho'
        date : str or date-like
            Passed to pandas.to_datetime for retreiving data.
        kind : str
            'offl' or 'nrti'

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash import tropomi
        >>> tr = tropomi('no2', '2023-09-01T17Z')
        >>> tr
        <tempodash.core.tropomi object at 0x14b3b06cc430>
        """
        keys = {
            'no2': [
                 f'tropomi.{kind}.no2.nitrogendioxide_tropospheric_column',
                 f'tropomi.{kind}.no2.nitrogendioxide_stratospheric_column',
            ],
            'hcho': [
                 (
                     f'tropomi.{kind}.hcho.'
                     'formaldehyde_tropospheric_vertical_column'
                 ),
            ],
        }[spc]
        source = f'tropomi_{kind}'
        dataloader.__init__(self, source, spc, date, keys)

    def get(self, geo=False):
        """
        Retrieve and open pandas.DataFrame using pyrsig

        Arguments
        ---------
        geo : bool
            If True, add geometry (see _geom) to output
        Returns
        -------
        df : pandas.DataFrame
            Results optionally with geometry

        Example
        -------
        >>> from tempodash import tropomi
        >>> tr = tropomi('no2', '2023-09-01T17Z')
        >>> df = tr.get()
        >>> df
                tropomi_offl_time  tropomi_offl_lon  ...  tropomi_offl_no2_sum
        0            1.693591e+09        -69.094490  ...          3.247371e+15
        1            1.693591e+09        -69.026848  ...          3.203644e+15
        ...                   ...               ...  ...                   ...
        279843       1.693591e+09       -126.145493  ...          3.447256e+15
        279844       1.693591e+09       -126.230492  ...          3.414751e+15

        [279845 rows x 6 columns]

        """
        df = dataloader.get(self, geo=geo)
        source = self.source
        if self.spc == 'no2' and f'{source}_no2_trop' in df.columns:
            df.eval(
                f'{source}_no2_sum = {source}_no2_trop + {source}_no2_strat',
                inplace=True
            )
        return df


class airnow(dataloader):
    def __init__(self, spc, date):
        """
        dataloader for airnow that is aware of keys and to drop string columns
        Otherwise, just a dataloader.

        Arguments
        ---------
        spc : str
            'no2' or 'hcho'
        date : str or date-like
            Passed to pandas.to_datetime for retreiving data.

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash import airnow
        >>> an = airnow('no2', '2023-09-01T17Z')
        >>> an
        <tempodash.core.airnow object at 0x14b39e1e0c10>
        """
        assert spc == 'no2'
        keys = ['airnow.no2']
        dataloader.__init__(self, 'airnow', spc, date, keys)

    def get(self, geo=False):
        """
        Retrieve and open pandas.DataFrame using pyrsig. Drop airnow_SITENAME

        Arguments
        ---------
        geo : bool
            If True, add geometry (see _geom) to output
        Returns
        -------
        df : pandas.DataFrame
            Results optionally with geometry

        Example
        -------
        >>> from tempodash import airnow
        >>> an = airnow('no2', '2023-09-01T17Z')
        >>> df = an.get()
        >>> df
              airnow_time  airnow_lon  airnow_lat  ...  airnow_no2_sfc  ...
        0    1.693588e+09  -66.645599   45.957298  ...           0.659  ...
        1    1.693588e+09  -66.008301   45.308800  ...           2.802  ...
        ..            ...         ...         ...  ...             ...  ...
        404  1.693588e+09 -109.789703   41.750599  ...           1.000  ...
        405  1.693588e+09 -110.830803   44.373100  ...           0.700  ...

        [406 rows x 6 columns]
        """
        df = dataloader.get(self, geo=geo).drop('airnow_SITE_NAME', axis=1)
        return df


class pandora(dataloader):
    def __init__(self, spc, start_date, end_date, verbose=0):
        """
        Pandora data is most efficiently read by site over long time periods.
        This dataloader reads many sites over a user defined period and then
        presents the get interface as though it was a single hour defined by
        bdate (see set_bdate). The pandora data loader initializes bdate to
        start_date.

        Arguments
        ---------
        spc : str
            'no2' or 'hcho'
        start_date : str or date-like
            Passed to pandas.to_datetime for retreiving data.
        end_date : str or date-like
            Passed to pandas.to_datetime for retreiving data.

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash import pandora
        >>> pm = pandora('no2', '2023-09-01T00Z', '2023-09-30T00Z')
        >>> pm
        <tempodash.core.pandora object at 0x14b3b06cab80>
        >>> df = pm.get()
        >>> df
                pandora_time  pandora_lon  pandora_lat  ... pandora_no2_total
        0       1.693527e+09    -122.3360      37.9130  ...      4.709014e+15
        1       1.693527e+09    -122.3360      37.9130  ...      4.707147e+15
        ...              ...          ...          ...  ...               ...
        174142  1.693527e+09     -97.1142      32.7316  ...      1.255918e+16
        174143  1.693527e+09     -97.1142      32.7316  ...      1.250498e+16

        [267 rows x 6 columns]

        """
        from .config import locconfigs
        vca = 'vertical_column_amount'
        key = {
            'no2': f'pandora.L2_rnvs3p1_8.nitrogen_dioxide_{vca}',
            'hcho': f'pandora.L2_rfus5p1_8.formaldehyde_total_{vca}',
        }[spc]
        self.verbose = verbose
        self.source = 'pandora'
        self.start_date = start_date
        self.end_date = end_date
        self.spc = spc
        cfgs = {k: v for k, v in locconfigs.items() if v.get('pandora', True)}
        if spc == 'hcho':
            cfgs = {
                k: v for k, v in cfgs.items()
                if v.get('pandora_hcho', True)
            }
        self.cfgs = cfgs
        self.key = key
        self.keys = [key]
        self.bdate = None
        self._load()
        self.set_bdate(self.start_date)

    def _load(self):
        """
        Load data from all sites. Called by __init__
        """
        import pandas as pd
        from .config import api
        bdate = pd.to_datetime(self.start_date)
        edate = pd.to_datetime(self.end_date) + pd.to_timedelta('3599s')
        qa = api.pandora_kw['minimum_quality']
        dfs = []
        if self.verbose > 0:
            print('Loading: ', end='', flush=True)
        for lockey, cfg in self.cfgs.items():
            if self.verbose > 0:
                print(lockey, end='.', flush=True)
            api.workdir = f'data/pandora/{lockey}/{qa}'
            api.bbox = cfg['bbox']
            df = api.to_dataframe(
                self.key, bdate=bdate, edate=edate,
                unit_keys=False, backend='xdr', verbose=self.verbose - 1
            )
            if df.shape[0] == 0:
                continue
            timekey = 'Timestamp'
            timeo = pd.to_datetime(df[timekey]).dt.tz_convert(None)
            df[timekey] = timeo.astype('i8') / 1e9
            df = df.loc[df['STATION'].astype('d') == float(cfg['pandoraid'])]
            dfs.append(df)

        if self.verbose > 0:
            print(flush=True)
        alldf = pd.concat(dfs, ignore_index=True)
        alldf = alldf.drop('NOTE', axis=1).rename(columns=self._renamer)
        self._alldf = alldf

    def set_bdate(self, bdate):
        """
        pandora is desigend to read many site-files at ones for a whole time
        range, so you must then set the bdate to match files it is being used
        with.

        Arguments
        ---------
        bdate : str or date-time
            Passed to pd.to_datetime

        Returns
        -------
        None

        Example
        -------
        >>> from tempodash import pandora
        >>> pm = pandora('no2', '2023-09-01T00Z', '2023-09-30T00Z')
        >>> pm.set_bdate('2023-09-01T17Z')
        >>> pm.get()
                pandora_time  pandora_lon  pandora_lat  ... pandora_no2_total
        79      1.693588e+09     -122.336       37.913  ...      3.276165e+15
        80      1.693588e+09     -122.336       37.913  ...      3.196372e+15
        ...              ...          ...          ...  ...               ...
        536832  1.693591e+09      -71.361       41.841  ...      5.436006e+15
        536833  1.693591e+09      -71.361       41.841  ...      5.434380e+15

        [2434 rows x 6 columns]
        >>> pm.set_bdate('2023-09-02T17Z')
        >>> pm.get()
                pandora_time  pandora_lon  pandora_lat  ... pandora_no2_total
        61082   1.693675e+09    -117.8811       34.960  ...      3.914332e+15
        61083   1.693675e+09    -117.8811       34.960  ...      3.864408e+15
        ...              ...          ...          ...  ...               ...
        537609  1.693677e+09     -71.3610       41.841  ...      6.200397e+15
        537610  1.693677e+09     -71.3610       41.841  ...      6.302773e+15

        [1914 rows x 6 columns]
        """
        import pandas as pd
        bdate = pd.to_datetime(bdate)
        if bdate != self.bdate:
            self.bdate = bdate
            edate = bdate + pd.to_timedelta('3599s')
            nbdate = bdate.to_numpy().astype('i8') / 1e9
            nedate = edate.to_numpy().astype('i8') / 1e9
            self._df = self._alldf.query(
                f'pandora_time >= {nbdate} and pandora_time <= {nedate}'
            )
            self._geodf = None
