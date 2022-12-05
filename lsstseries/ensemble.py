import pandas as pd
import vaex
import pyvo as vo
from .timeseries import timeseries
import time


class ensemble:
    """ensemble object is a collection of light curve ids"""

    def __init__(self, token=None):
        self.result = None  # holds the latest query
        self.token = token
        self.data = None

        self._id_col = 'object_id'
        self._time_col = 'midPointTai'
        self._flux_col = 'psFlux'
        self._err_col = 'psFluxErr'
        self._band_col = 'band'

    def count(self, sort=True, ascending=False):
        """Return the number of available measurements for each lightcurve"""
        count_df = self.data.groupby(self._id_col, 'count')
        if sort:
            count_df = count_df.sort('count', ascending=ascending)
        return count_df

    def dropna(self, threshold):
        """wrapper for vaex.dataframe.DataFrame.dropna"""
        self.data = self.data.dropna()
        return self

    def prune(self, threshold):
        """remove objects with less observations than a given threshold"""
        count = self.count(sort=False)
        subset_ids = count[count['count'] >= threshold][self._id_col]
        mask = self.data[self._id_col].isin(subset_ids.to_numpy())
        self.data = self.data[mask]
        return self

    def batch(self, func):
        """Run a function from lsstseries.timeseries on the available ids"""
        raise NotImplementedError

    def from_parquet(self, file, id_col=None, time_col=None, flux_col=None,
                     err_col=None, band_col=None):

        """Read in a parquet file"""
        self.data = vaex.open(file)

        # Track critical column changes
        if id_col is not None:
            self._id_col = id_col
        if time_col is not None:
            self._time_col = time_col
        if flux_col is not None:
            self._flux_col = flux_col
        if err_col is not None:
            self._err_col = err_col
        if band_col is not None:
            self._band_col = band_col

    def tap_token(self, token):
        """Add/update a TAP token to the class, enables querying
        Read here for information on TAP access: https://data.lsst.cloud/api-aspect

        Parameters
        ----------
        token : `str`
            Token string
        """
        self.token = token

    def query_tap(self, query, maxrec=None, debug=False):
        """Query the TAP service

        Parameters
        ----------
        query : `str`
            Query is an ADQL formatted string
        maxrec: `int`
            Max number of results returned

        Returns
        ----------
        result: `pd.df`
            Result of the query, as pandas dataframe
        """
        cred = vo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", self.token)
        service = vo.dal.TAPService(
            "https://data.lsst.cloud/api/tap", cred.get("ivo://ivoa.net/sso#BasicAA")
        )
        time0 = time.time()
        results = service.search(query, maxrec=maxrec)
        time1 = time.time()
        if debug:
            print(f"Query Time: {time1-time0} (s)")
        result = results.to_table().to_pandas()
        self.result = result
        return result

    def query_ids(
        self,
        ids,
        time_col="midPointTai",
        flux_col="psFlux",
        err_col="psFluxErr",
        add_cols=None,
        id_field="diaObjectId",
        catalog="dp02_dc2_catalogs",
        table="DiaSource",
        to_mag=True,
        maxrec=None,
    ):
        """Query based on a list of object ids; applicable for DP0.2

        Parameters
        ----------
        ids: `int`
            Ids of object
        time_col: `str`
            Column to retrieve and use for time
        flux_col: `str`
            Column to retrieve and use for flux (or magnitude or any "signal")
        err_col: `str`
            Column to retrieve and use for errors
        add_cols: `list` of `str`
            Additional columns to retreive
        id_field: `str`
            Which Id is being queried
        catalog: `str`
            Source catalog
        table: `str`
            Source table

        Returns
        ----------
        result: `pd.df`
            Result of the query, as pandas dataframe
        """
        cols = [time_col, flux_col, err_col]

        if to_mag:
            flux_query, flux_label = self.flux_to_mag([flux_col])
            flux_col = flux_label[0]
            if err_col is not None:
                err_query, err_label = self.flux_to_mag([err_col])
                err_col = err_label[0]

            query_cols = [time_col] + flux_query + err_query
            cols = [time_col, flux_col, err_col]

        else:
            query_cols = cols

        if add_cols is not None:
            cols = cols + add_cols
            query_cols = query_cols + add_cols

        idx_cols = ["diaObjectId", "filterName"]

        result = pd.DataFrame(columns=idx_cols + cols)
        select_cols = ",".join(idx_cols) + "," + ",".join(query_cols)
        str_ids = [str(obj_id) for obj_id in ids]
        id_list = "(" + ",".join(str_ids) + ")"

        result = self.query_tap(
            f"SELECT {select_cols} "
            f"FROM {catalog}.{table} "
            f"WHERE {id_field} IN {id_list}",
            maxrec=maxrec,
        )
        index = self._build_index(result["diaObjectId"], result["filterName"])
        result.index = index
        result = result[cols].sort_index()
        self.result = result

        self._time_col = time_col
        self._flux_col = flux_col
        self._err_col = err_col

        return result

    def to_timeseries(self, target, id_col=None, time_col=None,
                      flux_col=None, err_col=None, band_col=None):
        """Construct a timeseries object from one target object_id, assumes that the result
        is a collection of lightcurves (output from query_ids)

        Parameters
        ----------
        dataframe: `pd.df`
            Ensemble object
        target: `int`
            Id of a source to be extracted

        Returns
        ----------
        ts: `timeseries`
            Timeseries for a single object
        """

        # Without a specified column, use defaults (which are updated via query_id)
        if id_col is None:
            id_col = self._id_col
        if time_col is None:
            time_col = self._time_col
        if flux_col is None:
            flux_col = self._flux_col
        if err_col is None:
            err_col = self._err_col
        if band_col is None:
            band_col = self._band_col

        df = self.data
        df = df[(df[self._id_col] == target)].to_pandas_df()
        ts = timeseries()._from_ensemble(data=df, object_id=target, time_label=time_col,
                                         flux_label=flux_col, err_label=err_col, band_label=band_col)
        return ts

    def flux_to_mag(self, cols):
        """Transforms TAP query from fluxes to magnitudes

         Parameters
        ----------
        cols: `list` of `str`
            List of columns to be queried, containing Flux in the name

        Returns:
        ----------
        cols_mag `list` of `str`
            List of columns to be queried, replaced with magnitudes
        cols_label 'list' of 'str'
            List of column labels for the returned quantities
        """

        cols_mag = []
        cols_label = []
        for col in cols:
            pos_flux = col.find("Flux")
            if pos_flux == -1:
                cols_mag.append(col)
                cols_label.append(col)
            else:
                pre_var, post_var = col[:pos_flux], col[pos_flux + len("Flux"):]
                flux_str = pre_var + "Flux"
                mag_str = pre_var + "AbMag"
                if col.find("Err") != -1:
                    flux_str_err = pre_var + "Flux" + post_var
                    mag_str_err = pre_var + "AbMag" + post_var
                    cols_mag.append(
                        "scisql_nanojanskyToAbMagSigma("
                        + flux_str
                        + ","
                        + flux_str_err
                        + ") AS "
                        + mag_str_err
                    )
                    cols_label.append(mag_str_err)
                else:
                    cols_mag.append(
                        "scisql_nanojanskyToAbMag(" + flux_str + ") AS " + mag_str
                    )
                    cols_label.append(mag_str)
        return cols_mag, cols_label

    def _build_index(self, obj_id, band):
        """Build pandas multiindex from object_ids and bands"""
        count_dict = {}
        idx = []
        for o, b in zip(obj_id, band):
            if f"{o},{b}" in count_dict:
                idx.append(count_dict[f"{o},{b}"])
                count_dict[f"{o},{b}"] += 1
            else:
                idx.append(0)
                count_dict[f"{o},{b}"] = 1
        tuples = zip(obj_id, band, idx)
        index = pd.MultiIndex.from_tuples(tuples, names=["object_id", "band", "index"])
        return index
