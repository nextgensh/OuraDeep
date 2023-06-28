#!/usr/bin/env python3

from sensorfabric.athena import athena
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from math import floor, ceil


class TemperatureDataset(Dataset):
    """
    Description
    -----------
    This base class returns temperature dataset for the given pids.
    Currently it returns *1 data point per participant*. This means that
    all temperature data for a single participant is concatenated and returned
    as a single vector.

    TODO: Support groupBy = days.

    X And Y for dataset.
    --------------------
    X : [Xt, Xt+1, Xt+2, .....]
        Xt => Skin tempeature at time t.
        A tensor of raw skin temperature values in Celcius.
        Values are sorted in ascending order inside the tensor. That is the
        earliest timestamp is first. If you would like to have this reveresed
        then set reverseOrder=True when creating this dataset.
    Y : [Yt, Yt+1, Yt+2, .....]
        Yt => (gestational age in days) + abs(labor offset in days) for each corresponding point Xt.
        A vector of the (gestational age in days) + abs(labor offset in days) for each point Xt in the
        input vector Xt.

    Parameters
    -----------
    1. database : The name of the database to pull minute temperature data from.
    2. offlineCache : True (default) to cache results offline. It is highly recommended to leave this on
                      unless you have data that changes frequently.
    3. groupBy : 'pid' to group input and output vectors as a single datapoint per participant.
                 'day' (not implemented) to group the datapoints by each day per participant.
    4. reverseOrder : If True then input and output vectors are ordered in a descending order of time.
                      False (default) then they are organized in ascending order of time.
    4. skipNan : If True it drops data points (entire pid records) which have nan values in
                 any of the series. False it keeps them (False).
    5. pids : An array of pid values to use for pull temperature data.

    Returns
    --------
    Indexing elements of this class will return a tuple of (X: tensor, Y: tensor)
    """

    def __init__(self,
                 database='',
                 offlineCache=True,
                 groupBy='pid',
                 reverseOrder=False,
                 skipNan=False,
                 pids=[]):

        self.offlineCache = offlineCache
        self.db = athena(database=database, offlineCache=self.offlineCache)
        self.pids = pids
        self.length = len(self.pids)
        # We build the offline cache by fetching the temperature data for each pid.
        self.query = """
                    WITH sublabor AS (
                SELECT pid,
                    labordate,
                    laborprogress
                FROM labor
                WHERE pid = {pid}
            ),
            gestation AS (
                SELECT redcap.pid, (redcap.gestational_age_weeks*7+redcap.gestational_age_days) as gdays
                FROM redcap
                WHERE pid = {pid}
                    AND redcap.gestational_age_days IS NOT NULL
            ),
            jointable AS (
                SELECT temperature.pid AS pid,
                    temperature.unixtimestamp as utime,
                    temperature.skintemp AS skintemp,
                    sublabor.labordate AS labordate,
                    sublabor.laborprogress AS laborprogress,
                    CAST(
                        SPLIT_PART(
                            CAST(
                                (
                                    date(from_unixtime(temperature.unixtimestamp)) - sublabor.labordate
                                ) as VARCHAR
                            ),
                            ' ',
                            1
                        ) as INT
                    ) as laboroffset
                FROM temperature
                    FULL JOIN sublabor ON sublabor.pid = temperature.pid
                WHERE temperature.pid = {pid}
            ),
            laborskin AS (
                SELECT jointable.pid,
                    utime,
                    skintemp,
                    laboroffset,
                    gestation.gdays + ABS(laboroffset) as adjustedoffset
                FROM jointable
                FULL JOIN gestation ON gestation.pid = jointable.pid
            )
            SELECT pid, utime, skintemp, adjustedoffset, laboroffset FROM laborskin
            WHERE laboroffset <= 0
            ORDER BY utime ASC
        """
        # Since some pid don't have data we store the ones with data in this new array.
        updated_pid = []
        for pid in self.pids:
            results = self.db.execQuery(self.query.format(pid=pid), cached=self.offlineCache)
            #print('Preparing pid - {pid}'.format(pid=pid))
            # We have some pids with nan values, that are currently skipped if skipNan=True.
            # pid = [15, 16] are treated in that maner.
            dropFlag = False
            if results.shape[0] > 0:
                if results.isna().any().any():
                    if skipNan:
                        print('Dropped pid = {pid} because it contained Nan'.format(pid=pid))
                        dropFlag = True
                    else:
                        print('Warning : pid = {} has nan values. You have left skipNan=False. This will return nan values which may lead to further errors unless handled.')
            else:
                print('No data found for pid = {pid}. Dropped them'.format(pid=pid))
                dropFlag = True

            if not dropFlag:
                updated_pid += [pid]

        self.pids = updated_pid

    def __len__(self):
        return self.pids.length

    def __getitem__(self, idx):
        # Get the pid corresponding to the index inside the pid list.
        pid = self.pids[idx]
        # Run a query that will pull from cache results.
        result = self.db.execQuery(self.query.format(pid=pid), cached=self.offlineCache)
        X = result['skintemp'].values
        Y = result['adjustedoffset'].values
        laboroffset = result['laboroffset'].values

        # Convert them into tensors before sending them back.
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y, laboroffset, pid

class TemperatureTrain(TemperatureDataset):
    """
    Convinience wrapper to get training data for temperature.
    Extends the base class TemperatureDataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TemperatureVal(TemperatureDataset):
    """
    Convinience wrapper to get validation data for temperature.
    Extends the base class TemperatureDataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TemperatureUtils():
    """
    Utility class for helper functions.
    """
    def __init__(self, database='', offlineCache=False):
        self.database = database
        self.offlineCache = offlineCache
        self.db = athena(database=database, offlineCache=self.offlineCache)

    def getPids(self):
        query = """
        SELECT redcap_pid.pid FROM redcap_pid
            INNER JOIN temperature_pid ON temperature_pid.pid = redcap_pid.pid
        """
        results = self.db.execQuery(query, cached=self.offlineCache)
        pids = results['pid'].values

        return np.array(pids)

    def getSpontaneousPids(self):
        """
        Method returns pids for only those participants that underwent spontaneous birth.
        """
        query = """
        WITH labor_spon AS (
            SELECT pid
            from redcap
            WHERE laborprogress = 1.0
                OR laborprogress = 2.0
        ),
        temperature_pid AS (
            SELECT DISTINCT(pid) as pid
            FROM temperature
        )
        SELECT DISTINCT(labor_spon.pid)
        FROM labor_spon
            INNER JOIN temperature ON temperature.pid = labor_spon.pid
        """

        results = self.db.execQuery(query, cached=self.offlineCache)
        pids = results['pid'].values

        return np.array(pids)

    def split(self, pids, train=0.8, val=0.2):
        np.random.shuffle(pids)
        length = pids.size

        return (pids[0 : floor(train*length)],
                pids[floor(train*length) : max(floor(train*length)+ceil(val*length), length)])
