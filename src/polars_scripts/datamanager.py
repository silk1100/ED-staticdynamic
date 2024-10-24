import os
import sys

# MAIN_DIR = os.getenv("EDStaticDynamic")
# sys.path.insert(1, MAIN_DIR)

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import polars as pl
import numpy as np
from const import constants
import joblib
import datetime as dt


class CustomCrossFold:
    '''
     
    '''
    def __init__(self, train_period_in_days, test_period_in_days, step_in_days, date_col, max_folds=None):
        self.tr = int(train_period_in_days)
        self.te = int(test_period_in_days)
        self.step = step_in_days
        if self.step is None:
            self.step = self.te
        self.step = int(self.step)
        self.date_col = date_col

    def get_latest_batch(self, X, n_days=None):
        start_date = X[self.date_col].min()
        end_date = X[self.date_col].max()
        date_ranges = pl.datetime_range(start_date, end_date, interval=f'{self.step}d', eager=True)
        date_ranges = list(map(lambda x: x.date(), date_ranges))

        X = X.with_columns(
            pl.col(self.date_col).dt.date().alias(f'{self.date_col}_date')
        )

        if n_days is None:
            n_days = self.tr

        Xtre = X.filter( (pl.col(f'{self.date_col}_date')<=end_date)&(pl.col(f'{self.date_col}_date')>=end_date-dt.timedelta(days=n_days)) )
        return Xtre.drop(f'{self.date_col}_date')
        

    def reverse_split(self, X, y=0, max_iter=None):
        start_date = X[self.date_col].min()
        end_date = X[self.date_col].max()
        date_ranges = pl.datetime_range(start_date, end_date, interval=f'{self.step}d', eager=True)
        date_ranges = list(map(lambda x: x.date(), date_ranges))
        X = X.with_columns(
            pl.col(self.date_col).dt.date().alias(f'{self.date_col}_date')
        )
        date_ranges = date_ranges[::-1]
        idx = 0
        for idx in range(0, len(date_ranges)):
            Xte = X.filter( (pl.col(f'{self.date_col}_date')<=date_ranges[idx])&(pl.col(f'{self.date_col}_date')>=date_ranges[idx]-dt.timedelta(days=self.te)) )
            Xtr = X.filter( (pl.col(f'{self.date_col}_date')<date_ranges[idx]-dt.timedelta(days=self.te))&
                          (pl.col(f'{self.date_col}_date')>=date_ranges[idx]-dt.timedelta(days=self.tr)-dt.timedelta(days=self.te)) )
            yield Xtr.drop(f'{self.date_col}_date'), Xte.drop(f'{self.date_col}_date')
            idx += 1
            if max_iter is not None and idx>=max_iter:
                break
        
    
    def split(self, X,  y=0, max_iter=None):
        start_date = X[self.date_col].min()
        end_date = X[self.date_col].max()
        date_ranges = pl.datetime_range(start_date, end_date, interval=f'{self.step}d', eager=True)
        date_ranges = list(map(lambda x: x.date(), date_ranges))
        X = X.with_columns(
            pl.col(self.date_col).dt.date().alias(f'{self.date_col}_date')
        )
        idx = 0
        for idx in range(0, len(date_ranges)):
            Xtr = X.filter( (pl.col(f'{self.date_col}_date')>=date_ranges[idx])&
                          (pl.col(f'{self.date_col}_date')<=date_ranges[idx]+dt.timedelta(days=self.tr)))
            Xte = X.filter( (pl.col(f'{self.date_col}_date')>date_ranges[idx]+dt.timedelta(days=self.tr))&
                          (pl.col(f'{self.date_col}_date')<=date_ranges[idx]+dt.timedelta(days=self.tr+self.te)))
            yield Xtr.drop(f'{self.date_col}_date'), Xte.drop(f'{self.date_col}_date')
            idx += 1
            if max_iter is not None and idx>=max_iter:
                break

class CustomEDSampling:
    def __init__(self, max_num_pts, is_time_based, sample_col, step):
        '''
        - max_num_pts: can denote number of events if `is_time_based` is False, or can be the longest number 
        of minutes in the ED if `is_time_based` is True 
        - step: can denote number of events if `is_time_based` is False, or can be the longest number 
        of minutes in the ED if `is_time_based` is True 

        '''
        self.is_time_based = is_time_based
        self.max_num_pts = max_num_pts
        self.sample_col = sample_col
        self.step = step

    def split(self, X, y=None):
        X = X.with_columns(
            (pl.col(self.sample_col)-pl.col(self.sample_col).first()).dt.minutes().over('PAT_ENC_CSN_ID').alias('minutes')
        )
        time_range = np.arange(self.step, self.max_num_pts+self.step, self.step)
        for t in time_range:
            Xp = X.filter(pl.col('minutes')<=t)
            yield Xp


if __name__ == "__main__":
    with open(constants.CLEAN_DATA, 'rb') as f:
        df = joblib.load(f)
    ccf = CustomCrossFold(30*6, 30, 30*2, 'Arrived_Time')
    ceds = CustomEDSampling(60*48, True, "Calculated_DateTime", 30)
    df = df.with_columns(
        (pl.col("Calculated_DateTime")-pl.col("Calculated_DateTime").first()).dt.minutes().over('PAT_ENC_CSN_ID').alias('minutes')
    )
    time_range = np.arange(30, 60*24*2+30, 30)

    for Xtr, Xte in ccf.split(df):
        for t in time_range:
            Xtr_t = Xtr.filter(pl.col('minutes')<=t)
            Xte_t = Xte.filter(pl.col('minutes')<=t)
            
            x = 0