import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')
import polars as pl
import numpy as np
from const import constants
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
from typing import List
import joblib


class CustomOneHotEncoding:
    def __init__(self, single_val_cols, multi_val_cols, vocabthresh=100, cumprob_inc_thresh=0.99,
                 null_vals=[]):
        '''
            multi_val_cols are expected to be passed in a list datastructure. null values in the data should be mapped to empty list
            for multi_val_cols
        '''
        self.single_val_cols = single_val_cols
        self.multi_val_cols = multi_val_cols
        self.vocab_thr = vocabthresh
        self.inc_thr = cumprob_inc_thresh
        self.fitted = False
        self.null_vals = null_vals

    def _build_vocab_col(self, dd, colname):
        dd = dd.with_columns(
            pl.col('count').cum_sum().alias('cumsum')
        ).with_columns(
            (pl.col('cumsum')/pl.col('cumsum').max()).alias('prob')
        ).with_columns(
            (pl.col('prob')<=self.inc_thr).alias('included')
        )
        inc_df = dd.filter(pl.col('included'))
        inc_vals = inc_df[colname].to_list()
        inc_dict = {}
        inc_dict[f'null'] = 0
        if self.inc_thr < 1.0:
            inc_dict[f'unk'] = 1
        for c in inc_vals:
            inc_dict[c] = len(inc_dict)
        return inc_dict

    def _build_vocab(self, df):
        multival_vocab_ = {}
        for colname in self.multi_val_cols:
            dd = df.explode(colname)[colname].value_counts().sort(by='count', descending=True)
            vocab = self._build_vocab_col(dd, colname)
            multival_vocab_[colname] = vocab
            
        singleval_vocab_ = {}
        for colname in self.single_val_cols:
            vocab = self._build_vocab_col(df, colname)
            singleval_vocab_[colname] = vocab
    
        return singleval_vocab_, multival_vocab_

    def fit(self, X: pl.DataFrame, y=None):
        for col in self.single_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a single valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
        
        for col in self.multi_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a multi valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
            
        # Create dictionaries
        self.singleval_vocab_, self.multi_val_cols = self._build_vocab(df)
        self.fitted  = True
        
        return self
    
    def _transform_single_val(series, vocab):
        X = np.zeros((len(series), len(vocab)), dtype=np.uint16)
        for idx, v in enumerate(series):
            if v in null_vals:
                X[idx, 0] += 1
            else:
                X[idx, vocab.get(v, 1)] += 1
        return X
    
    def transform(self, X, y=None):
        if not self.fitted:
            raise ValueError("You need to run .fit() method first")
        X_    
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

if __name__ == '__main__':
    pass