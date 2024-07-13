import os
import sys

MAIN_DIR = os.getenv("EDStaticDynamic")
sys.path.insert(1, MAIN_DIR)

import polars as pl
import numpy as np
from const import constants
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

    def _vocab_from_list(self, inc_series):
        inc_vals = set(inc_series)
        for v in self.null_vals:
            if v in inc_vals: inc_vals.remove(v)
        inc_dict = {}
        inc_dict[f'null'] = 0
        if self.inc_thr < 1.0:
            inc_dict[f'unk'] = 1
        for c in inc_vals:
            inc_dict[c] = len(inc_dict)
        return inc_dict

    def _build_vocab_col(self, dd, colname):
        if len(dd) <= self.vocab_thr:
            return self._vocab_from_list(dd[colname])

        dd = dd.with_columns(
            pl.col('count').cum_sum().alias('cumsum')
        ).with_columns(
            (pl.col('cumsum')/pl.col('cumsum').max()).alias('prob')
        ).with_columns(
            (pl.col('prob')<=self.inc_thr).alias('included')
        )
        inc_df = dd.filter(pl.col('included'))

        return self._vocab_from_list(inc_df[colname])

    def _build_vocab(self, df):
        multival_vocab_ = {}
        for colname in self.multi_val_cols:
            dd = df.explode(colname)[colname].value_counts().sort(by='count', descending=True)
            vocab = self._build_vocab_col(dd, colname)
            multival_vocab_[colname] = vocab
            
        singleval_vocab_ = {}
        for colname in self.single_val_cols:
            dd = df[colname].value_counts().sort(by='count', descending=True)
            vocab = self._build_vocab_col(dd, colname)
            singleval_vocab_[colname] = vocab
    
        return singleval_vocab_, multival_vocab_

    def fit(self, X: pl.DataFrame, y=None):
        for col in self.single_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a single valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
        
        for col in self.multi_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a multi valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
            
        # Create dictionaries
        self.singleval_vocab_, self.multi_val_vocab_ = self._build_vocab(X)
        self.fitted  = True
        
        return self
    
    def _transform_single_val(self, series, vocab):
        X = np.zeros((len(series), len(vocab)), dtype=np.uint16)
        for idx, v in enumerate(series):
            if v in constants.NULL_LIST:
                X[idx, 0] += 1
            else:
                X[idx, vocab.get(v, 1)] += 1
        return X

    def _transform_mult_val(self, series, vocab):
        X = np.zeros((len(series), len(vocab)), dtype=np.uint16)
        for idx, ll in enumerate(series):
            if ll is None or len(ll) == 0:
                X[idx, 0] += 1
                continue
            for v in ll:
                X[idx, vocab.get(v, 1)] += 1
        return X
    
    def _update_colnames(self, colname, vocab):
        colnames = [""]*len(vocab)
        for k, v in vocab.items():
            colnames[v] = f"{colname}_{k}"
        return colnames

    def transform(self, X, y=None):
        if not self.fitted:
            raise ValueError("You need to run .fit() method first")

        X_singval_res = []
        s_colnames = []
        for col, vocab in self.singleval_vocab_.items():
            s_colnames.extend(self._update_colnames(col, vocab))
            X_singval_res.append(self._transform_single_val(X[col], vocab))
        X_singval = np.hstack(X_singval_res) if len(X_singval_res)>0 else np.empty((len(X), 0), dtype=np.uint16)

        X_multval_res = []
        m_colnames = []
        for col, vocab in self.multi_val_vocab_.items():
            m_colnames.extend(self._update_colnames(col, vocab))
            X_multval_res.append(self._transform_mult_val(X[col], vocab))
        X_multval = np.hstack(X_multval_res) if len(X_multval_res)>0 else np.empty((len(X), 0), dtype=np.uint16)

        df_singval = pl.DataFrame(X_singval, schema=s_colnames)
        df_multval = pl.DataFrame(X_multval, schema=m_colnames)
        return pl.concat([df_singval, df_multval], how='horizontal')

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class CustomLabelEncoding:
    def __init__(self, single_val_cols, vocabthresh=100, cumprob_inc_thresh=0.99,
                 null_vals=[]):
        self.single_val_cols = single_val_cols
        self.vocab_thr = vocabthresh
        self.inc_thr = cumprob_inc_thresh
        self.fitted = False
        self.null_vals = null_vals

    def _vocab_from_list(self, inc_series):
        inc_vals = set(inc_series)
        for v in self.null_vals:
            if v in inc_vals: inc_vals.remove(v)
        inc_dict = {}
        inc_dict[f'null'] = 0
        if self.inc_thr < 1.0:
            inc_dict[f'unk'] = 1
        for c in inc_vals:
            inc_dict[c] = len(inc_dict)
        return inc_dict

    def _build_vocab_col(self, dd, colname):
        if len(dd) <= self.vocab_thr:
            return self._vocab_from_list(dd[colname])

        dd = dd.with_columns(
            pl.col('count').cum_sum().alias('cumsum')
        ).with_columns(
            (pl.col('cumsum')/pl.col('cumsum').max()).alias('prob')
        ).with_columns(
            (pl.col('prob')<=self.inc_thr).alias('included')
        )
        inc_df = dd.filter(pl.col('included'))

        return self._vocab_from_list(inc_df[colname])

    def _build_vocab(self, df):
        singleval_vocab_ = {}
        for colname in self.single_val_cols:
            dd = df[colname].value_counts().sort(by='count', descending=True)
            vocab = self._build_vocab_col(dd, colname)
            singleval_vocab_[colname] = vocab
    
        return singleval_vocab_

    def fit(self, X, y=None):
        for col in self.single_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a single valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
            
        # Create dictionaries
        self.singleval_vocab_= self._build_vocab(X)
        self.fitted  = True
        
        return self

    def _transform_mult_val(self, series, vocab):
        X = np.zeros((len(series), len(vocab)), dtype=np.uint16)
        for idx, ll in enumerate(series):
            if ll is None or len(ll) == 0:
                X[idx, 0] += 1
                continue
            for v in ll:
                X[idx, vocab.get(v, 1)] += 1
        return X
    
    def _update_colnames(self, colname, vocab):
        colnames = [""]*len(vocab)
        for k, v in vocab.items():
            colnames[v] = f"{colname}_{k}"
        return colnames
    
    def _map_values_(self, val, column):
        return self.singleval_vocab_[column].get(val, 1)

    def transform(self, X: pl.DataFrame, y=None):
        if not self.fitted:
            raise ValueError("You need to run .fit() method first")

        for c in self.singleval_vocab_:
            if X[c].dtype == pl.String:
                cond = (pl.col(c).is_in(self.null_vals))|(pl.col(c).is_null())
            else:
                cond = pl.col(c).is_null()
            expr = pl.when(cond).then(pl.lit(0)).\
                otherwise(pl.col(c).map_elements(lambda x:self._map_values_(x, c), return_dtype=pl.UInt16)).alias(f'{c}_le') 
            X = X.with_columns(
                expr
            )

        return X.select([f'{c}_le' for c in self.singleval_vocab_])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


def select_pat_by_arrtime(df, pat_col, arrtime_col):
    df_pat_arr = df.groupby(pat_col).agg(pl.col(arrtime_col).last().alias('arr_time')).sort(by='arr_time')
    n_train_pat = int(len(df_pat_arr)*0.8)
    train_pat = df_pat_arr[:n_train_pat][pat_col]
    test_pat = df_pat_arr[n_train_pat:][pat_col]
    st_tr_time = df_pat_arr['arr_time'][0]
    en_tr_time = df_pat_arr['arr_time'][n_train_pat-1]
    st_te_time = df_pat_arr['arr_time'][n_train_pat]
    en_te_time = df_pat_arr['arr_time'][-1]
    print(f'Training data from: {st_tr_time} to {en_tr_time}\nTesting data from {st_te_time} to {en_te_time} ...')
    return train_pat, test_pat
    
if __name__ == '__main__':
    with open(constants.CLEAN_DATA, 'rb') as f:
        df = joblib.load(f)
    df_70_static = df.filter(pl.col("event_idx")<70).select(constants.static_cols+constants.id_cols)
    cu = CustomOneHotEncoding(
        single_val_cols=constants.static_singleval_cat_cols,
        multi_val_cols=constants.static_multval_cat_cols,
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST,
        vocabthresh=100
    )
    cle = CustomLabelEncoding(
        single_val_cols=constants.static_singleval_cat_cols,
        vocabthresh=100,
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST
    )
    df_70_static = df_70_static.sort(by='Arrived_Time')
    train_pat, test_pat = select_pat_by_arrtime(df_70_static, 'PAT_ENC_CSN_ID', 'Arrived_Time')
    df_train = df_70_static.filter(pl.col("PAT_ENC_CSN_ID").is_in(train_pat))
    df_test = df_70_static.filter(pl.col("PAT_ENC_CSN_ID").is_in(test_pat))
    dftr_le = cle.fit_transform(df_train)
    dfte_le = cle.transform(df_test)
    x = 0
    cu = CustomOneHotEncoding(
        single_val_cols=['A', 'B'],
        multi_val_cols=['C'],
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST,
        vocabthresh=100
    )
    cue = CustomLabelEncoding(
        single_val_cols=['A', 'B'],
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST,
        vocabthresh=100
    )
    dff = pl.DataFrame(
        {
            'A':['a', 'b', 'null', None, 'z', 'aa'],
            'B':[12, 12, None, 1, 15, 2],
            'C': [[], ['a', 'c'], ['a'], ['z'], ['a', 'b', 'c'], None]
        }
    )
    dff_k = pl.DataFrame(
        {
            'A':['a', 'b', 'null', None, 'zz', '1a'],
            'B':[12, 12, None, 1, 19, 2],
            'C': [[], ['a', 'c'], ['a'], ['k'], ['a', 'k', 'c'], None]
        }
    )

    dff_ohe = cu.fit_transform(dff)
    dff_k_ohe = cu.transform(dff_k)
    dff_le = cue.fit_transform(dff)
    dffk_le = cue.transform(dff_k)
    x = 0 

    
    
