import os
import sys

MAIN_DIR = os.getenv("EDStaticDynamic")
sys.path.insert(1, MAIN_DIR)

import polars as pl
import numpy as np
from const import constants
import joblib
import time
from warnings import warn

class CustomDynamicOneHotEncoding:
    def __init__(self, single_val_cols, multi_val_cols, dep_col_dict, num_cols=[],
                 id_col ='PAT_ENC_CSN_ID',
                 skip_indp_val={'vitals'},
                 vocabthresh=100, cumprob_inc_thresh=0.99, num_norm_methods = {},
                 null_vals=[]):
        '''
            multi_val_cols are expected to be passed in a list datastructure. null values in the data should be mapped to empty list
            for multi_val_cols
        '''
        self.id_col = id_col
        self.single_val_cols = single_val_cols
        self.multi_val_cols = multi_val_cols
        self.dep_col_dict = dep_col_dict
        self.num_cols = num_cols
        self.num_norm_method = num_norm_methods
        self._adjust_num_norm_method(['std', 'minmax'])

        self.vocab_thr = vocabthresh
        self.inc_thr = cumprob_inc_thresh
        self.skip_indp_val = skip_indp_val
        self.fitted = False
        self.null_vals = null_vals


    def _adjust_num_norm_method(self, allowed_methods: list):
        for c in self.num_cols:
            if c not in self.num_norm_method:
                self.num_norm_method[c] = 'std'
                warn(f"{c} is supposed to be processed as a numerical variable. However the processing method is not "
                     "specified in the num_norm_method dictionary. Therefore, it is assigned to 'std' ...")
            elif self.num_norm_method[c] not in allowed_methods:
                raise ValueError(f"Numerical preprocessing methods supproted are {allowed_methods}. "
                                 f"{self.num_norm_method[c]} is passed for column {c} ...")

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

    def _build_vocab_for_dependent_cols(self, df:pl.DataFrame,
                                        dep_dict={'Type_NORM':['EVENT_NORM_NORM']}):
        '''
        indep_cols and dep_cols must be of the same size.
        Dependencies assumes
        ''' 
        dep_dict_res = {}
        for indep, dep_list in dep_dict.items():
            vals = df[indep].value_counts().sort(by='count', descending=True)[indep]
            dep_dict_res[indep] = {}
            for c in dep_list:
                dep_dict_res[indep][c] = {}
                for val in vals:
                    dff = df.filter(pl.col(indep)==val)
                    if dff[c].dtype == pl.List(pl.String):
                        colval = dff[c].explode().value_counts().sort(by='count', descending=True)
                    else:
                        colval = dff[c].value_counts().sort(by='count', descending=True)
                    dep_dict_res[indep][c][val] = self._build_vocab_col(colval, c)
        return dep_dict_res

    def _calc_norm(self, X:pl.DataFrame):
        num_cols = {}
        for c in self.num_cols:
            if self.num_norm_method[c] == 'std':
                num_cols[c] = (X[c].mean(), X[c].std(ddof=1), 'std')
            elif self.num_norm_method[c] == 'minmax':
                num_cols[c] = (X[c].min(), X[c].max(), 'minmax')
            else:
                raise ValueError(f"Normalization methods supported are ['std', 'minmax']. {self.num_norm_method} is passed ...")
        
        return num_cols

    def fit(self, X: pl.DataFrame, y=None):
        for col in self.single_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a single valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
        
        for col in self.multi_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a multi valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'

        for col in self.num_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a numerical valued column using {self.num_norm_method}. However it doesnt exist in the passed X'

        for col in self.dep_col_dict:
            assert col in X.columns, f'{col} is supposed to be processed as an independent column using CustomOneHotEncoding. However it doesnt exist in the passed X'
            for c in self.dep_col_dict[col]:
                assert c in X.columns, f'{c} is supposed to be processed as a dependent valued column on {col} column using CustomOneHotEncoding. However it doesnt exist in the passed X'
     
        # Create dictionaries
        self.singleval_vocab_, self.multi_val_vocab_ = self._build_vocab(X)
        self.num_dict = self._calc_norm(X)
        
        '''
        self.dep_vocab[{indepent_colname}][{dependent_colname}][{indepent_value}][{'depenedent_vocab'}]
        '''
        self.dep_vocab_ = self._build_vocab_for_dependent_cols(X, self.dep_col_dict)
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

    def _transform_dep_cols(self,
                            df:pl.DataFrame,
                            indep_col: str,
                            dep_col: str,
                            vocab_dict:dict,
                            prefix:str
                            ):
        cols = []
        X_list = []
        for indkey, dep_dict in vocab_dict.items():
            if indkey in self.skip_indp_val:
                continue
            X = np.zeros((len(df), len(dep_dict)), dtype=np.uint16)
            cols.extend(self._update_colnames(indkey, dep_dict, prefix))
            for idx, row in enumerate(df.iter_rows(named=True)):
                if row[indep_col] != indkey: continue
                if row[dep_col] in self.null_vals:
                    X[idx, 0] += 1
                else:
                    X[idx, dep_dict.get(row[dep_col], 1)] += 1
            X_list.append(X)
        # Xstacked = np.hstack(X_list, dtype=np.uint16)
        Xstacked = np.hstack(X_list)#, dtype=np.uint16)
        return Xstacked, cols

    def _update_colnames(self, colname, vocab, prefix=""):
        colnames = [""]*len(vocab)
        for k, v in vocab.items():
            if len(prefix)==0:
                colnames[v] = f"{colname}_{k}"
            else:
                colnames[v] = f"{prefix}_{colname}_{k}"
        return colnames


    def _transfom_num_cols(self, X):
        for c, (v1, v2, method) in self.num_dict.items():
            if method == 'std':
                if v2 == 0:
                    X = X.with_columns(
                        pl.lit(0).alias(f'{c}_NUMNORM')
                    )
                else: 
                    X = X.with_columns(
                        ((pl.col(c)-v1)/(v2+(1e-9))).alias(f'{c}_NUMNORM')
                    )
            elif method == 'minmax':
                if v1 == v2:
                    X = X.with_columns(
                        pl.lit(0).alias(f'{c}_NUMNORM')
                    )
                else:
                    X = X.with_columns(
                        ((pl.col(c)-v1)/(v2-v1+1e-9)).alias(f'{c}_NUMNORM')
                    )
        return X
        

    def transform(self, X, y=None):
        if not self.fitted:
            raise ValueError("You need to run .fit() method first")

        if len(self.num_cols) > 0:
            X = self._transfom_num_cols(X)

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

        X_dep_res = []
        dep_colnames = []
        for col, dep_col_dict in self.dep_vocab_.items():
            for dep_col, indep_dep_dict in dep_col_dict.items():
                prefix = f'{col}__{dep_col}_'
                X_dr, coln = self._transform_dep_cols(X,
                                            col,
                                            dep_col,
                                            indep_dep_dict,
                                            prefix)
                X_dep_res.append(X_dr)
                dep_colnames.extend(coln)
        X_depval = np.hstack(X_dep_res) if len(X_dep_res)>0 else np.empty((len(X), 0), dtype=np.uint16)

        df_depval = pl.DataFrame(X_depval, schema=dep_colnames)
        df_singval = pl.DataFrame(X_singval, schema=s_colnames)
        df_multval = pl.DataFrame(X_multval, schema=m_colnames)

        if len(self.num_cols) > 0:
            self.num_norm_cols_ = [col+"_NUMNORM" for col in self.num_cols]
            df_transformed = pl.concat([df_singval, df_multval, df_depval, X.select(self.num_norm_cols_)], how='horizontal')
        else:
            df_transformed = pl.concat([df_singval, df_multval, df_depval], how='horizontal')
        # df_transformed = pl.concat([X, df_singval, df_multval, df_depval], how='horizontal')
        # df_transformed.group_by(self.id_col).agg(
        #     [
        #         pl.col(c).sum().alias(c)
        #         for c in dep_colnames+s_colnames+m_colnames
        #     ]+[
        #         pl.col(c).last().alias(c)
        #         for c in dep_colnames+s_colnames+m_colnames
        #     ]
        # )
        
        return df_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class CustomDynamicLabelEncoding:
    def __init__(self, single_val_cols, dep_col_dict={}, vocabthresh=100, cumprob_inc_thresh=0.99,
                 null_vals=[]):
        self.single_val_cols = single_val_cols
        self.vocab_thr = vocabthresh
        self.dep_col_dict = dep_col_dict
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

    def _build_vocab_for_dependent_cols(self, df:pl.DataFrame,
                                        dep_dict={'Type_NORM':['EVENT_NORM_NORM']}):
        '''
        indep_cols and dep_cols must be of the same size.
        Dependencies assumes
        ''' 
        dep_dict_res = {}
        for indep, dep_list in dep_dict.items():
            vals = df[indep].value_counts().sort(by='count', descending=True)[indep]
            dep_dict_res[indep] = {}
            for c in dep_list:
                dep_dict_res[indep][c] = {}
                for val in vals:
                    dff = df.filter(pl.col(indep)==val)
                    if dff[c].dtype == pl.List(pl.String):
                        colval = dff[c].explode().value_counts().sort(by='count', descending=True)
                    else:
                        colval = dff[c].value_counts().sort(by='count', descending=True)
                    dep_dict_res[indep][c][val] = self._build_vocab_col(colval, c)
        return dep_dict_res

    def fit(self, X, y=None):
        for col in self.single_val_cols:
            assert col in X.columns, f'{col} is supposed to be processed as a single valued column using CustomOneHotEncoding. However it doesnt exist in the passed X'
            
        # Create dictionaries
        self.singleval_vocab_= self._build_vocab(X)
        self.dep_vocab_ = self._build_vocab_for_dependent_cols(X, self.dep_col_dict)
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
        
        dep_cols = []
        for indep, dep_dict in self.dep_vocab_.items():
            for dep_col, val_dict in dep_dict.items():
                newcolname = f'{indep}__{dep_col}_le'
                dep_cols.append(newcolname)
                X = X.with_columns(
                    pl.lit(0).cast(pl.UInt16).alias(newcolname)
                )
                for indep_val, vocab_dict in val_dict.items():
                    X = X.with_columns(
                        pl.when(
                            (pl.col(indep)==indep_val)&(pl.col(dep_col).is_in(self.null_vals))
                        ).then(pl.lit(0)).when(
                            pl.col(indep)==indep_val
                        ).then(
                            (pl.col(dep_col).map_elements(lambda x:vocab_dict.get(x, 1), return_dtype=pl.UInt16))
                        ).otherwise(
                            pl.col(newcolname)
                        ).alias(newcolname)
                    )

        return X.select([f'{c}_le' for c in self.singleval_vocab_]+dep_cols)

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
    df_70_static = df.filter(pl.col("event_idx")<70).\
        select(constants.static_cols+constants.id_cols+['Type_NORM', 'EVENT_NAME_NORM'])
    cu = CustomDynamicOneHotEncoding(
        single_val_cols=constants.static_singleval_cat_cols,
        multi_val_cols=constants.static_multval_cat_cols,
        dep_col_dict={'Type_NORM':['EVENT_NAME_NORM']},
        skip_indp_val={'vitals'},
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
    # df_70_static = df_70_static.sort(by='Arrived_Time')
    # train_pat, test_pat = select_pat_by_arrtime(df_70_static, 'PAT_ENC_CSN_ID', 'Arrived_Time')
    # df_train = df_70_static.filter(pl.col("PAT_ENC_CSN_ID").is_in(train_pat))
    # df_test = df_70_static.filter(pl.col("PAT_ENC_CSN_ID").is_in(test_pat))
    # dftr_le = cu.fit_transform(df_train)
    # dfte_le = cu.transform(df_test)
    # x = 0
    cu = CustomOneHotEncoding(
        single_val_cols=['A', 'B'],
        multi_val_cols=['C'],
        dep_col_dict={'D':["E"]},
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST,
        vocabthresh=100
    )
    cue = CustomLabelEncoding(
        single_val_cols=['A', 'B', 'D'],
        dep_col_dict={'D':["E"]},
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST,
        vocabthresh=100
    )
    dff = pl.DataFrame(
        {
            'A':['a', 'b', 'null', None, 'z', 'aa'],
            'B':[12, 12, None, 1, 15, 2],
            'C': [[], ['a', 'c'], ['a'], ['z'], ['a', 'b', 'c'], None],
            'D':['A', 'A', 'A', 'B', 'B', 'B'],
            'E':['AA', 'AB', 'AC', 'BD', 'BE', 'BF']
        }
    )
    dff_k = pl.DataFrame(
        {
            'A':['a', 'b', 'null', None, 'zz', '1a'],
            'B':[12, 12, None, 1, 19, 2],
            'C': [[], ['a', 'c'], ['a'], ['k'], ['a', 'k', 'c'], None],
            'D':['A', 'A', 'A', 'B', 'B', 'B'],
            'E':['AA', 'AB', 'AE', 'BD', 'BE', None]
        }
    )

    dff_ohe = cu.fit_transform(dff)
    dff_k_ohe = cu.transform(dff_k)
    dff_le = cue.fit_transform(dff)
    dffk_le = cue.transform(dff_k)
    x = 0 

    
    
