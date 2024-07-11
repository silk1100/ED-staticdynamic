import os
import sys
sys.path.insert(1, os.path.join(os.path.abspath(os.path.curdir), 'src'))
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tqdm import tqdm
from numpy import ndarray
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
from const import constants
import time
import re

class MutiValPreprocessor(BaseEstimator, TransformerMixin):
    '''
    Preprocess categorical column that can hold more than one categorical value at each cell. This class is primarly
    designed to preprocess "Chief_Complaint_All" which can contain anything between 1 to 7 different chief complaints
    for a given patient.
    
    The following class enumerate all of the available categories in the training data, apply a threshold based on
    their occurances, Save the categorized variables list, and the rare variable list to be used for transforming 
    coming matrix
    '''
    # def __init__(self, colnames=[], col_sep=',', include_threshold=0.95,
    #              id='PAT_ENC_CSN_ID', timecol='Calculated_DateTime', reg_exp_filter=True) -> None:

    def __init__(self, mvp_dict:dict, # {'colnames':'colname', 'col_sep':',', 'red_thr':0.95 }
                  id='PAT_ENC_CSN_ID', timecol='Calculated_DateTime', reg_exp_filter=True) -> None:
        colnames = mvp_dict.get('colnames', None)
        col_sep = mvp_dict.get('col_sep', None)
        include_threshold = mvp_dict.get('red_thr', None)
        reg_exp_mvp = mvp_dict.get('reg_exp', None)
        
        if isinstance(colnames, str):
            self.singleCol = True
        else:
            self.singleCol = False 

        self.multicols = colnames
        self.threshold = include_threshold
        self.col_sep = col_sep
        self.id = id
        self.timecol = timecol

        self.included_cols_ = []
        self.included_cols_set_ = None 
        if reg_exp_mvp is None:
            self.reg_filt = reg_exp_filter
        else:
            self.reg_filt = reg_exp_mvp
            

    def _count_categories(self, X:pd.DataFrame, colname, sep):
        cntr_dict = defaultdict(int)
        X_ = X.sort_values(by=self.timecol)
        for idx, df_pat in X_.groupby(self.id):
            cc = df_pat[colname].iloc[-1]
            if isinstance(cc, float):
                cntr_dict[f'None_{colname}']+=1
            else:
                for val in cc.split(sep):
                    cntr_dict[val] += 1
        return cntr_dict

    def _count_categories_seq(self, X:pd.DataFrame, colname, sep):
        '''
        Count over the whole dataset not within subjects
        '''
        cntr_dict = defaultdict(int)
        X_ = X.sort_values(by=self.timecol).groupby(self.id)
        for pat_id, df_group in X_:
            col_val = df_group[colname].apply(lambda x: x.split(sep) if isinstance(x, str) else x)
            ss = set()
            for idx, dl in col_val.iteritems():
                if isinstance(dl, list):
                    for d in dl:
                        ss.add(d)
                else:
                    ss.add(f'None_{colname}')
            for i in ss:
                cntr_dict[i] += 1
        return cntr_dict

    def _single_col(self, X, colname, sep, thresh):
        cntr_dict = self._count_categories(X, colname, sep)
        sorted_list = sorted(cntr_dict.items(), key=lambda kv: kv[1], reverse=True)
        total_cnt = np.sum(list(map(lambda x: x[1], sorted_list)))
        normalized_sorted_list = list(map(lambda x: (x[0], x[1]/total_cnt), sorted_list))
        df = pd.DataFrame(normalized_sorted_list)
        df['cum'] = df[1].cumsum()
        categories = df.loc[df['cum']<=thresh, 0].tolist()
        if len(categories)<1:
            categories = [df.iloc[0][0]]
        print(f'{len(df)} is the total number of categories which is reduced to {len(categories)} based on {thresh} threshold')
        return categories
    
    def fit(self, X:pd.DataFrame, y=None, **fit_params):
        if not self.singleCol:
            for col in self.multicols:
                assert col in X.columns, f"{col} is not found in the passed DataFrame column {X.columns}"
        else:
            assert self.multicols in X.columns, f"{self.multicols} is not found in the passed DataFrame column {X.columns}"
        
        
        self.included_cols_ = dict()
        self.included_cols_set_ = dict()
        if self.singleCol:
            if self.reg_filt:
                X[self.multicols] = X[self.multicols].apply(lambda x: re.sub("[^a-zA-Z0-9_]+", "", x) if isinstance(x, str)
                                                            else f"None_{self.multicols}")
            categories = self._single_col(X, self.multicols, self.col_sep, self.threshold)
            self.included_cols_[self.multicols] = categories+[f'rare_{self.multicols}']
            self.included_cols_set_[self.multicols] = set(self.included_cols_[self.multicols])
        else:
            if isinstance(self.reg_filt, bool):
                reg_filter = [self.reg_filt for _ in range(len(self.multicols))]
            elif isinstance(self.reg_filt, (list, tuple)):
                reg_filter = self.reg_filt

            for idx, col in enumerate(self.multicols):
                if reg_filter[idx]:
                    X[col] = X[col].apply(lambda x: re.sub("[^a-zA-Z0-9_]+", "", x) if isinstance(x, str)
                                                            else f"None_{col}")
                categories = self._single_col(X, col, self.col_sep[idx], self.threshold[idx])
                self.included_cols_[col] = categories+[f'rare_{col}']
                self.included_cols_set_[col] = set(self.included_cols_[col])

    def transform(self, X, y=None):
        def fillmat(idx:str, df:pd.DataFrame, colname='Chief_Complaint_All'): 
            s = pd.Series(0, index=self.included_cols_[colname])
            val = df[colname].iloc[-1]

            if isinstance(val, float):
                s[f'None_{colname}'] += 1
                return idx, s

            for k in val.split(','):
                if k in self.included_cols_set_[colname]:
                    s[k]+=1
                else:
                    s[f'rare_{colname}']+=1
            return idx, s 
            
        if not self.singleCol:
            for col in self.multicols:
                assert col in X.columns, f"{col} is not found in the passed DataFrame column {X.columns}"
        else:
            assert self.multicols in X.columns, f"{self.multicols} is not found in the passed DataFrame column {X.columns}"
        
        pat_id = X[self.id].unique() 
        start = time.time()
        # SLower when parallel
        # with Parallel(n_jobs=1) as p:
        #     results=p(delayed(fillmat)(idx, df_pat)  for idx, df_pat in X.groupby('PAT_ENC_CSN_ID'))
        if isinstance(self.multicols, (list, tuple)):
            results = {col: [fillmat(idx, df_pat, col) for idx, df_pat in X.groupby(self.id) ] for col in self.multicols}
            binary_matrix_dict = {col: pd.DataFrame(0, index=pat_id, columns=self.included_cols_[col]) for col in self.multicols}

        elif isinstance(self.multicols, str): 
            results = {self.multicols: [fillmat(idx, df_pat, self.multicols) for idx, df_pat in X.groupby(self.id)]}
            binary_matrix_dict = {self.multicols: pd.DataFrame(0, index=pat_id, columns=self.included_cols_[self.multicols])} 

        for colname, key_val_list in results.items():
            for id, row in key_val_list:
                binary_matrix_dict[colname].loc[id] = row

        # print(f'total time taken 3 proc parallelizing the transformation: {time.time()-start}')
        self.bn_mat_dict = binary_matrix_dict
        df_out = pd.concat(binary_matrix_dict.values(), axis=1)
        return df_out

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

        
if __name__ == '__main__':
    DATA_PATH =  "/work/InternalMedicine/s223850/intermediate_data_filteredFlags/35_optim72123.csv"


    df = pd.read_csv(DATA_PATH, index_col=0)
    df['Arrived_Time'] = pd.to_datetime(df['Arrived_Time'])
    df['Calculated_DateTime'] = pd.to_datetime(df['Calculated_DateTime'])

    mp = MutiValPreprocessor('Chief_Complaint_All')
    ids = df['PAT_ENC_CSN_ID'].unique()
    train_ids = np.random.choice(ids, int(len(ids)*0.9), replace=False)
    test_ids = np.setdiff1d(ids, train_ids)
    df_train = df.loc[df['PAT_ENC_CSN_ID'].isin(train_ids)]
    df_test = df.loc[df['PAT_ENC_CSN_ID'].isin(test_ids)]
    mp.fit(df_train)
    pretrain = mp.transform(df_train)
    pretest = mp.transform(df_test)
    x = 0