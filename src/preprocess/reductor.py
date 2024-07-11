from time import time_ns
import pandas as pd
from collections import defaultdict
import numpy as np
import re

class Reductor:
    def __init__(self, cols=[], cnt_fn=None, red_thr=0.95, grpby_col='PAT_ENC_CSN_ID', sort_by_col = 'Calculated_DateTime',
                 nan_val='None_Chief_Complaint_All') -> None:
        '''Reduce the categories of the columns defines in cols using the funtion defined in cnt_fn
        cnt_fn expects 2 arguments cnt_fn(X, colname) such that X: pd.DataFrame, colname: str. and it
        returns a dictionary with the keys are the different categories within X[colname] and the values
        are the number of occurances in X[colname].
        '''
        self.thr = red_thr
        self.grpby_id = grpby_col
        self.sort_by = sort_by_col
        self.nan_val = nan_val
        self.columns = cols
        self.output_ = None
        self.count_fn=cnt_fn
        
    def _count_categories(self, X:pd.DataFrame, colname):
        cntr_dict = defaultdict(int)
        X_ = X.sort_values(by=self.sort_by)
        for idx, df_pat in X_.groupby(self.grpby_id):
            cc = df_pat[colname].iloc[-1]
            if isinstance(cc, float):
                cntr_dict[self.nan_val]+=1
            else:
                cntr_dict[cc] += 1
        return cntr_dict

    def _single_col(self, X, colname, nan_name=None):
        if self.count_fn is None:
            cntr_dict = self._count_categories(X, colname)
        else:
            cntr_dict = self.count_fn[colname](X, colname, nanamp=nan_name, time_idx=self.sort_by, grb_id=self.grpby_id)
        sorted_list = sorted(cntr_dict.items(), key=lambda kv: kv[1], reverse=True)
        total_cnt = np.sum(list(map(lambda x: x[1], sorted_list)))
        normalized_sorted_list = list(map(lambda x: (x[0], x[1]/total_cnt), sorted_list))
        df = pd.DataFrame(normalized_sorted_list)
        df['cum'] = df[1].cumsum()
        categories = df.loc[df['cum']<=self.thr, 0].tolist()
        if len(categories)<1:
            categories = [df.iloc[0][0]]
        print(f'{len(df)} is the total number of categories which is reduced to {len(categories)} based on {self.thr} threshold')
        return categories
    
    def _fast_counter(self, X, colname, nan_name=None):
        counts = X.groupby(self.grpby_id)[colname].value_counts()
    
    def reduce(self, X) -> dict:
        self.output_ = {}
        if isinstance(self.columns, str):
            self.output_[self.columns] = self._single_col(X, self.columns, nan_name=self.nan_val)
            return self.output_
        
        if isinstance(self.columns, (list, tuple)):
            for col in self.columns:
                self.output_[col] = self._single_col(X, col, nan_name=self.nan_val[col])
            return self.output_
        
        raise ValueError("Columns should be either a string, list, or tuple")

    # Implement those funtion to be able to place Reductor objects inside a sklearn pipeline
    def fit(self, X, **params):
        return self

    def transform(self, X):
        pass

    def fit_transform(self, X, **params):
        self.fit(X, **params)
        return self.transform(X)
        

if __name__ == '__main__':
    DATA_PATH =  "/work/InternalMedicine/s223850/intermediate_data_filteredFlags/35_optim72123.csv"

    df = pd.read_csv(DATA_PATH, index_col=0)
    df['Arrived_Time'] = pd.to_datetime(df['Arrived_Time'])
    df['Calculated_DateTime'] = pd.to_datetime(df['Calculated_DateTime'])

    df['Primary_DX_re'] = df['Primary_DX'].apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x) if isinstance(x, str) else f'None_Primary_DX')
    df['EVENT_NAME_re'] = df['EVENT_NAME'].apply(lambda x: re.sub('[^A-Za-z0-9_]+', '', x) if isinstance(x, str) else f'None_EVENT_NAME')
    
    robj = Reductor(['Primary_DX','EVENT_NAME'], {'Primary_DX': cnt_EVENT_NAME, 'EVENT_NAME': cnt_PDX})
    res = robj.reduce(df)

    robj = Reductor(['Primary_DX_re','EVENT_NAME_re'], {'Primary_DX_re': cnt_EVENT_NAME, 'EVENT_NAME_re': cnt_PDX})
    res = robj.reduce(df)
    x=0

