from multiprocessing.spawn import prepare
import os
import sys

sys.path.insert(1, os.path.join(os.path.abspath(os.path.curdir), 'src'))

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from tqdm import tqdm
from numpy import ndarray
from sklearn.exceptions import NotFittedError
from const import constants
from joblib import Parallel, delayed
from collections import defaultdict
from preprocess.reductor import Reductor
import time
import numpy as np
from utils.utils import count_seq


class PreprocessSequence(BaseEstimator, TransformerMixin):
    def __init__(self, col_proc_dict, timeidx, enc_id) -> None:
        '''This class will convert will apply one hot encoding on the sequence column by creating a separate column for each attribute. Then by going through
        all of the sequence, values will be assigned to attributes accordingly. The main difference between this class and a regular OneHotEncoder is that this
        class can have more than 1.0 value for the same subject.
        
        [Args]
            unique_attr (ArrayLike): should contain the unique values of the column you want to preprocess
            data_dict (dict): keys are the subjects, and values are the list of the sequence 
            colname (str): The sequence column to be preprocessed

        [Return] 
            None

        '''
        self.fitted=False
        self.unique_attr = None
        self.col_proc_dict = col_proc_dict # {'colname1':['red', reduce_fnc, reduce_thr]} colname1 will be reduced first and the processed, colname2 will be processed only

        self.reductor_ = {}
        self.event_seq_dict_ = defaultdict(list)
        self.col_seq_ = {}

        self.time_idx = timeidx
        self.enc_id = enc_id

    def fit(self, X:pd.DataFrame, y=None, **fit_params):
        '''X is the whole dataframe 
        '''
        for col, proc_list in self.col_proc_dict.items():
            if proc_list[0] == 'red':
                self.reductor_[col] = Reductor(cols=[col], cnt_fn={col: proc_list[1]}, nan_val={col: constants.AMPUTATION_DICT[col]},
                                                red_thr=proc_list[2]) 
                self.col_seq_.update(self.reductor_[col].reduce(X))
                self.col_seq_[col].append(f'rare_{col}')
            else:
                self.col_seq_[col] = X[col].unique().tolist()

        self.fitted=True
        return self

    def fit_transform(self, X, y= None, **fit_params) -> ndarray:
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def prepare_data_dict(self, key, df_pat):
        # Create a dict that iterates over each df_pat and parse the seq column (EVENT_NAME)
        # Create a list corresponding the found events
        #IF Reduced? # Filter this list based on the trained categories
        output_dict  = {col: pd.Series(0, index=cat_set) for col, cat_set in self.col_seq_.items()}
        for col, cat_list in self.col_seq_.items():
            seq_list = df_pat[col].tolist()
            cat_set = set(cat_list)
            for val in seq_list:
                if val in cat_set:
                    output_dict[col].loc[val]+=1
                elif isinstance(val, float):
                    output_dict[col].loc[f'None_{col}'] += 1
                else:
                    output_dict[col].loc[f'rare_{col}'] += 1
        return key, output_dict
                    
    def transform(self, X, y=None):
        if not self.fitted:
            raise NotFittedError("Run fit or fit_transform first")
        
        # Initializations
        col2df_dict = {col: pd.DataFrame(index=X[self.enc_id].unique(), columns=cat_list)
                       for col, cat_list in self.col_seq_.items()}
        grb = X.sort_values(by=self.time_idx).groupby(self.enc_id)

        # Processing unit
        # t1 = time.time()
        # key_seriesDict_res = [self.prepare_data_dict(id, df_pat) for id, df_pat in grb]
        # print(f"Time diff for 0 multiprocessing: {time.time()-t1}")

        t1 = time.time()
        with Parallel(n_jobs=20) as p:
            key_seriesDict_res0 = p([delayed(self.prepare_data_dict)(id, df_pat) for id, df_pat in grb])
        print(f"Time diff for 20 multiprocessing: {time.time()-t1}")

        # s1 = sorted(key_seriesDict_res, key=lambda x: x[0])
        s2 = sorted(key_seriesDict_res0, key=lambda x: x[0])
        for idx, (key, val) in enumerate(s2):
            for k, v in val.items():
                assert(key==s2[idx][0] and (v!=s2[idx][1][k]).sum()==0)
            
        # Filling into dataframes
        for key, seriesDict in key_seriesDict_res0:
            for col, s in seriesDict.items():
                col2df_dict[col].loc[key] = s
        
        # Concatenating into a single dataframe
        preprocc_list  = []
        for key, df in col2df_dict.items():
            preprocc_list.append(df)
        preprocess_df = pd.concat(preprocc_list, axis=1)

        return preprocess_df 
        
               # for col, cat_vals in self.col_seq_.items():
        
        #     bm_col = pd.DataFrame(0, index=X[self.enc_id].unique(), columns=cat_vals)
        #     st = time.time()
        #     print(f'{col} takes {time.time()-st} seconds without parallelization')

        # self.event_seq_dict_ = defaultdict(list)
        # self.dx_seq_dict_ = dict()
        # self.cc_seq_dict_ = defaultdict(list)

        # self.prepare_data_dict(X)

        # self._fix_rare_cc_all()

        # result1 = Parallel(n_jobs=22)(
        #     delayed(process_key)(key, seq, self.event_name_unique_) for key, seq in self.event_seq_dict_.items()
        # )
        # result2 = Parallel(n_jobs=22)(
        #     delayed(process_key)(key, seq, self.dx_unique_) for key, seq in self.dx_seq_dict_.items()
        # )
        # result3 = Parallel(n_jobs=22)(
        #     delayed(process_key)(key, seq, self.cc_all_) for key, seq in self.cc_seq_dict_.items()
        # )

        binary_matrix1 = pd.DataFrame(0, index=self.event_seq_dict_.keys(), columns=self.event_name_unique_)
        binary_matrix2 = pd.DataFrame(0, index=self.dx_seq_dict_.keys(), columns=self.dx_unique_)
        binary_matrix3 = pd.DataFrame(0, index=self.cc_seq_dict_.keys(), columns=self.cc_all_)
        for key, row in result1:
            binary_matrix1.loc[key] = row
        # for key, row in result2:
        #     binary_matrix2.loc[key] = row
        # for key, row in result3:
        #     binary_matrix3.loc[key] = row

        binary_matrix = pd.concat([binary_matrix1, binary_matrix2, binary_matrix3], axis=1)
        # Without joblib parallelization
        # for key, seq in tqdm(self.data_dict.items()):
        #     if isinstance(seq, list):
        #         if isinstance(seq[0], list):
        #             binary_matrix.loc[key, list(set(seq[0]))] = 1
        #         else:
        #             binary_matrix.loc[key, seq[0]] = 1

            
        #         if binary_matrix.isna().sum().sum()>0:
        #             x = 0
        #     else:
        #         binary_matrix.loc[key, seq] = 1

        # self.original_ = pd.concat([self.original_, binary_matrix], axis=1)
        # return self.original_
        return binary_matrix

def reduce_column_dims(df, colname, thresh, outlier_name):
    prob_event_name = pd.DataFrame(df[colname].value_counts().sort_values(ascending=False)/len(df[~df[colname].isna()]))
    prob_event_name['cumsum'] = prob_event_name.cumsum()
    included_cat = prob_event_name.index.get_loc(prob_event_name[prob_event_name['cumsum']>=thresh].index[0])
    rare_events = prob_event_name.iloc[included_cat:].index
    print(f"Reduction ratio: {(included_cat+1)/df[colname].nunique()}")
    print(f'included categories: {included_cat+1}')
    print(f'Total # of categories: {df[colname].nunique()}')
    df.loc[df[colname].isin(rare_events), colname] = outlier_name 
    return df


if __name__ == "__main__":
    DATA_PATH =  "/work/InternalMedicine/s223850/intermediate_data_filteredFlags/35_optim72123.csv"


    df = pd.read_csv(DATA_PATH, index_col=0)
    df['Arrived_Time'] = pd.to_datetime(df['Arrived_Time'])
    df['Calculated_DateTime'] = pd.to_datetime(df['Calculated_DateTime'])
    pat_list = df['PAT_ENC_CSN_ID'].unique()
    pat_tr = np.random.choice(pat_list, int(0.8*len(pat_list)), replace=False)
    pat_te = pat_list[~np.isin(pat_list, pat_tr)] 
    df_tr = df.loc[df['PAT_ENC_CSN_ID'].isin(pat_tr)]
    df_te = df.loc[df['PAT_ENC_CSN_ID'].isin(pat_te)]


    pobj = PreprocessSequence({'EVENT_NAME':[('red', count_seq)]}, 'Calculated_DateTime', 
                              'PAT_ENC_CSN_ID') 
    pobj.fit(df_tr)
    bin_tr = pobj.transform(df_tr)
    bin_te = pobj.transform(df_te)
    x=0