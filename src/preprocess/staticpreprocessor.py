import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from const import constants
from utils.utils import basic_preprocess, category_mappers_static
import re


class MultiValueProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, colnames, sep, thresh, regexp, apply_thresh):
        self.colnames = colnames # Column names
        self.sep = sep # spearator ","
        self.threshold = thresh # threshold 0.95
        self.regexp = regexp # True
        self.apply_thresh = apply_thresh
        assert isinstance(self.colnames, (list, tuple, np.ndarray)), \
        f'colnames should be one of the following (list, tuple, np.ndarray), {type(colnames)}'

    def _get_included_cat(self, X, idx):
        if self.regexp:
            all_list = X[f'{self.colnames[idx]}_NORM'].apply(lambda x: x.split(self.sep[idx]))
        vals = []
        for _, ll in all_list.items():
            ll = list(map(lambda x: x.strip(), ll))
            vals.extend(ll)
        
        counts = Counter(vals).most_common()

        if (self.threshold[idx]<1.0) and (self.apply_thresh[idx]<=len(counts)):
            df_counts = pd.DataFrame(counts, columns=['cc', 'count'])
            df_counts['prob'] = df_counts['count']/df_counts['count'].sum()
            df_counts['cumsum'] = df_counts['prob'].cumsum()
            df_counts_inc = df_counts[df_counts['cumsum']<=self.threshold[idx]]

            if len(df_counts_inc) == 0:
                cc = df_counts['cc']
            else:
                cc = df_counts_inc['cc'].tolist()

            if len(cc) < 10:
                cc = df_counts.loc[:10,'cc'].tolist()

        else:
            cc = list(map(lambda x: x[0], counts))
        
        return cc 
    
    def fit(self, X, y=None):
        self.included_cats = {}
        for idx, col in enumerate(self.colnames):
            assert col in X.columns, f'{col} is not found in the static features for fiting. {X.columns} is the available space ...'
            if self.regexp[idx]:
                X[f'{col}_NORM'] = X[col].apply(lambda x: re.sub('[^a-zA-Z0-9_]+', '', x))
            self.included_cats[col]  = self._get_included_cat(X, idx)
        return self 

    def transform(self, X, y=None):
        self.df_cols_dict = {}
        for idx, col in enumerate(self.colnames):
            assert col in X.columns, f'{col} is not found in the static features for transform. {X.columns} is the available space ...'
            inc_set = set(self.included_cats[col])
            if self.regexp[idx]:
                X[f'{col}_NORM'] = X[col].apply(lambda x: re.sub('[^a-zA-Z0-9_]+', '', x))    
                df_pre = pd.DataFrame(0, columns = self.included_cats[col]+[f'{col}_rare'], index=X.index)
                for ridx, val in X[f'{col}_NORM'].items():
                    v = val.split(self.sep[idx])
                    for vv in v:
                        if vv in inc_set:
                            df_pre.loc[ridx, vv] += 1
                        else:
                            df_pre.loc[ridx, f'{col}_rare'] += 1

                self.df_cols_dict[col] = df_pre.copy()
            else:
                df_pre = pd.DataFrame(0, columns = self.included_cats[col].tolist()+[f'{col}_rare'], index=X.index)
                for ridx, val in X[col].items():
                    v = val.split(self.sep[idx])
                    for vv in v:
                        if vv in inc_set:
                            df_pre.loc[ridx, vv] += 1
                        else:
                            df_pre.loc[ridx, f'{col}_rare'] += 1

                self.df_cols_dict[col] = df_pre.copy()
        return self.df_cols_dict


    def fit_transform(self, X, y=None,):
        self.fit(X, y)
        return self.transform(X, y)

        
class StaticTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pat_id, target, ohe_cols=None, le_cols=None, multival_dict=None, std_cols=None, minmax_cols=None):
        self.pat_id = pat_id
        self.target = target
        self.ohe_cols = ohe_cols
        self.le_cols = le_cols
        self.mv_dict = multival_dict
        self.std_cols = std_cols
        self.minmax_cols = minmax_cols

    def fit(self, X, y=None):
        if self.ohe_cols is not None:
            self.ohe_objs = {}
            for col in self.ohe_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                #TODO: Find a way to predefine in arguments for onehotencoder int he class arguments
                if col == 'arr_mnth': # Exceptions for datecolumn 
                    self.ohe_objs[col] = OneHotEncoder(categories=[list(range(1,13))])
                elif col == 'arr_hr':
                    self.ohe_objs[col] = OneHotEncoder(categories=[list(range(0,24))])
                elif col == 'arr_dow':
                    self.ohe_objs[col] = OneHotEncoder(categories=[list(range(0,7))])
                else:
                    self.ohe_objs[col] = OneHotEncoder(handle_unknown='ignore')
                     
                self.ohe_objs[col].fit(X[[col]])
        
        if self.le_cols is not None:
            self.le_objs = {}
            for col in self.le_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                self.le_objs[col] = LabelEncoder()
                self.le_objs[col].fit(X[[col]])

        if self.std_cols is not None:
            self.std_objs = {}
            for col in self.std_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                self.std_objs[col] = StandardScaler()
                self.std_objs[col].fit(X[[col]])

        if self.minmax_cols is not None:
            self.minmax_objs = {}
            for col in self.minmax_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                self.minmax_objs[col] = MinMaxScaler()
                self.minmax_objs[col].fit(X[[col]])
        
        if self.mv_dict is not None:
            self.mv_obj = MultiValueProcessor(self.mv_dict['colnames'], self.mv_dict.get('sep', None), self.mv_dict.get('thresh', 1.1), self.mv_dict.get('regex', False), self.mv_dict.get('apply_thresh', False))
            # self.mv_obj = MultiValueProcessor(self.mv_dict['colnames'], self.mv_dict['sep'], self.mv_dict['thresh'], self.mv_dict['regex'], self.mv_dict['apply_thresh'])
            self.mv_obj.fit(X, y)
        
        return self
    
    def fix_test_data_if_needed(self, X, cat_obj,col, default_categories=['Unknown', 'None']): #TODO: How to differentiate between different Unknown, None in different columns?
        try:
            mask = ~(X[col].isin(cat_obj.categories_[0])) # For OneHotEncoding
            categories = cat_obj.categories_[0]
        except Exception as e:
            mask = ~(X[col].isin(cat_obj.classes_)) # For LabelEncoding
            categories = cat_obj.classes_
            
        if mask.sum()>=1:
            for categ in default_categories:
                if categ in categories:
                    X.loc[mask, col] = categ
                    return X
        return X
    
    def transform(self, X, y=None):
        if self.ohe_cols is not None:
            self.ohe_df = {}
            for col, obj in self.ohe_objs.items():
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                X = self.fix_test_data_if_needed(X,self.ohe_objs[col], col, ['Unknown', 'None'])
                self.ohe_df[col] = pd.DataFrame(self.ohe_objs[col].transform(X[[col]]).toarray(), columns=list(map(lambda x: f'{col}_{x}', self.ohe_objs[col].categories_[0])), index=X.index)
        else:
            self.ohe_df = None
        
        if self.le_cols is not None:
            self.le_df = {}
            for col in self.le_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                X = self.fix_test_data_if_needed(X, self.le_objs[col], col, ['Unknown', 'None'])
                self.le_df[col] = pd.DataFrame(self.le_objs[col].transform(X[[col]]), columns=[col], index=X.index)
        else:
            self.le_df =None    
        
        if self.std_cols is not None:
            self.std_df = {}
            for col in self.std_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                self.std_df[col] = pd.DataFrame(self.std_objs[col].transform(X[[col]]), columns=[col], index=X.index)
        else:
            self.std_df = None

        if self.minmax_cols is not None:
            self.minmax_df = {}
            for col in self.minmax_cols:
                assert col in X.columns, f"{col} is not found in the dataframe columns ...: {X.columns}"
                self.minmax_df[col] = pd.DataFrame(self.minmax_objs[col].transform(X[[col]]), columns=[col], index=X.index)
        else:
            self.minmax_df = None
        
        if self.mv_dict is not None:
            self.mvp_df = self.mv_obj.transform(X)
        else:
            self.mvp_df = None
        
        self.transform_dict_ = {
            'ohe': self.ohe_df,
            'le': self.le_df,
            'minmax': self.minmax_df,
            'mvp': self.mvp_df
        }
        all_transforms = []
        for preproces_type, preprocess_cols_dict in self.transform_dict_.items():
            if preprocess_cols_dict is None:
                continue
            for col, df_transform in preprocess_cols_dict.items():
                if isinstance(df_transform, np.ndarray):
                    x = 0
                all_transforms.append(df_transform)
        
        df_all = pd.concat(all_transforms, axis=1)  
        df_all = df_all.merge(X[[self.pat_id, self.target]], how='outer', left_index=True, right_index=True)
        
        return df_all
    
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
        
            
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', 'static_100.csv'))
    df = df[df['Arrived_Time_appx']!='-1']
    df = basic_preprocess(df)
    df = category_mappers_static(df)
    df['arr_year'] = df['Arrived_Time_appx'].dt.year
    df['arr_month'] = df['Arrived_Time_appx'].dt.month
    df['arr_dow'] = df['Arrived_Time_appx'].dt.dayofweek
    df['arr_hr'] = df['Arrived_Time_appx'].dt.hour
    mvp = MultiValueProcessor(colnames=['Chief_Complaint_All'], sep=[','] ,thresh=[0.95], regexp=[True], apply_thresh=[100])
    static_transformer = StaticTransformer(ohe_cols=['Means_Of_Arrival', 'FirstRace', 'Ethnicity', 'arr_dow', 'arr_hr'],
                                           le_cols=['Acuity_Level', 'arr_month', 'MultiRacial'], std_cols=['arr_year', 'Patient_Age'], 
                                           minmax_cols=['Number of past appointments in last 60 days', 'Number of past inpatient admissions over ED visits in last three years'])
    transform_dict = static_transformer.fit_transform(df)
    x = 0
