import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from const import constants
from utils.utils import basic_preprocess, category_mappers_dynamic
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


'''
This class implements the new idea of preprocessing the dynamic events based on their clinical type. 
ED events will be handeled different than Lab - Orders, different than Medication - orders.

- Each type will be preprocessed independently and their categories will be reduced according to the frequency within that type.
- Each event will be stamped with the duration until it is 
'''


class DynamicTranformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_col, time_idx, type_col, event_col, time_elapsed_col_name, type_config={}, cols_config={}) -> None:
        
    # def __init__(self, id_col, time_idx,
    #              type_col, event_col, lab_result_type = 'Lab Order - Result',
    #              typeregex=True, type_reduce_threshold=0.95, lab_res_col='Result_Flag',
    #              colsdict={}):
        '''
        Example:

        typesdict = { # This dictionary should contain only those categories/types that requires extra processing
            type_attr : ['Lab Order - Results'], Specifiy types that you want to perform extra/special processing
            type_extra_col : ['Lab_Result'], Add extra column value to the event column
            type_extra_col_concat: ['_'], concatenation separator
            type_thresh : [0.95], 
            type_regex : [False],
            type_apply_thresh: [None] # always apply reduction
            type_global_thresh : 0.95, Set reduction threshold for all types
            type_global_regex: True, set regex for all types
            type_global_apply_thresh: 100 # Number of unique values to apply the threshold after (None means always apply)

        }
         colsdict = {
             'colnames': ['Primary_DX_Name', 'Primary_DX_ICD10', "Primary_DX_ICD10_First" ],
             'threshold': [0.95, 0.95, None], # To avoid doing any reduction to the feature space, then you can use None or >=1.
             'regex':[True, False, False],
             'sep':['_', '_', None]
             'apply_threshold": [100, 100, 100]
         }
        '''

        self.idx = id_col
        self.tidx = time_idx
        self.type_col = type_col
        self.event_col = event_col
        self.time_elapsed_colname = time_elapsed_col_name

        self.type_dict = type_config
        self.cols_dict = cols_config

        self.type_global_thresh = self.type_dict.get('type_global_thresh', 1.1)
        self.type_global_regex = self.type_dict.get('type_global_regex', False)
        self.type_global_apply_thresh = self.type_dict.get('type_global_apply_thresh', 0)

        self.special_type_dict = {}
        self.special_type_dict['type_attr'] = self.type_dict.get('type_attr', [])
        self.special_type_dict['type_extra_col'] = self.type_dict.get('type_extra_col', [])
        self.special_type_dict['type_regex'] = self.type_dict.get('type_regex', [])
        self.special_type_dict['type_extra_col_concat'] = self.type_dict.get('type_extra_col_concat', [])
        self.special_type_dict['type_thresh'] = self.type_dict.get('type_thresh', [])
        self.special_type_dict['type_apply_thresh'] = self.type_dict.get('type_apply_thresh',[]) 

        assert len(self.special_type_dict['type_attr']) == len(self.special_type_dict['type_extra_col'])\
            == len(self.special_type_dict['type_regex']) == len(self.special_type_dict['type_extra_col_concat'])\
            == len(self.special_type_dict['type_thresh']),\
            f"type_attr, type_extra_col, type_thresh, type_regex, type_extra_col_concat must be lists of the same length"

    def _get_multival_list(self, X, colname, regex):
        all_list = X[colname].apply(lambda x: x.split(self.sep))

        vals = []
        for _, ll in all_list.items():
            if regex:
                ll = list(map(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x)))
            vals.extend(ll)
        return vals
    
    def _get_singleval_list(self, X, colname, idx):
        if self.regexp:
            all_list = X[colname].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x)).tolist()
        else:
            all_list = X[colname].tolist()
        return  all_list

    def _get_included_cat(self, pat_groups, colname, multival=False, regex=False):
        #TODO: Add regex preprocessing if needed, or perform it outside the transformer and remove this todo statement
        
        if multival:
            all_list = self._get_multival_list(X, colname, regex) 
        else:
            all_list = self._get_singleval_list(X, colname, regex)

        counts = Counter(all_list).most_common()
        df_counts = pd.DataFrame(counts, columns=['cc', 'count'])
        df_counts['prob'] = df_counts['count']/df_counts['count'].sum()
        df_counts['cumsum'] = df_counts['prob'].cumsum()
        df_counts_inc = df_counts[df_counts['cumsum']<=self.ty]
        if len(df_counts_inc) == 0:
            cc = df_counts['cc']
        else:
            i = df_counts_inc.tail(1).index[0]
            cc = df_counts.loc[:i, 'cc']
        return cc 
    
    def _get_type_included_vals(self, X):
        type_vals_dict = {}
        for ttype in self.available_types_:
            if len(self.special_type_dict) > 0 and ttype == self.special_type_dict['type_attr']:
                continue
            if self.type_global_regex:
                counts = X.loc[X[self.type_col]==ttype, self.event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+', '', x))\
                    .value_counts().to_frame()
            else:
                counts = X.loc[X[self.type_col]==ttype, self.event_col].value_counts().to_frame()

            if (self.type_global_thresh<1.0) and \
                self.type_global_apply_thresh<=len(counts): # Condition to apply reduction
                counts['prob'] = counts[self.event_col]/counts[self.event_col].sum()
                counts['cumsum'] = counts['prob'].cumsum()
                row = counts[counts['cumsum']<=self.type_global_thresh]
                # if len(row) == 0:
                #     row = counts

                if len(row) < 10:
                    row = counts.iloc[:10]

                type_vals_dict[ttype] = row.index.tolist()
            else:
                # Include all values
                type_vals_dict[ttype] = counts.index.tolist()
        
        # Handle special types separately
        for idx, ttype in enumerate(self.special_type_dict['type_attr']):
            if self.special_type_dict['type_regex'][idx]:
                type_vals = X.loc[X[self.type_col]==ttype, self.event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+', '', x))
            else:
                type_vals = X.loc[X[self.type_col]==ttype, self.event_col]
            
            type_vals = type_vals+self.special_type_dict['type_extra_col_concat'][idx]+\
                X.loc[X[self.type_col]==ttype, self.special_type_dict['type_extra_col'][idx]]

            if self.special_type_dict['type_thresh'][idx] is not None and \
                self.special_type_dict['type_apply_thresh'][idx] is not None \
                and self.special_type_dict['type_apply_thresh'][idx] <= type_vals.nunique(): # Condition to perform reduction

                counts = type_vals.value_counts().to_frame()
                if self.event_col not in counts.columns:
                    counts.rename(columns={0:self.event_col}, inplace=True)
                counts['prob'] = counts[self.event_col]/counts[self.event_col].sum()
                counts['cumsum'] = counts['prob'].cumsum()
                row = counts[counts['cumsum']<=self.special_type_dict['type_thresh'][idx]]
                # if len(row) == 0:
                #     row = counts
                if len(row) < 10:
                    row = counts.iloc[:10]
                
                type_vals_dict[ttype] = row.index.tolist()

            else: # include all values
                type_vals_dict[ttype] = type_vals.unique().tolist()
            
        return type_vals_dict


    def _get_col_included_vals(self, pat_groups, idx):
        vals_dict = defaultdict(int)
        for pat_id, df_grp in pat_groups:
            vals = df_grp[self.cols_dict['colnames'][idx]].unique().tolist()
            if self.cols_dict['sep'][idx] is not None:
                for val in vals:
                    for v in val.split(self.cols_dict['sep'][idx]):
                        vals_dict[v.strip()]+=1
            else:
                vals_dict[vals]+=1
        print(f'{self.cols_dict["colnames"][idx]} has {len(vals_dict)} ...')

        if self.cols_dict['threshold'][idx] is not None and self.cols_dict['apply_threshold'][idx] is not None \
            and self.cols_dict['apply_threshold'][idx]<=len(vals_dict):
            counts = pd.DataFrame(vals_dict.items(), columns=['event', 'counts'])
            counts = counts.sort_values(by='counts', ascending=False)
            counts['prob'] = counts['counts']/counts['counts'].sum()
            counts['cumsum'] = counts['prob'].cumsum()
            selected_df = counts[counts['cumsum']<=self.cols_dict['threshold'][idx]]
            # if len(selected_df) < 0:
            #     selected_df = counts

            # Updated logic to include top 10 if the # of categories is less than 10
            if len(selected_df) < 10:
                selected_df = counts.iloc[:10]
            
            feats = selected_df['event'].tolist()
        else:
            feats = list(vals_dict.keys())
        return feats

    def _create_type_ohe(self, X, vals_dict):
        ohe_dict = {}
        X = self._set_rare_types(X, vals_dict)
        for type_, values in vals_dict.items():
            df_type = X.loc[X[self.type_col]==type_, [self.event_col]]
            # ohe_dict[type_] = OneHotEncoder(drop='first') # throws an error because I do not exclude the dropped value from categories_
            # ohe_dict[type_] = OneHotEncoder()
            ohe_dict[type_] = OneHotEncoder(categories=[values+[f'rare_{type_}']])
            ohe_dict[type_].fit(df_type)
        return ohe_dict

    def _create_cols_ohe(self, X, vals_dict):
        ohe_dict = {}
        X = self._set_rare_cols(X, vals_dict)
        for colname, values in vals_dict.items():
            df_col = X[[colname]]
            # ohe_dict[colname] = OneHotEncoder(drop='first') # throws an error because I do not exclude the dropped value from categories_
            # ohe_dict[colname] = OneHotEncoder() 
            ohe_dict[colname] = OneHotEncoder(categories=[values+[f'rare_{colname}']]) 
            ohe_dict[colname].fit(df_col)
        return ohe_dict

    def _set_rare_types(self, X, vals_dict):
        # X1 = X.copy()
        X1 = X

        # t1 = time.time()
        # for type_, values in vals_dict.items():
        #     df_type = X[X[self.type_col]==type_]
        #     rare_index = df_type[~df_type[self.event_col].isin(values)].index
        #     X.loc[rare_index, self.event_col] = f'rare_{type_}'
        # print(f'Traditional method took {time.time()-t1} seconds...')
         
        t1 = time.time()
        mask = pd.Series(False, index=X1.index)
        # Update the mask where the condition is met
        for type_, values in vals_dict.items():
            mask |= (X1[self.type_col] == type_) & (~X1[self.event_col].isin(values))

        # Use the mask to update the DataFrame
        X1.loc[mask, self.event_col] = X1.loc[mask, self.type_col].apply(lambda x: f'rare_{x}')
        # print(f'New method took {time.time()-t1} seconds...')
 
        return X
    
    def _set_rare_cols(self, X, vals_dict):
        for idx, (colname, values) in enumerate(vals_dict.items()):
            if self.cols_dict['sep'][idx] is None:
                rare_index = X.loc[~(X[colname].isin(values))].index
                X.loc[rare_index, colname] = f'rare_{colname}'
            else:
                dx_list = X[colname].apply(lambda x: x.split(self.cols_dict['sep'][idx]))
                dx_rare_s = pd.Series(index=dx_list.index, dtype=np.float64)
                values_set = set(values)
                for ridx, values_list in dx_list.items():
                    if len(values_list) == 1:
                        dx_rare_s.loc[ridx] = f'rare_{colname}' if values_list[0] not in values_set else values_list[0]
                    else:
                        rare_list = []
                        for val in values_list:
                            nv = f'rare_{colname}' if val.strip() not in values_set else val.strip()
                            rare_list.append(nv)
                        dx_rare_s.loc[ridx] = ','.join(rare_list)
                X.loc[:, colname] = dx_rare_s
        return X

    def _apply_regex_types(self, X, all_types):
        if self.type_global_regex:
            for attr in all_types:
                if attr not in self.special_type_dict['type_attr'] or\
                    (attr in self.special_type_dict['type_attr'] and self.special_type_dict['type_regex'][self.special_type_dict['type_attr'].index(attr)]):
                    # df_type = X.loc[X[self.type_col]==attr]
                    # df_type[self.event_col] = df_type[self.event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+', '', x).lower())
                    # X.loc[df_type.index, self.event_col] = df_type[self.event_col].tolist()
                    mask = X[self.type_col]==attr
                    X.loc[mask, self.event_col] = X.loc[mask, self.event_col].str.replace(r'[^a-zA-Z0-9_]+', '', regex=True).str.lower()
        else:
            for idx, attr in enumerate(self.special_type_dict['type_attr']):
                if self.special_type_dict['type_regex'][idx]:
                    # df_type = X.loc[X[self.type_col]==attr]
                    # df_type[self.event_col] = df_type[self.event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+', '', x).lower())
                    # X.loc[df_type.index, self.event_col] = df_type[self.event_col].tolist()
                    mask = X[self.type_col]==attr
                    X.loc[mask, self.event_col] = X.loc[mask, self.event_col].str.replace(r'[^a-zA-Z0-9_]+', '', regex=True).str.lower()
        return X


    def fit(self, X, y=0):
        if len(self.cols_dict) >= 1:
            for idx, col in enumerate(self.cols_dict['colnames']):
                assert col in X.columns, f'{col} is not found in the columns of the data passed to fit. {X.columns} are found ...'
                if self.cols_dict['regex'][idx]:
                    X[col] = X[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+','', x))

        self.available_types_ = X[self.type_col].unique()

        if len(self.special_type_dict) >= 1:
            for attr in self.special_type_dict['type_attr']:
                assert attr in self.available_types_, f'{attr} is not a valid attribute among the available attribute: {self.available_types_}'

        X = self._apply_regex_types(X, self.available_types_)
        
        X = X.sort_values(by=self.tidx)
        pat_groups = X.groupby(self.idx)

        self.included_types_dict_ = defaultdict(list) # {type1: [all_included_values], type2:[...], ...}
        self.included_cols_dict_ = defaultdict(list) # {col1: [all_included_values], col2:[...], ...}

        # X[self.time_elapsed_colname] = X.groupby(self.idx)[self.tidx].transform(lambda x: (x-x.iloc[0]).dt.total_seconds()/60)
        # Precompute the first timestamp in each group
        group_first_timestamp = X.groupby(self.idx)[self.tidx].transform('first')
        # Compute the difference in minutes
        X[self.time_elapsed_colname] = (X[self.tidx] - group_first_timestamp).dt.total_seconds() / 60


        self.types_values_dict_ = self._get_type_included_vals(X)

        self.ohe_types_dict_ = self._create_type_ohe(X, self.types_values_dict_)
        
        self.cols_values_dict_ = {}
        for idx, col in enumerate(self.cols_dict['colnames']):
            self.cols_values_dict_[col] = self._get_col_included_vals(pat_groups, idx)

        # self.ohe_cols_dict_ = self._create_cols_ohe(X, self.cols_values_dict_)
        self.ohe_cols_dict_ = {colname: None for colname in self.cols_dict['colnames']}

        return self


    def __fill_type_transform_dict(self, pat_id, df_grp):
            transform_dict ={}
            for type_, values in self.types_values_dict_.items():
                all_types = list(map(lambda x: f'{type_}_{x}', values))
                transform_dict[type_] = pd.Series(0, name=pat_id, index=all_types+[f'rare_{type_}'])
            
            for type_, values in self.types_values_dict_.items():
                df_type = df_grp[df_grp[self.type_col]==type_]
                for pid, type_val in df_type[self.event_col].items():
                    if 'rare' in type_val:
                        transform_dict[type_].loc[type_val] += 1
                    else:
                        transform_dict[type_].loc[f'{type_}_{type_val}'] += 1
            return transform_dict
    def __update_type_transform_dict(self, d, type_):
        self.type_transform_dict_[type_].loc[d[type_].name] = d[type_]
        return type_

    def transform(self, X, y=0):
        if len(self.cols_dict) >= 1:
            for idx, col in enumerate(self.cols_dict['colnames']):
                assert col in X.columns, f'{col} is not found in the columns of the data passed to fit. {X.columns} are found ...'
                if self.cols_dict['regex'][idx]: # Apply regex for columns
                    X[col] = X[col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+','', x))

        for type_ in X[self.type_col].unique():
            assert type_ in self.available_types_,  f'{type_} was not found in the fitted dataframe. Available types are: {self.available_types_} ...'            

        X = X.sort_values(by=self.tidx)
        X[self.time_elapsed_colname] = X.groupby(self.idx)[self.tidx].transform(lambda x: (x-x.iloc[0]).dt.total_seconds()/60)
        
        X = self._apply_regex_types(X, self.available_types_)

        X = self._set_rare_types(X, self.types_values_dict_)
        X = self._set_rare_cols(X, self.cols_values_dict_)
        
        grp = X.groupby(self.idx)

        self.type_transform_dict_ = {}
        # self.type_transform_dict1_ = {}
        for type_, values in self.types_values_dict_.items():
            type_values = list(map(lambda x: f'{type_}_{x.strip()}', values))
            type_values = type_values + [f'rare_{type_}']
            self.type_transform_dict_[type_] = pd.DataFrame(0, index=X['PAT_ENC_CSN_ID'].unique(),
                                                            columns=type_values)
            # self.type_transform_dict1_[type_] = pd.DataFrame(0, index=X['PAT_ENC_CSN_ID'].unique(),
            #                                                 columns=type_values)
    
        # dl = []
        # for idx, (pat_id, df_grp) in enumerate(grp):
        #     dl.append(self._fill_type_transform_dict(pat_id, df_grp))
        #     if idx ==10:break
        
        # for type_, values in self.types_values_dict_.items():
        #     for d in dl:
        #         self.type_transform_dict_[type_].loc[d[type_].name] = d[type_]

        
        #### NOT WORTH IT
        # t1 = time.time()
        # dl = []
        # with ThreadPoolExecutor(max_workers=25) as pool:
        #     futures = [pool.submit(self.__fill_type_transform_dict, pat_id, df_grp) for pat_id, df_grp in grp]
        #     for future in as_completed(futures):
        #         dl.append(future.result())

        #         # Refactored second loop with ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=25) as pool:
        #     futures = []
        #     for type_, values in self.types_values_dict_.items():
        #         for d in dl:
        #             future = pool.submit(self.__update_type_transform_dict, d, type_)
        #             futures.append(future)

        # # Waiting for all futures to complete (if needed)
        # for future in as_completed(futures):
        #     _ = future.result()  
        

        # # for type_, values in self.types_values_dict_.items():
        # #     for d in dl:
        # #         self.type_transform_dict_[type_].loc[d[type_].name] = d[type_]
        
        # print(f"Time taken for thread: {time.time()-t1} seconds ...")
        
        t1 = time.time()
        for pat_id, df_grp in grp:
            for type_, values in self.types_values_dict_.items():
                df_type = df_grp[df_grp[self.type_col]==type_]
                for pid, type_val in df_type[self.event_col].items():
                    if 'rare' in type_val:
                        self.type_transform_dict_[type_].loc[pat_id, type_val] += 1
                    else:
                        self.type_transform_dict_[type_].loc[pat_id, f'{type_}_{type_val}'] += 1
        # print(f"Time taken for seq: {time.time()-t1} seconds ...")
        x=0
          
        self.all_feature_values_ = []
        for type_, values in self.types_values_dict_.items():
            type_values = list(map(lambda x: f'{type_}_{x}', values))
            type_values = type_values + [f'rare_{type_}']
            self.all_feature_values_.extend(type_values)

        self.cols_transform_dict_ = {}
        for colname, values in self.cols_values_dict_.items():
            col_values = list(map(lambda x: f'{colname}_{x}', values))
            col_values = col_values + [f'rare_{colname}']
            self.cols_transform_dict_[colname] = pd.DataFrame(0, index=X[self.idx].unique(), columns=col_values)
            self.all_feature_values_.extend(col_values)
            
            
        for colname, ohe_obj in self.ohe_cols_dict_.items(): 
            for pid, df_grp in grp:
                d = df_grp.groupby(colname)['time_elapsed'].first()
                values = []
                for val in d.index:
                    if len(val.split(',')) == 1:
                        values.append(val.strip())
                    else:
                        for v in val.split(','):
                            values.append(v.strip())
                # self.cols_transform_dict_[colname].loc[pid, list(map(lambda x: x if 'rare' in x else f'{colname}_{x}', d.index))] += 1
                if len(set(values)) == len(values):
                    self.cols_transform_dict_[colname].loc[pid, list(map(lambda x: x if 'rare' in x else f'{colname}_{x}', values))] += 1
                else:
                    for val in values:
                        self.cols_transform_dict_[colname].loc[pid, val if 'rare' in val else f'{colname}_{val}'] += 1
                    
        
        df_transform_types_all = [X for _, X in self.type_transform_dict_.items()] 
        df_transform_cols_all = [X for _, X in self.cols_transform_dict_.items()] 
        df_transform_all = df_transform_types_all + df_transform_cols_all

        return pd.concat(df_transform_all, axis=1)


    def fit_transform(self, X, y=0):
        self.fit(X, y)
        return self.transform(X, y)


if __name__ == "__main__":
    parent_dir = os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel')
    df = pd.read_csv(os.path.join(parent_dir, 'dynamic_180.csv'))
    df = basic_preprocess(df)
    df = category_mappers_dynamic(df)
    # dt = DynamicTranformer('PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', 'EVENT_NAME')
    
    
    dt = DynamicTranformer('PAT_ENC_CSN_ID','Calculated_DateTime', 'Type', 'EVENT_NAME', 'time_elapsed',
                           type_config=dict(
                                type_attr = ['Lab Order - Result'],
                                type_extra_col = ['Result_Flag'],
                                type_extra_col_concat = ['_'], 
                                type_thresh = [0.95], 
                                type_apply_thresh = [500],
                                type_regex = [True],

                                type_global_thresh = 0.95, 
                                type_global_apply_thresh = 500,
                                type_global_regex= True
                           ),
                           cols_config={
                                'colnames': ['Primary_DX_ICD10'], # change it to DX_ICD10
                                'threshold': [0.95], # To avoid doing any reduction to the feature space, then you can use None or >=1.
                                'regex':[False],
                                'sep':[','],
                               'apply_threshold':[500]
                           })
    

                           
                        #    cols_config={
                        #         'colnames': ['Primary_DX_ICD10', 'Chief_Complaint'], # change it to DX_ICD10
                        #         'threshold': [0.95, None], # To avoid doing any reduction to the feature space, then you can use None or >=1.
                        #         'regex':[False],
                        #         'sep':[',', None]
                        #    })
    pats = np.random.choice(df['PAT_ENC_CSN_ID'].unique(), int(df['PAT_ENC_CSN_ID'].nunique()*0.02), replace=False)
    more_pats = [693774769, 693830255, 693877638, 693886579, 693575737, 693558304, 693550251, 693550085]
    pats = np.concatenate([pats, more_pats])
    df_sample = df[df['PAT_ENC_CSN_ID'].isin(pats)]
    df_transform_all = dt.fit_transform(df_sample)
    x= 0
