import re
import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import ast
import joblib

class CombinedDataset(Dataset):
    def __init__(self, df_static, df_dynamic, pat_id_col, time_col, label_col, trim_idx=99, static_transform=None, dynamic_transform=None,
                 max_cc_len=7, max_dx_len = 7):
        unamed_col = [col for col in df_static if 'unnam' in col.lower()]
        if len(unamed_col) > 0:
            df_static.drop(columns=unamed_col, inplace=True)

        unamed_col = [col for col in df_dynamic if 'unnam' in col.lower()]
        if len(unamed_col) > 0:
            df_dynamic.drop(columns=unamed_col, inplace=True)
        
        # df_dynamic.drop(pat_id_col, axis=1, inplace=True)
        self.trim_idx = trim_idx

        self.df_s = df_static
        self.df_d = df_dynamic.sort_values(by=time_col)

        self.max_cc_len = max_cc_len
        self.max_dx_len = max_dx_len
        
        self.pat_ids = df_static[pat_id_col]
        # self.labels = df_static.pop(label_col)

        self.pat_id_col = pat_id_col
        self.label_col = label_col

        self.static_transform = static_transform
        self.dynamic_transform = dynamic_transform 

    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, idx):
        pat_idx = self.pat_ids.iloc[idx]
        static_row = self.df_s[self.df_s[self.pat_id_col] == pat_idx]
        dynamic_mat = self.df_d[self.df_d[self.pat_id_col] == pat_idx]
        dynamic_mat = dynamic_mat.iloc[:self.trim_idx]
        label = static_row[self.label_col]

        if self.static_transform:
            static_row = self.static_transform(static_row)
        
        if self.dynamic_transform:
            dynamic_mat = self.dynamic_transform(dynamic_mat)
        
        # Static features (I am not including years because I train and test within 6 month)
        # Sequences
        cc_seq = static_row.loc[:, 'Chief_Complaint_All_seq'] # Can hold multivalues
        ethn_seq = static_row.loc[:, 'Ethnicity_seq'] 
        firstrace_seq = static_row.loc[:, 'FirstRace_seq']
        al_seq = static_row.loc[:, 'Acuity_Level_seq']
        cfcg_seq = static_row.loc[:, 'Coverage_Financial_Class_Grouper_seq']
        moa_seq = static_row.loc[:, 'Means_Of_Arrival_seq']
        hr_seq = static_row.loc[:, 'arr_hr_seq']
        mnth_seq = static_row.loc[:, 'arr_mnth_seq']
        dow_seq = static_row.loc[:, 'arr_dow_seq']
        # Numerical features
        stat_num_feats = static_row.loc[:, ['Sex', 'MultiRacial',  'Has Completed Appt in Last Seven Days', 'Has Hospital Encounter in Last Seven Days', 'is_holiday',  # Binary columns (0/1)
                                         'Patient_Age', 'Number of Inpatient Admissions in the last 30 Days',
                                         'Number of past appointments in last 60 days',
                                         'Number of past inpatient admissions over ED visits in last three years',
                                         'Count_of_Chief_Complaints']]
        assert all([len(x) == 1 for x in [cc_seq, ethn_seq, firstrace_seq, al_seq, cfcg_seq, moa_seq,
                                          hr_seq, mnth_seq, dow_seq, stat_num_feats]]), 'Static features should have only one row'
        
        num_feats_torch = torch.tensor(stat_num_feats.astype(np.float).values, dtype=torch.float).squeeze()
        ethn_seq_torch = torch.tensor(ethn_seq.iloc[0], dtype=torch.long)
        al_seq_torch = torch.tensor(al_seq.iloc[0], dtype=torch.long)
        cfcg_seq_torch = torch.tensor(cfcg_seq.iloc[0], dtype=torch.long)
        moa_seq_torch = torch.tensor(moa_seq.iloc[0], dtype=torch.long)
        hr_seq_torch = torch.tensor(hr_seq.iloc[0], dtype=torch.long)
        mnth_seq_torch = torch.tensor(mnth_seq.iloc[0], dtype=torch.long)
        dow_seq_torch = torch.tensor(dow_seq.iloc[0], dtype=torch.long)
        firstrace_seq_torch = torch.tensor(firstrace_seq.iloc[0], dtype=torch.long)
        
        cc_seq_torch = torch.zeros((self.max_cc_len), dtype=torch.long)
        cc_seq_l = cc_seq.iloc[0]
        if len(cc_seq_l) > self.max_cc_len:
            cc_seq_torch = torch.tensor(cc_seq_l[:self.max_cc_len])
        else:
            cc_seq_torch[:len(cc_seq_l)] = torch.tensor(cc_seq_l)

        # Dynamic features
        # Sequences
        met_seq = dynamic_mat.loc[:, 'Mixed_Event_Type_seq'] # Requires single value embedding
        pdx_seq = dynamic_mat.loc[:, 'Primary_DX_ICD10_seq'] # Requires Multival embeddings
        elapsed_time = dynamic_mat.loc[:, 'elapsed_time_hr']
        
        elapsed_time_torch = torch.tensor(elapsed_time.values, dtype=torch.float)
        dx_seq_torch = torch.zeros((len(pdx_seq), self.max_dx_len), dtype=torch.long)
        met_seq_torch = torch.tensor(met_seq.values, dtype=torch.long)

        for idx, (_, row) in enumerate(pdx_seq.items()):
            if len(row) > self.max_dx_len:
                dx_seq_torch[idx] = torch.tensor(row[:self.max_dd_len]).squeeze()
            else:
                dx_seq_torch[idx,:len(row)] = torch.tensor(row).squeeze()
        
        return {
            's_cc_seq': cc_seq_torch,
            's_ethn_seq': ethn_seq_torch,
            's_fr_seq': firstrace_seq_torch,
            's_al_seq': al_seq_torch,
            's_cfcg_seq': cfcg_seq_torch,
            's_ma_seq': moa_seq_torch,
            's_hr_seq': hr_seq_torch,
            's_mnth_seq': mnth_seq_torch,
            's_dow_seq': dow_seq_torch,
            's_num_feats': num_feats_torch,

            'd_ev_seq': met_seq_torch,
            'd_dx_seq': dx_seq_torch,
            'd_elt': elapsed_time_torch,
            'label': torch.tensor(label.iloc[0], dtype=torch.long)
        }

class DynamicTransformer:
    def __init__(self, df, idcol, col_dict):
        self.param_dict = self._get_params(df, idcol, col_dict)
        self.col_dict = col_dict
    
    def _get_params(self, df, idcol, col_dict):
        param_dict = {}
        for col, val in col_dict.items():
            mean=0;std=0
            minval=0;maxval=0
            for pid, df_grb in df.groupby(idcol):
                if val == 'std':
                    mean += df_grb[col].mean()
                    std += df_grb[col].std(ddof=1)
                elif val == 'minmax':
                    minval += df_grb[col].min()
                    maxval += df_grb[col].max()
                else:
                    raise ValueError('StaticTransfoermer only accepts std, or minmax transformation')
            if val == 'std':
                mean = mean/df[idcol].nunique()
                std = std/df[idcol].nunique()
                param_dict[col] = (mean, std)
            elif val =='minmax':
                minval = minval/df[idcol].nunique()
                maxval = maxval/df[idcol].nunique()
                param_dict[col] = (minval, maxval)
        return param_dict

    def __call__(self, x):
        for col, (v1, v2) in self.param_dict.items():
            if col in x.columns:
                if self.col_dict[col] == 'std':
                    x[col] = (x[col]-v1)/(v2+1e-7)
                elif self.col_dict[col] == 'minmax':
                    x[col] = (x[col]-v1)/(v2-v1+1e-7)
        return x
        
class StaticTransformer:
    def __init__(self, df, col_dict):
        self.col_dict = col_dict
        self.param_dict = self._get_params(df, col_dict)
        
    def _get_params(self, df, col_dict):
        param_dict = {}
        for col, val in col_dict.items():
            if val == 'std':
                param_dict[col] = (df[col].mean(), df[col].std(ddof=1))
            elif val == 'minmax':
                param_dict[col] = (df[col].min(), df[col].max())
            else:
                raise ValueError('StaticTransfoermer only accepts std, or minmax transformation')

        return param_dict 
    
    def __call__(self, x):
        for col, (v1, v2) in self.param_dict.items():
            if col in x.columns:
                if self.col_dict[col] == 'std':
                    x[col] = (x[col]-v1)/(v2+1e-7)
                elif self.col_dict[col] == 'minmax':
                    x[col] = (x[col]-v1)/(v2-v1+1e-7)
        return x

def clean_df(df):
    unamed = [col for col in df.columns if 'unnam' in col.lower()]
    if len(unamed) > 0:
        df.drop(columns=unamed, inplace=True)
    if 'Type' in df.columns:
        df.drop('Type', axis=1, inplace=True)
    if 'EVENT_NAME' in df.columns:
        df.drop('EVENT_NAME', axis=1, inplace=True)
    if 'Order_Status' in df.columns:
        df.drop('Order_Status', axis=1, inplace=True)
    if 'Result_Flag' in df.columns:
        df.drop('Result_Flag', axis=1, inplace=True)
    if 'Primary_DX_Name' in df.columns:
        df.drop('Primary_DX_Name', axis=1, inplace=True)
    return df

def ast_exp_cols(df_tr, ast_cols):
    for col in ast_cols:
        df_tr[col]    = df_tr[col].apply(lambda x: ast.literal_eval(x))
    return df_tr

def filter_patID_basedOn_eventIdx(df_tr, pat_id, validate_event_idx):
    pat_enc_included_tr = []
    for pid, df_grb in df_tr.groupby(pat_id):
        if len(df_grb)>=validate_event_idx:
            pat_enc_included_tr.append(pid)
    df_tr =  df_tr[df_tr[pat_id].isin(pat_enc_included_tr)]
    return df_tr

def filter_patID_basedOn_list(df_tr, pat_id, pat_idlist):
    df_tr = df_tr[df_tr[pat_id].isin(pat_idlist)]
    return df_tr

def preprocess_tr_df(df_tr, pat_id='PAT_ENC_CSN_ID', validate_event_idx=None):
    df_tr = clean_df(df_tr)
    if isinstance(validate_event_idx, int):
        pat_enc_included_tr = []
        for pid, df_grb in df_tr.groupby(pat_id):
            if len(df_grb)>=validate_event_idx:
                pat_enc_included_tr.append(pid)
        df_tr =  df_tr[df_tr[pat_id].isin(pat_enc_included_tr)]
    else:
        df_tr = df_tr[df_tr[pat_id].isin(validate_event_idx)]

    return df_tr

def preprocess_te_df(df_te, pat_id='PAT_ENC_CSN_ID', included_pat_id_list=[]):
    df_te = clean_df(df_te)
    df_te = df_te[df_te[pat_id].isin(included_pat_id_list)]
    return df_te

def load_vocab_dict(data_folder, event_idx, fold_idx):

    with open(os.path.join(data_folder, f'vocab_event_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_event = joblib.load(f) 

    with open(os.path.join(data_folder, f'vocab_dx_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
       vocab_dx =  joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_cc_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_cc_static = joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_eth_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_eth_static = joblib.load(f) 

    with open(os.path.join(data_folder, f'vocab_fr_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_fr_static = joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_al_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_al_static = joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_ma_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_ma_static=joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_hr_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_hr_static= joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_mnth_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_mnth_static =joblib.load( f) 

    with open(os.path.join(data_folder, f'vocab_dow_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_dow_static = joblib.load(f) 

    with open(os.path.join(data_folder, f'vocab_cfcg_static_{event_idx}_{fold_idx}.joblib'), 'rb') as f:
        vocab_dow_static = joblib.load(f) 

    return {
        'd_ev': vocab_event,
        'd_dx': vocab_dx,
        's_cc': vocab_cc_static,
        's_ethn': vocab_eth_static,
        's_firstrace': vocab_fr_static,
        's_al': vocab_al_static,
        's_moa': vocab_ma_static,
        's_hr': vocab_hr_static,
        's_mnth': vocab_mnth_static,
        's_dow': vocab_dow_static
    }

def get_data_loaders(data_folder='/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats',
                     pat_id = 'PAT_ENC_CSN_ID',
                     didx = 'Calculated_DateTime',
                     target_col='Admitted_YN',
                     event_index=99, fold_idx=0, batch_size=32, train=True,
                     dynamic_transformer_dict=None,
                     static_transformer_dict=None,
                     dynamic_transformer_path=None,
                     static_transformer_path=None,
                     dynamic_ast_cols=[],
                     static_ast_cols=[],
                    #  load_vocabs=False
                     ):
    if isinstance(data_folder, str):
        if train:
            df_dyn_tr = pd.read_csv(os.path.join(data_folder, f'df_dyn_tr_{event_index}_{fold_idx}.csv'))
            df_dyn_te = pd.read_csv(os.path.join(data_folder, f'df_dyn_te_{event_index}_{fold_idx}.csv'))
            df_s_tr   = pd.read_csv(os.path.join(data_folder, f'df_s_tr_{event_index}_{fold_idx}.csv'))
            df_s_te   = pd.read_csv(os.path.join(data_folder, f'df_s_te_{event_index}_{fold_idx}.csv'))
        else:
            df_dyn_tr = pd.read_csv(os.path.join(data_folder, f'df_dyn_tr_{event_index}_{fold_idx}_test.csv'))
            df_dyn_te = pd.read_csv(os.path.join(data_folder, f'df_dyn_te_{event_index}_{fold_idx}_test.csv'))
            df_s_tr   = pd.read_csv(os.path.join(data_folder, f'df_s_tr_{event_index}_{fold_idx}_test.csv'))
            df_s_te   = pd.read_csv(os.path.join(data_folder, f'df_s_te_{event_index}_{fold_idx}_test.csv'))
        df_dyn_tr = ast_exp_cols(df_dyn_tr, dynamic_ast_cols)
        df_dyn_te = ast_exp_cols(df_dyn_te, dynamic_ast_cols)
        df_s_tr = ast_exp_cols(df_s_tr, static_ast_cols)
        df_s_te = ast_exp_cols(df_s_te, static_ast_cols)
    elif isinstance(data_folder, dict):
        df_dyn_tr = data_folder['dynamic'][0]
        df_dyn_te = data_folder['dynamic'][1]
        df_s_tr = data_folder['static'][0]
        df_s_te = data_folder['static'][1]
    else:
        raise TypeError(f"data_folder can be either str, or dict. {type(data_folder)} was given...")

    df_dyn_tr = preprocess_tr_df(df_dyn_tr, pat_id,  event_index)
    df_s_tr = preprocess_te_df(df_s_tr, pat_id, df_dyn_tr['PAT_ENC_CSN_ID'].unique())

    df_dyn_te = preprocess_tr_df(df_dyn_te, pat_id, event_index) 
    df_s_te = preprocess_te_df(df_s_te, pat_id, df_dyn_te['PAT_ENC_CSN_ID'].unique())

    if train:
        if dynamic_transformer_dict is None:
            dt = DynamicTransformer(df_dyn_tr, pat_id, {'elapsed_time_hr': 'std'})
        else:
            dt = DynamicTransformer(df_dyn_tr, pat_id, dynamic_transformer_dict)
        
        with open(dynamic_transformer_path.format(event_index, fold_idx), 'wb') as f:
            joblib.dump(dt, f)
            
        if static_transformer_dict is None:
            ds = StaticTransformer(df_s_tr, {"Patient_Age":'minmax', 'Number of Inpatient Admissions in the last 30 Days': 'minmax',
                                            'Number of past appointments in last 60 days':'minmax',
                                            'Number of past inpatient admissions over ED visits in last three years':'minmax',
                                            'Count_of_Chief_Complaints':'minmax'})
        else:
            ds = StaticTransformer(df_s_tr, static_transformer_dict)

        with open(static_transformer_path.format(event_index, fold_idx), 'wb') as f:
            joblib.dump(ds, f)

    else:
        # Load transformers
        if isinstance(dynamic_transformer_path, str):
            with open(dynamic_transformer_path.format(event_index, fold_idx), 'rb') as f:
                dt = joblib.load(f)
        elif isinstance(dynamic_transformer_path, DynamicTransformer):
            dt = dynamic_transformer_path
        
        if isinstance(static_transformer_path, str):
            with open(static_transformer_path.format(event_index, fold_idx), 'rb') as f:
                ds = joblib.load(f)
        elif isinstance(static_transformer_path, StaticTransformer):
            ds = static_transformer_path

        
    ds_tr = CombinedDataset(df_s_tr  ,
                            df_dyn_tr,
                            pat_id,
                            didx,
                            target_col, event_index, ds, dt) 
    ds_te = CombinedDataset(df_s_te  ,
                            df_dyn_te,
                            pat_id,
                            didx,
                            target_col, event_index, ds, dt) 
    dl_tr = DataLoader(ds_tr, batch_size)
    dl_te = DataLoader(ds_te, batch_size)

    # if load_vocabs:
    #     vocabs_dict = load_vocab_dict(data_folder, event_idx=event_index, fold_idx=fold_idx)
    # else: vocabs_dict = None
    # return dl_tr, dl_te, vocabs_dict
    return dl_tr, dl_te
    


if __name__ == '__main__':
    df_dyn_tr = pd.read_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats/df_dyn_tr.csv')
    df_dyn_te = pd.read_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats/df_dyn_te.csv')
    df_s_tr   = pd.read_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats/df_s_tr.csv')
    df_s_te   = pd.read_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats/df_s_te.csv')

    df_dyn_tr['Primary_DX_ICD10_seq']    = df_dyn_tr['Primary_DX_ICD10_seq'].apply(lambda x: ast.literal_eval(x))
    df_dyn_te['Primary_DX_ICD10_seq']    = df_dyn_te['Primary_DX_ICD10_seq'].apply(lambda x: ast.literal_eval(x))
    df_s_tr  ['Chief_Complaint_All_seq'] = df_s_tr['Chief_Complaint_All_seq'].apply(lambda x: ast.literal_eval(x))
    df_s_te  ['Chief_Complaint_All_seq'] = df_s_te['Chief_Complaint_All_seq'].apply(lambda x: ast.literal_eval(x))
    
    df_dyn_tr = clean_df(df_dyn_tr)
    df_dyn_te = clean_df(df_dyn_te)

    df_s_tr = clean_df(df_s_tr)
    df_s_te = clean_df(df_s_te)
    
    pat_enc_included_tr = []
    for pid, df_grb in df_dyn_tr.groupby('PAT_ENC_CSN_ID'):
        if len(df_grb)>=99:
            pat_enc_included_tr.append(pid)

    pat_enc_included_te = []
    for pid, df_grb in df_dyn_te.groupby('PAT_ENC_CSN_ID'):
        if len(df_grb)>=99:
            pat_enc_included_te.append(pid)

    df_dyn_tr =  df_dyn_tr[df_dyn_tr['PAT_ENC_CSN_ID'].isin(pat_enc_included_tr)]
    df_dyn_te =  df_dyn_te[df_dyn_te['PAT_ENC_CSN_ID'].isin(pat_enc_included_te)]
    df_s_tr =  df_s_tr[df_s_tr['PAT_ENC_CSN_ID'].isin(pat_enc_included_tr)]
    df_s_te =  df_s_te[df_s_te['PAT_ENC_CSN_ID'].isin(pat_enc_included_te)]

    dt = DynamicTransformer(df_dyn_tr, 'PAT_ENC_CSN_ID', {'elapsed_time_hr': 'std'})
    ds = StaticTransformer(df_s_tr, {"Patient_Age":'minmax', 'Number of Inpatient Admissions in the last 30 Days': 'minmax',
                                     'Number of past appointments in last 60 days':'minmax',
                                     'Number of past inpatient admissions over ED visits in last three years':'minmax',
                                     'Count_of_Chief_Complaints':'minmax'})
    ds_tr = CombinedDataset(df_s_tr  ,
                            df_dyn_tr,
                            'PAT_ENC_CSN_ID',
                            'Calculated_DateTime',
                            'Admitted_YN', 99, ds, dt) 
    # dl_tr = DataLoader(ds_tr, 8, collate_fn=collate)
    dl_tr = DataLoader(ds_tr, 8)
    for data in dl_tr:
        print(data)
    
    x=0