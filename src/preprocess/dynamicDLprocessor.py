import os
import sys
from typing import Type

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

from collections import defaultdict, Counter
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from const import constants
from utils.utils import category_mappers_dynamic_dl, category_mappers_static_dl, basic_preprocess
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from utils.utils import clean_string
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def create_dictionary_static(df, colname, includeNull=False, regex=True, type_='str'):
    event_counter = defaultdict(int)
    if type_ == 'str':
        for id_, row in df.iterrows():
            if row[colname].startswith('<'):
                event_counter[row[colname]]+=1
            elif len(row[colname].split(',')) == 1:
                val = clean_string(row[colname])
                event_counter[val]+=1
            else:
                for val in row[colname].split(','):
                    val = clean_string(val)
                    event_counter[val] += 1
    else:
        for id_, row in df.iterrows():
            if np.isnan(row[colname]):
                event_counter['<NUL>'] += 1
            else:
                event_counter[str(row[colname])] += 1
                
    sorted_events = sorted(event_counter.items(), key=lambda kv: kv[1], reverse=True)
    
    if includeNull:
        vocab = {'<PAD>': 0, '<UNK>': 1, '<NUL>':2}
    else:
        vocab = {'<PAD>': 0, '<UNK>': 1}

    for key, _ in sorted_events:
        vocab[key] = len(vocab)
    return vocab

# Run on the training set
def create_dictionary(df, colname, id_col, includeNull=False, regex=True):
    event_counter = defaultdict(int)
    for pat_id, df_gr in df.groupby(id_col):
        for id, vals in df_gr[colname].items():
            if len(vals.split(','))==1:
                if vals.startswith('<'):
                    event_counter[vals]+=1
                    continue

                if regex:
                    vals = re.sub(r'[^a-zA-Z0-9_]+', '', vals.strip()).lower()
                else:
                    vals = vals.strip()
                event_counter[vals]+=1
            else:
                for val in vals.split(','):
                    if regex:
                        val = re.sub(r'[^a-zA-Z0-9_]+', '', val.strip()).lower()
                    else:
                        val = val.strip()
                    event_counter[val]+=1
                
    sorted_events = sorted(event_counter.items(), key=lambda kv: kv[1], reverse=True)
    
    if includeNull:
        vocab = {'<PAD>': 0, '<UNK>': 1, '<NUL>':2}
    else:
        vocab = {'<PAD>': 0, '<UNK>': 1}

    for key, _ in sorted_events:
        vocab[key] = len(vocab)
    return vocab

def event2seq(df, colname, vocab_dict, multi=True, regex=True, type_='str'):
    def process_multi(x):
        if len(x.split(','))==1:
            if regex:
                x = re.sub(r'[^a-zA-Z0-9_]+', '', x.strip()).lower()
                return [vocab_dict.get(x, 1)]
            else:
                return [vocab_dict.get(x.strip(),1)]
        o = []
        for val in x.split(','):
            if regex:
                val = re.sub(r'[^a-zA-Z0-9_]+', '', val.strip()).lower()
                o.append(vocab_dict.get(val, 1))
            else:
                o.append(vocab_dict.get(val.strip(), 1))
        return o

    if type_ == 'str':
        if multi:
            df[f'{colname}_seq'] = df[colname].apply(process_multi) 
        else:
            df[f'{colname}_seq'] = df[colname].apply(lambda x: vocab_dict.get(x.strip(), 1)) 
    else:
        try:
            df[f'{colname}_seq'] = df[colname].apply(lambda x: vocab_dict.get(str(x)))
        except Exception as e:
            x =0

    return df


def preprocess_df(df, colname):
    pass

def add_elapsedtime(df, time_col, id_col):
    # t1 = time.time() # 100 times slower
    # df = df.sort_values(by=time_col)
    # df['elapsed_time_hr'] = df.groupby(id_col)[time_col].transform(lambda x: (x-x.iloc[0]).dt.total_seconds()/(60*60))
    # print(f'Conventional method:{time.time()-t1} seconds...')

    t1 = time.time()
    df = df.sort_values(by=[time_col, id_col])
    first_stamps = df.groupby(id_col)[time_col].transform('first')
    df['elapsed_time_hr'] = (df[time_col]-first_stamps).dt.total_seconds()/3600
    print(f'Vectorized method:{time.time()-t1} seconds...')

    # print((df['elapsed_time_hr']!=df['elapsed_time_hr_eff']).sum())

    return df

'''
TODO: Handle splitting data into longer time periods such that you can have more training data for deep learning 
'''

def load_preprocess_df(ml_path, ds_path, event_idx, fold_idx):
    sf_i_tr = os.path.join(ml_path, f'static_{event_idx}_train_{fold_idx}.csv')
    sf_i_te = os.path.join(ml_path, f'static_{event_idx}_test_{fold_idx}.csv')
    dynamic_path = os.path.join(ds_path, f'dynamic_{event_idx}.csv')
    sf_i = os.path.join(ds_path, f'static_{event_idx}.csv')
    df_s_tr = pd.read_csv(sf_i_tr)
    df_s_te = pd.read_csv(sf_i_te)
    df_s  = pd.read_csv(sf_i)
    df_dyn =pd.read_csv(dynamic_path)
    
    df_s = df_s[df_s['Arrived_Time_appx']!='-1']
    df_s = basic_preprocess(df_s)
    df_dyn = basic_preprocess(df_dyn)
    df_dyn = category_mappers_dynamic_dl(df_dyn)
    df_s = category_mappers_static_dl(df_s)

    # Generic preprocessing
    indicies = df_dyn[df_dyn['Type']=='Lab Order - Result'].index
    df_dyn.loc[indicies, 'EVENT_NAME'] = df_dyn.loc[indicies, 'EVENT_NAME']+'_'+df_dyn.loc[indicies,'Result_Flag']
    df_dyn['Mixed_Event_Type'] = df_dyn['Type'] + '_' +df_dyn['EVENT_NAME']

    df_dyn['Mixed_Event_Type'] = df_dyn['Mixed_Event_Type'].\
        apply(lambda x: re.sub(r'[^a-zA-Z0-9_]+', '', x)).str.lower()

    df_dyn['Calculated_DateTime'] = pd.to_datetime(df_dyn['Calculated_DateTime'])
    df_dyn = add_elapsedtime(df_dyn, 'Calculated_DateTime', 'PAT_ENC_CSN_ID')

    # df_dyn.loc[df_dyn['Primary_DX_ICD10'].isna(), 'Primary_DX_ICD10'] = '<NUL>'
    # df_s.loc[df_s['Chief_Complaint_All'].isna(), 'Chief_Complaint_All'] = '<NUL>'

    df_s.loc[df_s['Sex']=='Male', 'Sex'] = 1
    df_s.loc[df_s['Sex']=='Female', 'Sex'] = 0
    df_s['Sex'] = df_s['Sex'].astype(np.int32)
    df_s.loc[df_s['Has Completed Appt in Last Seven Days']=='Yes', 'Has Completed Appt in Last Seven Days'] = 1
    df_s.loc[df_s['Has Completed Appt in Last Seven Days']=='No', 'Has Completed Appt in Last Seven Days'] = 0
    df_s['Has Completed Appt in Last Seven Days'] = df_s['Has Completed Appt in Last Seven Days'].astype(np.int32)

    df_s.loc[df_s['Has Hospital Encounter in Last Seven Days']=='Yes', 'Has Hospital Encounter in Last Seven Days'] = 1
    df_s.loc[df_s['Has Hospital Encounter in Last Seven Days']=='No', 'Has Hospital Encounter in Last Seven Days'] = 0
    df_s['Has Hospital Encounter in Last Seven Days'] = df_s['Has Hospital Encounter in Last Seven Days'].astype(np.int32)

    df_s['arr_hr'] =   df_s['Arrived_Time_appx'].dt.hour
    df_s['arr_dow'] =  df_s['Arrived_Time_appx'].dt.dayofweek
    df_s['arr_mnth'] = df_s['Arrived_Time_appx'].dt.month
    df_s['arr_year'] = df_s['Arrived_Time_appx'].dt.year

    cal = calendar()
    holidays = cal.holidays(start=df_s['Arrived_Time_appx'].min(), end=df_s['Arrived_Time_appx'].max())
    df_s['is_holiday'] = df_s['Arrived_Time_appx'].dt.date.isin(holidays.date)

    df_s.loc[df_s['is_holiday']==True,'is_holiday'] = 1 
    df_s.loc[df_s['is_holiday']==False,'is_holiday'] = 0
    df_s['is_holiday'] = df_s['is_holiday'].astype(np.int32)

    df_dyn_tr = df_dyn[df_dyn['PAT_ENC_CSN_ID'].isin(df_s_tr['PAT_ENC_CSN_ID'])]
    df_s_tr = df_s[df_s['PAT_ENC_CSN_ID'].isin(df_s_tr['PAT_ENC_CSN_ID'])]

    df_dyn_te = df_dyn[df_dyn['PAT_ENC_CSN_ID'].isin(df_s_te['PAT_ENC_CSN_ID'])]
    df_s_te = df_s[df_s['PAT_ENC_CSN_ID'].isin(df_s_te['PAT_ENC_CSN_ID'])]

    return {
        'static':[df_s_tr, df_s_te],
        'dynamic':[df_dyn_tr, df_dyn_te]
    }

def create_static_dictionaries(df, col_param_dict, hr_mnth_dow=True):
    '''
    col_param_dict: {dictkey: (colname: str[colnames], includeNull: bool, regex: bool, type_: str['str'/*] )} 
    '''
    output_dict = {}
    for key, values in col_param_dict.items():
        if isinstance(values, (list, tuple)):
            output_dict[key] = create_dictionary_static(df, *values)
        elif isinstance(values, dict):
            output_dict[key] = create_dictionary_static(df, **values)
        else:
            raise ValueError(f'col_param_dict values can only be either list/tuple, or dict. {type(values)} is recieved ...')

    # vocab_cfcg_static = create_dictionary_static(df_s_tr, 'Coverage_Financial_Class_Grouper',  False, True)
    # vocab_cc_static   = create_dictionary_static(df_s_tr, 'Chief_Complaint_All', False, True)
    # vocab_eth_static  = create_dictionary_static(df_s_tr, 'Ethnicity', False, True)
    # vocab_fr_static   = create_dictionary_static(df_s_tr, 'FirstRace', False, True)
    # vocab_al_static   = create_dictionary_static(df_s_tr, 'Acuity_Level', False, True)
    # vocab_ma_static   = create_dictionary_static(df_s_tr, 'Means_Of_Arrival', False, True)

    if hr_mnth_dow:
        vocab_hr_static = {'<PAD>':0}; vhs = {str(x):x+1 for x in range(0, 24)}; vocab_hr_static.update(vhs)
        vocab_mnth_static = {'<PAD>':0}; vms = {str(x):x for x in range(1, 13)}; vocab_mnth_static.update(vms)
        vocab_dow_static = {'<PAD>':0}; vds = {str(x):x+1 for x in range(0, 7)}; vocab_dow_static.update(vds)
        output_dict['hr'] = vocab_hr_static
        output_dict['mnth'] = vocab_mnth_static
        output_dict['dow'] = vocab_dow_static

    # return {
    #     'cfcg': vocab_cfcg_static,
    #     'cc': vocab_cc_static,
    #     'eth': vocab_eth_static,
    #     'fr': vocab_fr_static,
    #     'al': vocab_al_static,
    #     'ma': vocab_ma_static,
    #     'hr': vocab_hr_static,
    #     'mnth': vocab_mnth_static,
    #     'dow': vocab_dow_static
    # }
    return output_dict

def create_dynamic_dictionaries(df, col_param_dict):
    '''
    col_param_dict: {dictkey: (colname: str[colnames], pat_id, includeNull: bool, regex: bool)} 
    '''
    output_dict = {}
    for key, values in col_param_dict.items():
        if isinstance(values, (list, tuple)):
            output_dict[key] = create_dictionary(df, *values)
        elif isinstance(values, dict):
            output_dict[key] = create_dictionary(df, **values)
        else:
            raise ValueError(f'col_param_dict values can only be either list/tuple, or dict. {type(values)} is recieved ...')
            
        
    # vocab_event = create_dictionary(df_dyn_tr, 'Mixed_Event_Type', 'PAT_ENC_CSN_ID', False, False) # I have already applied regex to the whole column at the beginning of the preprocessing pipeline
    # vocab_dx    = create_dictionary(df_dyn_tr, 'Primary_DX_ICD10', 'PAT_ENC_CSN_ID', False, False)
    # return {
    #     'event': vocab_event,
    #     'dx': vocab_dx
    # }
    return output_dict
    
def save_dictionaries(dict_vocab_dict, event_idx, fold_idx, save_dir, type_):
    if save_dir is None:
        save_dir = constants.DL_FEATS_DIR
    for key, vocab in dict_vocab_dict.items():
        with open(os.path.join(save_dir, f'{key}_{type_}_{event_idx}_{fold_idx}.joblib'), 'wb') as f:
            joblib.dump(vocab, f) 

def event2seqall(df, event2seq_dict):
    '''
    event2eq_dict: {colname_key: (colname, vocab_dict, multival[True/False], regex[True/False], type_['str'/*])}
    '''
    new_col_names = {}
    for key, eventseqdict in event2seq_dict.items():
        new_col_names[key] = f'{key}_seq'
        if isinstance(eventseqdict, (list, tuple)):
            df   = event2seq(df, *eventseqdict)
        elif isinstance(eventseqdict, dict):
            df   = event2seq(df, **eventseqdict)
        else:
            raise ValueError(f'event2seq_dict values can be either list/tuple, or dict. {type(eventseqdict)}')
    return df

def convert_df_vocab_2_seq(df, vocab_seq_params):
    for key, values in vocab_seq_params.items():
        if isinstance(values, (list, tuple)):
            df = event2seq(df, *values)
        elif isinstance(values, dict):
            df = event2seq(df, **values)
        else:
            raise ValueError(f'vocab_seq_params values are expected to be either ')
    return df

def save_dl_preprocessed_df(data_dict, save_params, save_path=None):
    output_main_dir = ""
    if save_path is None:
        output_main_dir += constants.DL_FEATS_DIR
    else:
        output_main_dir += save_path
        
    for key in data_dict:
        for idx, df in enumerate(data_dict[key]):
            filename = "_".join(save_params[key][idx])
            df.to_csv(os.path.join(output_main_dir, 'df_'+filename+'.csv')) 

def preprocess_for_training(
    ml_dir,
    ds_dir,
    event_idx,
    fold_idx,
    static_dictionaries_params,
    dynamic_dictionaries_params,
    include_date_vocabs,
    output_path

):
    static_dynamic_df_dict =\
        load_preprocess_df(ml_dir, ds_dir, event_idx, fold_idx)
    
    static_dictionaries =\
        create_static_dictionaries(static_dynamic_df_dict['static'][0], static_dictionaries_params,
                                   include_date_vocabs)
    dynamic_dictionaries =\
        create_dynamic_dictionaries(static_dynamic_df_dict['dynamic'][0], dynamic_dictionaries_params)

    save_dictionaries(static_dictionaries, event_idx, fold_idx, output_path, type_='static')
    save_dictionaries(dynamic_dictionaries, event_idx, fold_idx, output_path, type_='dynamic')

    static_event2seqall_params = {
        #'key': (colname, vocab_dict, multi, regex, type_)
        'cfcg': ('Coverage_Financial_Class_Grouper', static_dictionaries['cfcg'], False, True, 'str'),
        'cc':   ('Chief_Complaint_All', static_dictionaries['cc'], True, True, 'str'),
        'ethn':  ('Ethnicity', static_dictionaries['ethn'], False, True, 'str'),
        'fr':   ('FirstRace', static_dictionaries['fr'], False, True, 'str'),
        'al':   ('Acuity_Level', static_dictionaries['al'], False, True, 'str'),
        'ma':   ('Means_Of_Arrival', static_dictionaries['ma'], False, True, 'str')
    }
    if include_date_vocabs:
        static_event2seqall_params.update(
            {
                'hr': ('arr_hr', static_dictionaries['hr'], False, False, 'int'),
                'dow': ('arr_dow', static_dictionaries['dow'], False, False, 'int'),
                'mnth': ('arr_mnth', static_dictionaries['mnth'], False, False, 'int'),
            }
        )

    dynamic_event2seqall_params = {
        #'key': (colname, vocab_dict, multi, regex, type_)
        'ev': ('Mixed_Event_Type', dynamic_dictionaries['ev'], False, False, 'str'), # Regex is false here because I have already applied it in the preprocessing step
        'dx':   ('Primary_DX_ICD10', dynamic_dictionaries['dx'], True, False, 'str'),
    }

    for key, vals in static_dynamic_df_dict.items():
        if key == 'static':
            for sidx, df in enumerate(vals):
                static_dynamic_df_dict[key][sidx] = convert_df_vocab_2_seq(df, static_event2seqall_params)
        elif key == 'dynamic':
            for didx, df in enumerate(vals):
                static_dynamic_df_dict[key][didx] = convert_df_vocab_2_seq(df, dynamic_event2seqall_params)

    save_dl_preprocessed_df(static_dynamic_df_dict, {
        'static': ( ('1s_tr', str(event_idx), str(fold_idx)), ('1s_te', str(event_idx), str(fold_idx))),
        'dynamic': ( ('1dyn_tr', str(event_idx), str(fold_idx)), ('1dyn_te', str(event_idx), str(fold_idx)) )

    }, output_path)
    return static_dynamic_df_dict, static_dictionaries, dynamic_dictionaries

def load_vocab_dict(data_path, type_, event_idx=None, fold_idx=None):
    if event_idx is None and fold_idx is None:
        files = [file for file in os.listdir(data_path) if (type_ in file) and file.endswith('.joblib') and not file.startswith('vocab')]
    else:
        files = [file for file in os.listdir(data_path) if f'{type_}_{event_idx}_{fold_idx}.joblib' in file and not file.startswith('vocab')]
    vocab_dict = {}
    for file in files:
        filepath = os.path.join(data_path, file)
        with open(filepath, 'rb') as f:
            vocab_dict[file.split('_')[0]] = joblib.load(f)
    return vocab_dict
            
def preprocess_for_testing(
    ml_dir,
    ds_dir,
    df_event_idx,
    df_fold_idx,
    vocab_event_idx,
    vocab_fold_idx,
    static_vocab_path,
    dynamic_vocab_path,
    include_date_vocabs=True,
    output_path=None 
):
    static_dynamic_df_dict =\
        load_preprocess_df(ml_dir, ds_dir, df_event_idx, df_fold_idx)
    
    if isinstance(static_vocab_path, str):
        static_dictionaries = load_vocab_dict(static_vocab_path, 'static', vocab_event_idx, vocab_fold_idx)
    elif isinstance(static_vocab_path, dict):
        static_dictionaries = static_vocab_path
    else:
        raise TypeError(f'static_vocab_path can be either str or dict. {type(static_vocab_path)} is recieved ...')

    if isinstance(dynamic_vocab_path, str):
        dynamic_dictionaries = load_vocab_dict(dynamic_vocab_path, 'dynamic', vocab_event_idx, vocab_fold_idx)
    elif isinstance(dynamic_vocab_path, dict):
        dynamic_dictionaries = dynamic_vocab_path
    else:
        raise TypeError(f'dynamic_vocab_path can be either str or dict. {type(dynamic_vocab_path)} is recieved ...')
        
    static_event2seqall_params = {
        #'key': (colname, vocab_dict, multi, regex, type_)
        'cfcg': ('Coverage_Financial_Class_Grouper', static_dictionaries['cfcg'], False, True, 'str'),
        'cc':   ('Chief_Complaint_All', static_dictionaries['cc'], True, True, 'str'),
        'eth':  ('Ethnicity', static_dictionaries['eth'], False, True, 'str'),
        'fr':   ('FirstRace', static_dictionaries['fr'], False, True, 'str'),
        'al':   ('Acuity_Level', static_dictionaries['al'], False, True, 'str'),
        'ma':   ('Means_Of_Arrival', static_dictionaries['ma'], False, True, 'str')
    }
    if include_date_vocabs:
        static_event2seqall_params.update(
            {
                'hr': ('arr_hr', static_dictionaries['hr'], False, False, 'int'),
                'dow': ('arr_dow', static_dictionaries['dow'], False, False, 'int'),
                'mnth': ('arr_mnth', static_dictionaries['mnth'], False, False, 'int'),
            }
        )
     
    dynamic_event2seqall_params = {
        #'key': (colname, vocab_dict, multi, regex, type_)
        'event': ('Mixed_Event_Type', dynamic_dictionaries['event'], False, False, 'str'), # Regex is false here because I have already applied it in the preprocessing step
        'dx':   ('Primary_DX_ICD10', dynamic_dictionaries['dx'], True, False, 'str'),
    }
    for key, vals in static_dynamic_df_dict.items():
        if key == 'static':
            for sidx, df in enumerate(vals):
                static_dynamic_df_dict[key][sidx] = convert_df_vocab_2_seq(df, static_event2seqall_params)
        elif key == 'dynamic':
            for didx, df in enumerate(vals):
                static_dynamic_df_dict[key][didx] = convert_df_vocab_2_seq(df, dynamic_event2seqall_params)

    save_dl_preprocessed_df(static_dynamic_df_dict, {
        'static': ( ('s_tr', str(df_event_idx), str(df_fold_idx), 'test'), ('1s_te', str(df_event_idx), str(df_fold_idx), 'test')),
        'dynamic': ( ('dyn_tr', str(df_event_idx), str(df_fold_idx), 'test'), ('1dyn_te', str(df_event_idx), str(df_fold_idx), 'test') )

    }, output_path) 
    return static_dynamic_df_dict, static_dictionaries, dynamic_dictionaries 
    

if __name__ == "__main__":
    tr_input_dict = dict(
        ml_dir = constants.ML_DATA_OUTPUT_ID,
        ds_dir = constants.DS_DATA_OUTPUT,
        fold_idx = 1,
        event_idx = 99,
        include_date_vocabs = True,
        static_dictionaries_params = {
            'cfcg': ('Coverage_Financial_Class_Grouper', False, True),
            'cc':   ('Chief_Complaint_All', False, True),
            'eth':  ('Ethnicity', False, True),
            'fr':   ('FirstRace', False, True),
            'al':   ('Acuity_Level', False, True),
            'ma':   ('Means_Of_Arrival', False, True)
        },
        dynamic_dictionaries_params = {
            'event': ('Mixed_Event_Type', 'PAT_ENC_CSN_ID', False, False),
            'dx':  ('Primary_DX_ICD10', 'PAT_ENC_CSN_ID', False, False)
        },
        output_path = constants.DL_FEATS_DIR
    )
    
    te_input_dict = dict(
        ml_dir = constants.ML_DATA_OUTPUT_ID,
        ds_dir = constants.DS_DATA_OUTPUT,
        fold_idx = 1,
        event_idx = 99,
        static_vocab_path=constants.DL_FEATS_DIR,
        dynamic_vocab_path=constants.DL_FEATS_DIR,
        include_date_vocabs = True,
        output_path = constants.DL_FEATS_DIR
    )

    data_dict1 = preprocess_for_training(**tr_input_dict)
    data_dict2, dynamic_dictionaries = preprocess_for_testing(**te_input_dict)