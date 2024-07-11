import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import numpy as np
import pandas as pd
import sys
from const import constants
import time
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


def get_target_flags(raw_file_path, event_col='EVENT_NAME', grb_col='Admitted_YN', flag='admi', normalize_text=True, corr_threshold=0.7):
    if isinstance(raw_file_path, str):
        df_raw = pd.read_csv(raw_file_path)
    else:
        df_raw = raw_file_path
    if normalize_text:
        df_raw[f'{event_col}_NORM'] = df_raw[event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x)).str.lower()
        event_col = f'{event_col}_NORM'

    flag_df = df_raw.loc[df_raw[event_col].str.lower().str.contains(flag), [event_col, grb_col]]
    flag_grb = flag_df.groupby(grb_col).value_counts()

    data = pd.DataFrame(0, index=flag_df[event_col].unique(), columns=flag_df[grb_col].unique())
    for (flag, event), value in flag_grb.items():
        data.loc[event, flag] = value
    
    data['total'] = data.sum(axis=1)
    # TODO: This is a hardcoded value of the target params and need to be changed manually
    data['perc_pos'] = data['Admitted']/data['total']

    target_events = data.loc[(data['perc_pos']>=corr_threshold)|(data['perc_pos']<=(1-corr_threshold))].index
    return target_events, data


def clean_target(raw_file_path, all_flags, event_col='EVENT_NAME', grb_id='PAT_ENC_CSN_ID',
                 event_time_col='Calculated_DateTime', orders_cols='Type', arrival_date_col='Arrived_Time', normalize_event_col=True,
                 min_date=None, max_date=None):
    if isinstance(raw_file_path, str):
        df_raw = pd.read_csv(raw_file_path)
    else:
        df_raw = raw_file_path
    df_raw[event_time_col] = pd.to_datetime(df_raw[event_time_col])
    df_raw[arrival_date_col] = pd.to_datetime(df_raw[arrival_date_col])

    if normalize_event_col:
        df_raw[f'{event_col}_NORM'] = df_raw[event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x)).str.lower()
        event_col = f'{event_col}_NORM'

    pat_groups = df_raw.sort_values(by=event_time_col).groupby(grb_id)

    # NOTE:
    #   1. Process only from the moment of arrival to ED, DO NOT FILTER ANY EVENTS PRIOR TO THE ARRIVAL [IGNORE FOR NOW BECAUSE OF THE ISSUE WITH ARRIVED_TIME]
    #   2. Looking for "Order - Admission" and "Order - Discharge" within the `orders_cols` as the actual flag in case none of the `all_flags` was encountered in the `eve`
    df_pat_list = []
    errors_list = []
    order_before_flag = []
    for pat_id, df_group in pat_groups:
        # df_group['time_since_arrival'] = (df_group[event_time_col]-df_group[arrival_date_col]).dt.total_seconds()/60
        # df_group = df_group[df_group['time_since_arrival']>=0]

        if min_date is not None and (df_group['Calculated_DateTime'].iloc[0]<min_date) or (df_group['Arrived_Time'].iloc[0]<min_date):
            continue
        df_group.reset_index(inplace=True)
        flag_d = df_group.loc[df_group[event_col].isin(all_flags)]
        order_d = df_group.loc[(df_group[orders_cols]=='Order - Admission')|(df_group[orders_cols]=='Order - Discharge')]
        if len(flag_d)>=1 and len(order_d)>=1:
            if flag_d.iloc[0]['Calculated_DateTime'] < order_d.iloc[0]['Calculated_DateTime']:
                idx = flag_d.index[0]
            else:
                idx = order_d.index[0]
                order_before_flag.append(pat_id)
        elif len(flag_d)>=1:
            idx = flag_d.index[0]
        elif len(order_d)>=1:
            idx = order_d.index[0]
        else:
            errors_list.append(pat_id)
            continue
        
        df_data = df_group.iloc[:idx]
        if (max_date is not None) and ((df_data['Calculated_DateTime'].iloc[-1]>max_date) or (df_data['Arrived_Time'].iloc[-1]<max_date)):
            continue

        df_pat_list.append(df_data.copy())

    return df_pat_list, errors_list, order_before_flag
        
def clean_target_withnoterminationflags(raw_file_path, all_flags, event_col='EVENT_NAME', grb_id='PAT_ENC_CSN_ID',
                 event_time_col='Calculated_DateTime', orders_cols='Type', arrival_date_col='Arrived_Time', normalize_event_col=True,
                 min_date=None, max_date=None):
    if isinstance(raw_file_path, str):
        df_raw = pd.read_csv(raw_file_path)
    else:
        df_raw = raw_file_path
    df_raw[event_time_col] = pd.to_datetime(df_raw[event_time_col])
    df_raw[arrival_date_col] = pd.to_datetime(df_raw[arrival_date_col])

    if normalize_event_col:
        df_raw[f'{event_col}_NORM'] = df_raw[event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x)).str.lower()
        event_col = f'{event_col}_NORM'

    pat_groups = df_raw.sort_values(by=event_time_col).groupby(grb_id)

    # NOTE:
    #   1. Process only from the moment of arrival to ED, DO NOT FILTER ANY EVENTS PRIOR TO THE ARRIVAL [IGNORE FOR NOW BECAUSE OF THE ISSUE WITH ARRIVED_TIME]
    #   2. Looking for "Order - Admission" and "Order - Discharge" within the `orders_cols` as the actual flag in case none of the `all_flags` was encountered in the `eve`
    df_pat_list = []
    errors_list = []
    order_before_flag = []
    for pat_id, df_group in pat_groups:
        if min_date is not None and (df_group['Calculated_DateTime'].iloc[0]<min_date) or (df_group['Arrived_Time'].iloc[0]<min_date):
            continue
        df_group.reset_index(inplace=True)
        if df_group['ED_Disposition'].iloc[-1] == 'Admitted':
            type_mask = df_group['Type'] == 'Order - Admission'
            flag_mask = df_group[event_col].isin(all_flags)
            if type_mask.sum()>0:
                ii = np.where(type_mask)[0][0]
                df_pat_list.append(df_group.iloc[:ii])
            elif flag_mask.sum()>0:
                ii = np.where(flag_mask)[0][0]
                df_pat_list.append(df_group.iloc[:ii])
            else:
                errors_list.append(pat_id)
        else:
            type_mask = df_group['Type'] == 'Order - Discharge'
            if type_mask.sum()>0:
                ii = np.where(type_mask)[0][0]
                df_pat_list.append(df_group.iloc[:ii])
            else:
                df_pat_list.append(df_group.copy())

        # flag_d = df_group.loc[df_group[event_col].isin(all_flags)]
        # order_d = df_group.loc[(df_group[orders_cols]=='Order - Admission')|(df_group[orders_cols]=='Order - Discharge')]
        # if len(flag_d)>=1 and len(order_d)>=1:
        #     if flag_d.iloc[0]['Calculated_DateTime'] < order_d.iloc[0]['Calculated_DateTime']:
        #         idx = flag_d.index[0]
        #     else:
        #         idx = order_d.index[0]
        #         order_before_flag.append(pat_id)
        # elif len(flag_d)>=1:
        #     idx = flag_d.index[0]
        # elif len(order_d)>=1:
        #     idx = order_d.index[0]
        # else:
        #     errors_list.append(pat_id)
        #     continue
        
        # df_data = df_group.iloc[:idx]
        # if (max_date is not None) and ((df_data['Calculated_DateTime'].iloc[-1]>max_date) or (df_data['Arrived_Time'].iloc[-1]<max_date)):
        #     continue

        # df_pat_list.append(df_data.copy())

    return df_pat_list, errors_list, order_before_flag

def clean_target_parallel(raw_file_path, all_flags, event_col='EVENT_NAME', grb_id='PAT_ENC_CSN_ID', event_time_col='Calculated_DateTime', orders_cols='Type', arrival_date_col='Arrived_Time', normalize_event_col=True, njobs=22):
    def clean_single_pat_id(df_group):
        df_group.reset_index(inplace=True)
        flag_d = df_group.loc[df_group[event_col].isin(all_flags)]
        order_d = df_group.loc[(df_group[orders_cols]=='Order - Admission')|(df_group[orders_cols]=='Order - Discharge')]
        if len(flag_d)>=1 and len(order_d)>=1:
            if flag_d.iloc[0]['Calculated_DateTime'] < order_d.iloc[0]['Calculated_DateTime']:
                idx = flag_d.index[0]
            else:
                idx = order_d.index[0]
        elif len(flag_d)>=1:
            idx = flag_d.index[0]
        elif len(order_d)>=1:
            idx = order_d.index[0]
        else:
            return None
        df_data = df_group.iloc[:idx]
        return df_data
        
                 
    df_raw = pd.read_csv(raw_file_path)
    df_raw[event_time_col] = pd.to_datetime(df_raw[event_time_col])
    df_raw[arrival_date_col] = pd.to_datetime(df_raw[arrival_date_col])

    if normalize_event_col:
        df_raw[f'{event_col}_NORM'] = df_raw[event_col].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x)).str.lower()
        event_col = f'{event_col}_NORM'

    pat_groups = df_raw.sort_values(by=event_time_col).groupby(grb_id)

    futures_list = []
    df_pat_list = []
    with ThreadPoolExecutor(max_workers=njobs) as executor:
        for pat_id, df_group in pat_groups:
            future = executor.submit(clean_single_pat_id, df_group)
            futures_list.append(future)
    
    for future in as_completed(futures_list):
        res = future.result()
        if future is not None:
            df_pat_list.append(res)
    
    return df_pat_list 


def load_clean_results(path):
    df_clean = pd.read_csv(os.path.join(path, 'df_clean.csv'))
    error_list = None
    orderbflag_list = None

    if os.path.exists(os.path.join(path, 'error_pat.json')):
        with open(os.path.join(path, 'error_pat.json'), 'r') as f:
            error_list = json.load(f)
    
    if os.path.exists(os.path.join(path, 'orderbflag.json')):
        with open(os.path.join(path, 'orderbflag.json'), 'r') as f:
            orderbflag_list = json.load(f)
    
    return df_clean, error_list, orderbflag_list
    

def exclude_admit_flags_prior_to_admission(df, admi_flags, pat_id, target_col, datecol, event_col):
    admi_flags_times = defaultdict(list) # in minutes
    df_admi = df[df[target_col]=='Admitted']
    group_admi = df_admi.groupby(pat_id)

    for grb, df_grb in group_admi:
        admi_mask = df_grb[event_col].isin(admi_flags)

        if admi_mask.sum() == 0:
            continue

        mask_admission = df_grb['Type'] == 'Order - Admission'
        if mask_admission.sum() > 0:
            admi_time = df_grb.iloc[np.where(mask_admission)[0][0]][datecol]
            for idx, row in df_grb[admi_mask].iterrows():
                admi_flags_times[row[event_col]].append(('admission', grb, (admi_time-row[datecol]).total_seconds()/60))
    
    
    flag_time_dict = defaultdict(list)
    for flag, data_list in admi_flags_times.items():
        for d in data_list:
            flag_time_dict[flag].append(d[-1])
    
    return list(map(lambda x: x[0], list(filter(lambda x: np.median(x[1])<=0, flag_time_dict.items()))))

def get_all_target_flags(raw_data_path, event_col, target_col, normalize_text, corr_thr, datecol='Calculated_DateTime', pat_id='PAT_ENC_CSN_ID'):
    if isinstance(raw_data_path, str):
        df = pd.read_csv(raw_data_path)
    else:
        df = raw_data_path
    df['Calculated_DateTime'] = pd.to_datetime(df['Calculated_DateTime'])
    df = df.sort_values(by='Calculated_DateTime')

    admi_flags, admi_data  = get_target_flags(df, event_col, target_col, 'admi', normalize_text, corr_thr)
    admi_flags = exclude_admit_flags_prior_to_admission(df, admi_flags, pat_id, target_col, datecol, event_col if not normalize_text else f'{event_col}_NORM')

    bed_flags, bed_data   = get_target_flags(df, event_col, target_col, 'bed', normalize_text, corr_thr)
    bed_flags = exclude_admit_flags_prior_to_admission(df, bed_flags, pat_id, target_col, datecol, event_col if not normalize_text else f'{event_col}_NORM')

    disch_flags, dishc_data = get_target_flags(df, event_col, target_col, 'disch', normalize_text, corr_thr)

    consu_flags, consu_data = get_target_flags(df, event_col, target_col, 'consu', normalize_text, corr_thr)
    consu_flags = exclude_admit_flags_prior_to_admission(df, consu_flags, pat_id, target_col, datecol, event_col if not normalize_text else f'{event_col}_NORM')

    obser_flags, obser_data = get_target_flags(df, event_col, target_col, 'observ', normalize_text, corr_thr)
    obser_flags = exclude_admit_flags_prior_to_admission(df, obser_flags, pat_id, target_col, datecol, event_col if not normalize_text else f'{event_col}_NORM')
    all_flags = np.concatenate([admi_flags, bed_flags, disch_flags, consu_flags, obser_flags])
    all_flags_list = list(set(all_flags))
    return all_flags_list

def set_admitted_discharged_only(df, target_col, admit_val, val2del, dropna=True):
    if dropna:
        print('Removing nans in the label if exists ...')
        print(f'There are {df[target_col].isna().sum()} target nan values ...')
        print(f'Size before removing nans: {df.shape} ...')
        df = df[~df[target_col].isna()]
        print(f'Size after removing nans: {df.shape} ...')

    if len(val2del) > 0:
        print(f"Target values to be excluded {val2del}")
        print(f'Size before removing {val2del}: {df.shape} ...')
        df = df.loc[~(df[target_col].isin(val2del))]
        print(f'Size after removing {val2del}: {df.shape} ...')

    df.loc[df[target_col]!=admit_val, target_col] = 'NotAdmitted'
    return df

# Driving code
if __name__ == "__main__":
    clean_date_folder = "12_1_23"
    print("target_cleaner.py has successfully been executed ...")
    if not os.path.exists(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'all_norm_flags.json')):
        admi_flags = get_target_flags(constants.RAW_DATA, 'EVENT_NAME', 'Admitted_YN', 'admi', True, 0.7)
        bed_flags = get_target_flags(constants.RAW_DATA, 'EVENT_NAME', 'Admitted_YN', 'bed', True, 0.7)
        disch_flags = get_target_flags(constants.RAW_DATA, 'EVENT_NAME', 'Admitted_YN', 'disch', True, 0.7)
        consu_flags = get_target_flags(constants.RAW_DATA, 'EVENT_NAME', 'Admitted_YN', 'consu', True, 0.7)
        obser_flags = get_target_flags(constants.RAW_DATA, 'EVENT_NAME', 'Admitted_YN', 'observ', True, 0.7)
        all_flags = np.concatenate([admi_flags, bed_flags, disch_flags, consu_flags, obser_flags])
        all_flags_list = list(set(all_flags))
        with open(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'all_norm_flags.json'), 'w') as f:
            json.dump(all_flags_list, f)
            
    else:
        with open(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'all_norm_flags.json'), 'r') as f:
            all_flags_list = json.load(f)

    print(f"Has successfully loaded all_flags_list which contains {len(all_flags_list)} flags....") 

    if not os.path.exists(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'df_clean.csv')):
        start = time.time()
        df_clean_list, error_list, orderbflag_list = clean_target(constants.RAW_DATA, all_flags_list, 'EVENT_NAME', 'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', 'Arrived_Time', True)
        print(f'Sequential processing took: {time.time()-start} sec. ...')
        with open(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'error_pat.json'), 'w') as f:
            json.dump(error_list, f)
        with open(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'orderbflag.json'), 'w') as f:
            json.dump(orderbflag_list, f)
        df_clean = pd.concat(df_clean_list)
        df_clean.to_csv(os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'df_clean.csv'))
    else:
        df_clean, error_pat, orderbflag =\
            load_clean_results(path=os.path.join(constants.OUTPUT_DIR, 'clean_target', clean_date_folder, 'df_clean.csv'))

    x = 0

    # SLOWER THAN SEQUENTIAL
    # start = time.time()
    # njobs = 25
    # df_clean_parallel_list = clean_target_parallel(constants.RAW_DATA, all_flags_list, 'EVENT_NAME', 'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', 'Arrived_Time', True, njobs=njobs)
    # print(f'Parallel processing took with {njobs} threads: {time.time()-start} sec. ...')
    # df_clean_parallel = pd.concat(df_clean_parallel_list)

    # df_clean_parallel.to_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/clean_target/df_clean_parallel.csv')
