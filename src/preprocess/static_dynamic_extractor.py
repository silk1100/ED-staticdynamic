import os
import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import pandas as pd
import numpy as np
from const import constants
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

def extract_static_dynamic(df_clean, static_feats, dynamic_feats, dropped_cols, event_idx,
                           event_time_col='Calculated_DateTime', grp_col='PAT_ENC_CSN_ID',
                           pat_id2include=None):
    if dropped_cols is not None:
        df_clean = df_clean.drop(columns=dropped_cols)
        for col in dropped_cols:
            if col in static_feats:
                static_feats.remove(col)
            if col in dynamic_feats:
                dynamic_feats.remove(col)
    df_clean = df_clean.sort_values(by=event_time_col)
    groups = df_clean.groupby(grp_col)
    stats_list = []
    dynamic_list = []
    for pat_id, df_grp in groups:
        if pat_id2include is not None and pat_id not in pat_id2include:
            continue
        if len(df_grp) > event_idx:
            stats_list.append(df_grp[static_feats].iloc[event_idx].to_frame().T)
            dynamic_list.append(df_grp[dynamic_feats].iloc[:event_idx])

        else:
            stats_list.append(df_grp[static_feats].iloc[-1].to_frame().T)
            dynamic_list.append(df_grp[dynamic_feats])

    
    df_static = pd.concat(stats_list)
    df_dynamic = pd.concat(dynamic_list)
    
    return df_static, df_dynamic

def extract_static_dynamic_pallel(pat_id, df_grp, static_feats, dynamic_feats, event_idx,
                           pat_id2include=None):
    if pat_id2include is not None and pat_id not in pat_id2include:
        return None, None
    if len(df_grp) > event_idx:
        df_grb_static = df_grp[static_feats].iloc[event_idx].to_frame().T
        df_grb_dynamic = df_grp[dynamic_feats].iloc[:event_idx]

    else:
        df_grb_static = df_grp[static_feats].iloc[-1].to_frame().T
        df_grb_dynamic = df_grp[dynamic_feats]

    return df_grb_static, df_grb_dynamic


def main_parallel(df_clean, static_feats, dynamic_feats, dropped_cols, event_idx,
                           event_time_col='Calculated_DateTime', grp_col='PAT_ENC_CSN_ID',
                           pat_id2include=None, max_workers=25, parallel_type='proc'):
    if dropped_cols is not None:
        df_clean = df_clean.drop(columns=dropped_cols)
        for col in dropped_cols:
            if col in static_feats:
                static_feats.remove(col)
            if col in dynamic_feats:
                dynamic_feats.remove(col)
    df_clean = df_clean.sort_values(by=event_time_col)
    groups = df_clean.groupby(grp_col)
    stats_list = []
    dynamic_list = []
    if parallel_type == 'proc':
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(extract_static_dynamic_pallel, pat_id, df_grb, static_feats,  dynamic_feats, event_idx,
                            pat_id2include) for pat_id, df_grb in groups]
            for future in as_completed(futures):
                df_grp_static, df_grp_dynamic = future.result()
                if df_grp_static is not None:
                    stats_list.append(df_grp_static)
                    dynamic_list.append(df_grp_dynamic)
    elif parallel_type == 'thread':
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(extract_static_dynamic_pallel, pat_id, df_grb, static_feats,  dynamic_feats, event_idx,
                            pat_id2include) for pat_id, df_grb in groups]
            for future in as_completed(futures):
                df_grp_static, df_grp_dynamic = future.result()
                if df_grp_static is not None:
                    stats_list.append(df_grp_static)
                    dynamic_list.append(df_grp_dynamic)
    
    df_static = pd.concat(stats_list)
    df_dynamic = pd.concat(dynamic_list)
    
    return df_static, df_dynamic

def main_sequential(df_clean, static_feats, dynamic_feats, dropped_cols, event_idx,
                           event_time_col='Calculated_DateTime', grp_col='PAT_ENC_CSN_ID',
                           pat_id2include=None):

    return extract_static_dynamic(df_clean, static_feats, dynamic_feats, dropped_cols, event_idx,
                           event_time_col, grp_col,
                           pat_id2include)

if __name__ == "__main__":
    df_clean = pd.read_csv(constants.CLEAN_DATA)
    df_clean['Arrived_Time'] = pd.to_datetime(df_clean['Arrived_Time'])
    df_clean['Calculated_DateTime'] = pd.to_datetime(df_clean['Calculated_DateTime'])

    df_clean
    