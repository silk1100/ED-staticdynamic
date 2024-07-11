from collections import defaultdict 
import pandas as pd
import numpy as np
import time
import re

from const import constants

# Always make sure to keep the count_* methods signature consistant
def count_seq(X:pd.DataFrame, colname='EVENT_NAME', nanamp:str='None_EVENT_NAME', time_idx='Calculated_DateTime', grb_id='PAT_ENC_CSN_ID'):
    cntr_dict = defaultdict(int)
    grp = X.sort_values(by=time_idx).groupby(grb_id)
    for idx, df_pat in grp:
        ee = df_pat[colname].tolist()
        for e in ee:
            if nanamp is None:
                assert (e is not None) and (not isinstance(e, float)), f"PAT_ID {idx} has a nan EVENT_NAME which is not expected to happen"
                cntr_dict[e] += 1
            else:
                if isinstance(e, float) or e is None:
                    cntr_dict[nanamp] +=1
                else:
                    cntr_dict[e] += 1
    return cntr_dict

def count_seq_fast(X:pd.DataFrame, colname='EVENT_NAME', nanamp:str='None_EVENT_NAME', time_idx='Calculated_DateTime', grb_id='PAT_ENC_CSN_ID'):
    '''
    Works only for sequential columns that does not require grouping.
    '''
    f = X[colname].value_counts()
    return f.to_dict()


def count_cat(X:pd.DataFrame, colname='Primary_DX', nanamp='None_Primary_DX', time_idx='Calculated_DateTime', grb_id = 'PAT_ENC_CSN_ID'):
    cntr_dict = defaultdict(int)
    grps = X.sort_values(by=time_idx).groupby(grb_id)
    for idx, df_pat in grps:
        ee = df_pat[colname].iloc[-1]
        if isinstance(ee, float):
            cntr_dict[nanamp] += 1
        elif isinstance(ee, str):
            cntr_dict[ee] += 1
        else:
            raise f"PAT_ID {idx} has a value which is neither a float nor a string in the {colname} column"
    return cntr_dict

def count_cat_fast(X:pd.DataFrame, colname='Primary_DX', nanamp='None_Primary_DX', time_idx='Calculated_DateTime', grb_id = 'PAT_ENC_CSN_ID'):
    X[colname] = X[colname].fillna(value=nanamp)
    # f = X.groupby(grb_id)[colname].value_counts()
    # f.name = 'count'
    # f = f.reset_index(level=1)
    # f['count'] = 1
    # return f[colname].value_counts().to_dict()
    return X.sort_values(by=time_idx).groupby(grb_id)[colname].last().value_counts().to_dict()


def category_mappers_static(df):
    if df['Patient_Age'].isna().sum() > 0:
        df.loc[df['Patient_Age'].isna(), 'Patient_Age'] = np.median(df['Patient_Age'])
    df['Patient_Age'] = df['Patient_Age'].astype('float')

    if df['Ethnicity'].isna().sum() > 0:
        df.loc[df['Ethnicity'].isna(), 'Ethnicity'] = 'Unknown'
    if (df['Ethnicity'] == 'Declined').sum() > 0:
        df.loc[df['Ethnicity'] == 'Declined', 'Ethnicity'] = 'Unknown'

    if (df['Ethnicity'] == '*Unspecified').sum() > 0:
        df.loc[df['Ethnicity'] == '*Unspecified', 'Ethnicity'] = 'Unknown'

    has_cols = [col for col in df.columns if 'has' in col.lower()]
    for col in has_cols:
        df.loc[df[col]=='Yes', col] = 1
        df.loc[df[col]=='No', col] = 1
        df[col] = df[col].astype('int')

    for col in df.columns:
        if 'number of' in col.lower():
            if df[col].isna().sum() > 0:
                df.loc[df[col].isna(), col] = 0
            df[col] = df[col].astype('int')
            
     
    if df['Means_Of_Arrival'].isna().sum() > 0:
        df.loc[df['Means_Of_arrival']=='NULL', 'Means_Of_Arrival'] = 'Other'
        df.loc[df['Means_Of_Arrival'].isna(), 'Means_Of_Arrival'] = 'Other'
    
    if df['Coverage_Financial_Class_Grouper'].isna().sum() > 0:
        df.loc[df['Coverage_Financial_Class_Grouper'].isna(), 'Coverage_Financial_Class_Grouper'] = 'None'

    if df['Sex'].isna().sum() > 0:
        df.loc[df['Sex'].isna(), 'Sex'] = 'Unknown'
    
    if df['Acuity_Level'].isna().sum() > 0:
        df.loc[df['Acuity_Level'].isna(), 'Acuity_Level'] = 'Unknown'
    
    if df['Chief_Complaint_All'].isna().sum() > 0:
        df.loc[df['Chief_Complaint_All'].isna(), 'Chief_Complaint_All'] = 'Unknown'
    
    if df['FirstRace'].isna().sum() > 0:
        df.loc[df['FirstRace'].isna(), 'FirstRace'] = 'Unknown'
    if (df['FirstRace'] == 'Unavailable/Unknown').sum() > 0:
        df.loc[df['FirstRace'] == 'Unavailable/Unknown', 'FirstRace'] = 'Unknown'
    if (df['FirstRace'] == 'Declined').sum() > 0:
        df.loc[df['FirstRace'] == 'Declined', 'FirstRace'] = 'Unknown'
    
    return df


def category_mappers_dynamic(df):
    for col in constants.DYNAMIC_FIELDS: #df.columns:
        if col in ['PAT_ENC_CSN_ID', 'Calculated_DateTime']:
            continue

        if df[col].isna().sum() > 0:
            df.loc[df[col].isna(), col] = 'Unknown'

    return df

def basic_preprocess(df):
    unamed_cols = [col for col in df.columns if 'unnamed' in col.lower()]
    if len(unamed_cols) >= 1:
        df.drop(columns = unamed_cols, inplace=True)
    if 'Calculated_DateTime' in df.columns:
        df['Calculated_DateTime'] = pd.to_datetime(df['Calculated_DateTime'])
    if 'Arrived_Time_appx' in df.columns:
        df['Arrived_Time_appx'] = pd.to_datetime(df['Arrived_Time_appx'])
    if 'Admitted_YN' in df.columns:
        if df['Admitted_YN'].dtype == 'O':
            df.loc[df['Admitted_YN']=='Admitted', 'Admitted_YN'] = 1
            df.loc[df['Admitted_YN']=='Not Admitted', 'Admitted_YN'] = 0
            df['Admitted_YN'] = df['Admitted_YN'].astype(np.int16)
    if 'ED_Disposition' in df.columns:
        if df['ED_Disposition'].dtype == 'O':
            df.loc[df['ED_Disposition']!='Admitted', 'ED_Disposition'] = 0
            df.loc[df['ED_Disposition']=='Admitted', 'ED_Disposition'] = 1
            df['ED_Disposition'] = df['ED_Disposition'].astype(np.int16)
            
    return df


def category_mappers_static_dl(df):
    if df['Ethnicity'].isna().sum() > 0:
        df.loc[df['Ethnicity'].isna(), 'Ethnicity'] = '<NUL>'
    if (df['Ethnicity'] == 'Declined').sum() > 0:
        df.loc[df['Ethnicity'] == 'Declined', 'Ethnicity'] = '<UNK>'
    if (df['Ethnicity'] == '*Unspecified').sum() > 0:
        df.loc[df['Ethnicity'] == '*Unspecified', 'Ethnicity'] = '<UNK>'
     
    if df['Means_Of_Arrival'].isna().sum() > 0:
        df.loc[df['Means_Of_Arrival'].isna(), 'Means_Of_Arrival'] = 'Car'
    
    if df['Coverage_Financial_Class_Grouper'].isna().sum() > 0:
        df.loc[df['Coverage_Financial_Class_Grouper'].isna(), 'Coverage_Financial_Class_Grouper'] = '<NUL>'

    if df['Sex'].isna().sum() > 0:
        df.loc[df['Sex'].isna(), 'Sex'] = '<NUL>'

    # df.loc[df['Sex']=='Unknown', 'Sex'] = '<UNK>'
    df.loc[df['Sex']=='Unknown', 'Sex'] = df['Sex'].mode()[0] # Since I want to convert it to 0's and ones
    
    
    if df['Acuity_Level'].isna().sum() > 0:
        df.loc[df['Acuity_Level'].isna(), 'Acuity_Level'] = '<NUL>'
    
    if df['Chief_Complaint_All'].isna().sum() > 0:
        df.loc[df['Chief_Complaint_All'].isna(), 'Chief_Complaint_All'] = '<NUL>'
    
    if df['FirstRace'].isna().sum() > 0:
        df.loc[df['FirstRace'].isna(), 'FirstRace'] = '<NUL>'
    if (df['FirstRace'] == 'Unavailable/Unknown').sum() > 0:
        df.loc[df['FirstRace'] == 'Unavailable/Unknown', 'FirstRace'] = '<UNK>'
    if (df['FirstRace'] == 'Declined').sum() > 0:
        df.loc[df['FirstRace'] == 'Declined', 'FirstRace'] = '<UNK>'
    
    return df


def category_mappers_dynamic_dl(df):
    for col in df.columns:
        if col in ['PAT_ENC_CSN_ID', 'Calculated_DateTime']:
            continue

        if df[col].isna().sum() > 0:
            df.loc[df[col].isna(), col] = '<NUL>'

    return df

def add_date_features(df, timecol):
    prefix = timecol.split('_')[0]
    df[f'{prefix}_hr'] = df[timecol].dt.hour
    df[f'{prefix}_dow'] = df[timecol].dt.dayofweek
    df[f'{prefix}_mnth'] = df[timecol].dt.month
    df[f'{prefix}_year'] = df[timecol].dt.year
    return df

    
def clean_string(val, regex=True, lowering=True, striping=True):
    
    if striping:
        val = val.strip()
    if regex:
        val = re.sub(r'[^a-zA-Z0-9_]+', '', val) 
    return val.lower() if lowering else val