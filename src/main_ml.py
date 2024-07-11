'''
This file is obselete and only kept for reference

To run data preprocessing, execute main_ml_data.py
to run training/testing, execute main_ml_model.py
'''
from distutils.log import warn
from glob import glob
import os
from re import T
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from const import constants
import json
from collections import defaultdict
from ml.customcrossvalidator import CustomCrossValidator1
from preprocess.target_cleaner import clean_target, get_all_target_flags, set_admitted_discharged_only

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from preprocess.staticpreprocessor import StaticTransformer
from preprocess.dynamicprocessor import DynamicTranformer
from preprocess import static_dynamic_extractor
from utils.utils import category_mappers_dynamic, category_mappers_static, basic_preprocess

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from pathlib import Path
import time

from warnings import simplefilter
simplefilter('ignore')

#TODO: Remove this method once the correct data is received with the corred `Arrived_Time`
def estimate_arrived_time(df): # df is assumed to have 'Calculated_DateTime' as a datetime[ns] datatype
    groups = df.sort_values(by='Calculated_DateTime').groupby('PAT_ENC_CSN_ID')
    arrival_time_per_pat = {}
    no_arrival_event = []
    for pat, df_group in groups:
        if sum(df_group['EVENT_NAME'].str.lower().str.contains('arrive')) >= 1:
            time = df_group.loc[(df_group['EVENT_NAME'].str.lower().str.contains('arrive')), 'Calculated_DateTime'].iloc[0]
        elif sum(df_group['EVENT_NAME'].str.lower().str.contains('triage direct to room')) >= 1:
            time = df_group.loc[(df_group['EVENT_NAME'].str.lower().str.contains('triage direct to room')), 'Calculated_DateTime'].iloc[0]
        elif sum(df_group['EVENT_NAME'].str.lower().str.contains('ed secondary triage')) >= 1:
            time = df_group.loc[(df_group['EVENT_NAME'].str.lower().str.contains('ed secondary triage')), 'Calculated_DateTime'].iloc[0]
        elif sum(df_group['EVENT_NAME'].str.lower().str.contains('emergency encounter created')) >= 1:
            time = df_group.loc[(df_group['EVENT_NAME'].str.lower().str.contains('emergency encounter created')), 'Calculated_DateTime'].iloc[0]
        else:
            no_arrival_event.append(pat)
            continue

        arrival_time_per_pat[pat] = time
    return arrival_time_per_pat, no_arrival_event


def split_static_dynamic_parallel(idx, pat_id_2include):

    # Filter df_clean for encounters with less than 6 events (Estimated from 01_clean_target_analysis.ipynb)
    df_static, df_dynamic = static_dynamic_extractor.main_sequential(df_clean, constants.STATIONARY_FIELDS,
                                                        constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
                                                        idx, event_time_col='Calculated_DateTime',
                                                        grp_col='PAT_ENC_CSN_ID',
                                                        pat_id2include=set(pat_id_2include))
        
    df_static.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'static_{idx}.csv'))
    df_dynamic.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'dynamic_{idx}.csv'))
    return os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'static_{idx}.csv'), os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'dynamic_{idx}.csv')

def sort_files(files):
    files_dict = {}
    for file in files:
        idx  = int(file.split('_')[-1].split('.')[0])
        files_dict[idx] = file
    sorted_files = sorted(files_dict.items(), key=lambda kv: kv[0], reverse=False)
    return list(map(lambda x: x[1], sorted_files)) 


def process_static_dynamic_parallel(sf, df, enc2remove, output_path):
    assert sf.split('_')[-1].split('.')[0] == df.split('_')[-1].split('.')[0], f"Wrong static-dynamic file correspondance. {sf} matched {df}"
    df_static = pd.read_csv (os.path.join(parent_dir, sf))
    df_dynamic = pd.read_csv(os.path.join(parent_dir, df))

    #TODO: Remove once Arrived_time issue is resolved
    df_static = df_static[~(df_static['PAT_ENC_CSN_ID'].isin(enc2remove))]
    df_dynamic = df_dynamic[~(df_dynamic['PAT_ENC_CSN_ID'].isin(enc2remove))]
    
    assert (set(df_static['PAT_ENC_CSN_ID'].unique()) == set(df_dynamic['PAT_ENC_CSN_ID'].unique())) and \
        df_static['PAT_ENC_CSN_ID'].nunique() == df_dynamic['PAT_ENC_CSN_ID'].nunique(), f'files {sf} and {df} do not have the same number of PAT_ENC_CSN_ID'
    df_static = basic_preprocess(df_static)
    df_dynamic = basic_preprocess(df_dynamic)
    
    df_static['arr_hr'] = df_static['Arrived_Time_appx'].dt.hour
    df_static['arr_dow'] = df_static['Arrived_Time_appx'].dt.dayofweek
    df_static['arr_mnth'] = df_static['Arrived_Time_appx'].dt.month
    df_static['arr_year'] = df_static['Arrived_Time_appx'].dt.year

    cal = calendar()
    holidays = cal.holidays(start=df_static['Arrived_Time_appx'].min(), end=df_static['Arrived_Time_appx'].max())
    df_static['is_holiday'] = df_static['Arrived_Time_appx'].dt.date.isin(holidays.date)

    #TODO: The following mapping should have occurred for the clean_df and before the splitting of static and dynamic multiple events ds
    #TODO: Consider adding it in the following run to the clean_df instead of doing it to every static and dynamic file
    df_static = category_mappers_static(df_static)
    df_dynamic = category_mappers_dynamic(df_dynamic)

    cc = CustomCrossValidator1(constants.TRAINING_PERIOD, constants.TESTING_PERIOD, 'Arrived_Time_appx')
    list_idx = []
    for idx, (df_static_train, df_static_test) in enumerate(cc.split(df_static)):
        if  os.path.exists((os.path.join(output_path, sf.split('.')[0]+f'_test_{idx}.csv'))) and \
            os.path.exists((os.path.join(output_path, sf.split('.')[0]+f'_train_{idx}.csv'))) and \
            os.path.exists((os.path.join(output_path, df.split('.')[0]+f'_test_{idx}.csv'))) and \
            os.path.exists((os.path.join(output_path, df.split('.')[0]+f'_train_{idx}.csv'))):
            list_idx.append((os.path.join(output_path, sf.split('.')[0]+f'_train_{idx}.csv'),
                        os.path.join(output_path, sf.split('.')[0]+f'_test_{idx}.csv'),
                        os.path.join(output_path, df.split('.')[0]+f'_train_{idx}.csv'),
                        os.path.join(output_path, df.split('.')[0]+f'_test_{idx}.csv')))

            continue
        train_enc = df_static_train['PAT_ENC_CSN_ID'].unique()
        test_enc = df_static_test['PAT_ENC_CSN_ID'].unique()

        assert len(train_enc) == len(df_static_train), f'df_static_train does not have 1 to 1 relationship with the PAT_ENC_CSN_ID. There are {len(train_enc)} unique PAT_ENC_CSN_ID, and there are {len(df_static_train)} rows in the df_static_train_ml'
        assert len(test_enc) == len(df_static_test), f'df_static_test does not have 1 to 1 relationship with the PAT_ENC_CSN_ID. There are {len(test_enc)} unique PAT_ENC_CSN_ID, and there are {len(df_static_test)} rows in the df_static_test_ml'

        df_dynamic_train = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(train_enc)]
        df_dynamic_test = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(test_enc)]


        static_transformer = StaticTransformer(ohe_cols=['Means_Of_Arrival', 'FirstRace', 'Ethnicity', 'arr_dow', 'arr_hr', 'arr_mnth'],
                                        le_cols=['Acuity_Level', 'MultiRacial', 'is_holiday'],
                                        std_cols=['arr_year', 'Patient_Age'], 
                                        minmax_cols=['Number of past appointments in last 60 days',
                                                    'Number of past inpatient admissions over ED visits in last three years'],
                                        multival_dict={ # Inside the logic of reduction, if the # of categories after reduction is less than 10, then I take the most common 10
                                            'colnames': ['Chief_Complaint_All'],
                                            'sep': [','],
                                            'regex':[True],
                                            'thresh':[0.95],
                                            'apply_thresh':[100]
                                        })

        dynamic_transformer = DynamicTranformer('PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', # Inside the logic of reduction, if the # of categories after reduction is less than 10, then I take the most common 10
                                                'EVENT_NAME', 'time_elapsed',
                        type_config=dict(
                            type_attr = ['Lab Order - Result'],
                            type_extra_col = ['Result_Flag'],
                            type_extra_col_concat= ['_'], 
                            type_thresh = [0.95], 
                            type_regex = [True],
                            type_apply_thresh = [100],
                            
                            type_global_thresh = 0.95, 
                            type_global_apply_thresh = 100,
                            type_global_regex= True
                        ),
                        
                        cols_config={
                            'colnames': ['Primary_DX_ICD10'],
                            'threshold': [0.95], # To avoid doing any reduction to the feature space, then you can use None or >=1.
                            'regex':[False],
                            'sep':[','],
                            'apply_threshold':[100]
                        })
        
        df_static_train_ml = static_transformer.fit_transform(df_static_train)
        df_static_test_ml = static_transformer.transform(df_static_test)

        df_dynamic_train_ml = dynamic_transformer.fit_transform(df_dynamic_train)
        df_dynamic_test_ml =  dynamic_transformer.transform(df_dynamic_test)

        df_static_train_ml.to_csv(os.path.join(output_path, sf.split('.')[0]+f'_train_{idx}.csv'))
        df_static_test_ml.to_csv(os.path.join(output_path, sf.split('.')[0]+f'_test_{idx}.csv'))
        df_dynamic_train_ml.to_csv(os.path.join(output_path, df.split('.')[0]+f'_train_{idx}.csv'))
        df_dynamic_test_ml.to_csv(os.path.join(output_path, df.split('.')[0]+f'_test_{idx}.csv'))

        # list_idx.append((df_static_train_ml, df_static_test_ml, df_dynamic_train_ml, df_dynamic_test_ml))
        list_idx.append((
            os.path.join(output_path, sf.split('.')[0]+f'_train_{idx}.csv'),
                        os.path.join(output_path, sf.split('.')[0]+f'_test_{idx}.csv'),
                        os.path.join(output_path, df.split('.')[0]+f'_train_{idx}.csv'),
                        os.path.join(output_path, df.split('.')[0]+f'_test_{idx}.csv')
                        ))
    
    return list_idx

def cross_validate(model, Xtr, ytr):
    cv = StratifiedKFold(5, shuffle=True, random_state=41)
    train_score_list = defaultdict(list)
    test_score_list = defaultdict(list)
    for train_idx, test_idx in cv.split(Xtr, ytr):
        Xtrain = Xtr.iloc[train_idx]
        ytrain = ytr.iloc[train_idx]
        Xte = Xtr.iloc[test_idx]
        yte = ytr.iloc[test_idx]
        if 'catboost' in str(model):
            model.fit(Xtrain, ytrain, eval_set=(Xte, yte))
        else:
            model.fit(Xtrain, ytrain)
        ytrainh = model.predict(Xtrain)
        yhat = model.predict(Xte)
        train_score_list['bacc'].append(balanced_accuracy_score(ytrain, ytrainh))
        train_score_list['ppv'].append(precision_score(ytrain, ytrainh))
        train_score_list['roc'].append(roc_auc_score(ytrain, ytrainh))

        test_score_list['bacc'].append(balanced_accuracy_score(yte, yhat))
        test_score_list['ppv'].append(precision_score(yte, yhat))
        test_score_list['roc'].append(roc_auc_score(yte, yhat))

    train_score_avg = {}
    test_score_avg = {}
    train_score_std = {}
    test_score_std = {}
    train_score_avg['bacc'] = np.mean(train_score_list['bacc'])
    train_score_avg['ppv'] = np.mean(train_score_list['ppv'])
    train_score_avg['roc'] = np.mean(train_score_list['roc'])
    test_score_avg['bacc'] = np.mean(test_score_list['bacc'])
    test_score_avg['ppv'] = np.mean(test_score_list['ppv'])
    test_score_avg['roc'] = np.mean(test_score_list['roc'])
    train_score_std['bacc'] = np.std(train_score_list['bacc'], ddof=1)
    train_score_std['ppv'] =  np.std(train_score_list['ppv'], ddof=1)
    train_score_std['roc'] =  np.std(train_score_list['roc'], ddof=1)
    test_score_std['bacc'] =  np.std(test_score_list['bacc'], ddof=1)
    test_score_std['ppv'] =   np.std(test_score_list['ppv'], ddof=1)
    test_score_std['roc'] =   np.std(test_score_list['roc'], ddof=1)
    return train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list


def process_raw_data(raw_data, output_path):
        df = pd.read_csv(raw_data)
        subdir_name = os.path.basename(raw_data).split('-')[-1].strip().split('.csv')[0].replace('.','_')
        output_dir = Path(output_path) / Path(subdir_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        vals2del = [val for val in df['ED_Disposition'].unique() if val not in ['Admitted', 'Discharged']]
        df = set_admitted_discharged_only(df, 'ED_Disposition', 'Admitted', vals2del)
        all_flags_list = get_all_target_flags(df, 'EVENT_NAME', 'ED_Disposition', True, 0.7)
        df_clean_list, error_list, orderbflag_list = clean_target(raw_data, all_flags_list, 'EVENT_NAME', 'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', 'Arrived_Time', True)

        df_clean = pd.concat(df_clean_list)
        df_clean.to_csv(os.path.join(output_dir, 'df_clean.csv'))
        with open(os.path.join(output_dir, 'error_list.joblib'), 'wb') as f:
            joblib.dump(error_list, f)
        with open(os.path.join(output_dir, 'orderbflag.joblib'), 'wb') as f:
            joblib.dump(orderbflag_list, f)

        return df_clean, error_list, orderbflag_list, output_dir


def train_ml_folder(feat_csv_dir,  output_dir, run_idx, tr_te_dir= None,
                    pat_id='PAT_ENC_CSN_ID', target='ED_Disposition'):
    if tr_te_dir is None:
        tr_te_dir = f'tr_{constants.TRAINING_PERIOD}_te_{constants.TESTING_PERIOD}'

    full_path = Path(output_dir) / Path(str(run_idx))
    full_path.mkdir(parents=True, exist_ok=True)

    input_dir = os.path.join(feat_csv_dir, tr_te_dir)

    static_files = [file for file in os.listdir(input_dir) if file.startswith('static') and file.endswith('.csv')] 
    dynamic_files = [file for file in os.listdir(input_dir) if file.startswith('dynamic') and file.endswith('.csv')] 
    s_static = sorted(static_files, key = lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]), x.split('_')[2]), reverse=True)
    s_dynamic = sorted(dynamic_files, key = lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]), x.split('_')[2]), reverse=True)
    flist = list(zip(s_static, s_dynamic))
    for idx in range(0, len(flist), 2):
        s_f_train, d_f_train  = flist[idx]
        s_f_test, d_f_test  = flist[idx+1]
        df_s_train = pd.read_csv(os.path.join(input_dir, s_f_train), index_col=0)
        df_s_test = pd.read_csv(os.path.join (input_dir, s_f_test), index_col=0)
        df_d_train = pd.read_csv(os.path.join(input_dir, d_f_train))
        df_d_test = pd.read_csv(os.path.join (input_dir, d_f_test))

        df_all_tr = df_s_train.merge(df_d_train, how='outer', left_on=pat_id, right_on='Unnamed: 0')
        df_all_te = df_s_test.merge(df_d_test, how='outer', left_on=pat_id, right_on='Unnamed: 0')

        static_id = df_all_tr.pop('Unnamed: 0')
        dynamic_id = df_all_tr.pop(pat_id)
        static_id_te = df_all_te.pop('Unnamed: 0')
        dynamic_id_te = df_all_te.pop(pat_id)
        df_all_tr.loc[df_all_tr[target]=='Admitted', target] = 1
        df_all_tr.loc[df_all_tr[target]=='Not Admitted', target] = 0
        df_all_tr[target] = df_all_tr[target].astype(np.int32)

        df_all_te.loc[df_all_te[target]=='Admitted',     target] = 1
        df_all_te.loc[df_all_te[target]=='Not Admitted', target] = 0
        df_all_te[target] = df_all_te[target].astype(np.int32)

        
        Xtr = df_all_tr.drop(target, axis=1)
        ytr = df_all_tr[target]

        Xte = df_all_te.drop(target, axis=1)
        yte = df_all_te[target]
        
        lgbm = LGBMClassifier() 
        lsvm = LinearSVC()
        svm = SVC()
        xgb = XGBClassifier()
        cat = CatBoostClassifier(4000, 0.03, 5)
        lg = LogisticRegression()
        train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
        cross_validate(cat, Xtr, ytr)
        print('Catboost results ...')
        print(train_score_avg)
        print(test_score_avg)
        print(train_score_std)
        print(test_score_std)
        print(f'Catboost Test Scores: ', score(cat, Xte, yte))
        
        train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
        cross_validate(lsvm, Xtr, ytr)
        print('lsvm results ...')
        print(train_score_avg)
        print(test_score_avg)
        print(train_score_std)
        print(test_score_std)
        print(f'lsvm Test Scores: ', score(lsvm, Xte, yte))
        

        train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
        cross_validate(xgb, Xtr, ytr)
        print('xgb results ...')
        print(train_score_avg)
        print(test_score_avg)
        print(train_score_std)
        print(test_score_std)
        print(f'xgb Test Scores: ', score(xgb, Xte, yte))
        
        train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
        cross_validate(lgbm, Xtr, ytr)
        print('lgbm results ...')
        print(train_score_avg)
        print(test_score_avg)
        print(train_score_std)
        print(test_score_std)
        print(f'lgbm Test Scores: ', score(lgbm, Xte, yte))


        train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
        cross_validate(lg, Xtr, ytr)
        print('logisticregression results ...')
        print(train_score_avg)
        print(test_score_avg)
        print(train_score_std)
        print(test_score_std)
        print(f'lg Test Scores: ', score(lg, Xte, yte))
        x = 0

def score(model, Xte, yte):
    yhat = model.predict(Xte)
    bacc = balanced_accuracy_score(yte, yhat)
    roc = roc_auc_score(yte, yhat)
    ppv = precision_score(yte, yhat)
    return {'bacc': bacc, 'roc': roc, 'ppv': ppv}


def filter_patients_encounters(df_clean, arrival_time_diff_min, min_event_sz):
    if isinstance(df_clean, str):
        df_clean = pd.read_csv(df_clean, index_col=0)

    df_clean['Arrived_Time'] = pd.to_datetime(df_clean['Arrived_Time'])  # Arrived_Time is still corrupt
    df_clean['Calculated_DateTime'] = pd.to_datetime(df_clean['Calculated_DateTime'])
    df_cc = df_clean.sort_values(by='PAT_ENC_CSN_ID').sort_values('Calculated_DateTime')

    df_cc['elapsed_time_min'] = (df_cc['Calculated_DateTime']-df_cc['Arrived_Time']).dt.total_seconds()/(60)
    if arrival_time_diff_min is not None:
        print(f'Removing patients encounters with arrival time prior to first Calculated encounter by {arrival_time_diff_min} min ...')
        print(f'Size before removing encounters: {df_cc.shape} ...')
        pat_id = df_cc[df_cc['elapsed_time_min']<(-arrival_time_diff_min)]['PAT_ENC_CSN_ID'].unique()
        df_cc = df_cc.loc[~(df_cc['PAT_ENC_CSN_ID'].isin(pat_id))]
        print(f'Size after removing encounters: {df_cc.shape} ...')

    
    df_len = df_cc.groupby('PAT_ENC_CSN_ID').apply(lambda x: len(x))
    pat_id_2include = df_len[df_len>=min_event_sz].index
    excluded_pats = np.setdiff1d(set(df_cc['PAT_ENC_CSN_ID'].unique()), pat_id_2include)
    df_cc = df_cc[df_cc['PAT_ENC_CSN_ID'].isin(pat_id_2include)]
    return df_cc, excluded_pats 


def split_data_into_static_dynamic(df_cc, idx, static_cols, dynamic_cols, dropped_fields,
                                   event_col_time, grp_col):
    
    df_static, df_dynamic = static_dynamic_extractor.main_sequential(df_clean, constants.STATIONARY_FIELDS,
                                                        constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
                                                        idx, event_time_col='Calculated_DateTime',
                                                        grp_col='PAT_ENC_CSN_ID',
                                                        pat_id2include=set(pat_id_2include))
        
    df_static.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'static_{idx}.csv'))
    df_dynamic.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'dynamic_{idx}.csv'))
    return os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'static_{idx}.csv'), os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'dynamic_{idx}.csv')



def process_static_dynamic_parallel2(df_static, eidx, df_dynamic, arr_time, training_period, testing_period, output_path):
    output_dir = Path(output_path) / Path(f'tr_{training_period}_te_{testing_period}') 
    output_dir.mkdir(parents=True, exist_ok=True)

    df_static = basic_preprocess(df_static)
    df_dynamic = basic_preprocess(df_dynamic)
    
    df_static['arr_hr'] =   df_static[arr_time].dt.hour
    df_static['arr_dow'] =  df_static[arr_time].dt.dayofweek
    df_static['arr_mnth'] = df_static[arr_time].dt.month
    df_static['arr_year'] = df_static[arr_time].dt.year

    cal = calendar()
    holidays = cal.holidays(start=df_static[arr_time].min(), end=df_static[arr_time].max())
    df_static['is_holiday'] = df_static[arr_time].dt.date.isin(holidays.date)

    cc = CustomCrossValidator1(training_period, testing_period, arr_time)
    list_idx = []
    for idx, (df_static_train, df_static_test) in enumerate(cc.split2(df_static)):
        train_enc = df_static_train['PAT_ENC_CSN_ID'].unique()
        test_enc = df_static_test['PAT_ENC_CSN_ID'].unique()

        assert len(train_enc) == len(df_static_train), f'df_static_train does not have 1 to 1 relationship with the PAT_ENC_CSN_ID. There are {len(train_enc)} unique PAT_ENC_CSN_ID, and there are {len(df_static_train)} rows in the df_static_train_ml'
        assert len(test_enc) == len(df_static_test), f'df_static_test does not have 1 to 1 relationship with the PAT_ENC_CSN_ID. There are {len(test_enc)} unique PAT_ENC_CSN_ID, and there are {len(df_static_test)} rows in the df_static_test_ml'
        assert len(np.setdiff1d(train_enc, test_enc)) == len(train_enc), f"There is a leakage detected in the range of training: ({df_static_train[arr_time].min()}, {df_static_train[arr_time].max()}), and testing: ({df_static_test[arr_time].min()},{df_static_test[arr_time].max()})"

        df_dynamic_train = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(train_enc)]
        df_dynamic_test = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(test_enc)]


        static_transformer = StaticTransformer(pat_id='PAT_ENC_CSN_ID', target='ED_Disposition',
            ohe_cols=['Means_Of_Arrival', 'FirstRace', 'Ethnicity', 'arr_dow', 'arr_hr', 'arr_mnth'],
                                        le_cols=['Acuity_Level', 'MultiRacial', 'is_holiday'],
                                        std_cols=['arr_year', 'Patient_Age'], 
                                        minmax_cols=['Number of past appointments in last 60 days',
                                                    'Number of past inpatient admissions over ED visits in last three years'],
                                        multival_dict={ # Inside the logic of reduction, if the # of categories after reduction is less than 10, then I take the most common 10
                                            'colnames': ['Chief_Complaint_All'],
                                            'sep': [','],
                                            'regex':[True],
                                            'thresh':[0.95],
                                            'apply_thresh':[100]
                                        })

        dynamic_transformer = DynamicTranformer('PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', # Inside the logic of reduction, if the # of categories after reduction is less than 10, then I take the most common 10
                                                'EVENT_NAME', 'time_elapsed',
                        type_config=dict(
                            type_attr = ['Lab Order - Result'],
                            type_extra_col = ['Result_Flag'],
                            type_extra_col_concat= ['_'], 
                            type_thresh = [0.95], 
                            type_regex = [True],
                            type_apply_thresh = [100],
                            
                            type_global_thresh = 0.95, 
                            type_global_apply_thresh = 100,
                            type_global_regex= True
                        ),
                        
                        cols_config={
                            'colnames': ['Primary_DX_ICD10'],
                            'threshold': [0.95], # To avoid doing any reduction to the feature space, then you can use None or >=1.
                            'regex':[False],
                            'sep':[','],
                            'apply_threshold':[100]
                        })
        
        df_static_train_ml = static_transformer.fit_transform(df_static_train)
        df_static_test_ml = static_transformer.transform(df_static_test)

        df_dynamic_train_ml = dynamic_transformer.fit_transform(df_dynamic_train)
        df_dynamic_test_ml =  dynamic_transformer.transform(df_dynamic_test)

        df_static_train_ml.to_csv(os.path.join(output_dir,  f'static_{eidx}'+f'_train_{idx}.csv'))
        df_static_test_ml.to_csv(os.path.join(output_dir,   f'static_{eidx}'+f'_test_{idx}.csv'))
        df_dynamic_train_ml.to_csv(os.path.join(output_dir, f'dynamic_{eidx}'+f'_train_{idx}.csv'))
        df_dynamic_test_ml.to_csv(os.path.join(output_dir,  f'dynamic_{eidx}'+f'_test_{idx}.csv'))

        # list_idx.append((df_static_train_ml, df_static_test_ml, df_dynamic_train_ml, df_dynamic_test_ml))
        list_idx.append((
                        os.path.join(output_path,f'static_{eidx}'+f'_train_{idx}.csv'),
                        os.path.join(output_path,f'static_{eidx}'+f'_test_{idx}.csv'),
                        os.path.join(output_path,f'dynamic_{eidx}'+f'_train_{idx}.csv'),
                        os.path.join(output_path,f'dynamic_{eidx}'+f'_test_{idx}.csv')
                        ))
    
    return list_idx

def process_one_eidx(groups, idx, static_feats, dynamic_feats, arr_time_col, training_period, testing_period, output_path):
    stats_list = []
    dynamic_list = []
    for pat_id, df_grp in groups:
        if len(df_grp) > idx:
            stats_list.append(df_grp[static_feats].iloc[idx].to_frame().T)
            dynamic_list.append(df_grp[dynamic_feats].iloc[:idx])

        else:
            stats_list.append(df_grp[static_feats].iloc[-1].to_frame().T)
            dynamic_list.append(df_grp[dynamic_feats])

    df_static = pd.concat(stats_list)
    df_dynamic = pd.concat(dynamic_list)

    return process_static_dynamic_parallel2(df_static, idx, df_dynamic, arr_time_col, training_period, testing_period, output_path)

def create_static_dynamic_feats(df_clean, min_event_sz, max_event_sz,
                                static_feats, dynamic_feats, dropped_cols,
                                grp_col, event_time_col, arr_time_col, target, training_period, testing_period, output_path, njobs=-1):
    if dropped_cols is not None:
        df_clean = df_clean.drop(columns=dropped_cols)
        for col in dropped_cols:
            if col in static_feats:
                static_feats.remove(col)
            if col in dynamic_feats:
                dynamic_feats.remove(col)
    
    if grp_col not in static_feats:
        static_feats.append(grp_col)
    if grp_col not in dynamic_feats:
        dynamic_feats.append(grp_col)

    if target not in static_feats:
        static_feats.append(target)
    if target not in dynamic_feats:
        dynamic_feats.append(target)
    
    df_clean = df_clean.sort_values(by=event_time_col)
    groups = df_clean.groupby(grp_col)

    # if njobs is None or njobs==0:
    #     for idx in range(min_event_sz, max_event_sz+1):
    #         for pat_id, df_grp in groups:
    #             if len(df_grp) > idx:
    #                 stats_list.append(df_grp[static_feats].iloc[idx].to_frame().T)
    #                 dynamic_list.append(df_grp[dynamic_feats].iloc[:idx])

    #             else:
    #                 stats_list.append(df_grp[static_feats].iloc[-1].to_frame().T)
    #                 dynamic_list.append(df_grp[dynamic_feats])
            
    #         df_static = pd.concat(stats_list, ignore_index=True)
    #         df_dynamic = pd.concat(dynamic_list, ignore_index=True)
            
    #         process_static_dynamic_parallel2(df_static, idx, df_dynamic, arr_time_col, training_period, testing_period, output_path)
    #         break
    # else:
    #     with ThreadPoolExecutor(max_workers=njobs) as pool:
    #         futures = [pool.submit(process_one_eidx, groups, idx, static_feats, dynamic_feats, arr_time_col, training_period, testing_period, output_path) for idx in range(min_event_sz, max_event_sz+1)]
   
    t1 = time.time() 
    for idx in range(min_event_sz, max_event_sz+1):
    # for idx in [10, 30, 70, 120, 200]:
        stats_list = []
        dynamic_list = []
        for pat_id, df_grp in groups:
            if len(df_grp) > idx:
                stats_list.append(df_grp[static_feats].iloc[idx].to_frame().T)
                dynamic_list.append(df_grp[dynamic_feats].iloc[:idx])
            else:
                stats_list.append(df_grp[static_feats].iloc[-1].to_frame().T)
                dynamic_list.append(df_grp[dynamic_feats])
        df_static = pd.concat(stats_list, ignore_index=True)
        df_dynamic = pd.concat(dynamic_list, ignore_index=True)
        process_static_dynamic_parallel2(df_static, idx, df_dynamic, arr_time_col, training_period, testing_period, output_path)
    print(f'Time taken by Sequential loop: {time.time()-t1} seconds')
    
    # t1 = time.time()    
    # with ThreadPoolExecutor(max_workers=15) as pool:
    #     # futures = [pool.submit(process_one_eidx, groups, idx, static_feats, dynamic_feats, arr_time_col, training_period, testing_period, output_path) for idx in range(min_event_sz, max_event_sz+1)]
    #     futures = [pool.submit(process_one_eidx, groups, idx, static_feats, dynamic_feats, arr_time_col, training_period, testing_period, '/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_feats_ID1_thread') for idx in [10, 30, 70, 120, 200]]
    # for future in as_completed(futures):
    #     result = future.result()
    # print(f'Time taken by multithreading loop: {time.time()-t1} seconds')

    # t1 = time.time()    
    # with ProcessPoolExecutor(max_workers=15) as pool:
    #     # futures = [pool.submit(process_one_eidx, groups, idx, static_feats, dynamic_feats, arr_time_col, training_period, testing_period, output_path) for idx in range(min_event_sz, max_event_sz+1)]
    #     futures = [pool.submit(process_one_eidx, groups, idx, static_feats, dynamic_feats, arr_time_col, training_period, testing_period, '/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_feats_ID1_parallel') for idx in [10, 30, 70, 120, 200]]
    # for future in as_completed(futures):
    #     result = future.result()
    # print(f'Time taken by multiprocessing loop: {time.time()-t1} seconds')
        

if __name__ == "__main__":
    # Read Raw dataset
    # Process and clean events
    # Save the processed/cleaned dataset
    # folders = list(filter(os.path.isdir, glob(constants.CLEAN_DATA_DIR+'/'+"*")))
    # folders.sort(key=lambda x: os.path.getmtime(x))

    # subdir_name = os.path.basename(constants.RAW_DATA).split('-')[-1].strip().split('.csv')[0].replace('.','_')
    # full_clean_df = Path(constants.CLEAN_DATA_DIR) / Path(subdir_name) / Path('df_clean.csv')
    # if not os.path.exists(full_clean_df):
    #     df_clean, error_list, orderbflag_list, output_dir = process_raw_data(constants.RAW_DATA, constants.CLEAN_DATA_DIR)
    #     # Clean features by removing redundant flag e.g. (Ethnicity: (Unknown, *Unspecified, Declined)
    #     # all should be mapped to the same category)
    #     # save clean_updated_feature
    #     df_clean = category_mappers_dynamic(df_clean)
    #     df_clean = category_mappers_static(df_clean)
    #     df_clean.to_csv(os.path.join(output_dir, 'df_clean.csv'))
    # else:
    #     df_clean = pd.read_csv(full_clean_df, index_col=0)
        

    # Load/read cleaned dataset [Already been made with the step above]
    # Clean features by removing redundant flag e.g. (Ethnicity: (Unknown, *Unspecified, Declined) all should be mapped to the same category)
    # save clean_updated_feature
    # df_clean = category_mappers_dynamic(df_clean)
    # df_clean = category_mappers_static(df_clean)

    
    # Load/read cleaned dataset
    # Split data into stationary features, and dynamic features
    # Save 2 dataframes: Stationary and dynamic
    # arrival_event_time_min = 60*48
    # min_event_sz = 6
    # # max_event_sz = 260

    # idx_trimming_threshold = 0.9993
    # df_cc, pat2exc = filter_patients_encounters(df_clean, arrival_event_time_min, min_event_sz)

    # df_len = df_cc.groupby('PAT_ENC_CSN_ID').apply(lambda x: len(x))
    # # idx_end = int(np.percentile(df_len, idx_trimming_threshold*100))
    # max_event_sz = int(np.percentile(df_len, idx_trimming_threshold*100))

    # print(f"There are {len(pat2exc)} excluded from the study since they have less than {min_event_sz} events ...")
    # print(f'Feature extraction will begin on a data of size {df_cc.shape} with {df_cc["PAT_ENC_CSN_ID"].nunique()}' 
    #       f' unique encounter ...')
    # df_cc = df_cc[df_cc['Arrived_Time'].dt.year>=2021]
    # Path(constants.ML_DATA_OUTPUT_ID).mkdir(parents=True, exist_ok=True)

    # # The following two lines are only uncommented for testing purposes
    # # sample_patid = np.random.choice(df_cc['PAT_ENC_CSN_ID'].unique(), 83000, False)
    # # df_cc = df_cc[df_cc['PAT_ENC_CSN_ID'].isin(sample_patid)]
    # warn("From 8 to 50")
    # create_static_dynamic_feats(df_cc, 8, 50,
    #                             constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #                             'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Arrived_Time', 'ED_Disposition', constants.TRAINING_PERIOD, constants.TESTING_PERIOD, constants.ML_DATA_OUTPUT_ID, 0)
    # warn("From 51 to 100") 
    # create_static_dynamic_feats(df_cc, 51, 100,
    #                             constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #                             'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Arrived_Time', 'ED_Disposition', constants.TRAINING_PERIOD, constants.TESTING_PERIOD, constants.ML_DATA_OUTPUT_ID, 0)
     
    
    # warn("From 101 to 151") 
    # create_static_dynamic_feats(df_cc, 101, 151,
    #                             constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #                             'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Arrived_Time', 'ED_Disposition', constants.TRAINING_PERIOD, constants.TESTING_PERIOD, constants.ML_DATA_OUTPUT_ID, 0)

    # warn("From 152 to 201") 
    # create_static_dynamic_feats(df_cc, 152, 201,
    #                             constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #                             'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Arrived_Time', 'ED_Disposition', constants.TRAINING_PERIOD, constants.TESTING_PERIOD, constants.ML_DATA_OUTPUT_ID, 0)

    # warn("From 202 to 250") 
    # create_static_dynamic_feats(df_cc, 202, 250,
    #                             constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #                             'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Arrived_Time', 'ED_Disposition', constants.TRAINING_PERIOD, constants.TESTING_PERIOD, constants.ML_DATA_OUTPUT_ID, 0)


    train_ml_folder(constants.ML_DATA_OUTPUT_ID, constants.ML_RESULTS_OUTPUT, 0, 'tr_8_te_4')

# (df_clean, constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
#                                                         idx, event_time_col='Calculated_DateTime',
#                                                         grp_col='PAT_ENC_CSN_ID',
#                                                         pat_id2include=set(pat_id_2include))

    # idx_range = [min_event_sz, ]
    # split_data_into_static_dynamic(
    #     df_cc, idx 
    # )
    # if njobs is not None and njobs !=0:
    #     with ProcessPoolExecutor(max_workers=njobs) as pool:

    #         futures = [pool.submit(split_static_dynamic_parallel, idx, pat_id_2include) for idx in range(min_event_sz, 990)]

    # if not os.path.exists(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel', f'static_6.csv')):
    #     df_clean = pd.read_csv(os.path.join(constants.CLEAN_DATA_DIR, '12_1_23', 'df_clean.csv'))
    #     # df_clean['Arrived_Time'] = pd.to_datetime(df_clean['Arrived_Time'])  # Arrived_Time is still corrupt
    #     df_clean['Calculated_DateTime'] = pd.to_datetime(df_clean['Calculated_DateTime'])

    #     # TODO: Remove the following block of code once the new data with the updated arrived_Time arrives
    #     arr_times, errors = estimate_arrived_time(df_clean)
    #     df_clean['Arrived_Time_appx'] = -1
    #     for pat, datetime in arr_times.items():
    #         df_clean.loc[df_clean['PAT_ENC_CSN_ID'] == pat, 'Arrived_Time_appx'] = datetime

    #     df_len = df_clean.groupby('PAT_ENC_CSN_ID').apply(lambda x: len(x))
    #     pat_id_2include = df_len[df_len>=6].index

    #     with ProcessPoolExecutor(max_workers=32) as pool:
    #         futures = [pool.submit(split_static_dynamic_parallel, idx, pat_id_2include) for idx in range(6, 290)]

    #         output = []
    #         for future in as_completed(futures):
    #             output.append(future.result())
    #             print(output)

    # parent_dir = os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds_parallel')
    # static_files = [file for file in os.listdir(parent_dir) if 'static' in file and 'csv' in file]
    # dynamic_files = [file for file in os.listdir(parent_dir) if 'dynamic' in file and 'csv' in file]
    # static_files = sort_files(static_files) 
    # dynamic_files = sort_files(dynamic_files)

    # # # # TODO: Start from where you stopped
    # static_files = static_files[190:]
    # dynamic_files = dynamic_files[190:]
    # # # # #TODO: Run it when you want to sbatch the static_dynamic_feats files
    # # # # #TODO: Remove once Arrived_time issue is resolved
    # df_static = pd.read_csv(os.path.join(parent_dir, static_files[0]))
    # enc2remove = df_static[df_static['Arrived_Time_appx'] == '-1']['PAT_ENC_CSN_ID']
    # results = []
    # with ThreadPoolExecutor(max_workers=21) as pool:
    #     # futures = [pool.submit(process_static_dynamic_parallel, sf, df, enc2remove) for sf, df in zip(static_files, dynamic_files)]
    #     futures = {pool.submit(process_static_dynamic_parallel, sf, df,
    #                            enc2remove, constants.ML_DATA_OUTPUT_ID): (sf, df) for sf, df in zip(static_files, dynamic_files)}

    #     for future in as_completed(futures):
    #         sf, df = futures[future]
    #         results.append((sf, df, future.result()))
    
    # nruns = len(os.listdir(constants.ML_RESULTS_OUTPUT))
    # train_ml_folder(constants.ML_DATA_OUTPUT_ID, constants.ML_RESULTS_OUTPUT, run_idx=nruns)
    # # x=0 
    # # df_s = pd.read_csv(os.path.join(parent_dir, 'static_205_train.csv'))
    # if os.path.exists(os.path.join(constants.ML_DATA_OUTPUT_ID, 'static_99_train_2.csv')):
    #     for i in range(3):
    #         sf_i_tr = os.path.join(constants.ML_DATA_OUTPUT_ID, f'static_99_train_{i}.csv')
    #         df_i_tr = os.path.join(constants.ML_DATA_OUTPUT_ID, f'dynamic_99_train_{i}.csv')
    #         sf_i_te = os.path.join(constants.ML_DATA_OUTPUT_ID, f'static_99_test_{i}.csv')
    #         df_i_te = os.path.join(constants.ML_DATA_OUTPUT_ID, f'dynamic_99_test_{i}.csv')
    #         df_sf_i_train = pd.read_csv(sf_i_tr, index_col=0)
    #         df_df_i_train = pd.read_csv(df_i_tr)
    #         df_sf_i_test = pd.read_csv(sf_i_te, index_col=0)
    #         df_df_i_test = pd.read_csv(df_i_te)
    #         df_all_tr = df_sf_i_train.merge(df_df_i_train, how='outer', left_on='PAT_ENC_CSN_ID', right_on='Unnamed: 0')
    #         df_all_te = df_sf_i_test.merge(df_df_i_test, how='outer', left_on='PAT_ENC_CSN_ID', right_on='Unnamed: 0')
            
    #         static_id = df_all_tr.pop('Unnamed: 0')
    #         dynamic_id = df_all_tr.pop('PAT_ENC_CSN_ID')
    #         df_all_tr.loc[df_all_tr['Admitted_YN']=='Admitted', 'Admitted_YN'] = 1
    #         df_all_tr.loc[df_all_tr['Admitted_YN']=='Not Admitted', 'Admitted_YN'] = 0
    #         df_all_tr['Admitted_YN'] = df_all_tr['Admitted_YN'].astype(np.int32)
            
    #         Xtr = df_all_tr.drop('Admitted_YN', axis=1)
    #         ytr = df_all_tr['Admitted_YN']

    #         Xte = df_all_te.drop('Admitted_YN', axis=1)
    #         yte = df_all_te['Admitted_YN']
            
    #         lgbm = LGBMClassifier() 
    #         lsvm = LinearSVC()
    #         svm = SVC()
    #         xgb = XGBClassifier()
    #         cat = CatBoostClassifier()
    #         lg = LogisticRegression()

    #         train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
    #             cross_validate(lgbm, Xtr, ytr)
    #         print('lgbm results ...')
    #         print(train_score_avg)
    #         print(test_score_avg)
    #         print(train_score_std)
    #         print(test_score_std)
    #         print('---------------------------------------------------')

    #         train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
    #             cross_validate(lsvm, Xtr, ytr)
    #         print('lsvm results ...')
    #         print(train_score_avg)
    #         print(test_score_avg)
    #         print(train_score_std)
    #         print(test_score_std)
    #         print('---------------------------------------------------')
    #         # train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
    #         #     cross_validate(svm, Xtr, ytr)
    #         # print('svm results ...')
    #         # print(train_score_avg)
    #         # print(test_score_avg)
    #         # print(train_score_std)
    #         # print(test_score_std)
    #         # print('---------------------------------------------------')
    #         train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
    #             cross_validate(xgb, Xtr, ytr)
    #         print('xgb results ...')
    #         print(train_score_avg)
    #         print(test_score_avg)
    #         print(train_score_std)
    #         print(test_score_std)
    #         print('---------------------------------------------------')
    #         train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
    #             cross_validate(cat, Xtr, ytr)
    #         print('cat results ...')
    #         print(train_score_avg)
    #         print(test_score_avg)
    #         print(train_score_std)
    #         print(test_score_std)
    #         print('---------------------------------------------------')
    #         train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
    #             cross_validate(lg, Xtr, ytr)
    #         print('lg results ...')
    #         print(train_score_avg)
    #         print(test_score_avg)
    #         print(train_score_std)
    #         print(test_score_std)
    #         print('---------------------------------------------------')
            

    #         x=0
        

    # for sf, df in zip(static_files, dynamic_files):
    #     assert sf.split('_')[-1].split('.')[0] == df.split('_')[-1].split('.')[0], f"Wrong static-dynamic file correspondance. {sf} matched {df}"
    #     df_static = pd.read_csv(os.path.join(parent_dir, sf))
    #     df_dynamic = pd.read_csv(os.path.join(parent_dir,df))

    #     #TODO: Remove once Arrived_time issue is resolved
    #     df_static = df_static[~(df_static['PAT_ENC_CSN_ID'].isin(enc2remove))]
    #     df_dynamic = df_dynamic[~(df_dynamic['PAT_ENC_CSN_ID'].isin(enc2remove))]

        
    #     assert (set(df_static['PAT_ENC_CSN_ID'].unique()) == set(df_dynamic['PAT_ENC_CSN_ID'].unique())) and \
    #         df_static['PAT_ENC_CSN_ID'].nunique() == df_dynamic['PAT_ENC_CSN_ID'].nunique(), f'files {sf} and {df} do not have the same number of PAT_ENC_CSN_ID'
    #     df_static = basic_preprocess(df_static)
    #     df_dynamic = basic_preprocess(df_dynamic)
        
    #     df_static['arr_hr'] = df_static['Arrived_Time_appx'].dt.hour
    #     df_static['arr_dow'] = df_static['Arrived_Time_appx'].dt.dayofweek
    #     df_static['arr_mnth'] = df_static['Arrived_Time_appx'].dt.month
    #     df_static['arr_year'] = df_static['Arrived_Time_appx'].dt.year

    #     cal = calendar()
    #     holidays = cal.holidays(start=df_static['Arrived_Time_appx'].min(), end=df_static['Arrived_Time_appx'].max())
    #     df_static['is_holiday'] = df_static['Arrived_Time_appx'].dt.date.isin(holidays.date)
    
    #     #TODO: The following mapping should have occurred for the clean_df and before the splitting of static and dynamic multiple events ds
    #     #TODO: Consider adding it in the following run to the clean_df instead of doing it to every static and dynamic file
    #     df_static = category_mappers_static(df_static)
    #     df_dynamic = category_mappers_dynamic(df_dynamic)

    #     cc = CustomCrossValidator1(constants.TRAINING_PERIOD, constants.TESTING_PERIOD, 'Arrived_Time_appx')
    #     for df_static_train, df_static_test in cc.split(df_static):
    #         train_enc = df_static_train['PAT_ENC_CSN_ID'].unique()
    #         test_enc = df_static_test['PAT_ENC_CSN_ID'].unique()

    #         assert len(train_enc) == len(df_static_train), f'df_static_train does not have 1 to 1 relationship with the PAT_ENC_CSN_ID. There are {len(train_enc)} unique PAT_ENC_CSN_ID, and there are {len(df_static_train)} rows in the df_static_train_ml'
    #         assert len(test_enc) == len(df_static_test), f'df_static_test does not have 1 to 1 relationship with the PAT_ENC_CSN_ID. There are {len(test_enc)} unique PAT_ENC_CSN_ID, and there are {len(df_static_test)} rows in the df_static_test_ml'

    #         df_dynamic_train = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(train_enc)]
    #         df_dynamic_test = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(test_enc)]


    #         static_transformer = StaticTransformer(ohe_cols=['Means_Of_Arrival', 'FirstRace', 'Ethnicity', 'arr_dow', 'arr_hr', 'arr_mnth'],
    #                                        le_cols=['Acuity_Level', 'MultiRacial', 'is_holiday'],
    #                                        std_cols=['arr_year', 'Patient_Age'], 
    #                                        minmax_cols=['Number of past appointments in last 60 days',
    #                                                     'Number of past inpatient admissions over ED visits in last three years'],
    #                                        multival_dict={ # Inside the logic of reduction, if the # of categories after reduction is less than 10, then I take the most common 10
    #                                            'colnames': ['Chief_Complaint_All'],
    #                                            'sep': [','],
    #                                            'regex':[True],
    #                                            'thresh':[0.95],
    #                                            'apply_thresh':[100]
    #                                        })

    #         dynamic_transformer = DynamicTranformer('PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', # Inside the logic of reduction, if the # of categories after reduction is less than 10, then I take the most common 10
    #                                                 'EVENT_NAME', 'time_elapsed',
    #                        type_config=dict(
    #                             type_attr = ['Lab Order - Result'],
    #                             type_extra_col = ['Result_Flag'],
    #                             type_extra_col_concat= ['_'], 
    #                             type_thresh = [0.95], 
    #                             type_regex = [True],
    #                             type_apply_thresh = [100],
                                
    #                             type_global_thresh = 0.95, 
    #                             type_global_apply_thresh = 100,
    #                             type_global_regex= True
    #                        ),
                           
    #                        cols_config={
    #                             'colnames': ['Primary_DX_ICD10'],
    #                             'threshold': [0.95], # To avoid doing any reduction to the feature space, then you can use None or >=1.
    #                             'regex':[False],
    #                             'sep':[','],
    #                             'apply_threshold':[100]
    #                        })
            
    #         df_static_train_ml = static_transformer.fit_transform(df_static_train)
    #         df_static_test_ml = static_transformer.transform(df_static_test)

    #         df_dynamic_train_ml = dynamic_transformer.fit_transform(df_dynamic_train)
    #         df_dynamic_test_ml =  dynamic_transformer.transform(df_dynamic_test)

    #         ### ML model

            
            
    #         x = 0

        

    # for idx in range(6, 260):

    # if not os.path.exists(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds', f'static_{idx}.csv')):
        # df_clean = pd.read_csv(constants.CLEAN_DATA)
        # df_clean['Arrived_Time'] = pd.to_datetime(df_clean['Arrived_Time'])
        # df_clean['Calculated_DateTime'] = pd.to_datetime(df_clean['Calculated_DateTime'])

        # # TODO: Remove the following block of code once the new data with the updated arrived_Time arrives
        # arr_times, errors = estimate_arrived_time(df_clean)
        # df_clean['Arrived_Time_appx'] = -1
        # for pat, datetime in arr_times.items():
        #     df_clean.loc[df_clean['PAT_ENC_CSN_ID'] == pat, 'Arrived_Time_appx'] = datetime


    #     # Filter df_clean for encounters with less than 6 events (Estimated from 01_clean_target_analysis.ipynb)
    #     df_len = df_clean.groupby('PAT_ENC_CSN_ID').apply(lambda x: len(x))
    #     pat_id_2include = df_len[df_len>=6].index
    #     # df_static, df_dynamic = static_dynamic_extractor.extract_static_dynamic(df_clean, constants.STATIONARY_FIELDS,
    #     #                                                     constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #     #                                                     idx, event_time_col='Calculated_DateTime',
    #     #                                                     grp_col='PAT_ENC_CSN_ID',
    #     #                                                     pat_id2include=set(pat_id_2include))
    #     df_static, df_dynamic = static_dynamic_extractor.main_sequential(df_clean, constants.STATIONARY_FIELDS,
    #                                                         constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #                                                         idx, event_time_col='Calculated_DateTime',
    #                                                         grp_col='PAT_ENC_CSN_ID',
    #                                                         pat_id2include=set(pat_id_2include))
    #     # df_static, df_dynamic = static_dynamic_extractor.main_parallel(df_clean, constants.STATIONARY_FIELDS,
    #     #                                                     constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #     #                                                     idx, event_time_col='Calculated_DateTime',
    #     #                                                     grp_col='PAT_ENC_CSN_ID',
    #     #                                                     pat_id2include=set(pat_id_2include), max_workers=25, parallel_type='proc')
    #     # df_static, df_dynamic = static_dynamic_extractor.main_parallel(df_clean, constants.STATIONARY_FIELDS,
    #     #                                                     constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
    #     #                                                     idx, event_time_col='Calculated_DateTime',
    #     #                                                     grp_col='PAT_ENC_CSN_ID',
    #     #                                                     pat_id2include=set(pat_id_2include), max_workers=25, parallel_type='thread')
         
    #     df_static.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds', f'static_{idx}_proc.csv'))
    #     df_dynamic.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds', f'dynamic_{idx}_proc.csv'))
    # else:
    #     df_static = pd.read_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds', f'static_{idx}_proc.csv'))
    #     df_dynamic = pd.read_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds', f'dynamic_{idx}_proc.csv'))

        # df_static = basic_preprocess(df_static) 
        # df_dynamic = basic_preprocess(df_dynamic) 
    
    # TODO: Remove the following block of code once the new data with the updated arrived_Time arrives
    # arr_times, errors = estimate_arrived_time(df_dynamic)
    # df_static['Arrived_Time_appx'] = -1
    # for pat, datetime in arr_times.items():
    #     assert (sum(df_static['PAT_ENC_CSN_ID'] == pat)==1)
    #     df_static.loc[df_static['PAT_ENC_CSN_ID'] == pat, 'Arrived_Time_appx'] = datetime
    # df_static.to_csv(os.path.join(constants.OUTPUT_DIR, 'static_dynamic_ds', f'static_{idx}.csv'))
    # print('done!')
    # x = 0
    
    # Read/Load stationary, and Dynamic datasets
    # Add engineered features to the static_csv (split time cols to (day, month, year, hour, dow, holiday_YN))
    # Add engineered features
    

    # Read/Load stationary, and Dynamic datasets
    # Split data into training/testing
    # Preprocess stationary features
    # Preprocess dynamic features
    # Save the preprocessed stationary and dynamic datasets
    
    # Load/Read the preprocessed stationary and dynamic datasets
    # Dynamic features contain the event/timestamp information which should be either: 
    #   1. Split into different time intervals, and predict at each time interval
    #   2. Split at different events, and predict at each event.
    #
    # DoOneOf{
    #   1. Concatenate the datasets and feed them to ML
    #   2. Feed each one of them to ML separately and aggregate the decisions
    #   3. Feed each one of them to neural network and aggregate the extracted features and pass it to FC layers for classification
    # }
    
    