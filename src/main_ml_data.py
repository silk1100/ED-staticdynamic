from distutils.log import warn
from glob import glob
import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import numpy as np
import pandas as pd
import polars as pl
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from const import constants
from collections import defaultdict
from ml.customcrossvalidator import CustomCrossValidator1
from preprocess.target_cleaner import clean_target, clean_target_withnoterminationflags, get_all_target_flags, set_admitted_discharged_only

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

def process_raw_data(raw_data, output_path, noflags, writedata=False):
        df = pd.read_csv(raw_data)
        subdir_name = os.path.basename(raw_data).split('-')[-1].strip().split('.csv')[0].replace('.','_')
        output_dir = Path(output_path) / Path(subdir_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        vals2del = [val for val in df['ED_Disposition'].unique() if val not in ['Admitted', 'Discharged']]
        df = set_admitted_discharged_only(df, 'ED_Disposition', 'Admitted', vals2del)
        if not noflags:
            all_flags_list = get_all_target_flags(df, 'EVENT_NAME', 'ED_Disposition', True, 1.0)
            # df_clean_list, error_list, orderbflag_list = clean_target(raw_data, all_flags_list, 'EVENT_NAME', 'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', 'Arrived_Time', True)
            df_clean_list, error_list, orderbflag_list = clean_target_withnoterminationflags(df, all_flags_list, 'EVENT_NAME', 'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Type', 'Arrived_Time', True, min_date=pd.Timestamp(year=2021, month=1, day=1))
            df_clean = pd.concat(df_clean_list)
        else:
            df_clean=df
            error_list=[]
            orderbflag_list=[]

        if writedata:
            df_clean.to_csv(os.path.join(output_dir, 'df_clean.csv'))
            with open(os.path.join(output_dir, 'error_list.joblib'), 'wb') as f:
                joblib.dump(error_list, f)
            with open(os.path.join(output_dir, 'orderbflag.joblib'), 'wb') as f:
                joblib.dump(orderbflag_list, f)

        return df_clean, error_list, orderbflag_list, output_dir

def trim_pat_encounter(df_cc, min_event_sz):
    '''
    This function implements the same logic as `filter_patients_encounters`, without
    any filtration on the data.
    '''

    df_len = df_cc.groupby('PAT_ENC_CSN_ID').apply(lambda x: len(x))
    pat_id_2include = df_len[df_len>=min_event_sz].index
    excluded_pats = np.setdiff1d(set(df_cc['PAT_ENC_CSN_ID'].unique()), pat_id_2include)
    df_cc = df_cc[df_cc['PAT_ENC_CSN_ID'].isin(pat_id_2include)]
    return df_cc, excluded_pats 


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
    for idx, (df_static_train, df_static_test) in enumerate(cc.split(df_static)):
        if os.path.exists(os.path.join(output_dir,  f'static_{eidx}'+f'_train_{idx}.csv')) and\
            os.path.exists(os.path.join(output_dir,  f'static_{eidx}'+f'_test_{idx}.csv')) and\
            os.path.exists(os.path.join(output_dir,  f'dynamic_{eidx}'+f'_train_{idx}.csv')) and\
            os.path.exists(os.path.join(output_dir,  f'dynamic_{eidx}'+f'_test_{idx}.csv')):
            continue
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

    t1 = time.time() 
    for idx in range(min_event_sz, max_event_sz+1):
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
    

if __name__ == '__main__':
#   Read Raw dataset
#     Process and clean events
#     Save the processed/cleaned dataset

    '''
    The following code is replaced by the code in notebook `notebooks/11-InDepth-...ipynb`
    '''
    # folders = list(filter(os.path.isdir, glob(constants.CLEAN_DATA_DIR+'/'+"*")))
    # folders.sort(key=lambda x: os.path.getmtime(x))

    # subdir_name = os.path.basename(constants.RAW_DATA).split('-')[-1].strip().split('.csv')[0].replace('.','_')
    # full_clean_df = Path(constants.CLEAN_DATA_DIR) / Path(subdir_name) / Path('df_clean_notargetflags11.csv')
    # if not os.path.exists(full_clean_df):
        df_clean, error_list, orderbflag_list, output_dir = process_raw_data(constants.RAW_DATA, constants.CLEAN_DATA_DIR, noflags=False)
    #     # Clean features by removing redundant flag e.g. (Ethnicity: (Unknown, *Unspecified, Declined)
    #     # all should be mapped to the same category)
    #     # save clean_updated_feature
        df_clean = category_mappers_dynamic(df_clean)
    #     df_clean = category_mappers_static(df_clean)
    #     df_clean.to_csv(os.path.join(output_dir, 'df_clean_notargetflags.csv'))
    # else:
    #     df_clean = pd.read_csv(full_clean_df, index_col=0)
        
    # df_clean = pd.read_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/clean_target/filtered_truncated_12.21.23.csv',
    #                        index_col=0)
    # df_clean['Calculated_DateTime'] = pd.to_datetime(df_clean['Calculated_DateTime'])
    # df_clean['Arrived_Time'] = pd.to_datetime(df_clean['Arrived_Time'])
    # df_clean = df_clean.sort_values(by='Calculated_DateTime')
    # Load/read cleaned dataset
    # Split data into stationary features, and dynamic features
    # Save 2 dataframes: Stationary and dynamic
    arrival_event_time_min = 60*48
    min_event_sz = 6
    rare_thresold = 0.05
    late_eidx_threshold = 150
    # max_event_sz = 260
    
    idx_trimming_threshold = 0.9993
    # df_cc, pat2exc = filter_patients_encounters(df_clean, arrival_event_time_min, min_event_sz)
    # df_cc, pat2exc = trim_pat_encounter(df_clean, min_event_sz)

    # df_len = df_cc.groupby('PAT_ENC_CSN_ID').apply(lambda x: len(x))
    # idx_end = int(np.percentile(df_len, idx_trimming_threshold*100))
    # max_event_sz = int(np.percentile(df_len, idx_trimming_threshold*100))

    # print(f"There are {len(pat2exc)} excluded from the study since they have less than {min_event_sz} events ...")
    # print(f'Feature extraction will begin on a data of size {df_cc.shape} with {df_cc["PAT_ENC_CSN_ID"].nunique()}' 
        #   f' unique encounter ...')
    # df_cc = df_cc[df_cc['Arrived_Time'].dt.year>=2021]
    Path(constants.ML_DATA_OUTPUT_ID).mkdir(parents=True, exist_ok=True)


    # This file contains the cumprob of every event in column `cumprob`, you can use it directly
    # to label the rare events without the need to manually do it in the StaticPreprocessor
    # and DynamicPreprocesor
    df_clean = pl.read_csv('/work/InternalMedicine/s223850/ED-StaticDynamic/clean_target/filtered_truncated_withRareMedEv_12.21.23.csv',
                           infer_schema_length=int(5e6),
                           null_values=dict(Patient_Age='NULL'),
                           dtypes=dict(Patient_Age=pl.Float64))
    df_clean = df_clean.with_columns(
        [
            pl.col('Arrived_Time').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
            pl.col('Calculated_DateTime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
        ]
    )
    pat_len = df_clean.group_by('PAT_ENC_CSN_ID').agg(pl.col("EVENT_NAME").len().alias("seq_len"))
    df_cc = df_clean.join(pat_len, on="PAT_ENC_CSN_ID", how="outer").filter(pl.col("seq_len")>=min_event_sz).drop("processing_time_min")

    df_clean = df_clean.sort(by='Calculated_DateTime')
    df_clean = df_clean.with_columns(
        (pl.col("cumprob")>(1-rare_thresold)).alias('is_rare_eventname'),
        (pl.col('median_event_idx')>late_eidx_threshold).alias('is_late_eventname')
    )
    df_clean = df_clean.with_columns([
        pl.when(
            (pl.col("is_rare_eventname")&pl.col("is_late_eventname"))
            ).then(
                pl.col("Type")+pl.lit("_")+pl.lit("rare_late")
            ).when(
                (pl.col("is_rare_eventname")&~pl.col("is_late_eventname"))
            ).then(
                pl.col("Type")+pl.lit("_")+pl.lit("rare_early")
            ).otherwise(
                pl.col("Type")+pl.lit("_")+pl.col("EVENT_NAME")
            ).alias("Type_EVENT_NAME")
    ])

    df_clean = df_clean.with_columns(
        pl.col('Type_EVENT_NAME').str.replace_all(r"[^a-zA-Z0-9_]+", "", literal=False).str.to_lowercase().alias("Type_EVENT_NAME_NORM")
    )

    df_clean = df_clean.drop(columns=['cumprob', 'first_time', 'processing_time_min', 'is_rare_eventname', 'is_late_eventname'])
    x = 0
    
    '''
    Used for testing purposes
    '''
    # # The following two lines are only uncommented for testing purposes
    # # sample_patid = np.random.choice(df_cc['PAT_ENC_CSN_ID'].unique(), 83000, False)
    # # df_cc = df_cc[df_cc['PAT_ENC_CSN_ID'].isin(sample_patid)]

    '''
    Uncomment each of the following statements and run the src/scripts/01_1_sbatch_main_data.sh
    To run the data synchronously on 4 clusters
    '''
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

    warn("From 202 to 250") 
    create_static_dynamic_feats(df_cc, 202, 250,
                                constants.STATIONARY_FIELDS, constants.DYNAMIC_FIELDS, constants.DROPPED_FIELDS,
                                'PAT_ENC_CSN_ID', 'Calculated_DateTime', 'Arrived_Time', 'ED_Disposition', constants.TRAINING_PERIOD, constants.TESTING_PERIOD, constants.ML_DATA_OUTPUT_ID, 0)
