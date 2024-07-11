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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from pathlib import Path
import time
import wandb

from warnings import simplefilter
simplefilter('ignore')

def agg_eidx_dict(files_list):
    dd = defaultdict(list)
    for file in files_list:
        dd[int(file.split('_')[1])].append(file)
    return dd

def cross_validate_all(Xtr, ytr, Xte, yte):
    lgbm = LGBMClassifier() 
    # lsvm = LinearSVC()
    # svm = SVC()
    xgb = XGBClassifier()
    cat = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=7, silent=True)
    lg = LogisticRegression()
    rf = RandomForestClassifier()
    models_dict = {
        # 'lsvm':lsvm,
        'lgbm':lgbm,
        # 'svm':svm,
        'xgb':xgb,
        'lg':lg,
        'cat':cat,
        'rf':rf
    }
    results_dict = dict.fromkeys(models_dict.keys())
        
    for model_name, model in models_dict.items():
        train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list =\
        cross_validate(model, Xtr, ytr)
        test_results = score(model, Xte, yte)
        warn(f'{model_name} results ...')
        warn(train_score_avg)
        warn(test_score_avg)
        warn(train_score_std)
        warn(test_score_std)
        warn(f'{model_name} Test Scores: {test_results}')
        warn('-------------------------------')
        yhat = model.predict_proba(Xte)
        yhat_train = model.predict_proba(Xtr)
        results_dict[model_name] = {
            'model':model,
            'train_score_avg':train_score_avg,
            'val_score_avg':test_score_avg,
            'train_score_std':train_score_std,
            'val_score_std':test_score_std,
            'train_score_list':train_score_list,
            'val_score_list':test_score_list,
            'test_results': test_results,
            'ytr': ytr,
            'yte': yte,
            'yhat_te': yhat,
            'yhat_tr':yhat_train,
            'classes': model.classes_
        }
    return results_dict


def train_ml_folder(feat_csv_dir,  output_dir, run_idx, tr_te_dir= None,
                    pat_id_s='PAT_ENC_CSN_ID', pat_id_d='Unnamed: 0', target='ED_Disposition', is_wandb=True):

    if tr_te_dir is None:
        tr_te_dir = f'tr_{constants.TRAINING_PERIOD}_te_{constants.TESTING_PERIOD}'

    full_path = Path(output_dir) / Path(tr_te_dir) / Path(f'run_{run_idx}')
    full_path.mkdir(parents=True, exist_ok=True)

    input_dir = os.path.join(feat_csv_dir, tr_te_dir)

    static_files = [file for file in os.listdir(input_dir) if file.startswith('static') and file.endswith('.csv')] 
    dynamic_files = [file for file in os.listdir(input_dir) if file.startswith('dynamic') and file.endswith('.csv')] 
    s_static = sorted(static_files, key = lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]), x.split('_')[2]))#, reverse=True)
    # s_static = sorted(static_files, key = lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]), x.split('_')[2]), reverse=False)
    s_dict = agg_eidx_dict(s_static)

    # Add reverse in order to run the code starting from the last event
    s_dynamic = sorted(dynamic_files, key = lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]), x.split('_')[2]))#, reverse=True)
    # s_dynamic = sorted(dynamic_files, key = lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]), x.split('_')[2]), reverse=False)
    d_dict = agg_eidx_dict(s_dynamic)

    nr_l = []
    for key in s_dict:
        if key not in d_dict or len(s_dict[key])!=len(d_dict[key]):
            warn(f'{key} is not complete yet')
            nr_l.append(key)
            with open(os.path.join(full_path, f'not_completed_event_idx_{key}.json'), 'w') as f:
                json.dump(nr_l, f)
            continue        
        
        s_files_in_order = sorted(s_dict[key], key=lambda x: (int(x.split('_')[3].split('.')[0]), x.split('_')[2]), reverse=True)
        d_files_in_order = sorted(d_dict[key], key=lambda x: (int(x.split('_')[3].split('.')[0]), x.split('_')[2]), reverse=True)
        sd_files = list(zip(s_files_in_order, d_files_in_order))

        if os.path.exists(os.path.join(full_path, f'sd_files_{key}_ziplist.joblib')):
            warn(f'sd_files_{key}_ziplist.joblib Already exists ...')
            continue

        with open(os.path.join(full_path, f'sd_files_{key}_ziplist.joblib'), 'wb') as f:
            joblib.dump(sd_files, f)

        for idx in range(0, len(sd_files), 2):
            static_full_path = full_path / Path('static')
            dynamic_full_path = full_path / Path('dynamic')
            comb_full_path = full_path / Path('comb')
            all_full_path = full_path / Path('all')


            static_full_path.mkdir(parents=True, exist_ok=True)
            dynamic_full_path.mkdir(parents=True, exist_ok=True)
            comb_full_path.mkdir(parents=True, exist_ok=True)
            all_full_path.mkdir(parents=True, exist_ok=True)

            if os.path.exists(os.path.join(static_full_path, f'static_results_{key}_{idx}.joblib')) and\
               os.path.exists(os.path.join(dynamic_full_path, f'dynamic_results_{key}_{idx}.joblib')) and\
               os.path.exists(os.path.join(comb_full_path, f'comb_results_{key}_{idx}.joblib')) and\
               os.path.exists(os.path.join(all_full_path, f'all_results_{key}_{idx}.joblib')):
                   warn(f'all_results_{key}_{idx}.joblib Already exists ...')
                   continue

            warn(f'Processing {sd_files[idx]} ...')
            s_f_train, d_f_train  = sd_files[idx]
            s_f_test, d_f_test  = sd_files[idx+1]
            assert ("_".join(s_f_train.split('_')[1:]) == "_".join(d_f_train.split('_')[1:])) and 'train' in s_f_train
            assert ("_".join(s_f_test.split('_')[1:]) == "_".join(d_f_test.split('_')[1:])) and 'test' in s_f_test
            
            df_s_train = pd.read_csv(os.path.join(input_dir, s_f_train), index_col=0)
            df_s_test = pd.read_csv(os.path.join (input_dir, s_f_test), index_col=0)
            df_d_train = pd.read_csv(os.path.join(input_dir, d_f_train))
            df_d_test = pd.read_csv(os.path.join (input_dir, d_f_test))

            # Train using only static
            # static_results_path = full_path / Path('static_results')
            # static_results_path.mkdir(parents=True, exist_ok=True)
            warn('Static model performance ...')
            X_s_train = df_s_train.drop([pat_id_s, target], axis=1)
            y_s_train = df_s_train[target]

            X_s_test = df_s_test.drop([pat_id_s, target], axis=1)
            y_s_test = df_s_test[target]

            static_results_dict = cross_validate_all(X_s_train, y_s_train, X_s_test, y_s_test)
            with open(os.path.join(static_full_path, f'static_results_{key}_{idx}.joblib'), 'wb') as f:
                joblib.dump(static_results_dict, f)
            warn('==================================================')

            # Train using only dynamic
            # pat_id_s_train = df_d_train.pop(pat_id_d)
            warn('Dynamic model performance ...')
            pat_id_s_train = df_d_train[pat_id_d]
            X_d_train = df_d_train.drop(pat_id_d, axis=1)
            y_d_train = pd.merge(pat_id_s_train, df_s_train[[pat_id_s, target]], left_on=pat_id_d, right_on=pat_id_s)
            assert((y_d_train[pat_id_s] != pat_id_s_train).sum()==0)
            y_d_train = y_d_train[target]

            # pat_id_s_test = df_d_test.pop(pat_id_d)
            pat_id_s_test = df_d_test[pat_id_d]
            X_d_test = df_d_test.drop(pat_id_d, axis=1)
            y_d_test = pd.merge(pat_id_s_test, df_s_test[[pat_id_s, target]], left_on=pat_id_d, right_on=pat_id_s)
            assert((y_d_test[pat_id_s] != pat_id_s_test).sum()==0)
            y_d_test = y_d_test[target]
            dynamic_results_dict = cross_validate_all(X_d_train, y_d_train, X_d_test, y_d_test)
            with open(os.path.join(dynamic_full_path, f'dynamic_results_{key}_{idx}.joblib'), 'wb') as f:
                joblib.dump(dynamic_results_dict, f)
            warn('==================================================')

            # Merge static, and dynamic decisions
            # static_dynamic_model_names = list(zip(static_results_dict.keys(), dynamic_results_dict.keys()))
            warn('Combined model applied on testing data ...')
            df_d_test = pd.merge(df_d_test, df_s_test[[pat_id_s, target]], how='inner', left_on=pat_id_d, right_on=pat_id_s)
            df_s_test = df_s_test.sort_values(by=pat_id_s)
            df_d_test = df_d_test.sort_values(by=pat_id_d)

            X_s_test = df_s_test.drop([pat_id_s, target], axis=1)
            X_d_test = df_d_test.drop([pat_id_d,pat_id_s,target], axis=1)
            y_s_test = df_s_test[target]
            y_d_test = df_d_test[target]
            assert(all(y_s_test.values==y_d_test))
            com_results = {'d_d_test':df_d_test, 'df_s_test': df_s_test}

            for d_model_name in dynamic_results_dict:
                s_model = static_results_dict[d_model_name]['model']
                d_model = dynamic_results_dict[d_model_name]['model']

                y_s_pred = s_model.predict_proba(X_s_test)
                y_d_pred = d_model.predict_proba(X_d_test)
                sidx = np.where(s_model.classes_==1)[0]
                didx = np.where(d_model.classes_==1)[0]
                preds = (y_s_pred[:, sidx] + y_d_pred[:, didx])/2.0
                ypreds = np.where(preds>=0.5, 1, 0)
                bacc = balanced_accuracy_score(y_s_test.values, ypreds)
                roc = roc_auc_score(y_s_test.values, ypreds)
                ppv = precision_score(y_s_test.values, ypreds)
                com_results[d_model_name] = {
                    'bacc':bacc,
                    'roc': roc,
                    'ppv': ppv,
                    'y_s_preds': y_s_pred,
                    'y_d_preds': y_d_pred,
                    's_classes_': s_model.classes_,
                    'd_classes_': d_model.classes_
                }
                warn(f'{d_model}: bacc={bacc}, roc={roc}, ppv={ppv}')
            
            with open(os.path.join(comb_full_path, f'comb_results_{key}_{idx}.joblib'), 'wb') as f:
                joblib.dump(com_results, f)

            warn('===================================================================')
            # Concatenate both and train an aggregated model
            warn('All features are combined and fed to the models .....')
            df_train_all = df_d_train.merge(df_s_train, how='inner', left_on=pat_id_d, right_on=pat_id_s)
            df_test_all = df_d_test.drop(columns=[pat_id_s, target]).merge(df_s_test, how='inner', left_on=pat_id_d, right_on=pat_id_s)

            X_train_all = df_train_all.drop(columns=[pat_id_s, pat_id_d, target])
            y_train_all = df_train_all[target]
            
            X_test_all = df_test_all.drop(columns=[pat_id_s, pat_id_d, target])
            y_test_all = df_test_all[target]

            all_results_dict = cross_validate_all(X_train_all, y_train_all, X_test_all, y_test_all)
            with open(os.path.join(all_full_path, f'all_results_{key}_{idx}.joblib'), 'wb') as f:
                joblib.dump(all_results_dict, f)

            warn('===================================================================')
            warn('===================================================================')
            
    return
            
            
def score(model, Xte, yte):
    yhat = model.predict(Xte)
    bacc = balanced_accuracy_score(yte, yhat)
    roc = roc_auc_score(yte, yhat)
    ppv = precision_score(yte, yhat)
    return {'bacc': bacc, 'roc': roc, 'ppv': ppv}


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

    model.fit(Xtr, ytr)

    return train_score_avg, test_score_avg, train_score_std, test_score_std, train_score_list, test_score_list


def get_run_idx(ml_path):
    return len(os.listdir(ml_path))

'''
Input:
    1. constants.ML_DATA_OUTPUT_ID / tr_{constants.TRAINING_PERIOD}_te_{constants.TESTING_PERIOD} 
    2. Files are structured as follow:
        a. static_{events_idx_included}_{train/test}_{fold_idx}.csv
        a. dynamic_{events_idx_included}_{train/test}_{fold_idx}.csv

Objectives:
    - For each events_idx_included get the average performance over all of the folds
    - What is the performance using only static features?
    - What is the performance using only dynamic features?

output:
    - Create a folder for every events_idx
    - Within each events_idx folder get the top performing model metrics on that fold
    **Bonus**
        - Use the first fold to estimate hyper parameters and feature selection
        - User the hyperparameters and selected features for the second fold and refine the 
        hyper parameters and feature selection
        - Propoagate the enhancement over the folds
        - Test your enhancements over a large batch of the training/testing, and test on the rest
    - Aggregate the metrics over all events_idx to get an overall performance of the model along with
    the most important features, and confidence at every event_idx

'''

if __name__ == "__main__":
    feats_dir = constants.ML_DATA_OUTPUT_ID
    tr_te_file = f'tr_{constants.TRAINING_PERIOD}_te_{constants.TESTING_PERIOD}'
    Path(constants.ML_RESULTS_OUTPUT).mkdir(parents=True, exist_ok=True)
    idx = get_run_idx(constants.ML_RESULTS_OUTPUT)
    config_dict = dict(
        feats_dir = feats_dir,
        tr_te_file=tr_te_file,
        classifiers_configs = dict(
            lgbm='default',
            xgb='default',
            lg='default',
            cat=dict(
                iterations=4000,
                learning_rate=0.03,
                depth=5
            )
        )
    )

    # wandb.init(project='ED-StaticDynamic ML NO HyperParams',
    #             config= config_dict,
    #             name=f'Run_{idx}'
    #            )

    train_ml_folder(feats_dir,
                    constants.ML_RESULTS_OUTPUT,
                    idx,
                    tr_te_file,
                    'PAT_ENC_CSN_ID',
                    'Unnamed: 0',
                    'ED_Disposition',
                    is_wandb=False)
    
    warn('Successfully done!')