from collections import defaultdict
from multiprocessing.sharedctypes import Value
from operator import contains
import os
from pathlib import Path
import sys
from turtle import st

# MAIN_DIR = os.getenv("EDStaticDynamic")
# sys.path.insert(1, MAIN_DIR)
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import polars as pl
import numpy as np
from const import constants
import joblib
import datetime as dt
from polars_scripts.datamanager import CustomCrossFold
from polars_scripts.static_transformers import CustomOneHotEncoding
from polars_scripts.dynamic_transformers import CustomDynamicOneHotEncoding
from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
# import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             recall_score, precision_score, roc_auc_score)
from warnings import warn
from polars_scripts.preprocessor import CascadePipeline, Preprocessor
from polars_scripts.feature_selector import static_dynamic_rfcev, static_dynamic_rfcev01
from polars_scripts.datacleaner import DataCleaner

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import yaml

def run_fsv01(model_dict: dict, cascadepipeline_output: dict, with_imputation:bool, target_col='has_admit_order', id_col='PAT_ENC_CSN_ID', model_type='all'):
    if with_imputation:
        if cascadepipeline_output['static'] is not None:
            X_train_static = cascadepipeline_output['static'].drop([target_col])
            y_train = cascadepipeline_output['static'][target_col]
        else:
            X_train_static = None
        if cascadepipeline_output['dynamic'] is not None:
            X_train_dynamic = cascadepipeline_output['dynamic'].drop([target_col]),
            y_train = cascadepipeline_output['dynamic'][target_col]
        else:
            X_train_dynamic = None

        if X_train_dynamic is not None and X_train_static is not None:
            assert all(X_train_static[id_col]==X_train_dynamic[id_col]), 'static and dynamic preprocessed dataframe are not aligned with respect to PAT_ENC_CSN_ID'
            X_train_comb = pl.concat([X_train_static, X_train_dynamic.drop([id_col])], how='horizontal')
        else:
            X_train_comb = None

    else:
        if cascadepipeline_output['out']['static'] is not None:
            X_train_static = cascadepipeline_output['out']['static'].drop([target_col])
            y_train = cascadepipeline_output['out']['static'][target_col]
        else:
            X_train_static = None
        if cascadepipeline_output['out']['dynamic'] is not None:
            X_train_dynamic = cascadepipeline_output['out']['dynamic'].drop([target_col]),
            y_train = cascadepipeline_output['out']['dynamic'][target_col]
        else:
            X_train_dynamic = None

        if X_train_dynamic is not None and X_train_static is not None:
            assert all(X_train_static[id_col]==X_train_dynamic[id_col]), 'static and dynamic preprocessed dataframe are not aligned with respect to PAT_ENC_CSN_ID'
            X_train_comb = pl.concat([X_train_static, X_train_dynamic.drop([id_col])], how='horizontal')
        else:
            X_train_comb = None

    fs_out = static_dynamic_rfcev01(model_dict, (X_train_static, X_train_dynamic, X_train_comb, y_train),
                                    id_col=id_col, feat_types=model_type)
    
    return fs_out 

def run_fs(model: str, cascadepipeline_output: dict, with_imputation:bool):
    if with_imputation:
        (X_train_static, X_train_dynamic, y_train) = (cascadepipeline_output['static'].drop(['has_admit_order']), 
                                                        cascadepipeline_output['dynamic'].drop(['has_admit_order']),
                                                        cascadepipeline_output['dynamic']['has_admit_order'])
        assert all(X_train_static['PAT_ENC_CSN_ID']==X_train_dynamic['PAT_ENC_CSN_ID']), 'static and dynamic preprocessed dataframe are not aligned with respect to PAT_ENC_CSN_ID'
        X_train_comb = pl.concat([X_train_static, X_train_dynamic.drop(['PAT_ENC_CSN_ID'])], how='horizontal')

        # (X_test_static, X_test_dynamic, y_train) =  (test_out_dict['static'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']), 
        #                                                 test_out_dict['dynamic'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']),
        #                                                 test_out_dict['dynamic']['has_admit_order'])
        # X_test_comb = pl.concat([X_test_static, X_test_dynamic], how='horizontal')
            
    else:
        (X_train_static, X_train_dynamic, y_train) = (cascadepipeline_output['out']['static'].drop([ 'has_admit_order']), 
                                                        cascadepipeline_output['out']['dynamic'].drop(['has_admit_order']),
                                                        cascadepipeline_output['out']['dynamic']['has_admit_order'])
        assert all(X_train_static['PAT_ENC_CSN_ID']==X_train_dynamic['PAT_ENC_CSN_ID']), 'static and dynamic preprocessed dataframe are not aligned with respect to PAT_ENC_CSN_ID'
        X_train_comb = pl.concat([X_train_static, X_train_dynamic.drop(['PAT_ENC_CSN_ID'])], how='horizontal')

        # (X_test_static_noimp, X_test_dynamic_noimp, y_train) =  (test_out_dict['out']['static'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']), 
        #                                                 test_out_dict['out']['dynamic'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']),
        #                                                 test_out_dict['out']['dynamic']['has_admit_order'])
        # X_test_comb_noimp = pl.concat([X_test_static_noimp, X_test_dynamic_noimp], how='horizontal')
        
    if model.lower() in ['lr' ,'lg', 'logistic', 'logisticregression', 'logistic_regression']:
        fs_out = static_dynamic_rfcev(LogisticRegression(max_iter=500), LogisticRegression(max_iter=500), LogisticRegression(max_iter=500),
                                     (X_train_static, X_train_dynamic, X_train_comb, y_train), None, "logisticregression") 
    elif model.lower() in ['cat', 'catboost']:
        fs_out = static_dynamic_rfcev(CatBoostClassifier(verbose=0, thread_count=12, iterations=500),
                                     CatBoostClassifier(verbose=0, thread_count=12, iterations=500),
                                     CatBoostClassifier(verbose=0, thread_count=12, iterations=500),
                                     (X_train_static, X_train_dynamic, X_train_comb, y_train), None, "catboost") 
    else:
        raise ValueError(f"Only ['catboost', 'logisticregression'] are supported. {model} is given ...")
    
    return fs_out 

def apply_fsv01(mats, fs_out):
    """
    mats are excpected to be tuple of 3 or 4 in the following order (static_mat, dynamic_mat, comb_mat) 

    Args:
        mats (typle): (static_mat: pl.DataFrame, dynamic_mat: pl.DataFrame, comb_mat: pl.DataFrame)
        fs_out (dict): output from `feature_selector/static_dynamic_rfcev` method
    """
    if len(mats) == 3:
        X_train_static, X_train_dynamic, X_train_comb = mats
    elif len(mats) == 4:
        X_train_static, X_train_dynamic, X_train_comb, ytrain = mats
        
    if fs_out['static_model'] is not None and'static_model' in fs_out:
        selected_cols = np.array( X_train_static.columns)[fs_out['static_model'].support_].tolist()
        X_train_static = pl.from_numpy(fs_out['static_model'].transform(X_train_static), schema=selected_cols, orient='row')

    if fs_out['dynamic_model'] is not None and 'dynamic_model' in fs_out:
        selected_cols = np.array(X_train_dynamic.columns)[fs_out['dynamic_model'].support_].tolist()
        X_train_dynamic = pl.from_numpy(fs_out['dynamic_model'].transform(X_train_dynamic), schema=selected_cols, orient='row')
    
    if fs_out['comb_model'] is not None and 'comb_model' in fs_out:
        selected_cols = np.array(X_train_comb.columns)[fs_out['comb_model'].support_].tolist()
        X_train_comb = pl.from_numpy(fs_out['comb_model'].transform(X_train_comb), schema=selected_cols, orient='row')

    if len(mats) == 3:
        return X_train_static, X_train_dynamic, X_train_comb
    return X_train_static, X_train_dynamic, X_train_comb, ytrain

from typing import Tuple
def mat_tonumpy(mats: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series]):
    output = []
    for i in range(len(mats)):
        if mats[i] is not None:
            output.append(mats[i].to_numpy())
        else:
            output.append(None)

    return tuple(output)
        

def tupilize_cascadepiplize_output(cascadepipeline_output: dict,
                                   with_imputation: bool=False, return_id=True, id_col='PAT_ENC_CSN_ID', target_col='has_admit_order', dropcols=[]):
    def filter_dropped_cols(dff, dr_cols):
        dr = []
        for c in dr_cols:
            if c in dff.columns:
                dr.append(c)
        return dr

    if with_imputation:
        if cascadepipeline_output['static'] is not None:
            st_dr = filter_dropped_cols(cascadepipeline_output['static'], dropcols)
            X_train_static = cascadepipeline_output['static'].drop([id_col, target_col]+st_dr)
            y_train = cascadepipeline_output['static'][target_col]
            pat_id = cascadepipeline_output['static'][id_col]
        else:
            X_train_static = None
        if cascadepipeline_output['dynamic'] is not None:
            dt_dr = filter_dropped_cols(cascadepipeline_output['dynamic'], dropcols)
            X_train_dynamic = cascadepipeline_output['dynamic'].drop([id_col, target_col]+dt_dr),
            y_train = cascadepipeline_output['dynamic'][target_col]
            pat_id = cascadepipeline_output['dynamic'][id_col]
        else:
            X_train_dynamic = None

        if X_train_dynamic is not None and X_train_static is not None:
            assert all(X_train_static[id_col]==X_train_dynamic[id_col]), 'static and dynamic preprocessed dataframe are not aligned with respect to PAT_ENC_CSN_ID'
            X_train_comb = pl.concat([X_train_static.drop([id_col, target_col]), X_train_dynamic.drop([id_col, target_col])], how='horizontal')
        else:
            X_train_comb = None

    else:
        if cascadepipeline_output['out']['static'] is not None:
            st_dr = filter_dropped_cols(cascadepipeline_output['static'], dropcols)
            X_train_static = cascadepipeline_output['out']['static'].drop([id_col, target_col]+st_dr)
            y_train = cascadepipeline_output['out']['static'][target_col]
            pat_id = cascadepipeline_output['static'][id_col]
        else:
            X_train_static = None
        if cascadepipeline_output['out']['dynamic'] is not None:
            dt_dr = filter_dropped_cols(cascadepipeline_output['dynamic'], dropcols)
            X_train_dynamic = cascadepipeline_output['out']['dynamic'].drop([id_col, target_col]+dt_dr),
            y_train = cascadepipeline_output['out']['dynamic'][target_col]
            pat_id = cascadepipeline_output['dynamic'][id_col]
        else:
            X_train_dynamic = None

        if X_train_dynamic is not None and X_train_static is not None:
            assert all(X_train_static[id_col]==X_train_dynamic[id_col]), 'static and dynamic preprocessed dataframe are not aligned with respect to PAT_ENC_CSN_ID'
            X_train_comb = pl.concat([X_train_static.drop([id_col, target_col]), X_train_dynamic.drop([id_col, target_col])], how='horizontal')
        else:
            X_train_comb = None
    
    if return_id:
        return (X_train_static, X_train_dynamic, X_train_comb, y_train), pat_id
    return (X_train_static, X_train_dynamic, X_train_comb, y_train)


def get_metrics_single(
    ytrain,
    ytest,
    model,
    X_train,
    X_test
):

    y_tr_prob = model.predict_proba(X_train)
    y_te_prob = model.predict_proba(X_test)

    y_tr = model.predict(X_train)
    y_te = model.predict(X_test)
    
    acc_train = balanced_accuracy_score(ytrain, y_tr)
    recall_train = recall_score(ytrain, y_tr)
    conf_train = confusion_matrix(ytrain, y_tr)
    pre_train = precision_score(ytrain, y_tr)
    roc_train = roc_auc_score(ytrain, y_tr)

    acc_test = balanced_accuracy_score(ytest, y_te)
    recall_test = recall_score(ytest, y_te)
    conf_test = confusion_matrix(ytest, y_te)
    pre_test = precision_score(ytest, y_te)
    roc_test = roc_auc_score(ytest, y_te)

    return {
        'train':{
        'acc':acc_train,
        'roc':roc_train,
        'recall':recall_train,
        'pre':pre_train,
        'conf':conf_train,
        'preds': y_tr_prob
    },
    'test':{
        'acc':acc_test,
        'roc':roc_test,
        'recall':recall_test,
        'pre':pre_test,
        'conf':conf_test,
        'preds':y_te_prob
    }
    }


def get_metrics_dict(
    ytrain,
    ytest,
    static_model,
    dynamic_model,
    comb_model,
    X_static_train_np,
    X_dynamic_train_np,
    comb_train_np,
    X_static_test_np,
    X_dynamic_test_np,
    comb_test_np
):
    if static_model is not None:
        static_dict = get_metrics_single(ytrain, ytest, static_model, X_static_train_np, X_static_test_np)
    else:
        static_dict = {}
    if dynamic_model is not None:
        dynamic_dict = get_metrics_single(ytrain, ytest, dynamic_model, X_dynamic_train_np, X_dynamic_test_np)
    else:
        dynamic_dict = {}

    if comb_model is not None:
        comb_dict = get_metrics_single(ytrain, ytest, comb_model, comb_train_np, comb_test_np)
    else:
        comb_dict = {}

    return {
        'static': static_dict,
        'dynamic': dynamic_dict,
        'comb': comb_dict
    }

    # return {
        # 'acc':{
        #     'static_tr': acc_static_train,
        #     'dynamic_tr': acc_dynamic_train,
        #     'comb_tr': acc_comb_train,
        #     'static_te': acc_static_test,
        #     'dynamic_te': acc_dynamic_test,
        #     'comb_te': acc_comb_test
        # },
        # 'recall':{
        #     'static_tr': recall_static_train,
        #     'dynamic_tr': recall_dynamic_train,
        #     'comb_tr': recall_comb_train,
        #     'static_te': recall_static_test,
        #     'dynamic_te': recall_dynamic_test,
        #     'comb_te': recall_comb_test
        # },
        # 'ppv':{
        #     'static_tr': ppv_static_train,
        #     'dynamic_tr': ppv_dynamic_train,
        #     'comb_tr': ppv_comb_train,
        #     'static_te': ppv_static_test,
        #     'dynamic_te': ppv_dynamic_test,
        #     'comb_te': ppv_comb_test
        # },
        # 'confmat':{
        #     'static_tr': conf_static_train,
        #     'dynamic_tr': conf_dynamic_train,
        #     'comb_tr': conf_comb_train,
        #     'static_te': conf_static_test,
        #     'dynamic_te': conf_dynamic_test,
        #     'comb_te': conf_comb_test
        # },
        # 'AUC':{
        #     'static_tr': roc_static_train,
        #     'dynamic_tr': roc_dynamic_train,
        #     'comb_tr': roc_comb_train,
        #     'static_te': roc_static_test,
        #     'dynamic_te': roc_dynamic_test,
        #     'comb_te': roc_comb_test
        # },
        # 'preds':{
        #     'ytrain': ytrain,
        #     'ytest': ytest,
        #     'ystat_tr':ystat_tr,
        #     'ydyn_tr':ydyn_tr,
        #     'ycomb_tr':ycomb_tr,
        #     'ystat_te':ystat_te,
        #     'ydyn_te':ydyn_te,
        #     'ycomb_te':ycomb_te
        # }
    # }

def train1(
    static_model, dynamic_model, comb_model,
    train_mats, test_mats, tr_pat_id, te_pat_id, model_name = ""
):

    X_static_train_np, X_dynamic_train_np, comb_train_np, train_label_list = train_mats
    X_static_test_np, X_dynamic_test_np, comb_test_np, test_label_list = test_mats

    try:
        if static_model is not None:
            warn(f"Training {model_name} static model ...")
            static_model.fit(X_static_train_np, train_label_list, eval_set=(X_static_test_np, test_label_list)) 
        if dynamic_model is not None:
            warn(f"Training {model_name} dynamic model ...")
            dynamic_model.fit(X_dynamic_train_np, train_label_list, eval_set=(X_dynamic_test_np, test_label_list)) 
        if comb_model is not None:
            warn(f"Training {model_name} combine model ...")
            comb_model.fit(comb_train_np, train_label_list, eval_set=(comb_test_np, test_label_list))
    except Exception as e:
        if 'eval_set' in static_model.__dir__():
            if static_model is not None:
                warn(f"Training {model_name} static model ...")
                static_model.fit(X_static_train_np, train_label_list, eval_set=[(X_static_test_np, test_label_list)]) 
            if dynamic_model is not None:
                warn(f"Training {model_name} dynamic model ...")
                dynamic_model.fit(X_dynamic_train_np, train_label_list, eval_set=[(X_dynamic_test_np, test_label_list)]) 
            if comb_model is not None:
                warn(f"Training {model_name} combine model ...")
                comb_model.fit(comb_train_np, train_label_list, eval_set=([comb_test_np, test_label_list]))
        else:
            if static_model is not None:
                warn(f"Training {model_name} static model ...")
                static_model.fit(X_static_train_np, train_label_list) 
            if dynamic_model is not None:
                warn(f"Training {model_name} dynamic model ...")
                dynamic_model.fit(X_dynamic_train_np, train_label_list) 
            if comb_model is not None:
                warn(f"Training {model_name} combine model ...")
                comb_model.fit(comb_train_np, train_label_list)

    warn(f"Calculating performance ...")
    scores = get_metrics_dict(
        train_label_list,
        test_label_list,
        static_model,
        dynamic_model,
        comb_model,
        X_static_train_np,
        X_dynamic_train_np,
        comb_train_np,
        X_static_test_np,
        X_dynamic_test_np,
        comb_test_np
    )
    
    return {
       'scores':scores,
       'static_model':static_model,
       'dynamic_model':dynamic_model,
       'comb_model':comb_model,
       'train_pat_id': tr_pat_id,
       'test_pat_id': te_pat_id
    } 

def add_suffix_filename(filename, suffix):
    return filename if filename.endswith(suffix) else filename+suffix

def process_fold_notidx(config, df_tr, df_te, preproc, output_dir):
    train_out_dict = preproc.train_fit_transform(df_tr)
    test_out_dict = preproc.eval_transform(df_te)

    if config['data_preprocessing']['saveto'] is not None and output_dir is not None:
        filename = add_suffix_filename(config['data_preprocessing']['saveto'], '.joblib')
        with open(os.path.join(output_dir, filename), 'wb') as f:
            joblib.dump(preproc, f)
            
    if config['experiment']['apply_fs']:
        if config['feature_selector']['model_params']['method']=='RFECV':
            cat_out = run_fsv01(config['feature_selector']['model_params'],
                               train_out_dict,
                                with_imputation=config['feature_selector']['with_imputation'],
                                model_type=config['experiment']['model_type'],
                                target_col=config['experiment']['target_col']
                                )
            train_mat, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=config['feature_selector']['with_imputation'],
                                                                return_id=True, target_col=config['experiment']['target_col'], id_col=config['experiment']['id_col'], dropcols=['from_rwb'])
            test_mat, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=config['feature_selector']['with_imputation'],
                                                                return_id=True, target_col=config['experiment']['target_col'], id_col=config['experiment']['id_col'], dropcols=['from_rwb'])
            train_mat = apply_fsv01(train_mat, cat_out)
            train_mat_np = mat_tonumpy(train_mat)

            test_mat = apply_fsv01(test_mat, cat_out)
            test_mat_np = mat_tonumpy(test_mat)
        else:
            raise ValueError(f"Only RFECV is currently implemented, please make sure to set it in the YAML file at [feature_selector][model_params][method] ...")
    else:
        train_mat, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=config['feature_selector']['with_imputation'],
                                                            return_id=True, target_col=config['experiment']['target_col'], id_col=config['experiment']['id_col'], dropcols=['from_rwb'])
        test_mat, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=config['feature_selector']['with_imputation'],
                                                                return_id=True, target_col=config['experiment']['target_col'], id_col=config['experiment']['id_col'], dropcols=['from_rwb'])
        train_mat_np = mat_tonumpy(train_mat)
        test_mat_np = mat_tonumpy(test_mat)
        
    
    static_model = None
    dynamic_model = None
    comb_model = None
    if config['machine_learning']['static'] is not None: 
        static_model = build_model_from_dict(config['machine_learning']['static']) 
    if config['machine_learning']['dynamic'] is not None:
        dynamic_model = build_model_from_dict(config['machine_learning']['dynamic']) 
    if config['machine_learning']['comb'] is not None:
        comb_model = build_model_from_dict(config['machine_learning']['dynamic']) 
    out = train1(static_model, dynamic_model, comb_model, train_mat_np, test_mat_np, tr_pat_id, te_pat_id)
    if config['machine_learning']['saveto'] is not None and len(config['machine_learning']['saveto']) > 0:
        filename = config['machine_learning']['saveto']
        write_obj(out, output_dir, filename, '.joblib')

    df_tr = df_tr.with_columns(
        pl.Series('cat_preds', out['scores']['static']['train']['preds'])
    )
    df_te = df_te.with_columns(
        pl.Series('cat_preds', out['scores']['static']['test']['preds'])
    )
    df_tr.write_parquet(os.path.join(output_dir, 'df_tr_preds.parquet'))
    df_te.write_parquet(os.path.join(output_dir, 'df_te_preds.parquet'))
        

def process_fold_tidx(config, df_tr, df_te, preproc, output_dir, tidx_list):
    for tidx in tidx_list:
        output_tidx = os.path.join(output_dir, f'tidx_{tidx}')
        Path(output_tidx).mkdir(parents=True, exist_ok=True)

        df_tr_idx = df_tr.filter(pl.col('elapsed_time_min')<=tidx)
        df_te_idx = df_te.filter(pl.col('elapsed_time_min')<=tidx)
        train_out_dict = preproc.train_fit_transform(df_tr_idx)
        test_out_dict = preproc.eval_transform(df_te_idx)
         
        if config['data_preprocessing']['saveto'] is not None and output_dir is not None:
            filename = config['data_preprocessing']['saveto']
            write_obj(preproc, output_tidx, filename, '.joblib')
                
        # TODO: Handle the else condition of apply_fs similar to how it is handled in `process_fold_notidx`
        if config['experiment']['apply_fs']:
            if config['feature_selector']['model_params']['method']=='RFECV':
                cat_out = run_fsv01(config['feature_selector']['model_params'],
                                train_out_dict,
                                    with_imputation=config['feature_selector']['with_imputation'],
                                    model_type=config['experiment']['model_type']
                                    )
                train_mat, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=config['feature_selector']['with_imputation'],
                                                                    return_id=True, dropcols=['from_rwb'])
                test_mat, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=config['feature_selector']['with_imputation'],
                                                                    return_id=True, dropcols=['from_rwb'])
                train_mat = apply_fsv01(train_mat, cat_out)
                train_mat_np = mat_tonumpy(train_mat)

                test_mat = apply_fsv01(test_mat, cat_out)
                test_mat_np = mat_tonumpy(test_mat)
                if config['feature_selector']['saveto'] is not None and len(config['feature_selector']['saveto'])>0:
                    filename = config['feature_selector']['saveto']
                    write_obj(cat_out, output_tidx, filename, '.joblib')
            else:
                raise ValueError(f"Only RFECV is currently implemented, please make sure to set it in the YAML file at [feature_selector][model_params][method] ...")
        
        static_model = None
        dynamic_model = None
        comb_model = None
        if config['machine_learning']['static'] is not None: 
            static_model = build_model_from_dict(config['machine_learning']['static']) 
        if config['machine_learning']['dynamic'] is not None:
            dynamic_model = build_model_from_dict(config['machine_learning']['dynamic']) 
        if config['machine_learning']['comb'] is not None:
            comb_model = build_model_from_dict(config['machine_learning']['dynamic']) 
        out = train1(static_model, dynamic_model, comb_model, train_mat_np, test_mat_np, tr_pat_id, te_pat_id)
        if config['machine_learning']['saveto'] is not None and len(config['machine_learning']['saveto']) > 0:
            filename = config['machine_learning']['saveto']
            write_obj(out, output_tidx, filename, '.joblib')

def build_model_from_dict(ml_dict):
    if ml_dict['model_name'] == 'catboost':
        model = CatBoostClassifier(**ml_dict['hyperparameters'])
    elif ml_dict['model_name'] == 'xgboost':
        model = XGBClassifier(**ml_dict['hyperparameters'])
    elif ml_dict['model_name'] == 'logistic_regression':
        model = LogisticRegression(**ml_dict['hyperparameters'])
    elif ml_dict['model_name'] == 'random_forest':
        model = RandomForestClassifier(**ml_dict['hyperparameters'])
    else:
        raise ValueError(f"Only the following classifiers are supported: [catboost, xgboost, logistic_regression, random_forest]. {ml_dict['model_name']} is given ...")

    return model

def write_obj(obj, output_dir, filename, suffix):
    filename = add_suffix_filename(filename, suffix)
    with open(os.path.join(output_dir, filename), 'wb') as f:
        joblib.dump(obj, f)

def main():
    with open('./src/polars_scripts/configv01_rwb.yaml', 'r') as file:
        config = yaml.safe_load(file)

    main_out_dir = config['experiment']['output_dir']
    
    if main_out_dir is not None and config['experiment']['output_labelByDate']:
        now_ = dt.datetime.now().strftime("%Y%m%d%_%H-%M-%s")
        output_dir = os.path.join(main_out_dir, now_)
    if main_out_dir is not None and not config['experiment']['output_labelByDate']:
        if config['experiment']['output_labelCustom'] is None:
            raise ValueError(f"YAML file => output_labelCustom in experiment needs to have value or set output_labelByDate to True if "
                             f"output_dir is not None ...")
        output_dir = os.path.join(main_out_dir, config['experiment']['output_labelCustom'])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    if main_out_dir is None:
        output_dir = None
    
    if output_dir is not None:
        write_obj(obj=config, output_dir=output_dir, filename='config_file', suffix='.joblib')
    
    if config['experiment']['tidx_method'] is None:
        tidx_list = []
    elif config['experiment']['tidx_method'] == 'list':
        tidx_list = config['experiment']['tidx_list']

    elif config['experiment']['tidx_method'] == 'generator':
        tidx_list = list(range(config['experiment']['tidx_start'],
                               config['experiment']['tidx_end']+config['experiment']['tidx_step'],
                               config['experiment']['tidx_step']))
    else:
        raise ValueError(f'tidx_method is only allowed to be either ["generator", "list", None] ... {config["experiment"][""]}')

    dc = DataCleaner(config['data_cleaner']['filepath'], target=config['experiment']['target_col'], analysis_type=config['experiment']['model_type'],
                     dx_col='Primary_DX_Name', multi_col_separator=config['data_cleaner']['multi_col_separator'])
    df = dc.run()

    #TODO: Preprocess target [Discharge=1; Admitted=0]
    df = df.with_columns(
        pl.when(pl.col('ED_Disposition').str.to_lowercase().str.contains('discharge')).then(pl.lit('Discharged')).otherwise(pl.col('ED_Disposition')).alias('ED_Disposition')
    ).with_columns(
        pl.when(pl.col('ED_Disposition').str.to_lowercase().str.contains('admit')).then(pl.lit('Admitted')).otherwise(pl.col('ED_Disposition')).alias('ED_Disposition')
    )
    df = df.filter(pl.col('ED_Disposition').is_in(['Admitted','Discharged']))
    df = df.with_columns(
        pl.when(pl.col('ED_Disposition')=='Admitted').then(pl.lit(0)).otherwise(pl.lit(1)).cast(pl.Int32).alias('ED_Disposition')
    )
    
    if config['data_preprocessing']['preprocess'] == 'static':
        preproc = CascadePipeline(
            config['data_preprocessing']['static_preprocessor'],
            None,
            simp_dict={c: 'median' for c in config['data_preprocessing']['s_imp']['cols']},
            dimp_dict={c: 'median' for c in config['data_preprocessing']['d_imp']['cols']},
            target_col=config['experiment']['target_col'],
            id_col=config['experiment']['id_col']
            )
    elif config['data_preprocessing']['preprocess'] == 'dynamic':
        preproc = CascadePipeline(
            None,
            config['data_preprocessing']['dynamic_preprocessor'],
            simp_dict={c: 'median' for c in config['data_preprocessing']['s_imp']['cols']},
            dimp_dict={c: 'median' for c in config['data_preprocessing']['d_imp']['cols']},
            target_col=config['experiment']['target_col'],
            id_col=config['experiment']['id_col']
            )
    elif config['data_preprocessing']['preprocess'] == 'all':
        preproc = CascadePipeline(
            config['data_preprocessing']['static_preprocessor'],
            config['data_preprocessing']['dynamic_preprocessor'],
            simp_dict={c: 'median' for c in config['data_preprocessing']['s_imp']['cols']},
            dimp_dict={c: 'median' for c in config['data_preprocessing']['d_imp']['cols']},
            target_col=config['experiment']['target_col'],
            id_col=config['experiment']['id_col']
            )

    else:
        fd = config['data_preprocessing']['preprocess']
        raise ValueError(f"YAML file => data_preprocessing->preprocess can only be ['static', 'dynamic', 'all']; {fd} is found ...")

    if config['folds']['steps_in_months'] is not None:
        cv = CustomCrossFold(config['folds']['training_in_months']*30, config['folds']['testing_in_months']*30,
                                config['folds']['steps_in_months']*30, config['folds']['date_col'])
    else:
        cv = CustomCrossFold(config['folds']['training_in_months']*30, config['folds']['testing_in_months']*30,
                                None, config['folds']['date_col'])
    
    if config['experiment']['type']=='latest' and config['experiment']['no_validation']: 
        Xtre = cv.get_latest_batch(df)
        batches = [(Xtre, Xtre)]
    if config['experiment']['type']=='latest' and not config['experiment']['no_validation']: 
        batches = cv.reverse_split(df, max_iter=1)
    elif config['experiment']['type'] == 'fold':
        batches = cv.reverse_split(df)
    

    for fold_idx, (df_tr, df_te) in enumerate(batches):
        if tidx_list is None or len(tidx_list) == 0:
            process_fold_notidx(config, df_tr, df_te, preproc, output_dir)
        else:
            process_fold_tidx(config, df_tr, df_te, preproc, output_dir, tidx_list)
    
if __name__ == "__main__":
    main()