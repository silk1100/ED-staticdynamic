import os
from pathlib import Path
import sys

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
from polars_scripts.feature_selector import static_dynamic_rfcev

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def get_static_feats(X, pat_id):
    X_static_tr = X.select(constants.id_cols+constants.static_cols+constants.target_col)
    X_pat_static = X_static_tr.group_by(pat_id).agg(
        [
            pl.col(c).last().alias(c) for c in X_static_tr.columns if c != pat_id
        ]
    )
    return X_pat_static

def get_dynamic_feats(X):
    X_dynamic_tr = X.select(constants.id_cols+constants.dynamic_cols+constants.target_col)
    return X_dynamic_tr

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

    ystat_tr = static_model.predict(X_static_train_np)
    ydyn_tr = dynamic_model.predict(X_dynamic_train_np)
    ycomb_tr = comb_model.predict(comb_train_np)

    ystat_te = static_model.predict(X_static_test_np)
    ydyn_te = dynamic_model.predict(X_dynamic_test_np)
    ycomb_te = comb_model.predict(comb_test_np)

    acc_static_train = balanced_accuracy_score(ytrain, ystat_tr)
    acc_dynamic_train = balanced_accuracy_score(ytrain, ydyn_tr)
    acc_comb_train = balanced_accuracy_score(ytrain, ycomb_tr)
    acc_static_test = balanced_accuracy_score(ytest, ystat_te)
    acc_dynamic_test = balanced_accuracy_score(ytest, ydyn_te)
    acc_comb_test = balanced_accuracy_score(ytest, ycomb_te)

    recall_static_train = recall_score(ytrain, ystat_tr)
    recall_dynamic_train = recall_score(ytrain, ydyn_tr)
    recall_comb_train = recall_score(ytrain, ycomb_tr)
    recall_static_test = recall_score(ytest, ystat_te)
    recall_dynamic_test = recall_score(ytest, ydyn_te)
    recall_comb_test = recall_score(ytest, ycomb_te)

    ppv_static_train = precision_score(ytrain, ystat_tr)
    ppv_dynamic_train = precision_score(ytrain, ydyn_tr)
    ppv_comb_train = precision_score(ytrain, ycomb_tr)
    ppv_static_test = precision_score(ytest, ystat_te)
    ppv_dynamic_test = precision_score(ytest, ydyn_te)
    ppv_comb_test = precision_score(ytest, ycomb_te)

    conf_static_train = confusion_matrix(ytrain, ystat_tr)
    conf_dynamic_train = confusion_matrix(ytrain, ydyn_tr)
    conf_comb_train = confusion_matrix(ytrain, ycomb_tr)
    conf_static_test = confusion_matrix(ytest, ystat_te)
    conf_dynamic_test = confusion_matrix(ytest, ydyn_te)
    conf_comb_test = confusion_matrix(ytest, ycomb_te)

    roc_static_train = roc_auc_score(ytrain, ystat_tr)
    roc_dynamic_train = roc_auc_score(ytrain, ydyn_tr)
    roc_comb_train = roc_auc_score(ytrain, ycomb_tr)
    roc_static_test = roc_auc_score(ytest, ystat_te)
    roc_dynamic_test = roc_auc_score(ytest, ydyn_te)
    roc_comb_test = roc_auc_score(ytest, ycomb_te)

    return {
        'acc':{
            'static_tr': acc_static_train,
            'dynamic_tr': acc_dynamic_train,
            'comb_tr': acc_comb_train,
            'static_te': acc_static_test,
            'dynamic_te': acc_dynamic_test,
            'comb_te': acc_comb_test
        },
        'recall':{
            'static_tr': recall_static_train,
            'dynamic_tr': recall_dynamic_train,
            'comb_tr': recall_comb_train,
            'static_te': recall_static_test,
            'dynamic_te': recall_dynamic_test,
            'comb_te': recall_comb_test
        },
        'ppv':{
            'static_tr': ppv_static_train,
            'dynamic_tr': ppv_dynamic_train,
            'comb_tr': ppv_comb_train,
            'static_te': ppv_static_test,
            'dynamic_te': ppv_dynamic_test,
            'comb_te': ppv_comb_test
        },
        'confmat':{
            'static_tr': conf_static_train,
            'dynamic_tr': conf_dynamic_train,
            'comb_tr': conf_comb_train,
            'static_te': conf_static_test,
            'dynamic_te': conf_dynamic_test,
            'comb_te': conf_comb_test
        },
        'AUC':{
            'static_tr': roc_static_train,
            'dynamic_tr': roc_dynamic_train,
            'comb_tr': roc_comb_train,
            'static_te': roc_static_test,
            'dynamic_te': roc_dynamic_test,
            'comb_te': roc_comb_test
        },
        'preds':{
            'ytrain': ytrain,
            'ytest': ytest,
            'ystat_tr':ystat_tr,
            'ydyn_tr':ydyn_tr,
            'ycomb_tr':ycomb_tr,
            'ystat_te':ystat_te,
            'ydyn_te':ydyn_te,
            'ycomb_te':ycomb_te
        }
    }

def train1(
    static_model, dynamic_model, comb_model,
    train_mats, test_mats, tr_pat_id, te_pat_id, model_name = ""
):

    X_static_train_np, X_dynamic_train_np, comb_train_np, train_label_list = train_mats
    X_static_test_np, X_dynamic_test_np, comb_test_np, test_label_list = test_mats

    try:
        warn(f"Training {model_name} static model ...")
        static_model.fit(X_static_train_np, train_label_list, eval_set=(X_static_test_np, test_label_list)) 
        warn(f"Training {model_name} dynamic model ...")
        dynamic_model.fit(X_dynamic_train_np, train_label_list, eval_set=(X_dynamic_test_np, test_label_list)) 
        warn(f"Training {model_name} combine model ...")
        comb_model.fit(comb_train_np, train_label_list, eval_set=(comb_test_np, test_label_list))
    except Exception as e:
        if 'eval_set' in static_model.__dir__():
            warn(f"Training {model_name} static model ...")
            static_model.fit(X_static_train_np, train_label_list, eval_set=[(X_static_test_np, test_label_list)]) 
            warn(f"Training {model_name} dynamic model ...")
            dynamic_model.fit(X_dynamic_train_np, train_label_list, eval_set=[(X_dynamic_test_np, test_label_list)]) 
            warn(f"Training {model_name} combine model ...")
            comb_model.fit(comb_train_np, train_label_list, eval_set=([comb_test_np, test_label_list]))
        else:
            warn(f"Training {model_name} static model ...")
            static_model.fit(X_static_train_np, train_label_list) 
            warn(f"Training {model_name} dynamic model ...")
            dynamic_model.fit(X_dynamic_train_np, train_label_list) 
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
    
def impute_with_median(Xstr: pl.DataFrame, params=None):
    null_vocab = None
    if params is None:
        null_vocab = {}
        for c in Xstr.columns:
            if Xstr[c].is_null().sum() > 0:
                warn(f'{c} has {Xstr[c].is_null().sum()} missing vals ...')
                null_vocab[c] = Xstr[c].median()
        params = null_vocab
    
    for c, val in params.items():
        Xstr = Xstr.with_columns(
            pl.col(c).fill_null(value=val).alias(c)
        )
    return Xstr, null_vocab 

def apply_fs(mats, fs_out):
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
        
    if 'static_model' in fs_out:
        selected_cols = np.array(X_train_static.columns)[fs_out['static_model'].support_].tolist()
        X_train_static = pl.from_numpy(fs_out['static_model'].transform(X_train_static), schema=selected_cols, orient='row')
    if 'dynamic_model' in fs_out:
        selected_cols = np.array(X_train_dynamic.columns)[fs_out['dynamic_model'].support_].tolist()
        X_train_dynamic = pl.from_numpy(fs_out['dynamic_model'].transform(X_train_dynamic), schema=selected_cols, orient='row')
    if 'comb_model' in fs_out:
        selected_cols = np.array(X_train_comb.columns)[fs_out['comb_model'].support_].tolist()
        X_train_comb = pl.from_numpy(fs_out['comb_model'].transform(X_train_comb), schema=selected_cols, orient='row')

    if len(mats) == 3:
        return X_train_static, X_train_dynamic, X_train_comb
    return X_train_static, X_train_dynamic, X_train_comb, ytrain

def tupilize_cascadepiplize_output(cascadepipeline_output: dict,
                                   with_imputation: bool=False, return_id=True):

    # Check that all ids and targets match for all of the output
    assert all(cascadepipeline_output['static']['PAT_ENC_CSN_ID']==cascadepipeline_output['dynamic']['PAT_ENC_CSN_ID']),  f"Dropped column mismatch in training between index {i} and {i+1} in the tuple for column {dropped_cols[i][idx]} ..."
    pat_id = cascadepipeline_output['static']['PAT_ENC_CSN_ID'].to_numpy()
    
    if with_imputation:
        (X_train_static, X_train_dynamic, y_train) = (cascadepipeline_output['static'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']), 
                                                        cascadepipeline_output['dynamic'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']),
                                                        cascadepipeline_output['dynamic']['has_admit_order'])
        X_train_comb = pl.concat([X_train_static, X_train_dynamic], how='horizontal')

        # (X_test_static, X_test_dynamic, y_train) =  (test_out_dict['static'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']), 
        #                                                 test_out_dict['dynamic'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']),
        #                                                 test_out_dict['dynamic']['has_admit_order'])
        # X_test_comb = pl.concat([X_test_static, X_test_dynamic], how='horizontal')
            
    else:
        (X_train_static, X_train_dynamic, y_train) = (cascadepipeline_output['out']['static'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']), 
                                                        cascadepipeline_output['out']['dynamic'].drop(['PAT_ENC_CSN_ID', 'has_admit_order']),
                                                        cascadepipeline_output['out']['dynamic']['has_admit_order'])
        X_train_comb = pl.concat([X_train_static, X_train_dynamic], how='horizontal')
    
    if return_id:
        return (X_train_static, X_train_dynamic, X_train_comb, y_train), pat_id
    return (X_train_static, X_train_dynamic, X_train_comb, y_train)

from typing import Tuple, final
def mat_tonumpy(mats: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series]):
    output = []
    for i in range(len(mats)):
        output.append(mats[i].to_numpy())

    return tuple(output)
        

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


if __name__ == "__main__":
    # with open(constants.CLEAN_DATA, 'rb') as f:
    #     df = joblib.load(f)
    # df = pl.read_parquet(os.path.join(MAIN_DIR, "ED_clean.parquet"))    
    df = pl.read_parquet(constants.CLEAN_DATA_PARQUET)

    ccf = CustomCrossFold(30*6, 30*6, 30*6, 'Arrived_Time')
    df = df.with_columns(
        (pl.col("Calculated_DateTime")-pl.col("Calculated_DateTime").first()).dt.total_minutes().over('PAT_ENC_CSN_ID').alias('minutes')
    )
    # time_range = np.arange(30, 60*24*2+30, 30)
    
    
    perform_fs = False 
    time_prefix = dt.datetime.now().strftime("%y%m%d-%H%M")
    save_dict = {}
    output_path = os.path.join(constants.OUTPUT_DIR, "Clean_V1_OHE_ALL")
    final_path = os.path.join(constants.FINAL_MODEL_OUTPUT, time_prefix)
    Path(final_path).mkdir(parents=True, exist_ok=True)
    # time_range = np.arange(30, 12*60+30, 30)
    # time_range = np.arange(12*60+30, 24*60+30, 30)
    time_range = np.arange(24*60+30, 32*60+30, 30)
    # time_range = np.arange(32*60+30, 48*60+30, 30)

    static_ohe_dict = dict(
        single_val_cols=constants.static_singleval_cat_cols,
        multi_val_cols=constants.static_multval_cat_cols,
        num_cols=constants.static_num_cols,
        dep_col_dict={},
        num_norm_method=constants.static_num_norm_method,
        null_vals=constants.NULL_LIST,
        vocabthresh=100,
        cumprob_inc_thresh=0.99
    )

    static_ohe_obj = CustomOneHotEncoding(
        single_val_cols=constants.static_singleval_cat_cols,
        multi_val_cols=constants.static_multval_cat_cols,
        num_cols=constants.static_num_cols,
        num_norm_method=constants.static_num_norm_method,
        null_vals=constants.NULL_LIST,
        vocabthresh=100,
        cumprob_inc_thresh=0.99
    )
    '''
     If you are using onehotencoding with dependent features, then do not use the same features
     in the singleval or multival cols

     However, if you are processing them via LabelEncoding then make sure to include the independent
     column in the singleval cols

    '''
    dyn_ohe_dict = dict(
        single_val_cols=constants.dynamic_singleval_col,
        multi_val_cols=constants.dynamic_multival_col,
        id_col="PAT_ENC_CSN_ID",
        dep_col_dict={'Type_NORM':["EVENT_NAME_NORM"]},
        num_cols=constants.dynamic_num_cols,
        num_norm_methods=constants.dynamic_num_norm_method,
        skip_indp_val={'vitals'},
        vocabthresh=100,
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST
    )

    if 'Type_NORM' in constants.dynamic_singleval_col:
        constants.dynamic_singleval_col.remove('Type_NORM')
    if 'EVENT_NAME_NORM' in constants.dynamic_singleval_col:
        constants.dynamic_singleval_col.remove('EVENT_NAME_NORM')
    if 'Type_NORM' in constants.dynamic_multival_col:
        constants.dynamic_multival_col.remove('Type_NORM')
    if 'EVENT_NAME_NORM' in constants.dynamic_multival_col:
        constants.dynamic_multival_col.remove('EVENT_NAME_NORM')
    dyn_ohe_obj = CustomDynamicOneHotEncoding(
        single_val_cols=constants.dynamic_singleval_col,
        multi_val_cols=constants.dynamic_multival_col,
        id_col="PAT_ENC_CSN_ID",
        dep_col_dict={'Type_NORM':["EVENT_NAME_NORM"]},
        num_cols=constants.dynamic_num_cols,
        num_norm_methods=constants.dynamic_num_norm_method,
        skip_indp_val={'vitals'},
        vocabthresh=100,
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST
    )
    '''
    - Cool milestones check to make sure that your data is alright
    [*] Check the # of admitted vs discharged counters in your training and testing datasets
        - Xte_t.group_by('PAT_ENC_CSN_ID').agg(pl.col('has_admit_order').last())['has_admit_order'].value_counts()
        - Xtr_t.group_by('PAT_ENC_CSN_ID').agg(pl.col('has_admit_order').last())['has_admit_order'].value_counts()

    [*] Check the training and testing span to confirm that they are covering the period you assigned during training
        - Xtr_t['Arrived_Time'].min(); Xtr_t['Arrived_Time'].max();
        - Xte_t['Arrived_Time'].min(); Xte_t['Arrived_Time'].max();
        - Xtr_t['Arrived_Time'].dt.weekday().value_counts(); # Confirm that all days of the week are covered 1->7 polars
        - Xtr_t['Arrived_Time'].dt.day().value_counts(); # Confirm that all days of the month are covered 1->31 polars

    [*] Check the longest duration in the minutes to confirm that it is matching the t in time_range
        - Xtr_t.group_by("PAT_ENC_CSN_ID").agg(pl.col('minutes').last()).max()
    
        
    '''
    begin_time=time.time()


    # for Xtr, Xte in ccf.split(df):
    for data_idx, (Xtr, Xte) in enumerate(ccf.reverse_split(df)):
        start_training_date = Xtr['Arrived_Time'].min().date().strftime('%Y-%m-%d')
        end_training_date = Xtr['Arrived_Time'].max().date().strftime('%Y-%m-%d')
        assert len(set(Xtr['PAT_ENC_CSN_ID']).intersection(set(Xte['PAT_ENC_CSN_ID'])))==0, f'There is a data leakage in index {data_idx} training range: {start_training_date} -> {end_training_date}'
        warn(f"Beginning of train time: {Xtr['Arrived_Time'].min()} ...")
        warn(f"Beginning of train time: {Xtr['Arrived_Time'].max()} ...")

        warn(f"Beginning of train time: {Xte['Arrived_Time'].min()} ...")
        warn(f"Beginning of train time: {Xte['Arrived_Time'].max()} ...")
        t_idx = []
        train_indices = []
        test_indices = []
        cat_score_list = []
        rf_score_list = []
        xgb_score_list = []
        lg_score_list = []
        lsvm_score_list = []
        for t in time_range:
            Xtr_t = Xtr.filter(pl.col('minutes')<=t)
            Xte_t = Xte.filter(pl.col('minutes')<=t)
            # Sample pats            
            # tr_n = int(0.05*Xtr_t['PAT_ENC_CSN_ID'].n_unique())
            # te_n = int(0.05*Xte_t['PAT_ENC_CSN_ID'].n_unique())
            # tr_pat = np.random.choice(Xtr_t['PAT_ENC_CSN_ID'].unique(), tr_n, replace=False)
            # te_pat = np.random.choice(Xte_t['PAT_ENC_CSN_ID'].unique(), te_n, replace=False)
            # Xtr_t = Xtr_t.filter(pl.col("PAT_ENC_CSN_ID").is_in(tr_pat))
            # Xte_t = Xte_t.filter(pl.col("PAT_ENC_CSN_ID").is_in(te_pat))

            preproc = CascadePipeline(static_ohe_dict, dyn_ohe_dict,
                            simp_dict={c: 'median' for c in list(constants.static_num_norm_method.keys())},
                            dimp_dict={c: 'median' for c in list(constants.dynamic_num_norm_method.keys())})
            train_out_dict = preproc.train_fit_transform(Xtr_t)
            test_out_dict = preproc.eval_transform(Xte_t)

            if test_out_dict['static'] is None or len(test_out_dict['static'])==0 :
                continue

            if perform_fs:
                cat_out = run_fs('cat', train_out_dict, with_imputation=False)
                train_mat, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=False, return_id=True)
                train_mat = apply_fs(train_mat, cat_out)
                train_mat_np = mat_tonumpy(train_mat)

                train_mat_imp, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=True, return_id=True)
                train_mat_imp = apply_fs(train_mat_imp, cat_out)
                train_mat_imp_np = mat_tonumpy(train_mat_imp)

                test_mat, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=False, return_id=True)
                test_mat = apply_fs(test_mat, cat_out)
                test_mat_np = mat_tonumpy(test_mat)

                test_mat_imp, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=True, return_id=True)
                test_mat_imp = apply_fs(test_mat_imp, cat_out)
                test_mat_imp_np = mat_tonumpy(test_mat_imp)


                out_cat = train1(CatBoostClassifier(verbose=0), CatBoostClassifier(verbose=0), CatBoostClassifier(verbose=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "catboost")

                if data_idx == 0:
                    with open(os.path.join(constants.CUSTOM_MODEL_OUTPUT,
                            f'custom_fs-cat_ml-cat_tid-{t}_from-{start_training_date}_to-{end_training_date}.joblib'), 'wb') as f:
                        joblib.dump({
                        'preprocessor': preproc,
                        'fs':{
                            'comb_model': cat_out['comb_model'],
                            'comb_feats':cat_out['comb_feats'],
                            }
                            ,
                            'ml':{
                                'comb_model':out_cat['comb_model'],
                            }},f)
                               
                out_xgb = train1(XGBClassifier(verbosity=0), XGBClassifier(verbosity=0), XGBClassifier(verbosity=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "xgboost")
                out_rf = train1(RandomForestClassifier(verbose=0), RandomForestClassifier(verbose=0), CatBoostClassifier(verbose=0),
                       train_mat_imp_np, test_mat_imp_np, tr_pat_id, te_pat_id, "rf")
                out_lg = train1(LogisticRegression(max_iter=500), LogisticRegression(max_iter=500), LogisticRegression(max_iter=500),
                       train_mat_imp_np, test_mat_imp_np, tr_pat_id, te_pat_id, "lg")
                out_lsvm = train1(LinearSVC(verbose=0), LinearSVC(verbose=0), LinearSVC(verbose=0),
                       train_mat_imp_np, test_mat_imp_np, tr_pat_id, te_pat_id, "lsvm")
                
                with open(os.path.join(final_path,
                          f'fs-cat_ml_tid-{t}_from-{start_training_date}_to-{end_training_date}.joblib'), 'wb') as f:
                    joblib.dump({
                    'preprocessor': preproc,
                    'fs': cat_out,
                        'ml':{
                            'cat':out_cat,
                            'lg':out_lg,
                            'lsvm':out_lsvm,
                            'rf':out_rf,
                            'xgb':out_xgb
                        }},f)
                
                lg_out = run_fs('lg', train_out_dict, with_imputation=True)
                train_mat, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=True, return_id=True)
                train_mat = apply_fs(train_mat, lg_out)

                test_mat, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=True, return_id=True)
                test_mat = apply_fs(test_mat, lg_out)

                train_mat_np = mat_tonumpy(train_mat)
                test_mat_np = mat_tonumpy(test_mat)
                
                out_cat = train1(CatBoostClassifier(verbose=0), CatBoostClassifier(verbose=0), CatBoostClassifier(verbose=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "catboost")
                
                out_rf = train1(RandomForestClassifier(verbose=0), RandomForestClassifier(verbose=0), CatBoostClassifier(verbose=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "rf")
                out_lg = train1(LogisticRegression(max_iter=500), LogisticRegression(max_iter=500), LogisticRegression(max_iter=500),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "lg")
                out_xgb = train1(XGBClassifier(verbosity=0), XGBClassifier(verbosity=0), XGBClassifier(verbosity=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "xgboost")
                out_lsvm = train1(LinearSVC(verbose=0), LinearSVC(verbose=0), LinearSVC(verbose=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "lsvm")

                with open(os.path.join(final_path,
                          f'fs-lg_ml_tid-{t}_from-{start_training_date}_to-{end_training_date}.joblib'), 'wb') as f:
                    joblib.dump({
                    'preprocessor': preproc,
                    'fs': lg_out,
                        'ml':{
                            'cat':out_cat,
                            'lg':out_lg,
                            'lsvm':out_lsvm,
                            'rf':out_rf,
                            'xgb':out_xgb
                        }},f)
                
            else:
                train_mat, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=False, return_id=True)
                train_mat_np = mat_tonumpy(train_mat)
                train_mat_imp, tr_pat_id = tupilize_cascadepiplize_output(train_out_dict, with_imputation=True, return_id=True)
                train_mat_imp_np = mat_tonumpy(train_mat_imp)

                test_mat, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=False, return_id=True)
                test_mat_np = mat_tonumpy(test_mat)
                test_mat_imp, te_pat_id = tupilize_cascadepiplize_output(test_out_dict, with_imputation=True, return_id=True)
                test_mat_imp_np = mat_tonumpy(test_mat_imp)

                out_cat = train1(CatBoostClassifier(verbose=0), CatBoostClassifier(verbose=0), CatBoostClassifier(verbose=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "catboost")
                
                out_rf = train1(RandomForestClassifier(verbose=0), RandomForestClassifier(verbose=0), CatBoostClassifier(verbose=0),
                       train_mat_imp_np, test_mat_imp_np, tr_pat_id, te_pat_id, "rf")
                out_lg = train1(LogisticRegression(max_iter=500), LogisticRegression(max_iter=500), LogisticRegression(max_iter=500),
                       train_mat_imp_np, test_mat_imp_np, tr_pat_id, te_pat_id, "lg")
                out_xgb = train1(XGBClassifier(verbosity=0), XGBClassifier(verbosity=0), XGBClassifier(verbosity=0),
                       train_mat_np, test_mat_np, tr_pat_id, te_pat_id, "xgboost")
                out_lsvm = train1(LinearSVC(verbose=0), LinearSVC(verbose=0), LinearSVC(verbose=0),
                       train_mat_imp_np, test_mat_imp_np, tr_pat_id, te_pat_id, "lsvm")

                with open(os.path.join(final_path,
                          f'fs-null_ml_tid-{t}_from-{start_training_date}_to-{end_training_date}.joblib'), 'wb') as f:
                    joblib.dump({
                    'preprocessor': preproc,
                    'fs': {},
                        'ml':{
                            'cat':out_cat,
                            'lg':out_lg,
                            'lsvm':out_lsvm,
                            'rf':out_rf,
                            'xgb':out_xgb
                        }},f)
        print(f'time taken to process between {time_range[0]} to {time_range[-1]} sequentially is {time.time()-begin_time} seconds')
