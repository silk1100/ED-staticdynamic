
import os
import sys

# MAIN_DIR = os.getenv("EDStaticDynamic")
# sys.path.insert(1, MAIN_DIR)

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import polars as pl
import numpy as np
from const import constants
import joblib
import datetime as dt
from sklearn.feature_selection import RFECV
from sklearn.metrics import balanced_accuracy_score
from polars_scripts.datamanager import CustomCrossFold
from dynamic_transformers import CustomDynamicOneHotEncoding
from static_transformers import CustomOneHotEncoding 
import time
from warnings import warn
from catboost import CatBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_static_feats(X: pl.DataFrame, pat_id):
    X_static_tr = X.select(constants.id_cols+constants.static_cols+constants.target_col)
    X_pat_static = X_static_tr.group_by(pat_id).agg(
        [
            pl.col(c).last().alias(c) for c in X_static_tr.columns if c != pat_id
        ]
    )
    return X_pat_static

def get_dynamic_feats(X: pl.DataFrame):
    X_dynamic_tr = X.select(constants.id_cols+constants.dynamic_cols+constants.target_col)
    return X_dynamic_tr

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

def prepare_feats(
    Xstr, Xste, Xdtr, Xdte, snum_cols, dnum_cols, num_norm=False, allow_nulls=False
):
    train_label_list = Xstr['has_admit_order'].to_numpy()
    test_label_list  = Xste['has_admit_order'].to_numpy()

    if len(test_label_list)<1:
        return None, None

    if not allow_nulls:
        Xstr, sparams = impute_with_median(Xstr)
        if len(sparams)>0:
            Xste, _ = impute_with_median(Xste, sparams)
        Xdtr, dparams = impute_with_median(Xdtr)
        if len(dparams)>0:
            Xdte, _ = impute_with_median(Xdte, dparams)
    
    if not num_norm: 
        snum_cols_ = [f'{c}_NUMNORM' for c in snum_cols]
        dnum_cols_ = [f'{c}_NUMNORM' for c in dnum_cols]
    else:
        snum_cols_ = snum_cols
        dnum_cols_ = dnum_cols
    Xstatic_train = Xstr.drop(['PAT_ENC_CSN_ID', 'has_admit_order']+snum_cols_)
    X_static_test = Xste.drop(['PAT_ENC_CSN_ID', 'has_admit_order']+snum_cols_)
    X_dynamic_train = Xdtr.drop(['PAT_ENC_CSN_ID', 'has_admit_order']+dnum_cols_)
    X_dynamic_test = Xdte.drop(['PAT_ENC_CSN_ID', 'has_admit_order']+dnum_cols_)

    X_static_train_np = Xstatic_train.to_numpy()
    X_static_test_np = X_static_test.to_numpy()
    X_dynamic_train_np = X_dynamic_train.to_numpy()
    X_dynamic_test_np = X_dynamic_test.to_numpy()
    comb_train_np = np.hstack([X_static_train_np, X_dynamic_train_np])
    comb_test_np = np.hstack([X_static_test_np, X_dynamic_test_np])
    
    static_feats = Xstatic_train.columns
    dynamic_feats = X_dynamic_train.columns
    comb_feats = static_feats+dynamic_feats
    
    return (X_static_train_np, X_dynamic_train_np, comb_train_np, train_label_list),\
            (X_static_test_np, X_dynamic_test_np, comb_test_np, test_label_list), (static_feats, dynamic_feats, comb_feats)


def train_preprocess(Xtr_t, Xte_t, pat_id, static_preprocessor, dynamic_preprocessor):
    X_static_tr = get_static_feats(Xtr_t, pat_id)
    X_static_te = get_static_feats(Xte_t, pat_id)
    X_dynamic_tr = get_dynamic_feats(Xtr_t)
    X_dynamic_te = get_dynamic_feats(Xte_t)
    Xpre_static_tr = static_preprocessor.fit_transform(X_static_tr)
    Xpre_static_te = static_preprocessor.transform(X_static_te)
    Xpre_dynamic_tr = dynamic_preprocessor.fit_transform(X_dynamic_tr)
    Xpre_dynamic_te = dynamic_preprocessor.transform(X_dynamic_te)
    
    Xdtr = pl.concat([X_dynamic_tr.drop(["Type_NORM", "EVENT_NAME_NORM", "dxcode_list"]+dynamic_preprocessor.num_cols),
                      Xpre_dynamic_tr], how='horizontal')
    Xdte = pl.concat([X_dynamic_te.drop(["Type_NORM", "EVENT_NAME_NORM", "dxcode_list"]+dynamic_preprocessor.num_cols),
                      Xpre_dynamic_te], how='horizontal')
    
    Xstr = pl.concat([X_static_tr.drop(["Ethnicity", "FirstRace", "Sex", "Acuity_Level",
                                        "Means_Of_Arrival", "Coverage_Financial_Class_Grouper",
                                        "Arrived_Time", 'cc_list',
                                        "arr_month", "arr_day", "arr_dow", "holiday", 'arr_hour']+static_preprocessor.num_cols), Xpre_static_tr], how='horizontal')
    Xste = pl.concat([X_static_te.drop(["Ethnicity", "FirstRace", "Sex", "Acuity_Level",
                                        "Means_Of_Arrival", "Coverage_Financial_Class_Grouper",
                                        "Arrived_Time", 'cc_list',
                                        "arr_month", "arr_day", "arr_dow", "holiday", "arr_hour"]+static_preprocessor.num_cols), Xpre_static_te], how='horizontal')

    ncols = set(dynamic_preprocessor.num_cols + [f'{c}_NUMNORM' for c in dynamic_preprocessor.num_cols])
    Xdtr_pat = Xdtr.group_by('PAT_ENC_CSN_ID').agg(
        [pl.col(c).sum().alias(c) for c in Xpre_dynamic_tr.columns if c not in ncols]+[
            pl.col('elapsed_time_min').last().alias('elapsed_time_min'),
            pl.col('event_idx').last().alias('event_idx'),
            pl.col('ED_Location_YN').mean().alias('ED_Location_YN'),
            pl.col('has_admit_order').last().alias('has_admit_order')
        ]+[
            pl.col(c).last().alias(c) for c in X_dynamic_tr.columns if c.startswith('MEAS')
        ]+[
            pl.col(c).sum().alias(c) for c in X_dynamic_tr.columns if c.startswith('Order')
        ]+[
            pl.col(c).sum().alias(c) for c in X_dynamic_tr.columns if c.startswith('Result')
        ]
    )
    Xdte_pat = Xdte.group_by('PAT_ENC_CSN_ID').agg(
        [pl.col(c).sum().alias(c) for c in Xpre_dynamic_tr.columns if c not in ncols]+[
            pl.col('elapsed_time_min').last().alias('elapsed_time_min'),
            pl.col('event_idx').last().alias('event_idx'),
            pl.col('ED_Location_YN').mean().alias('ED_Location_YN'),
            pl.col('has_admit_order').last().alias('has_admit_order')
        ]+[
            pl.col(c).last().alias(c) for c in X_dynamic_tr.columns if c.startswith('MEAS')
        ]+[
            pl.col(c).sum().alias(c) for c in X_dynamic_tr.columns if c.startswith('Order')
        ]+[
            pl.col(c).sum().alias(c) for c in X_dynamic_tr.columns if c.startswith('Result')
        ]
    )
    '''
    - Test cases:
    [*] Test the alignment of PAT_ENC_CSN_ID and has_admit_order between Xstr, Xdtr_pat and Xste, Xdte_pat
        - (Xstr['PAT_ENC_CSN_ID']!=Xdtr_pat['PAT_ENC_CSN_ID']).sum()
        - (Xste['PAT_ENC_CSN_ID']!=Xdte_pat['PAT_ENC_CSN_ID']).sum()
        - (Xstr['has_admit_order']!=Xdtr_pat['has_admit_order']).sum()
        - (Xste['has_admit_order']!=Xdte_pat['has_admit_order']).sum()
    [*] Test that all columns are numeric
        - for c in Xstr.columns:
            if Xstr[c].dtype == pl.String:
                print(c)
        - for c in Xdtr_pat.columns:
            if Xdtr_pat[c].dtype == pl.String:
                print(c)
    '''

    Xstr, Xdtr_pat = pl.align_frames(Xstr, Xdtr_pat, on="PAT_ENC_CSN_ID")
    Xste, Xdte_pat = pl.align_frames(Xste, Xdte_pat, on="PAT_ENC_CSN_ID")

    # Remove all columns with 100% nans
    for c in Xstr.columns:
        if Xstr[c].is_null().sum() == Xstr.shape[0]:
            warn(f'Removing {c} because it is all nans in Static matrix...')
            Xstr = Xstr.drop(c)
            Xste = Xste.drop(c)

    for c in Xdtr_pat.columns:
        if Xdtr_pat[c].is_null().sum() == Xdtr_pat.shape[0]:
            warn(f'Removing {c} because it is all nans in Dynamic matrix...')
            Xdtr_pat = Xdtr_pat.drop(c)
            Xdte_pat = Xdte_pat.drop(c)
    
    return  (Xstr, Xdtr_pat), (Xste, Xdte_pat)

from typing import Tuple
def static_dynamic_rfcev(
    static_model,
    dynamic_model,
    comb_model,
    train_mats:Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list], colnames=None, model_name=""
):
    
    X_static_train, X_dynamic_train, X_comb_train, train_label_list = train_mats
    pat_ids = X_static_train['PAT_ENC_CSN_ID']
    pat_idd = X_dynamic_train['PAT_ENC_CSN_ID']
    pat_idc = X_comb_train['PAT_ENC_CSN_ID']
    assert all(pat_ids==pat_idd), 'PAT_ENC_CSN_ID of static and dynamic do not allign'
    assert all(pat_ids==pat_idc), 'PAT_ENC_CSN_ID of static and combined do not allign'
    
    
    
    static_cols = X_static_train.drop(['PAT_ENC_CSN_ID']).columns
    dynamic_cols = X_dynamic_train.drop(['PAT_ENC_CSN_ID']).columns
    comb_cols = X_comb_train.drop(['PAT_ENC_CSN_ID']).columns 
            
    static_step = int(0.05*X_static_train.shape[1])
    dynamic_step = int(0.05*X_dynamic_train.shape[1])
    comb_step = int(0.05*X_comb_train.shape[1])
    srfecv = RFECV(static_model, step=static_step, cv=3, scoring='balanced_accuracy')
    drfecv = RFECV(dynamic_model, step=dynamic_step, cv=3, scoring='balanced_accuracy')
    crfecv = RFECV(comb_model, step=comb_step, cv=3, scoring='balanced_accuracy')
    warn(f"Training {model_name} static feature selector ... ")
    srfecv.fit(X_static_train.drop(['PAT_ENC_CSN_ID']), train_label_list)
    warn(f"Training {model_name} dynamic feature selector ... ")
    drfecv.fit(X_dynamic_train.drop(['PAT_ENC_CSN_ID']), train_label_list)
    warn(f"Training {model_name} comb feature selector ... ")
    crfecv.fit(X_comb_train.drop(['PAT_ENC_CSN_ID']), train_label_list)

    return {
       'static_model':srfecv,
       'dynamic_model':drfecv,
       'comb_model':crfecv,
       'static_feats':static_cols,
       'dynamic_feats':dynamic_cols,
       'comb_feats':comb_cols,
       'train_pat_id': pat_idc.to_numpy(),
       'test_pat_id': pat_idc.to_numpy()
    }

if __name__ == "__main__":
    # with open(constants.CLEAN_DATA, 'rb') as f:
    #     df = joblib.load(f)
    # df = pl.read_parquet(os.path.join(MAIN_DIR, "ED_clean.parquet"))    
    df = pl.read_parquet(constants.CLEAN_DATA_PARQUET)
    ccf = CustomCrossFold(30*6, 30, 30*6, 'Arrived_Time')
    df = df.with_columns(
        (pl.col("Calculated_DateTime")-pl.col("Calculated_DateTime").first()).dt.total_minutes().over('PAT_ENC_CSN_ID').alias('minutes')
    )
    # time_range = np.arange(30, 60*24*2+30, 30)

    # time_range = np.arange(30, 12*60+30, 30)
    # time_range = np.arange(12*60+30, 24*60+30, 30)
    time_range = np.arange(24*60+30, 32*60+30, 30)
    # time_range = np.arange(32*60+30, 48*60+30, 30)

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
    train_indices = []
    test_indices = []
    cat_score_list = []
    rf_score_list = []
    xgb_score_list = []
    lg_score_list = []
    lsvm_score_list = []
    output_path = os.path.join(constants.FS_OUTPUT_DIR, "Clean_V1_OHE_ALL")

    for Xtr, Xte in ccf.split(df):
        warn(f"Beginning of train time: {Xtr['Arrived_Time'].min()} ...")
        warn(f"Beginning of train time: {Xtr['Arrived_Time'].max()} ...")

        warn(f"Beginning of train time: {Xte['Arrived_Time'].min()} ...")
        warn(f"Beginning of train time: {Xte['Arrived_Time'].max()} ...")
        t_idx = []
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

            # Feature
            (Xstr, Xdtr), (Xste, Xdte) = \
                train_preprocess(Xtr_t, Xte_t, "PAT_ENC_CSN_ID", static_ohe_obj, dyn_ohe_obj)
            
            snum_cols = static_ohe_obj.num_cols
            dnum_cols = dyn_ohe_obj.num_cols

            rf_train_mats, rf_test_mats, rf_colnames          = prepare_feats(Xstr, Xste, Xdtr, Xdte, snum_cols, dnum_cols, num_norm=False, allow_nulls=False)
            lin_train_mats, lin_test_mats, lin_colnames       = prepare_feats(Xstr, Xste, Xdtr, Xdte, snum_cols, dnum_cols, num_norm=True, allow_nulls=False)
            gtree_train_mats, gtree_test_mats, gtree_colnames = prepare_feats(Xstr, Xste, Xdtr, Xdte, snum_cols, dnum_cols, num_norm=False, allow_nulls=True)

            if rf_test_mats is None:
                pass

            # rf_score_model = train1(RandomForestClassifier(n_jobs=12), RandomForestClassifier(n_jobs=12), RandomForestClassifier(n_jobs=12),
            #                         rf_train_mats, rf_colnames, 'RandomForest')

            # lsvm_score_model = train1(LinearSVC(), LinearSVC(), LinearSVC(),
            #                          lin_train_mats, lin_colnames, "lsvm")
            
            # xgb_score_model = train1(XGBClassifier(verbosity=0, n_jobs=12), XGBClassifier(verbosity=0, n_jobs=12),
            #                          XGBClassifier(verbosity=0, n_jobs=12),
            #                         gtree_train_mats, gtree_colnames, "xgboost")

            cat_score_model = static_dynamic_rfcev(CatBoostClassifier(verbose=0, thread_count=12, iterations=500),
                                     CatBoostClassifier(verbose=0, thread_count=12, iterations=500),
                                     CatBoostClassifier(verbose=0, thread_count=12, iterations=500),
                                    gtree_train_mats, gtree_colnames, "catboost")

            lg_score_model = static_dynamic_rfcev(LogisticRegression(n_jobs=12), LogisticRegression(n_jobs=12), LogisticRegression(n_jobs=12),
                                    lin_train_mats, lin_colnames, 'logisticregression')

            cat_score_list.append(cat_score_model)
            # rf_score_list.append(rf_score_model)
            # xgb_score_list.append(xgb_score_model)
            lg_score_list.append(lg_score_model)
            # lsvm_score_list.append(lsvm_score_model)


            train_indices.append((Xtr['Arrived_Time'].min(), Xtr['Arrived_Time'].max()))
            test_indices.append((Xte['Arrived_Time'].min(), Xte['Arrived_Time'].max()))
            t_idx.append(t)
            
        
        with open(os.path.join(output_path, f'results_{time_range[0]}_{time_range[-1]}.joblib'), 'wb') as f:
            joblib.dump({
                'lsvm_score_list': lsvm_score_list,
                'rf_score_list': rf_score_list,
                'cat_score_list': cat_score_list,
                'lg_score_list': lg_score_list,
                'xgb_score_list': xgb_score_list,
                'train_indices':train_indices,
                'test_indices':test_indices,
                'time_idx':t_idx,
            }, f)
        print(f'time taken to process between {time_range[0]} to {time_range[-1]} sequentially is {time.time()-begin_time} seconds')
    # t=time.time()
    # for Xtr, Xte in ccf.split(df):
    #     with ProcessPoolExecutor(max_workers=19) as pool:
    #         pfunc = partial(train_loop,
    #                         static_model=CatBoostClassifier(), dynamic_model=CatBoostClassifier(), comb_model=CatBoostClassifier(), static_preprocessor=static_ohe_obj, dynamic_preprocessor=dyn_ohe_obj)
    #         pool.map(lambda x: partial(train_loop, CatBoostClassifier(), CatBoostClassifier(), CatBoostClassifier()))
    #     for t in [60, 240, 600]:
    #         Xtr_t = Xtr.filter(pl.col('minutes')<=t)
    #         Xte_t = Xte.filter(pl.col('minutes')<=t)
    #         train_loop(CatBoostClassifier(), CatBoostClassifier(), CatBoostClassifier(),
    #                    Xtr_t, Xte_t, "PAT_ENC_CSN_ID", static_ohe_obj, dyn_ohe_obj)
    #     break
    # print(f'time taken to process [60, 240, 600] sequentially is {time.time()-t} seconds')

