import os
import sys

MAIN_DIR = os.getenv("EDStaticDynamic")
sys.path.insert(1, MAIN_DIR)

import polars as pl
import numpy as np
from const import constants
import joblib
import datetime as dt
from polars_scripts.datamanager import CustomCrossFold
from polars_scripts.static_transformers import CustomOneHotEncoding
from polars_scripts.dynamic_transformers import CustomDynamicOneHotEncoding
from catboost import CatBoostClassifier

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

def train_loop(
        Xtr_t: pl.DataFrame,
        Xte_t: pl.DataFrame,
        pat_id: str,
        static_preprocessor,
        dynamic_preprocessor
        ):
    X_static_tr = get_static_feats(Xtr_t, pat_id)
    X_static_te = get_static_feats(Xte_t, pat_id)
    X_dynamic_tr = get_dynamic_feats(Xtr_t)
    X_dynamic_te = get_dynamic_feats(Xte_t)
    Xpre_static_tr = static_preprocessor.fit_transform(X_static_tr)
    Xpre_static_te = static_preprocessor.transform(X_static_te)
    Xpre_dynamic_tr = dynamic_preprocessor.fit_transform(X_dynamic_tr)
    Xpre_dynamic_te = dynamic_preprocessor.transform(X_dynamic_te)
    Xdtr = pl.concat([X_dynamic_tr.drop(["Type_NORM", "EVENT_NAME_NORM", "dxcode_list"]),
                      Xpre_dynamic_tr])
    Xdte = pl.concat([X_dynamic_te.drop(["Type_NORM", "EVENT_NAME_NORM", "dxcode_list"]),
                      Xpre_dynamic_te])
    Xstr = pl.concat([X_static_tr.drop(["Ethnicity", "FirstRace", "Sex", "Acuity_Level",
                                        "Means_Of_Arrival", "Coverage_Financial_Class_Grouper",
                                        "Arrived_Time",
                                        "arr_year", "arr_month", "arr_day", "arr_dow"])])
    
    x = 0

    
    


if __name__ == "__main__":
    with open(constants.CLEAN_DATA, 'rb') as f:
        df = joblib.load(f)
    ccf = CustomCrossFold(30*6, 30, 30*2, 'Arrived_Time')
    df = df.with_columns(
        (pl.col("Calculated_DateTime")-pl.col("Calculated_DateTime").first()).dt.minutes().over('PAT_ENC_CSN_ID').alias('minutes')
    )
    time_range = np.arange(30, 60*24*2+30, 30)
    static_ohe_obj = CustomOneHotEncoding(
        single_val_cols=constants.static_singleval_cat_cols,
        multi_val_cols=constants.static_multval_cat_cols,
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
        skip_indp_val={'vitals'},
        vocabthresh=100,
        cumprob_inc_thresh=0.99,
        null_vals=constants.NULL_LIST
    )
    for Xtr, Xte in ccf.split(df):
        for t in time_range:
            Xtr_t = Xtr.filter(pl.col('minutes')<=120)
            Xte_t = Xte.filter(pl.col('minutes')<=120)
            train_loop(Xtr_t, Xte_t, "PAT_ENC_CSN_ID", static_ohe_obj, dyn_ohe_obj)