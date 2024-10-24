import os
import sys


# MAIN_DIR = os.getenv("EDStaticDynamic")

MAIN_DIR = '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src'
sys.path.insert(1, MAIN_DIR)

import polars as pl
import numpy as np
from const import constants
import joblib
import time
from warnings import warn
from polars_scripts import static_transformers 
from polars_scripts import dynamic_transformers 
from sklearn.base import TransformerMixin, BaseEstimator
from datamanager import CustomCrossFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.exceptions import NotFittedError


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 static_preproc_dict,
                 dynamic_preproc_dict,
                 id_col='PAT_ENC_CSN_ID',
                 target_col='has_admit_order'):
        '''
        static_ohe_obj = CustomOneHotEncoding(
            single_val_cols=constants.static_singleval_cat_cols,
            multi_val_cols=constants.static_multval_cat_cols,
            num_cols=constants.static_num_cols,
            num_norm_method=constants.static_num_norm_method,
            null_vals=constants.NULL_LIST,
            vocabthresh=100,
            cumprob_inc_thresh=0.99
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
        self.pat_id = id_col
        self.target_col   = target_col
        if static_preproc_dict is not None:
            self.static_params = static_preproc_dict
            self.static_preprocess_method = static_preproc_dict.get('method', 'ohe')
            self.static_cols  = static_preproc_dict['single_val_cols']+static_preproc_dict['multi_val_cols']\
                +static_preproc_dict['num_cols']+self._flatten_indep_dep(static_preproc_dict['dep_col_dict'])
            self.staticobj = self._create_staticobj_fromdict(self.static_params)
        else:
            self.static_obj = None
            self.static_params = None
            self.static_preprocess_method = None
            self.static_cols = None

        if dynamic_preproc_dict is not None:
            self.dynamic_params = dynamic_preproc_dict 
            self.dynamic_preprocess_method = dynamic_preproc_dict.get('method', 'ohe')
            self.dynamic_cols = dynamic_preproc_dict['single_val_cols']+dynamic_preproc_dict['multi_val_cols']\
                +dynamic_preproc_dict['num_cols']+self._flatten_indep_dep(dynamic_preproc_dict['dep_col_dict'])
            self.dynamicobj = self._create_dynamicobj_fromdict(self.dynamic_params)
        else:
            self.dynamic_params = None
            self.dynamic_cols = None
            self.dynamic_preprocess_method = None
            self.dynamic_obj = None
            

    def _flatten_indep_dep(self, indep_dep_dict):
        dep_indep_cols = []
        if len(indep_dep_dict)>0:
            keys = list(indep_dep_dict.keys())
            vals = []
            for v in indep_dep_dict.values():
                vals.extend(v)
            dep_indep_cols.extend(keys)
            dep_indep_cols.extend(vals)
        return dep_indep_cols

    def _create_staticobj_fromdict(self, static_preproc_dict):
        self.static_preprocess_method = static_preproc_dict.get('method', 'ohe') # 'ohe', 'le'

        if 'method' in static_preproc_dict:
            del static_preproc_dict['method']

        if self.static_preprocess_method == 'ohe':
            self.static_obj = self._set_static_ohe(static_preproc_dict)
        elif self.static_preprocess_method == 'le':
            self.static_obj = self._set_static_le(static_preproc_dict)
        else:
            raise ValueError(f'Static Preprocessor only support `method in ["le", "ohe"]` ... {self.static_preprocess_method} is given ...')

    def _create_dynamicobj_fromdict(self, dynamic_param_dict):
        self.dynamic_preprocess_method = dynamic_param_dict.get('method', 'ohe') # 'ohe', 'le'

        if 'method' in dynamic_param_dict:
            del dynamic_param_dict['method']

        if self.dynamic_preprocess_method == 'ohe':
            self.dynamic_obj = self._set_dynamic_ohe(dynamic_param_dict)
        elif self.dynamic_preprocess_method == 'le':
            self.dynamic_obj = self._set_dynamic_le(dynamic_param_dict)
        else:
            raise ValueError(f'Dynamic Preprocessor only support `method in ["le", "ohe"]` ... {self.static_preprocess_method} is given ...')

    def _set_static_ohe(self, param_dict):
        return static_transformers.CustomOneHotEncoding(**param_dict)

    def _set_static_le(self, param_dict):
        return static_transformers.CustomOneHotEncoding(**param_dict)

    def _set_dynamic_ohe(self, param_dict):
        return dynamic_transformers.CustomDynamicOneHotEncoding(**param_dict)

    def _set_dynamic_le(self, param_dict):
        return dynamic_transformers.CustomDynamicLabelEncoding(**param_dict)


    def get_static_feats(self, X, pat_id):
        # X_static_tr = X.select(constants.id_cols+constants.static_cols+constants.target_col)
        X_static_tr = X.select([self.pat_id]+self.static_cols+[self.target_col])
        X_pat_static = X_static_tr.group_by(pat_id).agg(
            [
                pl.col(c).last().alias(c) for c in X_static_tr.columns if c != pat_id
            ]
        )
        return X_pat_static
    
    def get_dynamic_feats(self, X):
        # X_dynamic_tr = X.select(constants.id_cols+constants.dynamic_cols+constants.target_col)
        X_dynamic_tr = X.select([self.pat_id]+self.dynamic_cols+[self.target_col])
        return X_dynamic_tr


    def _prepare_data_dict(self, X_dict:pl.DataFrame, y=None):
        if self.static_cols is not None:
            X_static = self.get_static_feats(X_dict, self.pat_id)
        else:
            X_static = None
        if self.dynamic_cols is not None:
            X_dynamic = self.get_dynamic_feats(X_dict)
        else:
            X_dynamic = None
        return X_static, X_dynamic

    def fit(self, Xtrain:pl.DataFrame, y=None):
        X_static_tr, X_dynamic_tr = self._prepare_data_dict(Xtrain)
        if X_static_tr is not None:
            self.static_obj.fit(X_static_tr)
        if X_dynamic_tr is not None:
            self.dynamic_obj.fit(X_dynamic_tr)
                
        return self

    def transform(self, Xtest:pl.DataFrame, y=None):
        X_static_te, X_dynamic_te = self._prepare_data_dict(Xtest)
        if self.static_obj is not None:
            Xpre_static_te = self.static_obj.transform(X_static_te)
        else:
            Xpre_static_te = None
        if self.dynamic_obj is not None:
            Xpre_dynamic_te = self.dynamic_obj.transform(X_dynamic_te)
        else:
            Xpre_dynamic_te = None

        if Xpre_dynamic_te is not None:
            Xdte = pl.concat([X_dynamic_te.drop(["Type_NORM", "EVENT_NAME_NORM", "dxcode_list"]),
                        Xpre_dynamic_te], how='horizontal')
            ncols = set(self.dynamic_obj.num_cols + [f'{c}_NUMNORM' for c in self.dynamic_obj.num_cols])

            # Dynamic aggregator
            elapsed_time_expr = None
            if 'elapsed_time_min_NUMNORM' in Xdte.columns:
                elapsed_time_expr = pl.col('elapsed_time_min_NUMNORM').last().alias('elapsed_time_min_NUMNORM')
            elif 'elapsed_time_min':
                elapsed_time_expr = pl.col('elapsed_time_min').last().alias('elapsed_time_min')

            event_idx_expr = None
            if '' in Xdte.columns:
                event_idx_expr = pl.col('event_idx_NUMNORM').last().alias('event_idx_NUMNORM')
            elif 'elapsed_time_min':
                event_idx_expr = pl.col('event_idx').last().alias('event_idx')
            
            # MEAS_VALUE expr list
            meas_cols = [c for c in Xdte.columns if c.startswith("MEAS_VALUE") and c.endswith("_NUMNORM")]    
            if len(meas_cols) == 0:
                meas_cols = [c for c in Xdte.columns if c.startswith("MEAS_VALUE")]    
            meas_expr = [pl.col(c).last().alias(c) for c in meas_cols]        

            expr_list = []
            if elapsed_time_expr is not None:
                expr_list.append(elapsed_time_expr)
            if event_idx_expr is not None:
                expr_list.append(event_idx_expr)
            
            Xdte_pat = Xdte.group_by('PAT_ENC_CSN_ID').agg(
                [pl.col(c).sum().alias(c) for c in Xpre_dynamic_te.columns if c not in ncols]+[
                    pl.col('ED_Location_YN').mean().alias('ED_Location_YN'),
                    pl.col('has_admit_order').last().alias('has_admit_order')
                ]+expr_list+
                meas_expr
                +[
                    pl.col(c).sum().alias(c) for c in X_dynamic_te.columns if c.startswith('Order')
                ]+[
                    pl.col(c).sum().alias(c) for c in X_dynamic_te.columns if c.startswith('Result')
                ]
            )
        else:
            Xdte_pat = None

        if Xpre_static_te is not None:
            Xste = pl.concat([X_static_te.drop(["Ethnicity", "FirstRace", "Sex", "Acuity_Level",
                                        "Means_Of_Arrival", "Coverage_Financial_Class_Grouper",
                                        "Arrived_Time", 'cc_list',
                                        "arr_month", "arr_day", "arr_dow", "holiday", "arr_hour", "dx_list"]+self.static_obj.num_cols), Xpre_static_te], how='horizontal')
        else:
            Xste = None
        

        
        if Xste is not None and Xdte_pat is not None:
            Xste, Xdte_pat = pl.align_frames(Xste, Xdte_pat, on=self.pat_id)
            
        return dict(static=Xste, dynamic=Xdte_pat)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class Imputer(TransformerMixin, BaseEstimator):
    def __init__(self, imputation_dict={}, impute_norm=False, id_col='PAT_ENC_CSN_ID', target_col='has_admit_order'):
        self.imputation_dict = imputation_dict
        self.impute_norm = impute_norm
        self.id_col = id_col 
        self.target_col = target_col 
    
    def _get_impute_val(self, X, c, imputation_method):
        if imputation_method == 'median':
            return X[c].median() if X[c].median() is not None else 0
        elif imputation_method == 'mean':
            return X[c].mean() if X[c].mean() is not None else 0
        elif imputation_method == 'max':
            return X[c].max() if X[c].max() is not None else 0
        elif imputation_method == 'min':
            return X[c].min() if X[c].min() is not None else 0
        elif imputation_method == 'first':
            return X[c][0] if X[c][0] is not None else 0
        elif imputation_method == 'last':
            return X[c][-1] if X[c][-1] is not None else 0
        elif imputation_method in ['most_common', 'most common', 'mode']:
            return X[c].mode() if X[c].mode() is not None else 0
        else:
            raise ValueError(f"Imputation method can be one of the following: ['median', 'mean', 'min', 'max', 'first', 'last', 'mode'] ...")
    
    def fit(self, X, y=None):
        null_vocab = {}
        cols_list = set(X.columns)
        self.imputed_cols_ = []
        for c, imp_method in self.imputation_dict.items():
            col2impute = None
            if self.impute_norm:
                if c+'_NUMNORM' in cols_list and X[c+'_NUMNORM'].is_null().sum()>0:
                    col2impute = c+'_NUMNORM'
                elif c in cols_list and X[c].is_null().sum()>0:
                    col2impute = c
                elif c not in cols_list and c+'_NUMNORM' not in cols_list:
                    raise ValueError(f"Neither {c} nor {c}_NUMNORM are found in data columns: {cols_list}")
                if col2impute:
                    null_vocab[col2impute] = self._get_impute_val(X, c+'_NUMNORM', imp_method)
                    self.imputed_cols_.append(col2impute)
            else:
                if c not in cols_list:
                    raise ValueError(f"{c} is not found in data columns: {cols_list}")
                n_nulls = X[c].is_null().sum() 
                if n_nulls > 0:
                    print(f'{c} has {n_nulls} null vals ...')
                    null_vocab[c] = self._get_impute_val(X, c, imp_method)
                    self.imputed_cols_.append(col2impute)

        self.params_ = null_vocab
        return self
    
    def transform(self, X, y=None):
        for c, val in self.params_.items():
            X = X.with_columns(
                pl.col(c).fill_null(value=val).alias(c)
            )
        return X   

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

class ImputerPipeline(TransformerMixin, BaseEstimator):
    def __init__(self, preprocessor: Preprocessor, simputation_dict, dimputation_dict, id_col='PAT_ENC_CSN_ID', target_col='has_admit_order'):
        self.simputation_dict = simputation_dict
        self.dimputation_dict = dimputation_dict
        self.id_col = id_col
        self.target = target_col
        self.preprocessor = preprocessor
        self.pipeline = FeatureUnion([
            ("static", Pipeline([
                ("imputer", Imputer(simputation_dict, id_col=id_col, target_col=target_col))
            ])),
            ("dynamic", Pipeline([
                ("imputer", Imputer(dimputation_dict, id_col=id_col, target_col=target_col))
            ]))
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.pipeline.fit_transform(X)

class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, static_preproc_dict, dynamic_preproc_dict, simputation_dict, dimputation_dict, id_col='PAT_ENC_CSN_ID', target_col='has_admit_order'):
        self.preprocessor = Preprocessor(
            static_preproc_dict=static_preproc_dict,
            dynamic_preproc_dict=dynamic_preproc_dict
        )
        self.imputer_pipeline = ImputerPipeline(self.preprocessor, simputation_dict, dimputation_dict, id_col, target_col)

    def fit(self, X, y=None):
        self.imputer_pipeline.fit(X)
        return self

    def transform(self, X, y=None):
        return self.imputer_pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.imputer_pipeline.fit_transform(X)
        
    
class CascadePipeline:
    def __init__(self, static_dict, dynamic_dict, simp_dict=None, dimp_dict=None,
                 id_col='PAT_ENC_CSN_ID', target_col='has_admit_order', impute_normcols=False):
        self.static_dict = static_dict
        self.dynamic_dict = dynamic_dict 
        self.simp_dict = simp_dict 
        self.dimp_dict = dimp_dict
        self.impute_normcols = impute_normcols

        self.preprocessor = Preprocessor(static_dict, dynamic_dict, id_col, target_col)
        if self.simp_dict is not None:
            self.simputer = Imputer(simp_dict, id_col, target_col)
        else:
            self.simputer = None
        if self.dimp_dict is not None:
            self.dimputer = Imputer(dimp_dict, id_col, target_col)
        else:
            self.dimputer = None
        
        self.is_fitted = False

    def train_fit_transform(self, X, y=None):
        self.preprocessor.fit(X, y)
        out = self.preprocessor.transform(X, y)

        if out['static'] is not None:
            Xstr = self.simputer.fit_transform(out['static'])
        else:
            Xstr = None
        if out['dynamic'] is not None:
            Xdtr = self.dimputer.fit_transform(out['dynamic'])
        else:
            Xdtr = None

        self.is_fitted=True
        return {
            'out':out, # Preprocessed without imputer
            'static': Xstr, # Static imputed
            'dynamic': Xdtr  # Dynamic imputed
        }

    def eval_transform(self, X, y=None):
        if not self.is_fitted:
            raise NotFittedError(f'You need to run train_fit_transform() first ...')
        out = self.preprocessor.transform(X, y)
        if out['static'] is not None:
            Xstr = self.simputer.transform(out['static'])
        else:
            Xstr = None

        if out['dynamic'] is not None:
            Xdtr = self.dimputer.transform(out['dynamic'])
        else:
            Xdtr = None
            
        return {
            'out':out, # Preprocessed without imputer
            'static': Xstr, # Static imputed
            'dynamic': Xdtr  # Dynamic imputed
        }


if __name__ == "__main__":
    df = pl.read_parquet(constants.CLEAN_DATA_PARQUET)

    df = df.with_columns(
        (pl.col("Calculated_DateTime")-pl.col("Calculated_DateTime").first()).dt.total_minutes().over('PAT_ENC_CSN_ID').alias('minutes')
    )
    
    ccf = CustomCrossFold(30*6, 30*6, 30*6, 'Arrived_Time')
    time_range = np.arange(30, 12*60+30, 30)
    
    static_dict = dict(
        method='ohe',
        single_val_cols=constants.static_singleval_cat_cols,
        multi_val_cols=constants.static_multval_cat_cols,
        num_cols=constants.static_num_cols,
        num_norm_method=constants.static_num_norm_method,
        null_vals=constants.NULL_LIST,
        dep_col_dict={},
        vocabthresh=100,
        cumprob_inc_thresh=0.99
    )
    if 'Type_NORM' in constants.dynamic_singleval_col:
        constants.dynamic_singleval_col.remove('Type_NORM')
    if 'EVENT_NAME_NORM' in constants.dynamic_singleval_col:
        constants.dynamic_singleval_col.remove('EVENT_NAME_NORM')
    if 'Type_NORM' in constants.dynamic_multival_col:
        constants.dynamic_multival_col.remove('Type_NORM')
    if 'EVENT_NAME_NORM' in constants.dynamic_multival_col:
        constants.dynamic_multival_col.remove('EVENT_NAME_NORM')
    dynamic_dict = dict(
        method='ohe',
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
    
    
    preproc = Preprocessor(static_dict, dynamic_dict)
    for Xtr, Xte in ccf.split(df):
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
            tr_n = int(0.05*Xtr_t['PAT_ENC_CSN_ID'].n_unique())
            te_n = int(0.05*Xte_t['PAT_ENC_CSN_ID'].n_unique())
            tr_pat = np.random.choice(Xtr_t['PAT_ENC_CSN_ID'].unique(), tr_n, replace=False)
            te_pat = np.random.choice(Xte_t['PAT_ENC_CSN_ID'].unique(), te_n, replace=False)
            Xtr_t = Xtr_t.filter(pl.col("PAT_ENC_CSN_ID").is_in(tr_pat))
            Xte_t = Xte_t.filter(pl.col("PAT_ENC_CSN_ID").is_in(te_pat))
            
            # Get the numeric columns
            snum_cols = []
            for c in preproc.static_cols:
                if  Xtr_t[c].dtype not in [pl.String, pl.Date, pl.Datetime, pl.List, pl.Array]\
                    and c not in ['arr_day', 'arr_hour', 'arr_month', 'arr_dow']:#\
                    # and 'pat' not in c.lower() and 'id' not in c.lower() and 'admit' not in c.lower()\
                    # and 'disch' not in c.lower():
                    snum_cols.append(c)

            dnum_cols = []
            for c in preproc.dynamic_cols:
                if  Xtr_t[c].dtype not in [pl.String, pl.Date, pl.Datetime, pl.List, pl.Array]:#\
                    # and 'pat' not in c.lower() and 'id' not in c.lower() and 'admit' not in c.lower()\
                    # and 'disch' not in c.lower():
                    dnum_cols.append(c)


            # Feature
            # Use this if you do not want to impute nulls
            Xtrpre = preproc.fit_transform(Xtr)
            Xtepre = preproc.transform(Xte)

            # Use this if you want to preprocess and impute nulls
            pipeline = CascadePipeline(
                static_dict=static_dict,
                dynamic_dict=dynamic_dict,
                simp_dict={c: 'median' for c in snum_cols},
                dimp_dict={c: 'median' for c in dnum_cols},
                id_col = 'PAT_ENC_CSN_ID',
                target_col='has_admit_order'
            )
            tr_out = pipeline.train_fit_transform(Xtr)
            out = pipeline.eval_transform(Xte)
            x = 0

    x = 0