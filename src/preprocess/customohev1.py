import sys
sys.path.insert(1, "/home2/s223850/ED/UTSW_ED_EventBased-optimizedEventPreproc/src")

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from utils.utils import count_cat_fast
from preprocess.reductor import Reductor


class ReducedHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col, cnt_fnc, grb_col='PAT_ENC_CSN_ID', time_idx='Calculated_DateTime', red_thr=0.95, rare_name=None, none_name=None):
        self.col = col
        self.cnt_fnc = cnt_fnc
        self.red_thr = red_thr
        self.grb_col = grb_col
        self.time_idx = time_idx
        
        self.rare_name = self._handle_rare_name(rare_name) 
        self.none_name = self._handle_none_name(none_name) 
    
    def _handle_rare_name(self, rare_name):
        if isinstance(self.col, str):
            self.rare_name = rare_name if rare_name is not None else f"rare_{self.col}"
        elif isinstance(self.col, (tuple, list)):
            if self.rare_name is None:
                self.rare_name = {col: f"rare_{col}" for col in self.col}
            else:
                self.rare_name = rare_name
        return self.rare_name

    def _handle_none_name(self, none_name):
        if isinstance(self.col, str):
            self.none_name = none_name if none_name is not None else f"None_{self.col}"
        elif isinstance(self.col, (tuple, list)):
            if self.none_name is None:
                self.none_name = {col: f"None_{col}" for col in self.col}
            else:
                self.none_name = none_name
        return self.none_name
    
    def _reduce_df(self, X, reduced_dict):
        Xc = X.copy()
        if isinstance(self.col, str):
            Xc.loc[~(X[self.col].isin(reduced_dict[self.col])), self.col] = self.rare_name
        elif isinstance(self.col, (tuple, list)):
            for col in self.col:
                Xc.loc[~(X[col].isin(reduced_dict[col])), col] = self.rare_name[col]
        return Xc

    def _fit_ohe(self, X, cols, reduced_dict):
        Xc = self._reduce_df(X, reduced_dict)
        if isinstance(cols, str):
            ohe = OneHotEncoder(handle_unknown='infrequent_if_exist')
            ohe = ohe.fit(Xc[[cols]])
        elif isinstance(cols, (tuple, list)):
            ohe = {}
            for col in cols:
                ohe[col] = OneHotEncoder(handle_unknown='infrequent_if_exist')
                ohe[col].fit(Xc[[col]])
        return ohe

    def fit(self, X, y=None, **params):
        self.red_ = Reductor(self.col, self.cnt_fnc, self.red_thr, self.grb_col, self.time_idx, self.none_name)
        self.reduced_ = self.red_.reduce(X)
        self.ohe_ = self._fit_ohe(X, self.col, self.reduced_)
        return self 

    def transform(self, X, y=None):
        Xc = self._reduce_df(X, self.reduced_)

        if isinstance(self.col, str):
            Xpre = pd.DataFrame(self.ohe_.transform(Xc[[self.col]]).toarray(), columns=[f'{self.col}_{cat}' for cat in self.ohe_.categories_[0]], index = Xc.index)
        elif isinstance(self.col, (list, tuple)):
            Xpre_list = []
            for col in self.col:
                Xpre = pd.DataFrame(self.ohe_[col].transform(Xc[[col]]), columns=[f'{col}_{cat}' for cat in self.ohe_[col].categories_[0]], index = Xc.index)
                Xpre_list.append(Xpre)
            Xpre = pd.concat(Xpre_list, axis=1)

        return Xpre
    
    def fit_transform(self, X, y=None, **params):
        self.fit(X, y, **params)
        return self.transform(X, y)
        

class CustomOneHotEncoderPreprocessor(BaseEstimator, TransformerMixin):
    '''It accepts 3 types of transformations. Ordinal tranformation, OHE transformation, and reductor+rareevent transformation.
    All of the columns required to be transformed are passed as follow:
    ```
    obj = CustomOneHotEncoderProcessor({
        'Acuity_Level':'ordinal',
        'Ethnicity':'ohe',
        'Primary_DX':('red', 0.95)
    })
    ```
    For the previous call, an object will be created which accepts a datatframe that must contains at least 3 columns with the following names: 'Acuity_Level', 'Ethnicity', and 'Primary_DX' each will be processed in a different way
    '''
    def __init__(self, transform_dict:dict, enc_id='PAT_ENC_CSN_ID', time_id='Calculated_DateTime') -> None:
        super(CustomOneHotEncoderPreprocessor).__init__()
        self.feat_dict = transform_dict #{'colname': 'ordinal', 'colname2': 'ohe', 'colename3':{'cnt_fnc': cnt_fnc, 'none_name': None_col, 'rare_name': RareName, 'red_thr': Thre}}
        self.time_id = time_id
        self.enc_id = enc_id

        self.transform_dict_ = {}
        self.cols_obj_dict_ = {}
        self.encoded_mats_ = {}
        self.cols_red_ = {}
        
        
    def fit(self, X:pd.DataFrame, y=None, **fit_params):
        '''Expected to get the whole dataframe
        '''
        for key, val in self.feat_dict.items():
            assert(key in X.columns)

        for key, val in self.feat_dict.items():
            if isinstance(val, str):
                if 'ordin' in val.lower():
                    self.transform_dict_[key] = self._fit_ordinal(X, key)
                elif 'one' in val.lower() or 'ohe' in val.lower():
                    self.transform_dict_[key] = self._fit_hot(X, key, handle_unknown='infrequent_if_exist')
            elif isinstance(val, dict):
                self.transform_dict_[key] = self._fit_red(X, key, cnt_fnc={key: val['cnt_fnc']} , none_name=val['none_name'], rare_name=val['rare_name'], red_thr=val['red_thr']) # Continue writing the arguments
        return self 

    def _fit_ordinal(self, X, col):
        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=X[col].nunique()+1)
        le.fit(X[[col]])
        return le

    def _fit_hot(self, X, col, **params):
        ohe = OneHotEncoder(**params)
        ohe.fit(X[[col]])
        return ohe

    def _fit_red(self, X, col, **args):
        red = ReducedHotEncoder(col, args['cnt_fnc'], self.enc_id, self.time_id, args['red_thr'], args['rare_name'], args['none_name'])
        red.fit(X)
        return red

    def fit_transform(self, X, y = None, **fit_params): 
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def transform(self, X, y=None):
        # if not check_is_fitted(self, 'cols_obj_dict_'):
        if len(self.transform_dict_) < 1:
            raise NotFittedError("You need to run fit or fit_transform first")

        Xpre = {}
        # X_encid = X.sort_values(by=self.time_id).groupby(self.enc_id).last()
        X_encid = X
        for col, transformer in self.transform_dict_.items():
            if 'ordin' in str(transformer).lower():
                Xpre[col] = pd.DataFrame(transformer.transform(X_encid[[col]]), columns=[f'ordinal_{col}'], index=X_encid.index)
            elif 'reduced' in str(transformer).lower():
                Xpre[col] = transformer.transform(X_encid)
            elif 'onehot' in str(transformer).lower():
                Xpre[col] = pd.DataFrame(transformer.transform(X_encid[[col]]).toarray(), columns = [f'{col}_{cat}' for cat in transformer.categories_[0]], index=X_encid.index)

        return Xpre 
        
if __name__ == "__main__":
    output_path = '/work/InternalMedicine/s223850/intermediate_data_ff_1/preprocessed_test_sample.csv'
    df = pd.read_csv(output_path)
    ohe = CustomOneHotEncoderPreprocessor({'Acuity_Level':'ordinal', 'Primary_DX':{'red_thr':0.95, 'cnt_fnc':count_cat_fast, 'rare_name':'rare_Primary_DX', 'none_name':'None_Primary_DX'}, 'Ethnicity':'ohe'})
    ohe.fit(df)
    Xpre = ohe.transform(df)
    print(df.info())