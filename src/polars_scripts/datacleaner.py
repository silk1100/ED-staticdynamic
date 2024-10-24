import os
import sys

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
from typing import Union
import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar

USCal = USFederalHolidayCalendar()


class DataCleaner:
    def __init__(self, file: Union[str, pl.DataFrame, pd.DataFrame], analysis_type='comb', dx_col='Primary_DX_ICD10', target='ED_Disposition', multi_col_separator=',') -> None:
       self.original = file 
       self.target = target
       self.type = analysis_type
       self.dx_col = dx_col
       self.separator = multi_col_separator

    def __read_file(self, file):
        if isinstance(file, str):
            if file.endswith('.csv'):
                df = pl.read_csv(file,
                    infer_schema_length=int(20e6),
                    null_values='NULL',
                    dtypes=dict(Patient_Age=pl.Float64)
                )
            elif file.endswith('.parquet'):
                df = pl.read_parquet(file)
        elif isinstance(file, pd.DataFrame):
            df = pl.from_pandas(file)
        elif isinstance(file, pl.DataFrame):
            df = file
        else:
            raise ValueError(f'file is expected to be either [str, pl.DataFrame, pd.DataFrame] ...')
        df = df.filter(~pl.col(self.target).is_in(['Dismiss - Incorrect Chart', 'Send to L&D', 'LWBS']))
        df = self.__category_mappers_static(df)
        return df

    def __preprocess_timecols(self, df):
        if df['Arrived_Time'].dtype == pl.String:  
            df = df.with_columns([
                pl.col('Arrived_Time').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
                # pl.col('Calculated_DateTime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
            ])
    
        if self.type in ['comb', 'dynamic']:
            if df['Calculated_DateTime'].dtype == pl.String:
                df = df.with_columns([
                    # pl.col('Arrived_Time').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
                    pl.col('Calculated_DateTime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
                ])  
        
            df = df.sort(by=['PAT_ENC_CSN_ID', 'Calculated_DateTime'])
        return df
        
    def __category_mappers_static(self, df):
        if 'Procedure in the Last 4 Weeks' in df.columns:
            binary_cols = [c for c in df.columns if 'has' in c.lower() or 'yn' in c.lower()]+['Procedure in the Last 4 Weeks']+\
                [c for c in df.columns if df[c].n_unique() == 2]
        else:
            binary_cols = [c for c in df.columns if 'has' in c.lower() or 'yn' in c.lower()]+\
                [c for c in df.columns if df[c].n_unique() == 2]
        number_cols = [c for c in df.columns if 'number' in c.lower()] 


        eth_exp = [pl.when( (pl.col('Ethnicity').is_in(['Declinded', '*Unspecified'])|(pl.col('Ethnicity').is_null()) )).\
            then(pl.lit("Unknown")).otherwise(pl.col('Ethnicity')).alias('Ethnicity')]

        binary_col_expr = []
        for col in binary_cols:
            if df[col].dtype != pl.String: continue
            binary_col_expr.append(
                pl.when(pl.col(col).str.to_lowercase().is_in(["y",'yes'])).then(pl.lit(1)).\
                    when(pl.col(col).str.to_lowercase().is_in(["n",'no'])).then(pl.lit(0)).\
                        otherwise(pl.lit(None)).cast(pl.UInt8).alias(f'{col}')
            )

        num_col_expr = []
        for col in number_cols:
            num_col_expr.append(
                pl.col(col).cast(pl.UInt16)
            )
        
        moa_expr = [pl.when(pl.col('Means_Of_Arrival').is_in(['NULL', 'Other'])).then(pl.lit(None)).\
            otherwise(pl.col('Means_Of_Arrival')).alias('Means_Of_Arrival')]

        firstrace_exp = [pl.when(pl.col('FirstRace').is_in(['Unavailable/Unknown', 'Declined'])).then(pl.lit(None)).\
            otherwise(pl.col('FirstRace')).alias('FirstRace')]

        all_expr = eth_exp+binary_col_expr+num_col_expr+moa_expr+firstrace_exp
        df = df.with_columns(all_expr)
        
        return df
    
    def __trim_frame(self, df):
        df_trim = df.with_columns(
            (pl.col('Type')=='Order - Admission').cum_sum().over('PAT_ENC_CSN_ID').alias('n_admit_orders'),
            ( (pl.col('Type')=='Order - Discharge')|(pl.col("EVENT_NAME").is_in([
                    'AVS Printed',
                    'Discharge AVS Print Snapshot',
                    'Discharge Status Change',
                    'FAM Discharge AVS Print Snapshot'
                ]))).cum_sum().over('PAT_ENC_CSN_ID').alias('n_disch_orders'),
        ).filter( (pl.col('n_admit_orders')<1) & (pl.col('n_disch_orders')<1) )
        return df_trim
    
    def __create_features(self, df):
        if 'Type' in df.columns:
            df = df.with_columns(
                (pl.col("Type") == 'Order - Admission').sum().over('PAT_ENC_CSN_ID').alias('admit-order-cnt')
            ).with_columns(
                pl.when(pl.col('admit-order-cnt')==0).
                then(0).otherwise(1).alias('has_admit_order')
            ).drop(columns=['admit-order-cnt'])

        if self.type in ['comb', 'dynamic']:
            df = self.__trim_frame(df)
        
            df = df.with_columns([
                ((pl.col('Calculated_DateTime')-pl.col('Arrived_Time')).dt.total_seconds().cast(pl.Float32)/60.0).alias('elapsed_time_min')
            ])

            df = df.with_columns(
                pl.int_range(pl.len(), dtype=pl.UInt32).over('PAT_ENC_CSN_ID').alias('event_idx')
            )

            df = df.with_columns(
                pl.col('elapsed_time_min').last().over('PAT_ENC_CSN_ID').alias('tta'),
                pl.col('event_idx').last().over('PAT_ENC_CSN_ID').alias('eidx_ed')
            )

            df = df.with_columns(
                [
                    pl.col("EVENT_NAME").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').alias('EVENT_NAME_NORM'),
                    pl.col("Type").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').alias('Type_NORM'),
                ] 
            ).with_columns(
                (pl.col('Type_NORM')+'_'+pl.col('EVENT_NAME_NORM')).alias('type_name')
            )
            # Update the vital ranges
            for k, v in constants.vital_ranges_dict.items():
                df = df.with_columns(
                    pl.when((pl.col('EVENT_NAME') == k)&((pl.col("MEAS_VALUE")>v[1])|(pl.col("MEAS_VALUE")<v[0])))
                    .then(pl.lit(None)).otherwise(pl.col("MEAS_VALUE")).alias("MEAS_VALUE_UP")
                )

            # Split MEAS_VALUE based on the different vitals (11 fields)
            uniq_vitals = df.filter(pl.col('Type') == 'Vitals')['EVENT_NAME'].unique()
            df = df.with_columns(
                [
                    pl.when(
                        (pl.col("EVENT_NAME")==uq)&(pl.col("MEAS_VALUE_UP").is_not_null())
                    ).then(
                        pl.col("MEAS_VALUE_UP")
                    ).otherwise(pl.lit(None)).alias(f'MEAS_VALUE_{uq}')
                    for uq in uniq_vitals
                ]
            )

            # Split Result_Flag based on different Types
            uniq_flags = df.filter(~pl.col("Result_Flag").is_null())['Result_Flag'].unique()
            df = df.with_columns(
                pl.when(
                    pl.col('Result_Flag')==val
                ).then(
                    pl.lit(1)
                ).otherwise(0).cast(pl.UInt16).alias(f'Result_Flag_{val}')
                for val in uniq_flags
            )

            # Split Order_Status based on different Types
            uniq_flags = df.filter(~pl.col("Order_Status").is_null())['Order_Status'].unique()
            df = df.with_columns(
                pl.when(
                    pl.col('Order_Status')==val
                ).then(
                    pl.lit(1)
                ).otherwise(0).cast(pl.UInt16).alias(f'Order_Status_{val}')
                for val in uniq_flags
            )
        ### =============== End of comb and dynamic preprocessing =============== ###

        df = df.with_columns(
            [
                pl.col("Chief_Complaint_All").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').
                    map_elements(lambda x: x.split(self.separator), return_dtype=pl.List(pl.String)).alias('cc_list'),

                #TODO: Until I fix it in the config.yaml to separate static and dynamic features [Primary_DX_Name is hardcoded from RWB as a static feature for now]
                pl.col('Primary_DX_Name').str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+', '').map_elements(lambda x: x.split(self.separator), return_dtype=pl.List(pl.String)).alias('dx_list'),

                # pl.col("Primary_DX_Name").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').
                #     map_elements(lambda x: x.split(','), return_dtype=pl.List(pl.String)).alias('dx_list'),

                pl.col(self.dx_col).map_elements(lambda x: x.split(','), return_dtype=pl.List(pl.String)).
                    alias('dxcode_list'),
                pl.col('Arrived_Time').dt.day().alias('arr_day'),
                pl.col('Arrived_Time').dt.weekday().alias('arr_dow'),
                pl.col('Arrived_Time').dt.year().alias('arr_year'),
                pl.col('Arrived_Time').dt.month().alias('arr_month'),
                pl.col('Arrived_Time').dt.hour().alias('arr_hour'),
                # pl.col('Arrived_Time').map_elements(lambda x: get_holiday(x.date()), return_dtype=pl.String).alias('holiday')
            ]
        ) 

        df = self.__get_holiday_optimized(df, 'Arrived_Time')
        
        return df
    
    def __get_holiday_optimized(self, df, date_col):
        newdatecol = f'{date_col}_date'
        df = df.with_columns(
            pl.col(date_col).dt.date().alias(newdatecol)
        )
        start_date = df[newdatecol].to_pandas().min()
        end_date = df[newdatecol].to_pandas().max()
        
        df_holiday = USCal.holidays(start_date, end_date, return_name=True).to_frame().reset_index()
        df_holiday.columns = [newdatecol, 'holiday']
        df_holiday[newdatecol] = df_holiday[newdatecol].dt.date
        df_holiday = pl.DataFrame(df_holiday)
        df = df.join(df_holiday, on=newdatecol, how='outer')
        df = df.drop(newdatecol)
        if 'holiday_right' in df.columns:
            df = df.rename({'holiday_right': 'holiday'})
        return df
    
    def run(self):
        df = self.__read_file(self.original)
        df = self.__preprocess_timecols(df)
        df = self.__create_features(df)
        return df

if __name__ == "__main__":
    dc = DataCleaner(constants.RAW_DATA)
    df = dc.run()
    x =0