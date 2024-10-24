import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import polars as pl
import numpy as np
from const import constants
import datetime as dt
import joblib
import time
from warnings import warn

class RWBTransformer:
    def __init__(self, rwb_orig_mapping, zc_dir, zc_dict: dict[str, str]):
        self.rwb_orig_mapping = rwb_orig_mapping
        self.zc_dict = zc_dict
        self.zc_dir = zc_dir
        
    # def fit(self, X_rwb: pl.DataFrame, X_orig:pl.DataFrame, y=None): 
    def fit(self, X_rwb: pl.DataFrame, y=None): 
        self.rwb_schema_ = X_rwb.schema
        # self.orig = X_orig.schema

    def transform(self, X_rwb: pl.DataFrame, y=None):
        # Parse patient age and sex
        pattern_age = r"(\d{1,3})Y/O"  # Extracts the age
        pattern_sex = r"Y/O (\w)"  # Extracts the sex

        X_rwb = X_rwb.with_columns([
            pl.col("Patient Name/Age/Gender").str.extract(pattern_age, 1).cast(pl.Int64).alias("Patient_Age"),  # Extract age and cast to integer
            pl.col("Patient Name/Age/Gender").str.extract(pattern_sex, 1).alias("Sex")  # Extract sex
        ])

        X_rwb = X_rwb.with_columns(
            pl.col('Patient_Age').cast(pl.Float64)  # Cast to float
        )
        
        # Convert Arrived datetime col
        # start_time = pl.datetime(1840, 12, 31, 0, 0, 0)
        start_time = dt.datetime(1840, 12, 31, 0, 0, 0)

        # Add the timedelta to the 'Arrived' column (assuming it's in seconds)
        X_rwb = X_rwb.with_columns(
            pl.col("Arrived").str.replace(r'\[CDT', '').alias("Arrived")
        )

        # Add the timedelta to the 'Arrived' column (assuming it's in seconds)
        X_rwb = X_rwb.with_columns(
            (pl.lit(start_time) + pl.col('Arrived').cast(pl.Int64) * pl.duration(seconds=1)).cast(pl.Datetime).alias('Arrived_Time')
        )

        X_rwb = X_rwb.with_columns(
            pl.when(pl.col("Race").is_not_null())  # Check if not null
            .then(pl.col("Race").str.split("\n").list.first())  # Split by '\n' and get the first element
            .otherwise(pl.col("Race"))  # If null, keep the original value
            .alias("Race")  # Assign it back to the column
        )

        # Drop raw cols
        X_rwb = X_rwb.drop(columns=['Acuity', 'Arrival Time', 'Patient Name/Age/Gender', 'Arrived'])

        # Transform columns according to zc tables
        for colname, zc_path in self.zc_dict.items():
            df_zc = pl.read_csv(os.path.join(self.zc_dir, zc_path))
            col_dict = dict(zip(df_zc['INTERNAL_ID'].to_list(), df_zc['NAME'].to_list()))
            X_rwb = X_rwb.with_columns(
                pl.when(pl.col(colname).is_not_null())
                .then(pl.col(colname).cast(pl.Int64))
                .otherwise(-1)
                .map_dict(col_dict, default="unknown")
                .alias(colname)
            )

        X_rwb = X_rwb.rename(self.rwb_orig_mapping)
        return X_rwb 

    def fit_transform(self, X, y=0):
        self.fit(X, y)
        return self.transform(X, y)

        
if __name__ == "__main__":
    rename_dict = {
        'Race': 'FirstRace',
        'Acuity Abbr': 'Acuity_Level',
        'Arrival Method': 'Means_Of_Arrival',
        'CC': 'Chief_Complaint_All',
        'Primary FC': 'Coverage_Financial_Class_Grouper',
        'Primary Dx': 'Primary_DX_Name',
        'ED Disposition': 'ED_Disposition',
        '': 'index',
        'CSN': 'PAT_ENC_CSN_ID'
    }
    ZC_tables_dir = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/zc_tables/'

    zc_dict = {
        'Ethnicity':  'ClarityMirror_dbo_ZC_ETHNIC_GROUP.csv',
        'Race':  'ClarityMirror_dbo_ZC_PATIENT_RACE.csv',
        'Arrival Method':  'ClarityMirror_dbo_ZC_ARRIV_MEANS.csv',
        'Primary FC':  'ClarityMirror_dbo_ZC_FINANCIAL_CLASS.csv'
    }
    
    rwb_trans = RWBTransformer(rename_dict, ZC_tables_dir, zc_dict)
    
    
    RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ondemand-RWB1.csv'
    df_rwb = pl.read_csv(RAW_DATA, infer_schema_length=int(1e5), dtypes={'Arrived': pl.String})
    rwb_trans.fit(df_rwb)
    df_rwb_transform = rwb_trans.transform(df_rwb)
    df_o = pl.read_parquet('/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 6.6.24.parquet')
    df_os = df_o.select(['PAT_ENC_CSN_ID', 'FirstRace', 'Acuity_Level', 'Means_Of_Arrival', 'Arrived_Time', 'Chief_Complaint_All', 'Ethnicity',
             'Sex', 'Patient_Age', 'Coverage_Financial_Class_Grouper', 'Primary_DX_Name', 'ED_Disposition'])
    
    df_agg_s = df_os.group_by('PAT_ENC_CSN_ID').agg(
        pl.col('FirstRace').last().alias('FirstRace'),
        pl.col('Acuity_Level').last().alias('Acuity_Level'),
        pl.col('Means_Of_Arrival').last().alias('Means_Of_Arrival'),
        pl.col('Arrived_Time').last().alias('Arrived_Time'),
        pl.col('Chief_Complaint_All').last().alias('Chief_Complaint_All'),
        pl.col('Ethnicity').last().alias('Ethnicity'),
        pl.col('Sex').last().alias('Sex'),
        pl.col('Patient_Age').last().alias('Patient_Age'),
        pl.col('Coverage_Financial_Class_Grouper').last().alias('Coverage_Financial_Class_Grouper'),
        pl.col('Primary_DX_Name').last().alias('Primary_DX_Name'),
        pl.col('ED_Disposition').last().alias('ED_Disposition')
    )
    df_agg_s = df_agg_s.with_columns([pl.arange(1, len(df_agg_s)+1,1).alias('index'), pl.lit(0).cast(pl.Int64).alias('from_rwb')])
    df_agg_s = df_agg_s.with_columns(
        pl.col('Arrived_Time').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
    )

    dd = pl.concat([df_agg_s, df_rwb_transform])
    
    x = 0
    